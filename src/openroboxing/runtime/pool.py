"""Generation for two fighters (M3-T2).

Why two instances rather than ``batch=2``
-----------------------------------------
`WORKPLAN` M3-T2 asks for "one MotionBricks instance serving both fighters with ``batch=2`` and
shared weights". **Upstream cannot do that**, and the reason is in `full_agent.py`, not in the model:

- ``_generate_inbetween_frames`` opens with ``batch_size = 1`` (`:450`) and builds its context
  tensors at that shape.
- ``get_next_frame`` reads ``frames['mujoco_qpos'][0, ...]`` (`:561`) — only ever fighter zero.
- ``generate_new_frames`` truncates with ``num_pred_frames.item()`` (`:167`), which raises on a
  two-element tensor, and would in any case force both fighters to one plan length.

Supporting a real batch means a second patch to upstream, which `CLAUDE.md` invariant 3 makes a
stop-and-ask. Measured, it is not worth it: two independent generators cost **1513 MiB** of a 49 GiB
card, and a replan for both takes **29.6 ms**, which over a 0.5 s replan interval is **1.18 ms per
tick** against a 20 ms budget. The scarce resource batching would save is not scarce.

What is lost is a little latency. What is bought is that fighter isolation is *structural* — two
objects that share no tensors cannot leak into each other — rather than a property to be maintained
across a patched batch dimension. For a competitive game that is the better trade, and it is the one
:func:`GeneratorPool.independence_holds` exists to keep honest.

Conventions
-----------
- Fighters are addressed by **name** (``"red"`` / ``"blue"``), never by index.
- Each fighter gets its **own seed**, so a match is reproducible per fighter and so two fighters
  running the same style do not move identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.generator import (
    GeneratorConfig,
    GeneratorIntent,
    MotionBricksGenerator,
)
from openroboxing.spec.constants import QPOS_DIM

#: Seeds are derived from a single match seed so a whole match reproduces from one number, but must
#: differ per fighter or both fighters shadow-box identically. The offset is arbitrary and stated.
SEED_STRIDE = 1000


class PoolError(RuntimeError):
    """The generator pool could not be built or driven. Never recovered from silently."""


@dataclass
class PoolStats:
    """What the pool cost, for checking against the M1-T7 budget."""

    replans: int = 0
    replan_ms: list[float] = field(default_factory=list)

    @property
    def mean_replan_ms(self) -> float:
        return float(np.mean(self.replan_ms)) if self.replan_ms else 0.0

    def per_tick_ms(self, replan_interval_s: float, tick_hz: float) -> float:
        """Generation cost amortised over the ticks between replans."""
        ticks = max(1.0, replan_interval_s * tick_hz)
        return self.mean_replan_ms / ticks


def fighter_seed(match_seed: int, fighter: str) -> int:
    """A per-fighter seed derived from the match's. Same match seed, same fight."""
    if fighter not in FIGHTERS:
        raise PoolError(f"unknown fighter {fighter!r}; expected one of {FIGHTERS}")
    return match_seed + SEED_STRIDE * FIGHTERS.index(fighter)


class GeneratorPool:
    """One MotionBricks generator per fighter, driven together.

    Building this loads a checkpoint per fighter, so build it once per match, not per round.
    """

    def __init__(
        self,
        match_seed: int = 1234,
        fighters: tuple[str, ...] = FIGHTERS,
        generators: dict[str, MotionBricksGenerator] | None = None,
    ) -> None:
        if not fighters:
            raise PoolError("a pool needs at least one fighter")
        self.fighters = tuple(fighters)
        self.match_seed = match_seed
        self.stats = PoolStats()

        if generators is not None:
            missing = [f for f in self.fighters if f not in generators]
            if missing:
                raise PoolError(f"no generator supplied for {missing}")
            self._generators = dict(generators)
        else:
            self._generators = {
                fighter: MotionBricksGenerator(
                    GeneratorConfig(random_seed=fighter_seed(match_seed, fighter))
                )
                for fighter in self.fighters
            }

    def __getitem__(self, fighter: str) -> MotionBricksGenerator:
        if fighter not in self._generators:
            raise PoolError(f"no generator for {fighter!r}; pool has {sorted(self._generators)}")
        return self._generators[fighter]

    def reset(self, round_index: int = 0) -> None:
        """Reseed every fighter from the match seed, offset by the round.

        Rounds must not be seeded identically or a match plays the same round three times, which
        looks like determinism and is actually a bug. The offset is the round number, which stays
        inside :data:`SEED_STRIDE` for any format a bracket would run.
        """
        if not 0 <= round_index < SEED_STRIDE:
            raise PoolError(
                f"round_index {round_index} outside [0, {SEED_STRIDE}); beyond that a round's "
                "seeds collide with the next fighter's"
            )
        for fighter in self.fighters:
            self[fighter].reset(seed=fighter_seed(self.match_seed + round_index, fighter))

    # -- driving ------------------------------------------------------------------------------------
    def generate(self, intents: dict[str, GeneratorIntent], dt: float, *, force: bool = False):
        """Ask every fighter to (re)plan.

        Each generator gets **its own** intent and reads **its own** context. Nothing is shared, which
        is what makes one fighter's commit unable to reach the other.
        """
        import time

        missing = [f for f in self.fighters if f not in intents]
        if missing:
            raise PoolError(f"no intent for {missing}; every fighter must be driven every replan")

        start = time.perf_counter()
        for fighter in self.fighters:
            generator = self[fighter]
            generator.generate(intents[fighter], generator.context_qpos(), dt=dt, force=force)
        elapsed = (time.perf_counter() - start) * 1e3

        self.stats.replans += 1
        self.stats.replan_ms.append(elapsed)
        return elapsed

    def next_frames(self) -> dict[str, np.ndarray]:
        """One ``(36,)`` qpos per fighter."""
        frames = {fighter: self[fighter].next_frame() for fighter in self.fighters}
        for fighter, qpos in frames.items():
            if qpos.shape != (QPOS_DIM,):
                raise PoolError(f"{fighter}: expected a ({QPOS_DIM},) frame, got {qpos.shape}")
        return frames

    def plans(self) -> dict[str, np.ndarray]:
        """Each fighter's current plan in full, ``(N, 36)``."""
        return {fighter: self[fighter].plan() for fighter in self.fighters}

    # -- the isolation guarantee -----------------------------------------------------------------------
    def independence_holds(self) -> bool:
        """True if no two fighters share a generator, an agent or a model.

        Cheap enough to assert in a match's setup. Two fighters sharing any of these would let one
        fighter's commit steer the other, which is the failure this design exists to make impossible.
        """
        seen: set[int] = set()
        for fighter in self.fighters:
            generator = self[fighter]
            for obj in (generator, generator.agent, generator.agent._inferencer):
                if id(obj) in seen:
                    return False
                seen.add(id(obj))
        return True
