"""M3-T2 acceptance: generation for two fighters.

Acceptance criterion from WORKPLAN.md M3-T2:
  two fighters generate independent motion in one instance; a test asserts VRAM use and per-replan
  latency stay within the M1-T7 budget, and that fighter A's intents never affect fighter B's output
  (swap-order test).

The swap-order test is the important one and it runs against the **real** generators, because
isolation is exactly the property a stub cannot demonstrate.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_pool.py -v
    .venv_mb/bin/python -m pytest tests/test_pool.py -v -m slow   # needs a GPU
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.runtime.pool import (
    SEED_STRIDE,
    GeneratorPool,
    PoolError,
    PoolStats,
    fighter_seed,
)
from openroboxing.spec.constants import QPOS_DIM, TICK_DT, TICK_HZ

#: The M1-T7 budget: one 50 Hz tick. Generation must fit in what physics and inference leave.
TICK_BUDGET_MS = TICK_DT * 1e3

#: Measured in M3-T1: physics 3.76 ms (clinch) + inference 1.56 ms for both fighters.
MEASURED_OTHER_COSTS_MS = 3.76 + 1.56


class _StubGenerator:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.agent = object()
        self.agent = type("A", (), {"_inferencer": object()})()
        self.seen: list[GeneratorIntent] = []
        self.frame = 0
        self.seeds: list[int] = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.frame = 0

    def next_frame(self):
        qpos = np.zeros(QPOS_DIM)
        qpos[3] = 1.0
        qpos[7] = self.frame
        self.frame += 1
        return qpos

    def generate(self, intent, context_qpos, dt, *, force=False):
        self.seen.append(intent)

    def context_qpos(self):
        return np.zeros((4, QPOS_DIM))

    def plan(self):
        return np.zeros((10, QPOS_DIM))


def _pool(**kwargs) -> GeneratorPool:
    return GeneratorPool(
        generators={f: _StubGenerator(f) for f in FIGHTERS}, **kwargs
    )


def _intents(**overrides) -> dict[str, GeneratorIntent]:
    base = {f: GeneratorIntent(style="walk_boxing") for f in FIGHTERS}
    base.update(overrides)
    return base


# --- isolation ---------------------------------------------------------------------------------------
def test_no_two_fighters_share_anything() -> None:
    assert _pool().independence_holds()


def test_sharing_a_generator_is_detected() -> None:
    """The check has to be able to fail, or it is decoration."""
    shared = _StubGenerator("shared")
    pool = GeneratorPool(generators={f: shared for f in FIGHTERS})
    assert not pool.independence_holds()


def test_each_fighter_gets_only_its_own_intent() -> None:
    pool = _pool()
    red_intent = GeneratorIntent(style="walk", facing_angle=1.1)
    pool.generate(_intents(red=red_intent), dt=0.5)

    assert pool["red"].seen == [red_intent]
    assert pool["blue"].seen[0] is not red_intent
    assert pool["blue"].seen[0].style == "walk_boxing"


def test_every_fighter_must_be_driven() -> None:
    """A missing intent is a bug, not a fighter that stands still."""
    with pytest.raises(PoolError, match="no intent for"):
        _pool().generate({"red": GeneratorIntent()}, dt=0.5)


def test_an_unknown_fighter_raises() -> None:
    with pytest.raises(PoolError, match="no generator for"):
        _pool()["green"]


# --- seeds -------------------------------------------------------------------------------------------
def test_fighters_get_different_seeds_from_one_match_seed() -> None:
    seeds = [fighter_seed(77, f) for f in FIGHTERS]
    assert len(set(seeds)) == len(FIGHTERS), "both fighters would shadow-box identically"
    assert seeds[0] == 77
    assert seeds[1] == 77 + SEED_STRIDE


def test_reset_reseeds_every_fighter_from_the_match_seed() -> None:
    pool = _pool(match_seed=99)
    pool.reset()
    assert pool["red"].seeds == [fighter_seed(99, "red")]
    assert pool["blue"].seeds == [fighter_seed(99, "blue")]


def test_an_unknown_fighter_has_no_seed() -> None:
    with pytest.raises(PoolError, match="unknown fighter"):
        fighter_seed(1, "green")


# --- frames ------------------------------------------------------------------------------------------
def test_one_frame_per_fighter() -> None:
    frames = _pool().next_frames()
    assert set(frames) == set(FIGHTERS)
    assert all(q.shape == (QPOS_DIM,) for q in frames.values())


def test_plans_come_back_per_fighter() -> None:
    plans = _pool().plans()
    assert set(plans) == set(FIGHTERS)
    assert all(p.shape[1] == QPOS_DIM for p in plans.values())


# --- the budget ----------------------------------------------------------------------------------------
def test_generation_amortises_over_the_replan_interval() -> None:
    stats = PoolStats(replans=2, replan_ms=[29.6, 29.6])
    per_tick = stats.per_tick_ms(replan_interval_s=0.5, tick_hz=TICK_HZ)

    assert per_tick == pytest.approx(29.6 / 25.0)
    assert per_tick + MEASURED_OTHER_COSTS_MS < TICK_BUDGET_MS, (
        "generation plus physics plus inference does not fit in a tick"
    )


def test_stats_start_empty() -> None:
    pool = _pool()
    assert pool.stats.replans == 0 and pool.stats.mean_replan_ms == 0.0
    pool.generate(_intents(), dt=0.5)
    assert pool.stats.replans == 1


# --- the real generators ---------------------------------------------------------------------------------
@pytest.mark.slow
def test_swap_order_does_not_change_either_fighters_motion() -> None:
    """The claim a competitive match rests on: A's commits cannot reach B.

    Generated once in each order. If any state were shared, driving blue first would change red's
    output — which is precisely the bug that would be invisible until someone lost a match to it.
    """
    from openroboxing.runtime.pool import GeneratorPool as RealPool

    def run(order: tuple[str, ...]) -> dict[str, np.ndarray]:
        pool = RealPool(match_seed=1234, fighters=order)
        assert pool.independence_holds()
        pool.reset()
        intents = {
            "red": GeneratorIntent(style="walk_boxing", facing_angle=0.0),
            "blue": GeneratorIntent(style="walk", facing_angle=2.0),
        }
        for _ in range(3):
            pool.generate(intents, dt=0.5, force=True)
        return {f: pool.plans()[f].copy() for f in order}

    forward = run(("red", "blue"))
    reversed_ = run(("blue", "red"))

    for fighter in FIGHTERS:
        assert np.array_equal(forward[fighter], reversed_[fighter]), (
            f"{fighter}'s motion changed when the other fighter was generated first"
        )


@pytest.mark.slow
def test_two_fighters_stay_within_the_measured_budget() -> None:
    from openroboxing.runtime.pool import GeneratorPool as RealPool

    pool = RealPool(match_seed=1234)
    pool.reset()
    intents = {f: GeneratorIntent(style="walk_boxing") for f in FIGHTERS}
    for _ in range(6):
        pool.generate(intents, dt=0.5, force=True)

    per_tick = pool.stats.per_tick_ms(replan_interval_s=0.5, tick_hz=TICK_HZ)
    total = per_tick + MEASURED_OTHER_COSTS_MS
    assert total < TICK_BUDGET_MS, (
        f"a tick costs {total:.1f} ms of {TICK_BUDGET_MS:.1f} ms "
        f"(generation {per_tick:.2f} ms, replan {pool.stats.mean_replan_ms:.1f} ms)"
    )


@pytest.mark.slow
def test_two_fighters_fit_in_vram() -> None:
    import torch

    from openroboxing.runtime.pool import GeneratorPool as RealPool

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    torch.cuda.reset_peak_memory_stats()
    RealPool(match_seed=1234)
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20

    assert peak_mib < 0.25 * total_mib, (
        f"two fighters take {peak_mib:.0f} MiB of {total_mib:.0f} MiB; batching would be worth "
        "revisiting"
    )
