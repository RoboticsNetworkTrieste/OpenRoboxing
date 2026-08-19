"""The regression gate: a weight set must not silently break what already worked (S-T3).

`WORKPLAN` S-T3: *before any weight set ships, a fixed battery of upstream motions checked for
tracking regression, so a pose-specific finetune cannot silently degrade general behaviour.*
Acceptance: *a deliberately over-fitted checkpoint fails the regression gate.*

The failure this exists to catch
---------------------------------
A finetune on six boxing poses will make those six poses better. It can also make the fighter forget
how to walk, and nothing in the Studio's own metrics would notice — the poses are what is being
measured. This runs the **general** motions instead, under physics, and compares against a recorded
baseline.

What it measures
----------------
Per motion: mean and max joint tracking error, whether the fighter fell, and how far it travelled.
Falling is not a tolerance — a checkpoint that cannot stand fails outright, however good its numbers
were up to the moment it went down.

Baselines are recorded, not asserted
------------------------------------
`CLAUDE.md` standing rule 3: never invent a number. The gate compares against a **baseline file
produced by running the current weights**, and the tolerance is a stated *relative* margin over that.
There is no absolute "good tracking error" here, because nobody has ever established one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SPEC_VERSION = "0.1"

#: The battery. Chosen to span what the policy has to do outside a boxing ring, not to be exhaustive:
#: standing still, walking at three speeds, turning both ways, and the boxing gait itself as the
#: control (a finetune *should* improve that one).
DEFAULT_BATTERY: tuple[str, ...] = (
    "idle",
    "slow_walk",
    "walk",
    "walk_left",
    "walk_right",
    "walk_boxing",
)

#: How much worse than baseline a motion may get before it is a regression. 15% is a stated margin,
#: not a measured one — it is loose enough to absorb run-to-run variation in a contact simulation and
#: tight enough that forgetting how to walk shows up. Revisit once there are two real weight sets to
#: compare (`docs/ASSUMPTIONS.md`).
TOLERANCE = 0.15

#: Seconds of each motion. Long enough to walk several steps and fall over if it is going to.
SECONDS = 12.0

#: Root height below which a run is a fall, metres. Same as `run_single`'s default.
FALL_HEIGHT = 0.4


class RegressionError(RuntimeError):
    """The battery could not be run or compared. Never recovered from silently."""


@dataclass
class MotionResult:
    """One motion, run once."""

    style: str
    mean_joint_error_rad: float
    max_joint_error_rad: float
    min_root_height_m: float
    distance_m: float
    fell: bool
    ticks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Baseline:
    """What the battery scored on a known-good weight set."""

    label: str
    spec_version: str = SPEC_VERSION
    seconds: float = SECONDS
    seed: int = 1234
    results: dict[str, MotionResult] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "label": self.label,
            "seconds": self.seconds,
            "seed": self.seed,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "notes": self.notes,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> Baseline:
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RegressionError(f"{path}: cannot read the baseline ({exc})") from exc
        return cls(
            label=data["label"],
            spec_version=data.get("spec_version", SPEC_VERSION),
            seconds=data.get("seconds", SECONDS),
            seed=data.get("seed", 1234),
            results={k: MotionResult(**v) for k, v in data.get("results", {}).items()},
            notes=data.get("notes", {}),
        )


@dataclass
class Finding:
    """One motion that got worse."""

    style: str
    metric: str
    baseline: float
    candidate: float
    reason: str

    @property
    def ratio(self) -> float:
        return self.candidate / self.baseline if self.baseline else float("inf")


def run_motion(
    style: str,
    *,
    seconds: float = SECONDS,
    seed: int = 1234,
    policy=None,
    generator=None,
) -> MotionResult:
    """Run one motion under physics and measure it.

    Uses :class:`~openroboxing.runtime.world.SingleFighterWorld` — the general-behaviour check is
    about one fighter tracking a reference, not about a fight.
    """
    import numpy as np

    from openroboxing.runtime.world import SingleFighterWorld

    world = SingleFighterWorld(
        style=style, seed=seed, fall_height=FALL_HEIGHT, policy=policy, generator=generator
    )
    world.reset(seed=seed)
    log = world.run(seconds=seconds)
    summary = log.summary()

    distance = (
        float(np.linalg.norm(log.root_position[-1][:2] - log.root_position[0][:2]))
        if len(log.root_position) > 1
        else 0.0
    )
    return MotionResult(
        style=style,
        mean_joint_error_rad=summary["mean_joint_error_rad"],
        max_joint_error_rad=summary["max_joint_error_rad"],
        min_root_height_m=summary["min_root_height_m"],
        distance_m=distance,
        fell=bool(log.fell),
        ticks=len(log.tick),
    )


def run_battery(
    battery: Sequence[str] = DEFAULT_BATTERY,
    *,
    label: str = "current",
    seconds: float = SECONDS,
    seed: int = 1234,
    policy=None,
    on_result=None,
) -> Baseline:
    """Run every motion in the battery. Building the world per motion is deliberate: a leaked
    history between motions would make the second one depend on the first."""
    results: dict[str, MotionResult] = {}
    for style in battery:
        result = run_motion(style, seconds=seconds, seed=seed, policy=policy)
        results[style] = result
        if on_result is not None:
            on_result(result)
    return Baseline(label=label, seconds=seconds, seed=seed, results=results)


def compare(
    baseline: Baseline, candidate: Baseline, tolerance: float = TOLERANCE
) -> list[Finding]:
    """Every way the candidate is worse than the baseline. Empty means it passes.

    Falling is checked first and separately: a checkpoint that cannot stand has failed regardless of
    what its error numbers said before it went down.
    """
    if tolerance < 0:
        raise RegressionError(f"tolerance must not be negative, got {tolerance}")

    findings: list[Finding] = []
    missing = [s for s in baseline.results if s not in candidate.results]
    if missing:
        raise RegressionError(
            f"the candidate did not run {missing}; a partial battery cannot clear the gate"
        )

    for style, before in baseline.results.items():
        after = candidate.results[style]

        if after.fell and not before.fell:
            findings.append(
                Finding(style, "fell", 0.0, 1.0, "the fighter fell where the baseline stood")
            )
            continue  # every other number for this motion is about a falling robot

        for metric in ("mean_joint_error_rad", "max_joint_error_rad"):
            was = getattr(before, metric)
            now = getattr(after, metric)
            if was > 0 and now > was * (1.0 + tolerance):
                findings.append(
                    Finding(
                        style,
                        metric,
                        was,
                        now,
                        f"{(now / was - 1) * 100:.0f}% worse than baseline "
                        f"(tolerance {tolerance * 100:.0f}%)",
                    )
                )

    return findings


def format_report(
    baseline: Baseline, candidate: Baseline, findings: Sequence[Finding]
) -> str:
    lines = [
        f"regression: {candidate.label} against {baseline.label}",
        f"  {'motion':<14} {'mean err':>18} {'max err':>18} {'fell':>10} {'travelled':>10}",
    ]
    for style, before in baseline.results.items():
        after = candidate.results.get(style)
        if after is None:
            continue
        fell = "yes" if after.fell else "no"
        lines.append(
            f"  {style:<14} "
            f"{before.mean_joint_error_rad:>8.4f}->{after.mean_joint_error_rad:<9.4f} "
            f"{before.max_joint_error_rad:>8.4f}->{after.max_joint_error_rad:<9.4f} "
            f"{fell:>10} {after.distance_m:>9.2f}m"
        )

    lines.append("")
    if findings:
        lines.append(f"  FAILED — {len(findings)} regression(s):")
        for finding in findings:
            lines.append(f"    {finding.style:<14} {finding.metric:<22} {finding.reason}")
    else:
        lines.append("  PASSED — nothing regressed beyond tolerance.")
    return "\n".join(lines)


# -- proving the gate can fail ---------------------------------------------------------------------------
class DegradedPolicy:
    """A policy wrapper that returns deliberately worse actions.

    S-T3's acceptance is that *a deliberately over-fitted checkpoint fails the gate*. Producing a
    genuinely over-fitted checkpoint needs the finetune runner (S-T2) and TORC hardware, so this
    stands in for one: it degrades the action the way an over-fitted network would — biased and
    scaled wrong — while leaving the rest of the stack alone.

    A gate that has never been observed failing is not a gate.
    """

    def __init__(self, policy, scale: float = 0.75, bias: float = 0.05) -> None:
        self._policy = policy
        self.scale = scale
        self.bias = bias

    def __getattr__(self, name: str):
        return getattr(self._policy, name)

    def step(self, encoder_input, builder):
        action, tokens = self._policy.step(encoder_input, builder)
        return action * self.scale + self.bias, tokens


def default_baseline_path() -> Path:
    from openroboxing.paths import OPENROBOXING_ROOT

    return OPENROBOXING_ROOT / "studio" / "baselines" / "regression_v0.1.json"


def notes_for(manifest_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What produced a baseline, so it can be traced (`league/manifest.py`)."""
    return {
        "purpose": (
            "General-behaviour battery. A pose-specific finetune must not silently degrade these."
        ),
        "tolerance": TOLERANCE,
        "assets": dict(manifest_hashes or {}),
    }
