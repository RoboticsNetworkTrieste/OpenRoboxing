"""S-T3 acceptance: the regression gate.

Acceptance criterion from WORKPLAN.md S-T3:
  a deliberately over-fitted checkpoint fails the regression gate.

The fast tests here are about the gate's *logic* — that it catches what it claims to and does not
pass a partial run. The slow one is the criterion itself: a deliberately degraded policy must fail.

A gate nobody has watched fail is not a gate, which is why the failing case is a test and not a note.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_regression.py -v
    .venv_mb/bin/python -m pytest tests/test_regression.py -v -m slow   # needs a GPU
"""

from __future__ import annotations

import pytest

from openroboxing.studio.regression import (
    DEFAULT_BATTERY,
    TOLERANCE,
    Baseline,
    MotionResult,
    RegressionError,
    compare,
    format_report,
)


def _result(style: str, mean: float = 0.10, mx: float = 0.60, fell: bool = False) -> MotionResult:
    return MotionResult(
        style=style,
        mean_joint_error_rad=mean,
        max_joint_error_rad=mx,
        min_root_height_m=0.30 if fell else 0.70,
        distance_m=0.5 if fell else 6.0,
        fell=fell,
        ticks=600,
    )


def _baseline(label: str = "baseline", **overrides) -> Baseline:
    results = {style: _result(style) for style in DEFAULT_BATTERY}
    results.update(overrides)
    return Baseline(label=label, results=results)


# --- what the gate catches ------------------------------------------------------------------------------
def test_an_identical_run_passes() -> None:
    assert compare(_baseline(), _baseline("candidate")) == []


def test_an_improvement_passes() -> None:
    """A finetune is supposed to make some things better. That must never be a failure."""
    better = _baseline("candidate", walk_boxing=_result("walk_boxing", mean=0.04, mx=0.20))
    assert compare(_baseline(), better) == []


def test_error_within_tolerance_passes() -> None:
    """Contact simulation varies run to run; the margin exists to absorb that."""
    inside = _baseline(
        "candidate", walk=_result("walk", mean=0.10 * (1 + TOLERANCE * 0.9), mx=0.60)
    )
    assert compare(_baseline(), inside) == []


def test_error_beyond_tolerance_fails() -> None:
    worse = _baseline("candidate", walk=_result("walk", mean=0.10 * (1 + TOLERANCE * 2), mx=0.60))
    findings = compare(_baseline(), worse)

    assert len(findings) == 1
    assert findings[0].style == "walk"
    assert findings[0].metric == "mean_joint_error_rad"
    assert findings[0].ratio > 1 + TOLERANCE


def test_falling_fails_outright() -> None:
    """Not a tolerance. A checkpoint that cannot stand has failed however good its numbers were
    before it went down."""
    fallen = _baseline("candidate", walk=_result("walk", mean=0.02, mx=0.05, fell=True))
    findings = compare(_baseline(), fallen)

    assert [f.metric for f in findings] == ["fell"]
    assert "fell where the baseline stood" in findings[0].reason


def test_a_motion_that_already_fell_is_not_reported_again() -> None:
    """If the baseline itself falls on a motion, the candidate falling too is not a regression."""
    already = _baseline(walk=_result("walk", fell=True))
    candidate = Baseline(label="candidate", results=dict(already.results))
    assert compare(already, candidate) == []


def test_only_the_fall_is_reported_for_a_fallen_motion() -> None:
    """Every other number for a falling robot is about a falling robot."""
    fallen = _baseline("candidate", walk=_result("walk", mean=9.9, mx=9.9, fell=True))
    findings = compare(_baseline(), fallen)
    assert len(findings) == 1, f"expected only the fall, got {[f.metric for f in findings]}"


def test_several_motions_regressing_are_all_reported() -> None:
    worse = _baseline(
        "candidate",
        walk=_result("walk", mean=0.30, mx=1.50),
        idle=_result("idle", mean=0.30, mx=1.50),
    )
    styles = {f.style for f in compare(_baseline(), worse)}
    assert styles == {"walk", "idle"}


# --- the gate cannot be dodged ----------------------------------------------------------------------------
def test_a_partial_battery_cannot_clear_the_gate() -> None:
    """Running only the motions you expect to pass is the obvious way to cheat this."""
    partial = Baseline(label="candidate", results={"idle": _result("idle")})
    with pytest.raises(RegressionError, match="partial battery"):
        compare(_baseline(), partial)


def test_a_negative_tolerance_raises() -> None:
    with pytest.raises(RegressionError, match="must not be negative"):
        compare(_baseline(), _baseline("candidate"), tolerance=-0.1)


def test_a_zero_baseline_error_does_not_divide_by_zero() -> None:
    perfect = _baseline(walk=_result("walk", mean=0.0, mx=0.0))
    worse = _baseline("candidate", walk=_result("walk", mean=0.5, mx=0.5))
    assert compare(perfect, worse) == [], "a zero baseline has no ratio to exceed"


# --- reporting and storage ---------------------------------------------------------------------------------
def test_the_report_names_every_motion_and_the_verdict() -> None:
    passed = format_report(_baseline(), _baseline("candidate"), [])
    assert "PASSED" in passed
    assert all(style in passed for style in DEFAULT_BATTERY)

    worse = _baseline("candidate", walk=_result("walk", mean=0.9, mx=2.0))
    failed = format_report(_baseline(), worse, compare(_baseline(), worse))
    assert "FAILED" in failed and "walk" in failed


def test_a_baseline_round_trips_through_disk(tmp_path) -> None:
    baseline = _baseline()
    baseline.notes = {"tolerance": TOLERANCE}
    loaded = Baseline.load(baseline.save(tmp_path / "b.json"))

    assert loaded.label == baseline.label
    assert set(loaded.results) == set(baseline.results)
    assert loaded.results["walk"].mean_joint_error_rad == pytest.approx(0.10)
    assert compare(baseline, loaded) == []


def test_an_unreadable_baseline_raises(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(RegressionError, match="cannot read the baseline"):
        Baseline.load(bad)


def test_the_battery_covers_more_than_boxing() -> None:
    """The whole point: a boxing finetune must be checked against what it is *not* being trained on."""
    assert "walk_boxing" in DEFAULT_BATTERY, "the control motion should be present"
    assert len(set(DEFAULT_BATTERY) - {"walk_boxing"}) >= 4, "too little general behaviour"


# --- the acceptance criterion --------------------------------------------------------------------------------
@pytest.mark.slow
def test_a_deliberately_degraded_policy_fails_the_gate() -> None:
    """S-T3's criterion, with a stand-in for the over-fitted checkpoint S-T2 will eventually produce.

    Two short motions rather than the whole battery: the claim is that the gate *fires*, and that is
    established by one honest failure.
    """
    from openroboxing.runtime.policy import GearSonicPolicy
    from openroboxing.studio.regression import DegradedPolicy, run_battery

    battery = ("idle", "walk")
    good = run_battery(battery, label="baseline", seconds=6.0)
    degraded = run_battery(
        battery, label="degraded", seconds=6.0, policy=DegradedPolicy(GearSonicPolicy(), scale=0.5)
    )

    findings = compare(good, degraded)
    assert findings, (
        "a policy with its actions halved passed the regression gate; the gate is not working"
    )
