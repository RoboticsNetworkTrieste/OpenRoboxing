"""M5-T2: scoring a recorded match.

Acceptance criterion from WORKPLAN.md M5-T2:
  replaying ten recorded matches produces scores that a human watching the replays agrees with in at
  least eight cases; disagreements are documented as rule bugs, not code bugs.

The human half of that cannot be tested here. What *can* be pinned is that every rule in
`spec/scoring.md` does what the page says, so that when a human disagrees the argument is about the
rule and not about whether the code implements it. Each test below names the rule it covers.

Everything runs on dicts, because that is what a JSON archive holds — no simulator, no GPU.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_scoring.py -v
"""

from __future__ import annotations

import pytest

from openroboxing.league.scoring import (
    CONTACT_RANGE_M,
    DRAW_MARGIN,
    ENGAGEMENT_FALLOFF_M,
    MAX_AGGRESSION,
    POINTS_HEAVY,
    POINTS_ON_POINTS,
    POINTS_ONE_KNOCKDOWN,
    POINTS_WINNER,
    REGION_WEIGHTS,
    TARGET_COMMIT_RATE,
    ScoringError,
    aggression,
    damage,
    engagement,
    ring_control,
    score_match,
    score_round,
)
from openroboxing.runtime.contact import FightTrace
from openroboxing.spec.constants import TICK_HZ

FIGHTERS = ("red", "blue")


def _hit(attacker: str, region: str = "head", impulse: float = 1.0) -> dict:
    return {
        "attacker": attacker,
        "defender": "blue" if attacker == "red" else "red",
        "region": region,
        "impulse_ns": impulse,
        "peak_force_n": 100.0,
        "start_tick": 0,
        "end_tick": 1,
    }


def _commit(fighter: str, commit_at: int) -> dict:
    return {"fighter": fighter, "slot": "1", "issued_at": commit_at - 30, "commit_at": commit_at}


def _trace(separations: list[float], red_centre: list[float], blue_centre: list[float]) -> FightTrace:
    trace = FightTrace()
    trace.tick = list(range(len(separations)))
    trace.separation_m = list(separations)
    trace.centre_distance_m = {"red": list(red_centre), "blue": list(blue_centre)}
    return trace


def _round(index: int = 0, ticks: int = 3000, **kwargs) -> dict:
    base = {
        "index": index,
        "ticks": ticks,
        "ended_by": "bell",
        "knocked_out": None,
        "hits": [],
        "knockdowns": [],
        "commits": [],
    }
    base.update(kwargs)
    return base


# --- damage -------------------------------------------------------------------------------------------
def test_a_head_shot_is_worth_more_than_a_body_shot() -> None:
    assert damage([_hit("red", "head")], "red") > damage([_hit("red", "body")], "red")
    assert damage([_hit("red", "body")], "red") == pytest.approx(REGION_WEIGHTS["body"])


def test_a_punch_stopped_by_the_guard_scores_nothing() -> None:
    """`spec/scoring.md`: a blocked punch is the defender succeeding. Paying the attacker for it
    would reward throwing at a guard."""
    assert damage([_hit("red", "arm", impulse=50.0)], "red") == 0.0
    assert damage([_hit("red", "leg", impulse=50.0)], "red") == 0.0


def test_damage_is_impulse_not_a_count() -> None:
    one_hard = damage([_hit("red", "head", impulse=4.0)], "red")
    four_soft = damage([_hit("red", "head", impulse=1.0) for _ in range(4)], "red")
    assert one_hard == pytest.approx(four_soft)


def test_only_this_fighters_hits_count() -> None:
    hits = [_hit("red", "head", 2.0), _hit("blue", "head", 5.0)]
    assert damage(hits, "red") == pytest.approx(2.0)
    assert damage(hits, "blue") == pytest.approx(5.0)


def test_an_unknown_region_raises_rather_than_scoring_zero() -> None:
    """Silently weighting an unknown region at zero would lose real hits without saying so."""
    with pytest.raises(ScoringError, match="has no weight"):
        damage([_hit("red", "elbow_of_doom")], "red")


# --- engagement and ring control ---------------------------------------------------------------------------
def test_fighters_in_range_are_fully_engaged() -> None:
    assert engagement(0.0) == 1.0
    assert engagement(CONTACT_RANGE_M) == 1.0


def test_engagement_decays_to_nothing_by_starting_distance() -> None:
    assert engagement(CONTACT_RANGE_M + ENGAGEMENT_FALLOFF_M) == pytest.approx(0.0)
    assert engagement(5.0) == 0.0
    half = engagement(CONTACT_RANGE_M + ENGAGEMENT_FALLOFF_M / 2)
    assert half == pytest.approx(0.5)


def test_holding_the_centre_while_pressuring_scores_control() -> None:
    trace = _trace([0.5] * 10, red_centre=[0.2] * 10, blue_centre=[1.4] * 10)
    assert ring_control(trace, "red", "blue") == pytest.approx(1.0)
    assert ring_control(trace, "blue", "red") == pytest.approx(0.0)


def test_standing_in_the_middle_alone_is_not_control() -> None:
    """The correction that makes ring control mean something: pressure, not real estate."""
    far = _trace([3.0] * 10, red_centre=[0.1] * 10, blue_centre=[2.0] * 10)
    assert ring_control(far, "red", "blue") == pytest.approx(0.0)


def test_control_is_a_fraction_of_the_round_not_a_count() -> None:
    """A round cut short by a knockout must not score less for being short."""
    short = _trace([0.5] * 5, [0.2] * 5, [1.4] * 5)
    long = _trace([0.5] * 500, [0.2] * 500, [1.4] * 500)
    assert ring_control(short, "red", "blue") == pytest.approx(ring_control(long, "red", "blue"))


def test_an_empty_trace_yields_no_control() -> None:
    assert ring_control(FightTrace(), "red", "blue") == 0.0


def test_a_trace_missing_a_fighter_raises() -> None:
    trace = _trace([0.5] * 3, [0.2] * 3, [1.0] * 3)
    with pytest.raises(ScoringError, match="no centre distance"):
        ring_control(trace, "red", "green")


# --- aggression ---------------------------------------------------------------------------------------
def _minute_of_commits(count: int, separation: float) -> tuple[float, int, int]:
    ticks = int(60 * TICK_HZ)
    commits = [_commit("red", int(i * ticks / max(count, 1))) for i in range(count)]
    return aggression(commits, "red", ticks, [separation] * ticks)


def test_the_target_rate_scores_one() -> None:
    score, issued, in_range = _minute_of_commits(int(TARGET_COMMIT_RATE), 0.5)
    assert score == pytest.approx(1.0, abs=0.02)
    assert issued == in_range == int(TARGET_COMMIT_RATE)


def test_doing_nothing_scores_nothing() -> None:
    assert _minute_of_commits(0, 0.5)[0] == 0.0


def test_commit_spam_cannot_be_farmed_past_the_clamp() -> None:
    assert _minute_of_commits(200, 0.5)[0] == MAX_AGGRESSION


def test_committing_at_range_does_not_count() -> None:
    """`spec/scoring.md`: committing at range and closing is boxing; committing at range and
    staying there is not."""
    score, issued, in_range = _minute_of_commits(int(TARGET_COMMIT_RATE), separation=3.0)
    assert issued == int(TARGET_COMMIT_RATE)
    assert in_range == 0
    assert score == 0.0


def test_reach_is_measured_at_execution_not_at_issue() -> None:
    ticks = 1000
    separation = [3.0] * ticks
    separation[500] = 0.4  # in range only at the moment the move executes
    score, _, in_range = aggression([_commit("red", 500)], "red", ticks, separation)
    assert in_range == 1 and score > 0


def test_without_a_trace_every_commit_counts_and_is_an_upper_bound() -> None:
    _, issued, in_range = aggression([_commit("red", 10)] * 3, "red", 1000, None)
    assert issued == in_range == 3


def test_a_round_of_no_ticks_raises() -> None:
    with pytest.raises(ScoringError, match="at least one tick"):
        aggression([], "red", 0, None)


# --- the round score ---------------------------------------------------------------------------------------
def test_the_better_round_is_won_ten_nine() -> None:
    data = _round(hits=[_hit("red", "head", 5.0)])
    score = score_round(data, None, FIGHTERS)
    assert score.winner == "red"
    assert score.points == {"red": POINTS_WINNER, "blue": POINTS_ON_POINTS}


def test_a_knockdown_makes_it_ten_eight() -> None:
    data = _round(
        hits=[_hit("red", "head", 5.0)],
        knockdowns=[{"fighter": "blue", "start_tick": 10, "became_knockout": False}],
    )
    assert score_round(data, None, FIGHTERS).points == {
        "red": POINTS_WINNER,
        "blue": POINTS_ONE_KNOCKDOWN,
    }


def test_two_knockdowns_make_it_ten_seven() -> None:
    data = _round(
        hits=[_hit("red", "head", 5.0)],
        knockdowns=[
            {"fighter": "blue", "start_tick": 10, "became_knockout": False},
            {"fighter": "blue", "start_tick": 900, "became_knockout": False},
        ],
    )
    assert score_round(data, None, FIGHTERS).points["blue"] == POINTS_HEAVY


def test_a_knockout_round_is_ten_seven() -> None:
    data = _round(
        ended_by="knockout",
        knocked_out="blue",
        ticks=1200,
        hits=[_hit("red", "head", 5.0)],
        knockdowns=[{"fighter": "blue", "start_tick": 800, "became_knockout": True}],
    )
    score = score_round(data, None, FIGHTERS)
    assert score.points == {"red": POINTS_WINNER, "blue": POINTS_HEAVY}
    assert score.dimensions["red"].knockouts_against_opponent == 1


def test_an_even_round_is_drawn_ten_ten() -> None:
    """Without a margin, floating-point noise decides rounds that are visibly even."""
    data = _round(hits=[_hit("red", "head", 3.0), _hit("blue", "head", 3.0)])
    score = score_round(data, None, FIGHTERS)
    assert score.winner is None
    assert score.points == {"red": POINTS_WINNER, "blue": POINTS_WINNER}
    assert abs(score.margin) <= DRAW_MARGIN


def test_a_round_where_nothing_lands_splits_damage_evenly() -> None:
    score = score_round(_round(), None, FIGHTERS)
    assert score.winner is None, "nothing to separate them"


def test_damage_is_normalised_within_the_round() -> None:
    """A cagey round and a brawl are scored out of the same total."""
    cagey = score_round(_round(hits=[_hit("red", "head", 0.2)]), None, FIGHTERS)
    brawl = score_round(_round(hits=[_hit("red", "head", 90.0)]), None, FIGHTERS)
    assert cagey.margin == pytest.approx(brawl.margin)


def test_a_round_needs_exactly_two_fighters() -> None:
    with pytest.raises(ScoringError, match="two fighters"):
        score_round(_round(), None, ("red",))


# --- the match ------------------------------------------------------------------------------------------
def _record(rounds: list[dict]) -> dict:
    return {
        "match_id": "test",
        "fighters": {"red": {}, "blue": {}},
        "rounds": rounds,
    }


def test_the_match_goes_to_whoever_has_more_round_points() -> None:
    score = score_match(
        _record(
            [
                _round(0, hits=[_hit("red", "head", 5.0)]),
                _round(1, hits=[_hit("red", "head", 5.0)]),
                _round(2, hits=[_hit("blue", "head", 5.0)]),
            ]
        )
    )
    assert score.points == {"red": 29, "blue": 28}
    assert score.winner == "red"
    assert score.rounds_won("red") == 2


def test_equal_points_is_a_draw_with_no_countback() -> None:
    """`spec/scoring.md`: a countback invents a tiebreak nobody agreed to; a draw is honest."""
    score = score_match(
        _record(
            [
                _round(0, hits=[_hit("red", "head", 5.0)]),
                _round(1, hits=[_hit("blue", "head", 5.0)]),
            ]
        )
    )
    assert score.points["red"] == score.points["blue"]
    assert score.winner is None


def test_a_knockout_round_does_not_end_the_match_score() -> None:
    """The rule the whole format rests on: every round is fought and every round is scored."""
    score = score_match(
        _record(
            [
                _round(
                    0,
                    ended_by="knockout",
                    knocked_out="red",
                    ticks=900,
                    hits=[_hit("blue", "head", 5.0)],
                    knockdowns=[{"fighter": "red", "start_tick": 500, "became_knockout": True}],
                ),
                _round(1, hits=[_hit("red", "head", 5.0)]),
                _round(2, hits=[_hit("red", "head", 5.0)]),
            ]
        )
    )
    assert len(score.rounds) == 3
    assert score.points == {"red": 27, "blue": 28}
    assert score.winner == "blue", "the knockout round is worth 10-7 and that decided it"


def test_the_score_serialises_with_its_reasoning() -> None:
    score = score_match(_record([_round(0, hits=[_hit("red", "head", 5.0)])]))
    data = score.to_dict()

    assert set(data) == {"spec_version", "match_id", "points", "winner", "rounds"}
    assert set(data["rounds"][0]) == {"index", "points", "winner", "margin", "dimensions"}
    assert set(data["rounds"][0]["dimensions"]["red"]) == {
        "damage",
        "knockdowns_against_opponent",
        "knockouts_against_opponent",
        "control",
        "aggression",
        "commits",
        "commits_in_range",
    }


def test_scoring_never_touches_the_record() -> None:
    record = _record([_round(0, hits=[_hit("red", "head", 5.0)])])
    import copy

    before = copy.deepcopy(record)
    score_match(record)
    assert record == before, "a score is a derivation, written beside a record and never into it"
