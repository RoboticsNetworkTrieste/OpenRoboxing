"""M5-T1 acceptance: league services — ratings, pairing, the season.

Acceptance criterion from WORKPLAN.md M5-T1:
  a simulated 32-player, 10-week season runs end to end from scripted clients and produces a sane
  table; ratings converge and the 8-match threshold behaves as specified.

The rating tests are checked against **Glickman's published worked example**, not against numbers of
ours. That is the only reason to trust a rating implementation: it is the one people can look up.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_league.py -v
"""

from __future__ import annotations

import pytest

from openroboxing.league.pairing import (
    Fixture,
    PairingError,
    PairingState,
    pair_round,
    rematch_count,
)
from openroboxing.league.rating import (
    DEFAULT_RATING,
    DEFAULT_RD,
    Glicko2,
    Rating,
    RatingError,
    Result,
    win_probability,
)
from openroboxing.league.season import (
    MATCHES_TO_RANK,
    PLAYOFF_SIZE,
    MatchResult,
    Season,
    SeasonError,
)

# --- Glicko-2 against the paper ---------------------------------------------------------------------------
#: Glickman, "Example of the Glicko-2 system", glicko.net/glicko/glicko2.pdf. A 1500/200/0.06 player
#: beats a 1400, loses to a 1550 and a 1700. The paper's answers, to its own rounding.
PAPER_PLAYER = Rating(1500, 200, 0.06)
PAPER_RESULTS = (
    Result(Rating(1400, 30, 0.06), 1.0),
    Result(Rating(1550, 100, 0.06), 0.0),
    Result(Rating(1700, 300, 0.06), 0.0),
)


def test_glicko2_reproduces_the_published_worked_example() -> None:
    """The only test here that matters. Everything else is behaviour around a correct core.

    Two notes on tolerances, both about the paper rather than about us:

    - The walkthrough **rounds its intermediates for display** — it prints g(phi) and E to four
      decimals and carries those rounded numbers forward. Computing without that rounding gives
      v = 1.77898 against the paper's 1.7785 and delta = -0.48393 against -0.4834. The final
      answers still agree to the precision the paper prints, which is the thing being checked.
    - It reports sigma' = 0.05999 where the unrounded value is 0.05999598, i.e. **truncated**, not
      rounded. Hence 1e-5 here and not 5e-6.
    """
    new = Glicko2(tau=0.5).rate(PAPER_PLAYER, list(PAPER_RESULTS))

    # The paper prints mu' = -0.2069 and phi' = 0.8722 in the internal scale. Both exact.
    assert (new.rating - DEFAULT_RATING) / 173.7178 == pytest.approx(-0.2069, abs=5e-5)
    assert new.rd / 173.7178 == pytest.approx(0.8722, abs=5e-5)
    assert new.volatility == pytest.approx(0.05999, abs=1e-5)
    assert new.rating == pytest.approx(1464.06, abs=0.02)
    assert new.rd == pytest.approx(151.52, abs=0.02)


def test_the_intermediate_quantities_match_the_paper_to_its_own_rounding() -> None:
    """Localises a failure of the test above: if v or delta is wrong, the algorithm is wrong; if
    only the final volatility is wrong, the root find is."""
    glicko = Glicko2(tau=0.5)
    mu, _ = glicko._to_glicko2(PAPER_PLAYER)

    variance_inverse = 0.0
    delta_sum = 0.0
    for result in PAPER_RESULTS:
        mu_j, phi_j = glicko._to_glicko2(result.opponent)
        g = glicko._g(phi_j)
        e = glicko._e(mu, mu_j, phi_j)
        variance_inverse += g**2 * e * (1.0 - e)
        delta_sum += g * (result.score - e)

    variance = 1.0 / variance_inverse
    assert variance == pytest.approx(1.7785, abs=1e-3)
    assert variance * delta_sum == pytest.approx(-0.4834, abs=1e-3)


def test_winning_raises_a_rating_and_losing_lowers_it() -> None:
    glicko = Glicko2()
    peer = Rating()
    won = glicko.rate(Rating(), [Result(peer, 1.0)])
    lost = glicko.rate(Rating(), [Result(peer, 0.0)])

    assert won.rating > DEFAULT_RATING > lost.rating


def test_playing_anybody_at_all_shrinks_the_deviation() -> None:
    played = Glicko2().rate(Rating(), [Result(Rating(), 1.0)])
    assert played.rd < DEFAULT_RD, "a match should make the system more sure, not less"


def test_sitting_out_a_period_widens_the_deviation() -> None:
    """Step 6 of the paper: not playing makes the system less sure about you."""
    idle = Glicko2().rate(Rating(1600, 100, 0.06), [])
    assert idle.rating == pytest.approx(1600.0), "an idle fighter's rating does not drift"
    assert idle.rd > 100.0


def test_beating_a_stronger_opponent_moves_a_rating_further() -> None:
    glicko = Glicko2()
    over_strong = glicko.rate(Rating(), [Result(Rating(1900, 50, 0.06), 1.0)])
    over_weak = glicko.rate(Rating(), [Result(Rating(1100, 50, 0.06), 1.0)])
    assert over_strong.rating > over_weak.rating


def test_an_uncertain_opponent_moves_a_rating_less() -> None:
    """The whole point of RD: a result against somebody unknown says less."""
    glicko = Glicko2()
    known = glicko.rate(Rating(), [Result(Rating(1700, 30, 0.06), 1.0)])
    unknown = glicko.rate(Rating(), [Result(Rating(1700, 350, 0.06), 1.0)])
    assert known.rating > unknown.rating


def test_the_interval_is_two_deviations_either_side() -> None:
    rating = Rating(1600, 75, 0.06)
    assert rating.interval == (1450.0, 1750.0)
    assert rating.conservative == 1450.0


def test_the_table_order_prefers_the_proven_fighter() -> None:
    """Two fighters on 1600 are not equal if one has played 8 matches and the other 40."""
    proven = Rating(1600, 50, 0.06)
    unproven = Rating(1600, 300, 0.06)
    assert proven.conservative > unproven.conservative


def test_a_nonsense_score_raises() -> None:
    with pytest.raises(RatingError, match="score must be"):
        Result(Rating(), 0.75)


def test_a_nonsense_tau_raises() -> None:
    with pytest.raises(RatingError, match="outside a sane range"):
        Glicko2(tau=0.0)


def test_win_probability_is_even_between_equals() -> None:
    assert win_probability(Rating(), Rating()) == pytest.approx(0.5)
    assert win_probability(Rating(1800, 50), Rating(1200, 50)) > 0.9


# --- Swiss pairing ------------------------------------------------------------------------------------
def _state(handles, scores=None, played=None, byes=()) -> PairingState:
    return PairingState(
        scores=scores or {h: 0.0 for h in handles},
        conservative={h: 1500.0 for h in handles},
        played=played or {},
        byes=set(byes),
    )


def test_everybody_is_paired_exactly_once() -> None:
    handles = [f"f{i}" for i in range(8)]
    fixtures = pair_round(1, handles, _state(handles))

    assert len(fixtures) == 4
    paired = [h for f in fixtures for h in f.fighters()]
    assert sorted(paired) == sorted(handles)


def test_an_odd_entry_list_produces_exactly_one_bye() -> None:
    handles = [f"f{i}" for i in range(7)]
    fixtures = pair_round(1, handles, _state(handles))

    byes = [f for f in fixtures if f.is_bye]
    assert len(byes) == 1
    paired = [h for f in fixtures for h in f.fighters()]
    assert sorted(paired) == sorted(handles), "the bye still appears exactly once"


def test_similar_scores_are_paired_together() -> None:
    handles = ["a", "b", "c", "d"]
    scores = {"a": 3.0, "b": 3.0, "c": 0.0, "d": 0.0}
    fixtures = pair_round(1, handles, _state(handles, scores))

    pairs = {frozenset(f.fighters()) for f in fixtures}
    assert pairs == {frozenset({"a", "b"}), frozenset({"c", "d"})}


def test_a_rematch_is_avoided_when_anybody_else_is_available() -> None:
    handles = ["a", "b", "c", "d"]
    state = _state(handles, played={"a": {"b"}, "b": {"a"}})
    fixtures = pair_round(2, handles, state)

    assert rematch_count(fixtures, state) == 0
    pairs = {frozenset(f.fighters()) for f in fixtures}
    assert frozenset({"a", "b"}) not in pairs


def test_a_rematch_beats_leaving_somebody_unpaired() -> None:
    """`spec/season.md`: a fighter with no match gets nothing from the week, which is worse."""
    handles = ["a", "b"]
    state = _state(handles, played={"a": {"b"}, "b": {"a"}})
    fixtures = pair_round(3, handles, state)

    assert len(fixtures) == 1
    assert rematch_count(fixtures, state) == 1
    assert not fixtures[0].is_bye


def test_the_bye_goes_to_somebody_who_has_not_had_one() -> None:
    handles = ["a", "b", "c"]
    scores = {"a": 2.0, "b": 1.0, "c": 0.0}
    fixtures = pair_round(2, handles, _state(handles, scores, byes={"c"}))

    bye = next(f for f in fixtures if f.is_bye)
    assert bye.home == "b", "c already had one; the next lowest gets it"


def test_pairing_is_deterministic() -> None:
    handles = [f"f{i}" for i in range(10)]
    first = pair_round(1, handles, _state(handles))
    second = pair_round(1, list(reversed(handles)), _state(handles))
    assert [f.fighters() for f in first] == [f.fighters() for f in second]


def test_duplicate_handles_raise() -> None:
    with pytest.raises(PairingError, match="duplicate handles"):
        pair_round(1, ["a", "a"], _state(["a"]))


def test_nobody_registered_pairs_nothing() -> None:
    assert pair_round(1, [], _state([])) == []


def test_a_bye_prints_as_a_bye() -> None:
    assert "bye" in str(Fixture(week=1, home="a"))
    assert "v" in str(Fixture(week=1, home="a", away="b"))


# --- the season ----------------------------------------------------------------------------------------
def _season(entrants: int = 4, weeks: int = 10) -> Season:
    season = Season(weeks=weeks)
    for index in range(entrants):
        season.register(f"f{index}")
    return season


def test_a_registered_fighter_starts_at_the_default_rating() -> None:
    season = _season()
    assert season["f0"].rating.rating == DEFAULT_RATING
    assert season["f0"].rating.rd == DEFAULT_RD
    assert not season["f0"].is_ranked


def test_registering_twice_raises() -> None:
    season = _season()
    with pytest.raises(SeasonError, match="already registered"):
        season.register("f0")


def test_an_unregistered_fighter_cannot_be_looked_up() -> None:
    with pytest.raises(SeasonError, match="not registered"):
        _season()["nobody"]


def test_a_reported_win_moves_both_ratings() -> None:
    season = _season(2)
    fixtures = season.pair_week()
    season.report_week([MatchResult(fixture=fixtures[0], winner=fixtures[0].home)])

    winner, loser = season[fixtures[0].home], season[fixtures[0].away]
    assert winner.rating.rating > DEFAULT_RATING > loser.rating.rating
    assert winner.won == 1 and loser.lost == 1
    assert winner.score == 1.0 and loser.score == 0.0


def test_a_draw_splits_the_score() -> None:
    season = _season(2)
    fixtures = season.pair_week()
    season.report_week([MatchResult(fixture=fixtures[0], winner=None)])

    assert all(season[h].drawn == 1 for h in ("f0", "f1"))
    assert all(season[h].score == 0.5 for h in ("f0", "f1"))


def test_everyone_in_a_period_is_rated_against_the_ratings_held_at_its_start() -> None:
    """Rating in sequence would make the order of a week's results matter, which it must not."""
    forward = _season(4)
    backward = _season(4)
    fixtures = forward.pair_week()
    results = [MatchResult(fixture=f, winner=f.home) for f in fixtures]

    forward.report_week(results)
    backward.report_week(list(reversed(results)))

    for handle in forward.entrants:
        assert forward[handle].rating.rating == pytest.approx(backward[handle].rating.rating)


def test_the_eight_match_threshold_gates_the_table() -> None:
    """`spec/season.md`: counts matches played, not wins, and a bye does not count."""
    season = _season(2, weeks=MATCHES_TO_RANK + 2)
    for _ in range(MATCHES_TO_RANK - 1):
        fixtures = season.pair_week()
        season.report_week([MatchResult(fixture=f, winner=f.home) for f in fixtures])

    assert season["f0"].played == MATCHES_TO_RANK - 1
    assert season.table() == []
    assert len(season.provisional()) == 2

    fixtures = season.pair_week()
    season.report_week([MatchResult(fixture=f, winner=f.home) for f in fixtures])

    assert season["f0"].played == MATCHES_TO_RANK
    assert len(season.table()) == 2
    assert season.provisional() == []


def test_a_bye_scores_but_does_not_rank_or_rate() -> None:
    """A fighter cannot be rated on a match that never happened."""
    season = _season(3)
    fixtures = season.pair_week()
    bye = next(f for f in fixtures if f.is_bye)
    before = season[bye.home].rating.rating

    season.report_week([MatchResult(fixture=f, winner=None if f.is_bye else f.home) for f in fixtures])

    entrant = season[bye.home]
    assert entrant.score == 1.0
    assert entrant.byes == 1
    assert entrant.played == 0, "a bye is not a match played"
    assert entrant.rating.rating == pytest.approx(before), "a bye does not rate"


def test_a_provisional_fighter_still_has_a_rating() -> None:
    """Hidden, not excluded — so the match that ranks them arrives with a real number."""
    season = _season(2)
    fixtures = season.pair_week()
    season.report_week([MatchResult(fixture=fixtures[0], winner=fixtures[0].home)])

    assert not season[fixtures[0].home].is_ranked
    assert season[fixtures[0].home].rating.rating > DEFAULT_RATING


def test_the_table_is_ordered_by_the_bottom_of_the_interval() -> None:
    season = _season(2)
    season["f0"].played = season["f1"].played = MATCHES_TO_RANK
    season["f0"].rating = Rating(1600, 300, 0.06)
    season["f1"].rating = Rating(1550, 40, 0.06)

    assert [e.handle for e in season.table()] == ["f1", "f0"]


def test_a_result_naming_a_stranger_raises() -> None:
    season = _season(2)
    outside = Fixture(week=1, home="f0", away="ghost")
    with pytest.raises(SeasonError, match="not registered"):
        season.report_week([MatchResult(fixture=outside, winner="f0")])


def test_a_result_cannot_be_won_by_a_non_participant() -> None:
    with pytest.raises(SeasonError, match="won a fixture between"):
        MatchResult(fixture=Fixture(week=1, home="a", away="b"), winner="c")


def test_a_season_cannot_run_past_its_last_week() -> None:
    season = _season(2, weeks=1)
    season.report_week([MatchResult(fixture=f, winner=f.home) for f in season.pair_week()])
    with pytest.raises(SeasonError, match="is over"):
        season.pair_week()


def test_a_playoff_needs_four_ranked_fighters() -> None:
    season = _season(4)
    with pytest.raises(SeasonError, match=f"a playoff needs {PLAYOFF_SIZE}"):
        season.playoff()


def test_the_playoff_is_one_v_four_and_two_v_three() -> None:
    season = _season(4)
    for index, rating in enumerate((1800, 1700, 1600, 1500)):
        entrant = season[f"f{index}"]
        entrant.played = MATCHES_TO_RANK
        entrant.rating = Rating(rating, 50, 0.06)

    assert season.playoff() == [("f0", "f3"), ("f1", "f2")]


def test_a_season_serialises_with_its_table_and_results() -> None:
    season = _season(2)
    fixtures = season.pair_week()
    season.report_week([MatchResult(fixture=fixtures[0], winner=fixtures[0].home, match_id="m1")])
    data = season.to_dict()

    assert set(data) == {
        "spec_version",
        "name",
        "weeks",
        "weeks_run",
        "table",
        "provisional",
        "fixtures",
        "results",
    }
    assert data["results"][0]["match_id"] == "m1"
    assert data["provisional"][0]["ranked"] is False


# --- the acceptance criterion --------------------------------------------------------------------------
def test_a_thirty_two_player_ten_week_season_produces_a_sane_table() -> None:
    """M5-T1's criterion. "Sane" is made concrete: the table recovers the hidden strength order."""
    from openroboxing.tools.simulate_season import _convergence, simulate

    season, strengths = simulate(entrants=32, weeks=10, seed=1234, verbose=False)
    stats = _convergence(season, strengths)

    assert stats["ranked"] == 32, "everyone played 10 matches; everyone should be ranked"
    assert len(season.provisional()) == 0
    assert stats["spearman"] > 0.6, (
        f"the table barely tracks true strength (Spearman {stats['spearman']:.2f})"
    )
    assert season.playoff(), "a full season should produce a playoff"


def test_ratings_converge_over_a_season() -> None:
    """Deviation must fall a long way from its 350 starting point, or the table means nothing."""
    from openroboxing.tools.simulate_season import simulate

    season, _ = simulate(entrants=32, weeks=10, seed=1234, verbose=False)
    deviations = [e.rating.rd for e in season.table()]

    assert max(deviations) < DEFAULT_RD / 2
    assert all(e.rating.rd > 0 for e in season.table()), "certainty is never absolute"


def test_the_confidence_interval_is_honest() -> None:
    """The RD should be about the size of the error it claims to describe.

    A rating system whose stated uncertainty does not match its actual error is worse than one with
    no interval at all, because the table would look more authoritative than it is.
    """
    from openroboxing.tools.simulate_season import _convergence, simulate

    season, strengths = simulate(entrants=32, weeks=10, seed=1234, verbose=False)
    stats = _convergence(season, strengths)

    assert stats["mean_abs_rating_error"] < 2 * stats["mean_rd"], (
        f"error {stats['mean_abs_rating_error']:.0f} against a claimed RD of {stats['mean_rd']:.0f}"
    )


def test_the_threshold_holds_for_a_late_entrant() -> None:
    """The interesting threshold case: somebody who joins in week 6 must not be ranked at week 10."""
    from openroboxing.tools.simulate_season import simulate

    season, _ = simulate(entrants=8, weeks=10, seed=7, verbose=False)
    season.register("latecomer")
    assert not season["latecomer"].is_ranked
    assert "latecomer" in [e.handle for e in season.provisional()]
    assert "latecomer" not in [e.handle for e in season.table()]


def test_a_season_is_deterministic() -> None:
    from openroboxing.tools.simulate_season import simulate

    first, _ = simulate(entrants=16, weeks=5, seed=99, verbose=False)
    second, _ = simulate(entrants=16, weeks=5, seed=99, verbose=False)

    assert [e.handle for e in first.table()] == [e.handle for e in second.table()]
    assert [round(e.rating.rating, 6) for e in first.table()] == [
        round(e.rating.rating, 6) for e in second.table()
    ]
