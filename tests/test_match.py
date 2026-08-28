"""M3-T4 acceptance: the match loop.

Acceptance criterion from WORKPLAN.md M3-T4:
  a full three-round match runs headless and produces a match record containing every field in
  `spec/match_record.md`.

Everything except the last test runs against a scripted stub world. That is the point of the
``MatchWorld`` protocol: the rules — a knockout ends the round, a bell cuts a count short, a match
always runs three rounds — are decisions about numbers in a trace, and none of them needs a GPU to
be wrong. The stub is what lets a disputed knockdown be re-derived from a recording.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_match.py -v
    .venv_mb/bin/python -m pytest tests/test_match.py -v -m slow   # needs a GPU
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.contact import (
    DOWN_HEIGHT_FRACTION,
    STANDING_TORSO_HEIGHT_M,
    ContactTracker,
    FightTrace,
    HitEvent,
)
from openroboxing.runtime.match import (
    SCHEMA_VERSION,
    KnockdownDetector,
    Match,
    MatchError,
    MatchFormat,
    MatchRecord,
)
from openroboxing.spec.constants import TICK_HZ

#: A collapsed G1, from `runtime/contact.py`'s measurements.
DOWN_HEIGHT = 0.058
DOWN_UPRIGHT = 0.051

#: A short format, so a rules test is a handful of ticks rather than a minute of them. The rules do
#: not care about the numbers; `test_default_format_is_the_spec` is what pins the real ones.
SHORT = MatchFormat(rounds=3, round_ticks=40, get_up_window_ticks=10, tick_hz=float(TICK_HZ))


class _ScriptedWorld:
    """A ``MatchWorld`` that is told exactly when each fighter is down.

    Writes trace rows directly instead of reading a simulator, which is what makes every rule below
    testable in milliseconds.
    """

    def __init__(
        self,
        down: dict[str, set[int]] | None = None,
        hits: dict[int, HitEvent] | None = None,
        commits: list[dict] | None = None,
        qpos_dim: int = 72,
    ) -> None:
        self.down = down or {}
        self.hits = hits or {}
        self._commits = commits or []
        self.qpos_dim = qpos_dim
        self.rounds_reset: list[int] = []
        self.ticks_stepped: list[int] = []
        self.commits_read = 0

    def reset_round(self, index: int) -> None:
        self.rounds_reset.append(index)

    def step(self, tick: int) -> None:
        self.ticks_stepped.append(tick)

    def observe(self, tracker: ContactTracker, trace: FightTrace, tick: int) -> None:
        trace.tick.append(tick)
        trace.separation_m.append(1.2)
        for fighter in FIGHTERS:
            is_down = tick in self.down.get(fighter, set())
            trace.positions.setdefault(fighter, []).append(np.zeros(3))
            trace.centre_distance_m.setdefault(fighter, []).append(0.6)
            trace.torso_height_m.setdefault(fighter, []).append(
                DOWN_HEIGHT if is_down else STANDING_TORSO_HEIGHT_M
            )
            trace.torso_upright.setdefault(fighter, []).append(
                DOWN_UPRIGHT if is_down else 1.0
            )
            trace.torso_quat.setdefault(fighter, []).append(np.array([1.0, 0.0, 0.0, 0.0]))
        if tick in self.hits:
            tracker.events.append(self.hits[tick])

    def qpos(self) -> np.ndarray:
        return np.arange(self.qpos_dim, dtype=np.float64)

    def commits(self) -> list[dict]:
        self.commits_read += 1
        return list(self._commits)


def _trace(down_at: set[int], ticks: int, fighter: str = "red") -> FightTrace:
    """A trace where one fighter is down at the given ticks."""
    world = _ScriptedWorld(down={fighter: down_at})
    trace = FightTrace()
    for tick in range(ticks):
        world.observe(ContactTracker(), trace, tick)
    return trace


def _feed(detector: KnockdownDetector, trace: FightTrace) -> str | None:
    """Replay a whole trace through a detector. Returns the first fighter knocked out."""
    for index, tick in enumerate(trace.tick):
        out = detector.observe(trace, index, tick)
        if out is not None:
            return out
    return None


def _hit(attacker: str = "red", tick: int = 5) -> HitEvent:
    return HitEvent(
        attacker=attacker,
        defender="blue" if attacker == "red" else "red",
        attacker_body="left_wrist_yaw_link",
        defender_body="head_link",
        region="head",
        start_tick=tick,
        end_tick=tick + 1,
        peak_force_n=180.0,
        impulse_ns=0.9,
        position=(0.1, 0.0, 1.2),
    )


# --- the format -------------------------------------------------------------------------------------
def test_default_format_is_the_spec() -> None:
    """`spec/match_record.md` v0.1: three 60 s rounds, boxing's eight-count."""
    fmt = MatchFormat()
    assert fmt.rounds == 3
    assert fmt.round_ticks == 60 * TICK_HZ == 3000
    assert fmt.get_up_window_ticks == 8 * TICK_HZ == 400
    assert fmt.round_seconds == 60.0
    assert fmt.get_up_seconds == 8.0


def test_a_count_that_cannot_finish_is_refused() -> None:
    """A get-up window longer than a round makes a knockout unreachable, silently."""
    with pytest.raises(MatchError, match="no count could ever complete"):
        MatchFormat(rounds=3, round_ticks=100, get_up_window_ticks=200)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rounds": 0}, "at least one round"),
        ({"round_ticks": 0}, "round_ticks must be positive"),
        ({"get_up_window_ticks": 0}, "get_up_window_ticks must be positive"),
    ],
)
def test_a_nonsense_format_raises(kwargs, message) -> None:
    with pytest.raises(MatchError, match=message):
        MatchFormat(**kwargs)


# --- knockdowns -------------------------------------------------------------------------------------
def test_getting_up_inside_the_window_is_a_knockdown_not_a_knockout() -> None:
    detector = KnockdownDetector(get_up_window_ticks=10)
    knocked_out = _feed(detector, _trace(down_at=set(range(5, 14)), ticks=30))

    assert knocked_out is None
    events = detector.flush()
    assert len(events) == 1
    assert events[0].fighter == "red"
    assert events[0].became_knockout is False
    assert events[0].duration_ticks == 9


def test_staying_down_for_the_whole_window_is_a_knockout() -> None:
    detector = KnockdownDetector(get_up_window_ticks=10)
    knocked_out = _feed(detector, _trace(down_at=set(range(5, 30)), ticks=30))

    assert knocked_out == "red"
    assert detector.knocked_out == "red"
    events = detector.flush()
    assert len(events) == 1
    assert events[0].became_knockout is True
    assert events[0].duration_ticks == 10, "the count closes the moment it completes"
    assert events[0].start_tick == 5 and events[0].end_tick == 14


def test_the_count_starts_again_after_getting_up() -> None:
    """Two separate trips to the canvas, neither long enough. Boxing has no three-knockdown rule
    here — `spec/match_record.md` says so — so this is two events and no knockout."""
    detector = KnockdownDetector(get_up_window_ticks=10)
    down = set(range(2, 10)) | set(range(20, 28))
    assert _feed(detector, _trace(down_at=down, ticks=40)) is None

    events = detector.flush()
    assert [e.became_knockout for e in events] == [False, False]
    assert [(e.start_tick, e.end_tick) for e in events] == [(2, 9), (20, 27)]


def test_a_fighter_down_at_the_bell_was_not_counted_out() -> None:
    """The case the rule as stated does not cover. The round ended first."""
    detector = KnockdownDetector(get_up_window_ticks=10)
    assert _feed(detector, _trace(down_at={6, 7, 8}, ticks=9)) is None

    events = detector.flush()
    assert len(events) == 1
    assert events[0].became_knockout is False
    assert events[0].end_tick == 8


def test_an_episode_records_how_far_the_fighter_went_down() -> None:
    detector = KnockdownDetector(get_up_window_ticks=10)
    _feed(detector, _trace(down_at={3, 4}, ticks=10))

    event = detector.flush()[0]
    assert event.lowest_torso_height_m == pytest.approx(DOWN_HEIGHT)
    assert event.min_upright == pytest.approx(DOWN_UPRIGHT)
    assert event.lowest_torso_height_m < DOWN_HEIGHT_FRACTION * STANDING_TORSO_HEIGHT_M


def test_a_window_of_zero_is_refused() -> None:
    with pytest.raises(MatchError, match="must be positive"):
        KnockdownDetector(get_up_window_ticks=0)


def test_a_fighter_never_down_produces_nothing() -> None:
    detector = KnockdownDetector(get_up_window_ticks=10)
    assert _feed(detector, _trace(down_at=set(), ticks=30)) is None
    assert detector.flush() == []


# --- rounds -----------------------------------------------------------------------------------------
def test_a_quiet_round_runs_to_the_bell() -> None:
    world = _ScriptedWorld()
    record = Match(world, match_format=SHORT).run_round(0)

    assert record.ended_by == "bell"
    assert record.ticks == SHORT.round_ticks
    assert record.knocked_out is None
    assert world.ticks_stepped == list(range(SHORT.round_ticks))


def test_a_knockout_ends_the_round_there() -> None:
    world = _ScriptedWorld(down={"blue": set(range(5, 40))})
    record = Match(world, match_format=SHORT).run_round(0)

    assert record.ended_by == "knockout"
    assert record.knocked_out == "blue"
    assert record.ticks == 5 + SHORT.get_up_window_ticks
    assert len(world.ticks_stepped) == record.ticks, "the round kept running after the count"


def test_a_knockout_does_not_end_the_match() -> None:
    """The rule that makes OpenRoboxing not-quite-boxing. Every round is fought."""
    world = _ScriptedWorld(down={"red": set(range(0, 40))})
    record = Match(world, match_format=SHORT).run()

    assert len(record.rounds) == SHORT.rounds
    assert [r.ended_by for r in record.rounds] == ["knockout"] * SHORT.rounds
    assert record.knockouts() == [(0, "red"), (1, "red"), (2, "red")]
    assert world.rounds_reset == [0, 1, 2], "both fighters reset for the next round"


def test_the_trace_is_one_row_per_tick_actually_run() -> None:
    world = _ScriptedWorld(down={"blue": set(range(5, 40))})
    record = Match(world, match_format=SHORT).run_round(0)

    assert record.trace.shape == (record.ticks, world.qpos_dim)
    assert record.trace.dtype == np.float32
    assert record.trace.shape[0] < SHORT.round_ticks, "the round ended early; so should the trace"


def test_a_rounds_hits_and_commits_are_carried_through() -> None:
    world = _ScriptedWorld(
        hits={4: _hit("red", 4), 9: _hit("blue", 9)},
        commits=[{"fighter": "red", "slot": "1", "issued_at": 2}],
    )
    record = Match(world, match_format=SHORT).run_round(0)

    assert [h.attacker for h in record.hits] == ["red", "blue"]
    assert record.commits == [{"fighter": "red", "slot": "1", "issued_at": 2}]
    assert world.commits_read == 1, "the commit log is read once, at the bell"


def test_a_knockdown_survives_into_the_round_record() -> None:
    world = _ScriptedWorld(down={"red": set(range(3, 8))})
    record = Match(world, match_format=SHORT).run_round(0)

    assert record.ended_by == "bell"
    assert len(record.knockdowns) == 1
    assert record.knockdowns[0].became_knockout is False


# --- the record -------------------------------------------------------------------------------------
def _match_record(world: _ScriptedWorld | None = None) -> MatchRecord:
    match = Match(
        world or _ScriptedWorld(hits={4: _hit("red", 4)}),
        match_id="test-0001",
        match_format=SHORT,
        fighters={
            f: {"handle": f, "combinations": ["combo-a", "combo-b"]}
            for f in FIGHTERS
        },
        versions={"policy": "sonic-model12", "pose_library": "0.1", "rules": "0.1"},
        seeds={"match_seed": 1234, "red": 1234, "blue": 2234},
    )
    return match.run()


def test_the_record_holds_every_field_the_spec_names() -> None:
    """The M3-T4 acceptance criterion, field by field, against `spec/match_record.md`."""
    data = _match_record().to_dict(include_trace=True)

    assert set(data) == {
        "schema_version",
        "match_id",
        "format",
        "arena",
        "fighters",
        "versions",
        "seeds",
        "rounds",
    }
    assert data["schema_version"] == SCHEMA_VERSION
    assert set(data["format"]) == {"rounds", "round_ticks", "get_up_window_ticks", "tick_hz"}
    assert set(data["fighters"]) == set(FIGHTERS)
    assert set(data["fighters"]["red"]) == {"handle", "combinations"}

    for round_data in data["rounds"]:
        assert set(round_data) == {
            "index",
            "ticks",
            "ended_by",
            "knocked_out",
            "hits",
            "knockdowns",
            "commits",
            "trace",
        }
        assert round_data["ended_by"] in ("bell", "knockout")

    hit = data["rounds"][0]["hits"][0]
    assert set(hit) == {
        "attacker",
        "defender",
        "attacker_body",
        "defender_body",
        "region",
        "start_tick",
        "end_tick",
        "peak_force_n",
        "impulse_ns",
        "position",
    }


def test_a_knockdown_serialises_with_every_field() -> None:
    world = _ScriptedWorld(down={"red": set(range(3, 8))})
    data = _match_record(world).to_dict()

    knockdown = data["rounds"][0]["knockdowns"][0]
    assert set(knockdown) == {
        "fighter",
        "start_tick",
        "end_tick",
        "lowest_torso_height_m",
        "min_upright",
        "became_knockout",
    }


def test_the_format_travels_with_the_record() -> None:
    """An old match must stay readable against the numbers it was fought under."""
    data = _match_record().to_dict()
    assert data["format"]["get_up_window_ticks"] == SHORT.get_up_window_ticks
    assert data["format"]["round_ticks"] == SHORT.round_ticks


def test_hits_are_attributed_to_the_fighter_that_threw_them() -> None:
    record = _match_record(_ScriptedWorld(hits={4: _hit("red", 4), 9: _hit("blue", 9)}))
    assert len(record.hits_by("red")) == SHORT.rounds
    assert len(record.hits_by("blue")) == SHORT.rounds
    assert record.hits_by("green") == []


def test_the_trace_is_left_out_of_the_json_by_default() -> None:
    """2.6 MB of float32 per match. It goes in the npz, not inline."""
    data = _match_record().to_dict()
    assert "trace" not in data["rounds"][0]


def test_save_writes_the_json_and_the_trace_beside_it(tmp_path) -> None:
    record = _match_record()
    path = tmp_path / "matches" / "test-0001.json"
    trace_path = record.save(path)

    assert path.exists() and trace_path.exists()
    assert trace_path.name == "test-0001.trace.npz"

    data = json.loads(path.read_text())
    assert data["match_id"] == "test-0001"

    with np.load(trace_path) as traces:
        assert sorted(traces.files) == ["round_0", "round_1", "round_2"]
        assert traces["round_0"].shape == (SHORT.round_ticks, 72)


def test_the_trace_is_written_before_the_json(tmp_path, monkeypatch) -> None:
    """The trace is authoritative, so a JSON without one must be impossible.

    Written the other way round, a crash between the two writes leaves a record that reads as
    complete and replays wrong.
    """
    record = _match_record()
    path = tmp_path / "test-0001.json"

    monkeypatch.setattr(
        "openroboxing.runtime.match.json.dumps",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    with pytest.raises(RuntimeError, match="disk full"):
        record.save(path)

    assert path.with_suffix(".trace.npz").exists(), "the trace was lost with the JSON"
    assert not path.exists()


# --- the acceptance criterion ------------------------------------------------------------------------
@pytest.mark.slow
def test_a_full_three_round_match_runs_headless(tmp_path) -> None:
    """M3-T4, end to end: two G1s, physics, generators, policies, three rounds, one record.

    Runs the **real** format — three 60 s rounds — because the criterion says a full match, and a
    shortened one would not exercise the generator long enough to be evidence of anything.

    Ported for `spec/intent.md` 3.0's `D6` (task A6): both fighters draw from the whole on-disk
    combination library rather than a loadout, scripted the same way `tools/run_match.py` scripts
    a real match — reusing its `_script` rather than inventing a second copy of that spacing/orbit
    arithmetic. The on-disk library is all-draft (telegraph and tracking error have not been
    measured), so this passes `require_admitted=False`, exactly as `run_match.py --allow-draft`
    does.
    """
    from openroboxing.paths import COMBINATION_DIR
    from openroboxing.runtime.fight import FightWorld, ScriptedPilot
    from openroboxing.runtime.pool import fighter_seed
    from openroboxing.studio import combination_record as cr
    from openroboxing.tools.run_match import _script

    library = {p.stem: cr.load(p) for p in sorted(COMBINATION_DIR.glob("*.json"))}
    assert library, f"no combinations in {COMBINATION_DIR}; run tools.import_motions first"
    order = sorted(library)

    default_format = MatchFormat()
    world = FightWorld(
        libraries={f: library for f in FIGHTERS},
        pilots={
            f: ScriptedPilot(_script(f, order, library, default_format.round_ticks))
            for f in FIGHTERS
        },
        match_seed=1234,
        require_admitted=False,
    )
    match = Match(
        world,
        match_id="acceptance",
        match_format=default_format,
        fighters={f: {"handle": f, "combinations": order} for f in FIGHTERS},
        versions={"pose_library": library[order[0]].library_version, "rules": SCHEMA_VERSION},
        seeds={"match_seed": 1234, **{f: fighter_seed(1234, f) for f in FIGHTERS}},
    )
    record = match.run()

    assert len(record.rounds) == 3
    for index, round_record in enumerate(record.rounds):
        assert round_record.index == index
        assert round_record.ended_by in ("bell", "knockout")
        assert round_record.trace.shape == (round_record.ticks, world.model.nq)
        assert round_record.commits, f"round {index} recorded no commits; the pilots did nothing"

        # A round that reached the bell ran its full length; one that did not must name a fighter.
        if round_record.ended_by == "bell":
            assert round_record.ticks == match.format.round_ticks
            assert round_record.knocked_out is None
        else:
            assert round_record.ticks < match.format.round_ticks
            assert round_record.knocked_out in FIGHTERS

    # Not decoration: without this the test passes just as happily on two fighters who collapse in
    # the first second and never touch each other, which is exactly the regression worth catching.
    landed = sum(len(r.hits) for r in record.rounds)
    assert landed > 0, "three rounds and nobody landed anything; the fighters never engaged"

    trace_path = record.save(tmp_path / "acceptance.json")
    assert trace_path.exists()
    assert json.loads((tmp_path / "acceptance.json").read_text())["schema_version"] == (
        SCHEMA_VERSION
    )
