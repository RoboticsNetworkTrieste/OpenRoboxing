"""M5-T4: the agent API (rewritten for combinations, M6 Phase 3 / Part A-3).

Acceptance criterion from WORKPLAN.md M5-T4:
  a scripted baseline agent plays a full ranked-format match against a human client and appears in
  the exhibition results, not the table.

An agent is just a client, so most of what needs testing is that it stays one: it sees only what the
protocol sends, it is rate-limited like anyone else, and its results are kept out of the table.

Rewritten against `spec/intent.md` 3.0 and `spec/protocol.md` 0.6: there is no more loadout, no more
``place``/``stage`` messages and no more pose slots. A ``welcome`` carries the whole shared
combination library (`spec/protocol.md` §"welcome's combination library"), and a client stages a
combination and a ghost together in one ``intent`` message. ``BaselineAgent`` is rewritten to choose
from that library rather than from a dealt loadout; see its docstring for what changed and why.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_agent.py -v
"""

from __future__ import annotations

import math

import pytest

from openroboxing.server.agent import (
    AGENT_PREFIX,
    DECISION_BUDGET_S,
    MAX_MESSAGES_PER_SECOND,
    AgentStats,
    BaselineAgent,
    IdleAgent,
    RateLimiter,
    combination_kind,
    is_exhibition,
    run_decision,
)

#: An agent gets no slots any more — a combination has none to name (`spec/intent.md` `D6`) — but
#: `run_decision`/`Agent` keep the parameter for source-compatibility with `server/client.py`'s
#: `AgentConnection`, which still passes one. Tests call `decide()` the same way that call site does.
NO_SLOTS: list[str] = []


def _combo(
    name: str, *, reach_m: float = 3.0, seconds: float = 1.2, heading_delta: float = 0.0
) -> dict:
    """One entry of `welcome`'s ``combinations`` list (`spec/protocol.md` 0.6), not a
    `studio.combination_record.CombinationRecord` — the agent only ever sees the wire shape, the
    same as a browser client would."""
    return {
        "name": name,
        "seconds": seconds,
        "heading_delta": heading_delta,
        "reach_m": reach_m,
        "pose": {},
    }


#: A small library spanning the three prefixes `CLAUDE.md` names as real corpus families:
#: `shadow-boxing` (strikes), `ib-dodge` (defensive) and `ib-combat-turn-jog` (travel). Generous
#: reach so most scenarios are trivially reachable; tests that care about the reach guard build
#: their own tighter library instead of overriding this one.
DEFAULT_COMBOS = [
    _combo("shadow-boxing-r-001-a359-00", reach_m=3.0),
    _combo("shadow-boxing-r-001-a360-00", reach_m=3.0),
    _combo("shadow-boxing-r-002-a361-00", reach_m=3.0),
    _combo("ib-dodge-270-r-001-a437-00", reach_m=1.5),
    _combo("ib-combat-turn-jog-start-270-r-003-a437-00", reach_m=4.0),
]


def _welcome(combinations: list[dict]) -> dict:
    return {"type": "welcome", "spec_version": "0.6", "combinations": combinations}


def _agent(seed: int = 0, combos: list[dict] | None = None) -> BaselineAgent:
    """A baseline that has seen a welcome, so it knows what it may commit."""
    agent = BaselineAgent(seed=seed)
    agent.on_welcome(_welcome(DEFAULT_COMBOS if combos is None else combos))
    return agent


def _state(
    tick: int,
    can_commit: bool = True,
    phase: str = "fighting",
    anchor: tuple[float, float] | None = (0.0, 0.0),
    opponent: tuple[float, float] | None = (2.0, 0.0),
) -> dict:
    """One protocol `state` frame (`spec/protocol.md` 0.6). ``anchor``/``opponent`` are world
    positions; ``None`` drops them, which is what a state with no positions looks like."""
    seats = {
        "red": {
            "handle": "agent:baseline",
            "staged": None,
            "position": None if anchor is None else {"x": anchor[0], "y": anchor[1]},
            "ghost": None,
            "anchor": None if anchor is None else {"x": anchor[0], "y": anchor[1]},
            "queue": [],
            "queue_depth": 0,
            "can_commit": can_commit,
            "hits_landed": 0,
            "torso_height_m": 0.84,
            "down": False,
        },
        "blue": {
            "handle": "human",
            "staged": None,
            "position": None if opponent is None else {"x": opponent[0], "y": opponent[1]},
            "ghost": None,
            "anchor": None,
            "queue": [],
            "queue_depth": 0,
            "can_commit": True,
            "hits_landed": 0,
            "torso_height_m": 0.84,
            "down": False,
        },
    }
    separation = None
    if anchor is not None and opponent is not None:
        separation = math.hypot(opponent[0] - anchor[0], opponent[1] - anchor[1])
    return {
        "type": "state",
        "tick": tick,
        "round": 1,
        "clock_ticks": 3000 - tick,
        "phase": phase,
        "separation_m": separation,
        "seats": seats,
    }


# --- an agent sees only what a client sees --------------------------------------------------------
def test_the_baseline_agent_reads_nothing_privileged() -> None:
    """It is handed a state message and its own seat. There is nowhere else for it to look."""
    agent = _agent()
    messages = agent.decide(_state(tick=200), "red", NO_SLOTS)
    assert all(m["type"] in ("intent", "commit") for m in messages)


def test_it_commits_by_staging_an_intent_first() -> None:
    """`spec/intent.md` keeps staging and committing separate; an agent uses the same two steps,
    now with one `intent` message rather than 0.4-0.5's separate `stage` and `place`."""
    agent = _agent()
    messages = agent.decide(_state(tick=200), "red", NO_SLOTS)
    assert [m["type"] for m in messages] == ["intent", "commit"]

    staged = messages[0]
    assert staged["combination"] in {c["name"] for c in DEFAULT_COMBOS}
    assert len(staged["ghost"]) == 2
    assert all(math.isfinite(v) for v in staged["ghost"])


def test_it_does_not_commit_while_one_is_active() -> None:
    agent = _agent()
    assert agent.decide(_state(tick=200, can_commit=False), "red", NO_SLOTS) == []


def test_it_never_commits_into_a_full_queue() -> None:
    """The existing guard, unchanged in spirit: `can_commit` is exactly what the host computes from
    the queue bound, and an agent that ignored it would spam a rejection into the log."""
    agent = _agent()
    for _ in range(5):
        assert agent.decide(_state(tick=200, can_commit=False), "red", NO_SLOTS) == []


def test_it_waits_a_beat_between_commits() -> None:
    """Otherwise it is only patient because the rate limiter made it so."""
    agent = _agent()
    assert agent.decide(_state(tick=200), "red", NO_SLOTS), "the first commit should fire"
    assert agent.decide(_state(tick=205), "red", NO_SLOTS) == [], "too soon"
    assert agent.decide(_state(tick=200 + agent.RECOVERY_TICKS + 1), "red", NO_SLOTS)


def test_a_reset_agent_forgets_its_cooldown_but_keeps_its_library() -> None:
    """`reset()` is a new-round event; the library came from `welcome`, once per connection, and a
    round boundary must not make the agent forget what it may commit."""
    agent = _agent()
    agent.decide(_state(tick=200), "red", NO_SLOTS)
    agent.reset()
    assert agent.decide(_state(tick=5), "red", NO_SLOTS), "it should be ready again, immediately"


def test_the_idle_agent_never_acts() -> None:
    agent = IdleAgent()
    assert all(agent.decide(_state(tick=t), "red", NO_SLOTS) == [] for t in range(0, 500, 50))


def test_it_does_nothing_between_rounds() -> None:
    agent = _agent()
    assert agent.decide(_state(tick=500, phase="round_over"), "red", NO_SLOTS) == []


def test_it_still_works_without_positions() -> None:
    """A record replayed through an older protocol (or a malformed state) has no positions; the
    agent must not crash, and must not invent a ghost out of nothing (`CLAUDE.md` "fail loudly" read
    the other way round: an unknown placement is not a placement to guess at)."""
    agent = _agent()
    assert agent.decide(_state(200, anchor=None, opponent=None), "red", NO_SLOTS) == []
    assert agent.decide(_state(200, anchor=(0.0, 0.0), opponent=None), "red", NO_SLOTS) == []


# --- welcome carries the library, not a loadout (D6 / spec/protocol.md 0.6) ------------------------
def test_it_survives_a_welcome_with_an_empty_library() -> None:
    """`D6`'s whole-library `welcome` can in principle carry zero entries (a fresh season, a paging
    bug upstream) and an agent must not crash on it — it just has nothing to do."""
    agent = BaselineAgent(seed=0)
    agent.on_welcome(_welcome([]))
    assert agent.decide(_state(tick=200), "red", NO_SLOTS) == []


def test_an_agent_that_never_saw_a_welcome_does_nothing() -> None:
    agent = BaselineAgent(seed=0)
    assert agent.decide(_state(tick=200), "red", NO_SLOTS) == []


def test_two_agents_seeded_differently_open_with_different_moves() -> None:
    first = _agent(seed=0).decide(_state(tick=200), "red", NO_SLOTS)
    second = _agent(seed=1).decide(_state(tick=200), "red", NO_SLOTS)
    chosen = [next(m for m in ms if m["type"] == "intent")["combination"] for ms in (first, second)]
    assert chosen[0] != chosen[1]


# --- it only ever asks for a ghost its combination can reach (spec/intent.md 3.0 "Feasibility") ----
def test_it_only_picks_combinations_that_can_reach_the_ghost_it_wants() -> None:
    """The host rejects a ghost beyond a combination's own `reach_m`; an agent that ignored that
    would spend its whole match being told 'can't get there'. Only one combo in this library can
    cover the distance to a far opponent, so it must be the only one ever chosen."""
    combos = [
        _combo("shadow-boxing-close-00", reach_m=0.2),
        _combo("ib-dodge-close-00", reach_m=0.2),
        _combo("ib-combat-turn-jog-far-00", reach_m=5.0),
    ]
    agent = _agent(combos=combos)

    tick = 0
    chosen: set[str] = set()
    for _ in range(6):
        tick += agent.RECOVERY_TICKS + 1
        messages = agent.decide(_state(tick=tick, anchor=(0.0, 0.0), opponent=(3.0, 0.0)), "red", NO_SLOTS)
        staged = next(m for m in messages if m["type"] == "intent")
        chosen.add(staged["combination"])
        distance = math.hypot(staged["ghost"][0] - 0.0, staged["ghost"][1] - 0.0)
        record_reach = next(c["reach_m"] for c in combos if c["name"] == staged["combination"])
        assert distance <= record_reach + 1e-6, (
            f"{staged['combination']!r} was asked to reach {distance:.3f} m, its own reach is "
            f"{record_reach:.3f} m"
        )
    assert chosen == {"ib-combat-turn-jog-far-00"}, f"it committed something it could not reach: {chosen}"


def test_a_ghost_never_exceeds_its_own_combinations_reach_even_when_everything_reaches() -> None:
    """The same guarantee, the ordinary case: comfortably reachable, still bounded correctly."""
    agent = _agent()
    tick = 0
    for _ in range(6):
        tick += agent.RECOVERY_TICKS + 1
        messages = agent.decide(
            _state(tick=tick, anchor=(0.0, 0.0), opponent=(0.5, 0.0)), "red", NO_SLOTS
        )
        staged = next(m for m in messages if m["type"] == "intent")
        distance = math.hypot(*staged["ghost"])
        reach = next(c["reach_m"] for c in DEFAULT_COMBOS if c["name"] == staged["combination"])
        assert distance <= reach + 1e-6


def test_it_varies_its_choices_rather_than_repeating_one_combination() -> None:
    """A repeated move is the easiest thing in the game to read."""
    agent = _agent()
    chosen = []
    tick = 0
    for _ in range(8):
        tick += agent.RECOVERY_TICKS + 1
        messages = agent.decide(_state(tick=tick), "red", NO_SLOTS)
        chosen.append(next(m for m in messages if m["type"] == "intent")["combination"])
    assert len(set(chosen)) > 1, f"it repeated itself: {chosen}"


def test_it_aims_the_ghost_near_the_opponent() -> None:
    """The ghost is where the combination's own last keyframe should land, so it is placed near
    where the opponent actually is, not at the fighter's own feet or the ring origin."""
    agent = _agent()
    messages = agent.decide(_state(10, anchor=(0.0, 0.0), opponent=(2.0, 0.0)), "red", NO_SLOTS)
    staged = next(m for m in messages if m["type"] == "intent")

    reach = math.hypot(staged["ghost"][0] - 2.0, staged["ghost"][1] - 0.0)
    assert reach < 2.0, "it aimed the ghost at least some of the way to the opponent"


def test_it_measures_from_its_anchor_not_from_where_it_stands() -> None:
    """Under a queue the next move starts where the last one leaves off, so aiming from the live
    position aims at the past."""
    agent = _agent()
    state = _state(10, anchor=(0.0, 0.0), opponent=(3.0, 2.0))
    state["seats"]["red"]["anchor"] = {"x": 0.0, "y": 2.0}
    state["seats"]["red"]["position"] = {"x": -5.0, "y": -5.0}

    staged = next(
        m for m in agent.decide(state, "red", NO_SLOTS) if m["type"] == "intent"
    )
    assert staged["ghost"][1] == pytest.approx(2.0), "it does not drift back to the live position"


# --- classification is honest and available, per name prefix, if it helps ---------------------------
def test_combination_kind_reads_the_recorded_prefixes() -> None:
    """`CLAUDE.md`'s three real corpus families. Anything unrecognised is treated as a strike — the
    majority family and the safe default: an agent that cannot classify a move should still throw it
    rather than freeze up."""
    assert combination_kind("shadow-boxing-r-001-a359-00") == "strike"
    assert combination_kind("ib-dodge-270-r-001-a437-00") == "defensive"
    assert combination_kind("ib-combat-turn-jog-start-270-r-003-a437-00") == "travel"
    assert combination_kind("some-future-family-00") == "strike"


def test_it_backs_off_in_a_clinch_rather_than_walking_into_the_opponent() -> None:
    """No control is bound to `stage`/`place` any more, so "backing out of a clinch" is just placing
    the ghost further from the opponent than the fighter already is — the same target expression as
    closing, gone negative."""
    agent = _agent()
    messages = agent.decide(_state(10, anchor=(0.0, 0.0), opponent=(0.2, 0.0)), "red", NO_SLOTS)
    staged = next(m for m in messages if m["type"] == "intent")

    assert staged["ghost"][0] < 0.0, "it steps away from the opponent, not into it"


# --- the limits --------------------------------------------------------------------------------------
def test_the_rate_limiter_allows_a_reasonable_client() -> None:
    limiter = RateLimiter()
    assert all(limiter.allow(now=index * 0.2) for index in range(5)), "a human peaks near 5/s"
    assert limiter.dropped == 0


def test_the_rate_limiter_drops_a_flood() -> None:
    limiter = RateLimiter()
    allowed = sum(1 for index in range(500) if limiter.allow(now=index * 1e-4))
    assert allowed == MAX_MESSAGES_PER_SECOND
    assert limiter.dropped == 500 - MAX_MESSAGES_PER_SECOND


def test_the_window_slides() -> None:
    limiter = RateLimiter(limit=2, window_s=1.0)
    assert limiter.allow(now=0.0) and limiter.allow(now=0.1)
    assert not limiter.allow(now=0.2)
    assert limiter.allow(now=1.5), "the first two have aged out"


def test_a_slow_decision_is_counted_not_discarded() -> None:
    """Discarding would be dishonest: the message exists and the host applies it when it arrives.
    The budget measures whether the agent is keeping up, it does not censor it."""
    import time

    class _Slow:
        def reset(self) -> None: ...

        def decide(self, state, seat, slots):
            time.sleep(DECISION_BUDGET_S * 1.5)
            return [{"type": "commit"}]

    stats = AgentStats()
    messages = run_decision(_Slow(), _state(0), "red", NO_SLOTS, stats)

    assert messages == [{"type": "commit"}], "the decision was still delivered"
    assert stats.over_budget == 1
    assert stats.mean_decision_ms > DECISION_BUDGET_S * 1000


def test_a_fast_decision_is_within_budget() -> None:
    stats = AgentStats()
    run_decision(_agent(), _state(200), "red", NO_SLOTS, stats)
    assert stats.over_budget == 0
    assert stats.decisions == 1


# --- exhibition, not the table --------------------------------------------------------------------
def test_an_agent_handle_is_exhibition() -> None:
    """`WORKPLAN` M5-T4 and the project definition §8 put agents outside the Season 0 table."""
    assert is_exhibition(f"{AGENT_PREFIX}baseline")
    assert not is_exhibition("carlo")


def test_a_human_handle_is_not_kept_out_of_the_table() -> None:
    assert not is_exhibition("agentina"), "a handle that merely starts with 'agent' is not enough"
