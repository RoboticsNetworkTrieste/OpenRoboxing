"""M5-T4: the agent API.

Acceptance criterion from WORKPLAN.md M5-T4:
  a scripted baseline agent plays a full ranked-format match against a human client and appears in
  the exhibition results, not the table.

An agent is just a client, so most of what needs testing is that it stays one: it sees only what the
protocol sends, it is rate-limited like anyone else, and its results are kept out of the table.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_agent.py -v
    .venv_mb/bin/python -m pytest tests/test_agent.py -v -m slow   # needs a GPU
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
    is_exhibition,
    run_decision,
)

SLOTS = ["1", "2", "3", "4", "5", "6"]


def _state(
    tick: int,
    can_commit: bool = True,
    hits: int = 0,
    phase: str = "fighting",
    separation: float | None = 0.6,
    anchor: tuple[float, float] | None = (0.0, 0.0),
    opponent: tuple[float, float] | None = (0.6, 0.0),
) -> dict:
    """One protocol frame. ``anchor``/``opponent`` are world positions; ``None`` drops them, which is
    what an older record or a replay looks like."""
    seats = {
        "red": {
            "handle": "agent:baseline",
            "staged": None,
            "queue": [],
            "queue_depth": 0,
            "can_commit": can_commit,
            "hits_landed": hits,
            "torso_height_m": 0.84,
            "down": False,
            "anchor": None if anchor is None else {"x": anchor[0], "y": anchor[1], "heading": 0.0},
            "position": None if anchor is None else {"x": anchor[0], "y": anchor[1], "heading": 0.0},
        },
        "blue": {
            "handle": "human",
            "staged": None,
            "queue": [],
            "queue_depth": 0,
            "can_commit": True,
            "hits_landed": 0,
            "torso_height_m": 0.84,
            "down": False,
            "position": (
                None if opponent is None else {"x": opponent[0], "y": opponent[1], "heading": 0.0}
            ),
        },
    }
    return {
        "type": "state",
        "tick": tick,
        "round": 1,
        "clock_ticks": 3000 - tick,
        "phase": phase,
        "separation_m": separation,
        "seats": seats,
    }


def _agent(seed: int = 0) -> BaselineAgent:
    """A baseline that has seen a welcome, so it knows a guard from a hook."""
    agent = BaselineAgent(seed=seed)
    agent.on_welcome({"loadout": dict(zip(SLOTS, [
        "guard", "jab-left", "jab-right", "hook-left", "hook-right", "uppercut-left"
    ]))})
    return agent


# --- an agent sees only what a client sees --------------------------------------------------------------
def test_the_baseline_agent_reads_nothing_privileged() -> None:
    """It is handed a state message and its own seat. There is nowhere else for it to look."""
    agent = _agent()
    messages = agent.decide(_state(tick=200, hits=1), "red", SLOTS)
    assert all(m["type"] in ("place", "stage", "commit") for m in messages)


def test_it_commits_by_staging_first() -> None:
    """`spec/intent.md` keeps staging and committing separate; an agent uses the same two steps."""
    agent = _agent()
    messages = agent.decide(_state(tick=200, hits=1), "red", SLOTS)
    assert [m["type"] for m in messages] == ["place", "stage", "commit"]
    assert messages[1]["slot"] in SLOTS


def test_it_does_not_commit_while_one_is_active() -> None:
    agent = _agent()
    assert agent.decide(_state(tick=200, can_commit=False, hits=1), "red", SLOTS) == []


def test_it_waits_a_beat_between_commits() -> None:
    """Otherwise it is only patient because the rate limiter made it so."""
    agent = _agent()
    assert agent.decide(_state(tick=200, hits=1), "red", SLOTS), "the first commit should fire"
    assert agent.decide(_state(tick=205, hits=2), "red", SLOTS) == [], "too soon"
    assert agent.decide(_state(tick=200 + agent.RECOVERY_TICKS + 1, hits=3), "red", SLOTS)


def test_it_cycles_its_loadout_rather_than_spamming_one_slot() -> None:
    """A repeated move is the easiest thing in the game to read."""
    agent = _agent()
    chosen = []
    tick = 0
    for _ in range(len(SLOTS) - 1):  # the stance is not thrown
        tick += agent.RECOVERY_TICKS + 1
        messages = agent.decide(_state(tick=tick, hits=len(chosen) + 1), "red", SLOTS)
        chosen.append(next(m for m in messages if m["type"] == "stage")["slot"])
    assert len(set(chosen)) == len(agent._strike_slots), f"it repeated itself: {chosen}"


def test_it_never_throws_at_nothing() -> None:
    """It can see the range, so there is nothing to guess.

    Under `spec/intent.md` 1.1 "throwing at nothing" is no longer "throwing while far away" — a
    commit walks the whole way and lands its punch on arrival. It is *placing the punch where the
    opponent is not*. So the test is on the placement, not on the slot.
    """
    agent = _agent()
    far = agent.decide(_state(tick=1, opponent=(3.0, 0.0)), "red", SLOTS)

    placed = next(m for m in far if m["type"] == "place")
    reach = math.hypot(placed["x"] - 3.0, placed["y"] - 0.0)
    assert reach == pytest.approx(agent.STRIKE_RANGE_M), (
        f"it committed a punch that arrives {reach:.2f} m from the opponent"
    )


def test_it_does_nothing_between_rounds() -> None:
    agent = _agent()
    assert agent.decide(_state(tick=500, hits=3, phase="round_over"), "red", SLOTS) == []


def test_an_agent_with_no_loadout_does_nothing() -> None:
    assert _agent().decide(_state(tick=200, hits=1), "red", []) == []


def test_a_reset_agent_forgets_the_round() -> None:
    agent = _agent()
    agent.decide(_state(tick=200, hits=1), "red", SLOTS)
    agent.reset()
    assert agent.decide(_state(tick=5, hits=1), "red", SLOTS), "it should be ready again"


def test_the_idle_agent_never_acts() -> None:
    agent = IdleAgent()
    assert all(agent.decide(_state(tick=t, hits=t), "red", SLOTS) == [] for t in range(0, 500, 50))


def test_two_agents_seeded_differently_open_with_different_moves() -> None:
    first = _agent(seed=0).decide(_state(tick=200, hits=1), "red", SLOTS)
    second = _agent(seed=1).decide(_state(tick=200, hits=1), "red", SLOTS)
    staged = [next(m for m in ms if m["type"] == "stage")["slot"] for ms in (first, second)]
    assert staged[0] != staged[1]


# --- it manages distance by placing, not steering (spec/intent.md 1.0/1.1) -----------------------
def test_it_closes_and_punches_in_one_commit() -> None:
    """The whole shape of the model: a commit is "go here and do this", and since 1.1 it walks the
    whole way. Spending a separate commit on the approach would burn a queue slot for nothing."""
    agent = _agent()
    messages = agent.decide(_state(10, opponent=(3.0, 0.0)), "red", SLOTS)

    kinds = [m["type"] for m in messages]
    assert kinds == ["place", "stage", "commit"]
    assert messages[1]["slot"] in agent._strike_slots, "the walk is inside the punch, not before it"

    placed = messages[0]
    assert placed["x"] == pytest.approx(3.0 - agent.STRIKE_RANGE_M)
    assert placed["y"] == pytest.approx(0.0)
    assert placed["heading"] == pytest.approx(0.0), "facing the opponent"


def test_it_aims_the_step_along_the_line_to_the_opponent() -> None:
    agent = _agent()
    messages = agent.decide(_state(10, opponent=(0.0, 3.0)), "red", SLOTS)
    placed = next(m for m in messages if m["type"] == "place")

    assert placed["x"] == pytest.approx(0.0)
    assert placed["y"] == pytest.approx(3.0 - agent.STRIKE_RANGE_M)
    assert placed["heading"] == pytest.approx(math.pi / 2)


def test_it_measures_from_its_anchor_not_from_where_it_stands() -> None:
    """Under a queue the next move starts where the last one leaves off, so aiming from the live
    position aims at the past. Chosen so the two disagree about the *heading*, which a target on the
    line between them would hide."""
    agent = _agent()
    state = _state(10, anchor=(0.0, 0.0), opponent=(3.0, 2.0))
    state["seats"]["red"]["anchor"] = {"x": 0.0, "y": 2.0, "heading": 0.0}

    placed = next(m for m in agent.decide(state, "red", SLOTS) if m["type"] == "place")
    assert placed["heading"] == pytest.approx(0.0), "from the anchor the opponent is due +x"
    assert placed["y"] == pytest.approx(2.0), "it does not drift back to the live position"


def test_a_throw_also_carries_it_to_striking_distance() -> None:
    """A commit is "go here and do this", so a punch places itself where the punch can land rather
    than being thrown from wherever the fighter happens to be."""
    agent = _agent()
    messages = agent.decide(_state(10, opponent=(0.95, 0.0)), "red", SLOTS)

    staged = next(m for m in messages if m["type"] == "stage")
    placed = next(m for m in messages if m["type"] == "place")
    assert staged["slot"] in agent._strike_slots
    assert placed["x"] == pytest.approx(0.95 - agent.STRIKE_RANGE_M)


def test_it_backs_out_of_a_clinch() -> None:
    """The same target expression as closing, gone negative: end up at striking distance."""
    agent = _agent()
    messages = agent.decide(_state(10, opponent=(0.2, 0.0)), "red", SLOTS)
    placed = next(m for m in messages if m["type"] == "place")

    assert placed["x"] == pytest.approx(0.2 - agent.STRIKE_RANGE_M)
    assert placed["x"] < 0.0, "it steps away from the opponent, not into them"
    assert next(m for m in messages if m["type"] == "stage")["slot"] == agent._stance_slot


def test_it_throws_rather_than_shuffling_when_it_can_reach() -> None:
    agent = _agent()
    messages = agent.decide(_state(10, opponent=(0.8, 0.0)), "red", SLOTS)

    staged = next(m for m in messages if m["type"] == "stage")
    assert staged["slot"] in agent._strike_slots
    assert staged["slot"] != agent._stance_slot


def test_a_small_gap_is_not_worth_a_step() -> None:
    """Without a deadband it commits a stance for every few centimetres and never punches."""
    agent = _agent()
    just_short = agent.STRIKE_RANGE_M + agent.STEP_DEADBAND_M - 0.05  # inside the deadband
    staged = next(
        m for m in agent.decide(_state(10, opponent=(just_short, 0.0)), "red", SLOTS)
        if m["type"] == "stage"
    )
    assert staged["slot"] in agent._strike_slots


def test_without_a_stance_it_can_still_close_but_cannot_back_off() -> None:
    """At 1.0 a fighter with no stance could not walk at all, because closing was a separate commit.
    Since 1.1 the walk is inside the punch, so it closes fine — what it loses is the one commit that
    is *not* a punch: stepping back out of a clinch, where throwing is not an option."""
    agent = BaselineAgent()
    agent.on_welcome({"loadout": {"1": "jab-left", "2": "hook-right"}})
    assert agent._stance_slot is None

    far = agent.decide(_state(10, opponent=(3.0, 0.0)), "red", ["1", "2"])
    assert [m["type"] for m in far] == ["place", "stage", "commit"], "it can still close and punch"

    agent.reset()
    clinch = agent.decide(_state(10, opponent=(0.2, 0.0)), "red", ["1", "2"])
    assert clinch == [], "with nothing but punches there is no way to step back out"


def test_it_still_works_without_positions() -> None:
    """A record replayed through an older protocol has no positions; it must not crash, and it must
    not invent a placement out of nothing."""
    agent = _agent()
    messages = agent.decide(
        _state(200, hits=1, separation=0.6, anchor=None, opponent=None), "red", SLOTS
    )
    assert [m["type"] for m in messages] == ["stage", "commit"], "no placement it cannot know"

    far = agent.decide(
        _state(400, separation=3.0, anchor=None, opponent=None), "red", SLOTS
    )
    assert far == [], "and it does not throw at a range it cannot reach"


# --- the limits ----------------------------------------------------------------------------------------
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
    messages = run_decision(_Slow(), _state(0), "red", SLOTS, stats)

    assert messages == [{"type": "commit"}], "the decision was still delivered"
    assert stats.over_budget == 1
    assert stats.mean_decision_ms > DECISION_BUDGET_S * 1000


def test_a_fast_decision_is_within_budget() -> None:
    stats = AgentStats()
    run_decision(BaselineAgent(), _state(200, hits=1), "red", SLOTS, stats)
    assert stats.over_budget == 0
    assert stats.decisions == 1


# --- exhibition, not the table --------------------------------------------------------------------------
def test_an_agent_handle_is_exhibition() -> None:
    """`WORKPLAN` M5-T4 and the project definition §8 put agents outside the Season 0 table."""
    assert is_exhibition(f"{AGENT_PREFIX}baseline")
    assert not is_exhibition("carlo")


def test_a_human_handle_is_not_kept_out_of_the_table() -> None:
    assert not is_exhibition("agentina"), "a handle that merely starts with 'agent' is not enough"


# --- the whole thing ------------------------------------------------------------------------------------
@pytest.mark.slow
def test_a_baseline_agent_plays_a_full_match_and_commits() -> None:
    """M5-T4's acceptance, minus the human: the agent joins over a real socket, plays a real match
    and its commits reach the record."""
    import asyncio

    from openroboxing.paths import LOADOUT_DIR
    from openroboxing.runtime.arena import FIGHTERS
    from openroboxing.runtime.intents import Loadout
    from openroboxing.runtime.match import MatchFormat
    from openroboxing.server.client import play_match
    from openroboxing.server.host import MatchHost

    loadout = Loadout.load(LOADOUT_DIR / "orthodox.json")

    async def run():
        host = MatchHost(
            loadouts={f: loadout for f in FIGHTERS},
            match_format=MatchFormat(rounds=1, round_ticks=600, get_up_window_ticks=400),
            match_id="agent-test",
            render=False,
        )
        return await play_match(
            host,
            {"red": BaselineAgent(seed=0), "blue": IdleAgent()},
            handles={"red": f"{AGENT_PREFIX}baseline", "blue": "human"},
        )

    record, stats = asyncio.run(run())

    assert len(record.rounds) == 1
    commits = record.rounds[0].commits
    assert commits, "the agent never committed anything"
    assert all(c["fighter"] == "red" for c in commits), "the idle seat committed"
    assert stats["red"].decisions > 0
    assert stats["red"].over_budget == 0, "the baseline agent must fit its own budget"
