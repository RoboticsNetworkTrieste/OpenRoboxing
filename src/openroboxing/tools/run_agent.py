"""Connect an agent to a running match host (M5-T4).

Acceptance criterion from WORKPLAN.md M5-T4:
  a scripted baseline agent plays a full ranked-format match against a human client and appears in
  the exhibition results, not the table.

An agent is just a client. Start a host, open the browser for the human seat, then point this at the
other seat — the host cannot tell the difference and does not need to.

Usage
-----
    # terminal 1
    python -m openroboxing.tools.serve_match --no-wait
    # terminal 2
    python -m openroboxing.tools.run_agent --seat blue
    # then open http://localhost:8080/ and play red

    python -m openroboxing.tools.run_agent --seat blue --latency-ms 200   # handicap it
    python -m openroboxing.tools.run_agent --seat blue --agent idle       # a punchbag
"""

from __future__ import annotations

import argparse
import asyncio

from openroboxing.server.agent import AGENT_PREFIX, BaselineAgent, IdleAgent, is_exhibition

AGENTS = {"baseline": BaselineAgent, "idle": IdleAgent}


async def _play(url: str, seat: str, handle: str, kind: str, latency_ms: float, seed: int) -> dict:
    from aiohttp import ClientSession

    from openroboxing.server.client import AgentConnection

    agent = AGENTS[kind]() if kind == "idle" else AGENTS[kind](seed=seed)
    connection = AgentConnection(agent, seat, handle=handle, latency_ms=latency_ms)

    async with ClientSession() as session:
        stats = await connection.play(session, f"{url}/ws?seat={seat}")
    return stats.summary()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_agent", description="Play a seat with a scripted agent (M5-T4)."
    )
    parser.add_argument("--url", default="http://localhost:8080", help="the match host")
    parser.add_argument("--seat", default="blue", choices=("red", "blue"))
    parser.add_argument("--agent", default="baseline", choices=sorted(AGENTS))
    parser.add_argument("--handle", default=None, help=f"defaults to {AGENT_PREFIX}<agent>")
    parser.add_argument("--seed", type=int, default=0, help="which slot the agent opens with")
    parser.add_argument(
        "--latency-ms", type=float, default=0.0, help="artificial delay on this client's sends"
    )
    args = parser.parse_args(argv)

    handle = args.handle or f"{AGENT_PREFIX}{args.agent}"
    print(f"connecting {handle} to {args.url} as {args.seat}")
    if not is_exhibition(handle):
        print(
            f"  note: {handle!r} does not start with {AGENT_PREFIX!r}, so its results would count "
            "towards the Season 0 table. Agents belong in the exhibition track."
        )

    stats = asyncio.run(
        _play(args.url, args.seat, handle, args.agent, args.latency_ms, args.seed)
    )

    print("\nagent finished")
    print(f"  frames seen     : {stats['frames']:.0f}")
    print(f"  decisions       : {stats['decisions']:.0f}")
    print(f"  mean decision   : {stats['mean_decision_ms']:.3f} ms")
    print(f"  over budget     : {stats['over_budget']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
