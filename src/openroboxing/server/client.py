"""A headless websocket client, for agents and for tests (M5-T4).

The same socket a browser opens. Nothing here is privileged — it is the reference implementation of
``spec/protocol.md`` from the client side, and it is what `tools/run_agent.py` and the latency A/B
both drive.

Latency injection
-----------------
:class:`AgentConnection` can delay its own sends by a fixed amount. That is how `WORKPLAN` M4-T2's
requirement — "injecting 200 ms latency does not change match outcomes systematically" — is measured
without a network emulator: the delay is applied where a real one would land, between the decision
and the host receiving it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from openroboxing.server.agent import (
    Agent,
    AgentError,
    AgentStats,
    RateLimiter,
    run_decision,
)


class AgentConnection:
    """One agent in one seat, over one websocket.

    Args:
        agent: the decision-maker.
        seat: which fighter to ask for.
        handle: what to register as. `agent:` prefixed handles land in the exhibition list.
        latency_ms: artificial one-way delay on this client's sends, for the M4-T2 A/B.
    """

    def __init__(
        self,
        agent: Agent,
        seat: str,
        handle: str = "agent:baseline",
        latency_ms: float = 0.0,
    ) -> None:
        if latency_ms < 0:
            raise AgentError(f"latency_ms must not be negative, got {latency_ms}")
        self.agent = agent
        self.seat = seat
        self.handle = handle
        self.latency_ms = latency_ms
        self.stats = AgentStats()
        self.limiter = RateLimiter()
        self.slots: list[str] = []
        self.done = asyncio.Event()

    async def _send(self, socket, message: dict) -> None:
        """Send, honouring the injected latency and the rate limit.

        The limit is applied client-side too. The host enforces its own; doing it here as well means
        a well-behaved agent never trips it, and the drop count is visible to whoever wrote it.
        """
        if not self.limiter.allow():
            return
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
        await socket.send_str(json.dumps(message))

    async def play(self, session, url: str) -> AgentStats:
        """Connect, play until the match ends, and return what it cost."""
        async with session.ws_connect(url) as socket:
            await self._send(socket, {"type": "join", "handle": self.handle})

            async for raw in socket:
                if raw.type.name == "BINARY":
                    self.stats.frames += 1
                    continue
                if raw.type.name != "TEXT":
                    continue

                message = json.loads(raw.data)
                kind = message.get("type")

                if kind == "welcome":
                    self.slots = sorted(message.get("loadout", {}))
                    # Optional hook: an agent that wants to know *which* pose each slot holds gets
                    # the welcome. The `Agent` protocol stays two methods for anyone who does not.
                    if hasattr(self.agent, "on_welcome"):
                        self.agent.on_welcome(message)
                    self.agent.reset()
                elif kind == "state":
                    for reply in run_decision(
                        self.agent, message, self.seat, self.slots, self.stats
                    ):
                        await self._send(socket, reply)
                elif kind == "event" and message.get("event") == "round_end":
                    self.agent.reset()
                elif kind == "event" and message.get("event") == "match_end":
                    break

        self.done.set()
        return self.stats


async def play_match(
    host,
    agents: dict[str, Agent],
    handles: dict[str, str] | None = None,
    latency_ms: dict[str, float] | None = None,
    port: int = 0,
) -> tuple[Any, dict[str, AgentStats]]:
    """Run a whole match in-process with agents in every seat. Returns ``(record, stats)``.

    Used by the latency A/B and by the M5-T4 acceptance run. A real agent connects over a real
    network to :func:`~openroboxing.server.app.serve`; this is the same code path with the server
    and clients in one process, which is what makes a hundred matches practical.
    """
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer

    from openroboxing.server.app import build_app

    handles = handles or {}
    latency_ms = latency_ms or {}

    server = TestServer(build_app(host), port=port or None)
    await server.start_server()
    base = f"http://{server.host}:{server.port}"

    connections = {
        seat: AgentConnection(
            agent,
            seat,
            handle=handles.get(seat, f"agent:{seat}"),
            latency_ms=latency_ms.get(seat, 0.0),
        )
        for seat, agent in agents.items()
    }

    async with ClientSession() as session:
        players = [
            asyncio.create_task(conn.play(session, f"{base}/ws?seat={seat}"))
            for seat, conn in connections.items()
        ]
        # Let both sockets take their seat before the bell.
        await asyncio.sleep(0.4)
        record = await host.run()
        for task in players:
            task.cancel()
        await asyncio.gather(*players, return_exceptions=True)

    await server.close()
    return record, {seat: conn.stats for seat, conn in connections.items()}


def slots_of(loadout) -> Sequence[str]:
    return sorted(loadout.slots)
