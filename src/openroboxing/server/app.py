"""The websocket server around a :class:`~openroboxing.server.host.MatchHost` (M4-T2).

aiohttp, because it is already in the environment — a match host that needs a new dependency is a
match host nobody at a meetup can install.

One process, one match. A league runs many of these; `M5-T1` owns that and this does not pretend to.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from openroboxing.paths import OPENROBOXING_ROOT
from openroboxing.runtime.arena import FIGHTERS
from openroboxing.server import protocol
from openroboxing.server.host import MatchHost

CLIENT_DIR = OPENROBOXING_ROOT / "client"

#: How a browser asks for a seat: ``/ws?seat=red``. Omitting it takes the first free one.
SEAT_QUERY = "seat"

#: Ask for this instead of a fighter to watch without playing. Unlimited; occupies no seat.
SPECTATOR = "spectator"


def build_app(host: MatchHost, client_dir: Path = CLIENT_DIR):
    """An aiohttp application serving the client and one websocket per seat."""
    from aiohttp import WSMsgType, web

    # Typed keys rather than bare strings: aiohttp warns about the latter, and two apps in one
    # process (a league host runs several) must not collide in a shared namespace.
    key_host = web.AppKey("host", MatchHost)
    key_taken = web.AppKey("taken", set)

    app = web.Application()
    app[key_host] = host
    app[key_taken] = set()
    app["taken_key"] = key_taken  # exposed for tests that inspect occupancy
    mesh_blob: bytes | None = None

    async def index(request):
        return web.FileResponse(client_dir / "index.html")

    async def scene_json(request):
        """The static half of the world, once. See `spec/protocol.md` §"Scene description"."""
        return web.json_response(host.scene.description())

    async def meshes_bin(request):
        """Every unique mesh, concatenated. ~10 MB, fetched once and then cached by the browser.

        Built on demand rather than at startup so a headless match never pays for it, and cached in
        the closure so two clients joining together do not build it twice.
        """
        nonlocal mesh_blob
        if mesh_blob is None:
            mesh_blob = host.scene.mesh_blob()
        return web.Response(
            body=mesh_blob,
            content_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    async def websocket(request):
        socket = web.WebSocketResponse(heartbeat=20.0)
        await socket.prepare(request)

        requested = request.query.get(SEAT_QUERY)
        taken: set[str] = app[key_taken]

        # A spectator watches and cannot play. Unlimited of them, and none occupies a seat — the
        # fight-night screen (M5-T3) is one, and so is anybody who opens the page to look.
        if requested == SPECTATOR:
            host.subscribe(socket, viewer=None)
            await socket.send_json(host.spectator_welcome())
            await socket.send_json(host.state_message(None))
            try:
                async for message in socket:
                    if message.type is not WSMsgType.TEXT:
                        continue
                    try:
                        parsed = protocol.parse(json.loads(message.data))
                    except (ValueError, protocol.ProtocolError) as exc:
                        await socket.send_json(protocol.error(str(exc)))
                        continue
                    if parsed["type"] == "ping":
                        await socket.send_json(protocol.pong(parsed.get("t")))
                    else:
                        await socket.send_json(
                            protocol.error("spectators cannot play", rejected=parsed["type"])
                        )
            finally:
                host.unsubscribe(socket)
            return socket

        seat = requested if requested in FIGHTERS else next(
            (f for f in FIGHTERS if f not in taken), None
        )
        if seat is None or seat in taken:
            await socket.send_json(protocol.error(f"seat {seat or requested!r} is taken"))
            await socket.close()
            return socket

        taken.add(seat)
        host.subscribe(socket, viewer=seat)
        await socket.send_json(host.welcome_message(seat))
        await socket.send_json(host.state_message(seat))

        try:
            async for message in socket:
                if message.type is not WSMsgType.TEXT:
                    continue
                try:
                    parsed = protocol.parse(json.loads(message.data))
                except (ValueError, protocol.ProtocolError) as exc:
                    await socket.send_json(protocol.error(str(exc)))
                    continue
                reply = host.handle(seat, parsed)
                if reply is not None:
                    await socket.send_json(reply)
        finally:
            host.unsubscribe(socket)
            taken.discard(seat)
        return socket

    async def screen(request):
        return web.FileResponse(client_dir / "screen.html")

    app.router.add_get("/", index)
    app.router.add_get("/screen", screen)
    app.router.add_get("/ws", websocket)
    # The static half of the world (`spec/protocol.md` 0.4). Fetched once per client, then cached.
    app.router.add_get("/scene.json", scene_json)
    app.router.add_get("/meshes.bin", meshes_bin)
    app.router.add_static("/static/", path=client_dir, name="static")
    return app


async def serve(host: MatchHost, port: int = 8080, wait_for_players: bool = True) -> Any:
    """Serve the client, wait for both seats, then run the match to its end."""
    from aiohttp import web

    app = build_app(host)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"  play    http://localhost:{port}/        (hotseat: both seats in one page)")
    print(f"  project http://localhost:{port}/screen  (fight-night screen, no controls)")

    if wait_for_players:
        print("  waiting for both seats...")
        while len(app[app["taken_key"]]) < len(FIGHTERS):
            await asyncio.sleep(0.25)
        print(f"  {len(app[app['taken_key']])} seats taken; starting")

    try:
        return await host.run()
    finally:
        await runner.cleanup()
