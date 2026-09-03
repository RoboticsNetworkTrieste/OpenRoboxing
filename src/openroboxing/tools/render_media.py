"""Render the README's GIFs: the fight in the ring, and the punches that land.

What these are, and are not
---------------------------
They are the **simulation**, rendered offscreen by MuJoCo from the arena's own model — the same
world the browser client draws when it receives streamed body transforms. They are *not* a screen
capture of the web UI, so they carry none of its chrome: no picker, no queue, no minimap.

Two clips, each from its own fight, because they want opposite things:

- ``ring.gif`` — a broadcast view from the shipped starting distance, where the fighters box at
  range. This is what a match looks like.
- ``contacts.gif`` — a second fight started at punching distance, close in, with a marker at every
  punch that lands. At the shipped distance nothing lands at all (see ``CLOSE_START_M``), so a clip
  about contact has to begin inside it.

Why the contacts clip is rendered in a second pass
--------------------------------------------------
A marker has to sit where a *landed punch* is, and what counts as one is
``runtime/contact.py``'s decision, not this file's: a glove geom against an opposing fighter's body,
with a non-zero normal force. ``ContactTracker`` only yields a :class:`HitEvent` once the episode
closes, which is after the punch is visibly over — so drawing markers live would put them a few
frames late, and re-deriving "is this a punch" here would let the picture disagree with the scoring.

Instead the fight is simulated once, keeping a qpos snapshot per frame and the authoritative hit
list, and the frames are rendered afterwards from those snapshots. Nothing is simulated twice and
the markers are the real hits.

The library ships all-draft (nothing has measured telegraph or tracking error yet), so the world is
built with ``require_admitted=False``. That is legitimate here and would not be in a match: admission
gates *scoring*, not whether the recorded motion is real.

Encoding is ffmpeg's two-pass palette pipeline — one pass to build a palette from the whole clip,
one to apply it. A single-pass GIF quantises per frame and visibly shimmers on the canvas.

Run: ``.venv_mb/bin/python -m openroboxing.tools.render_media --help``
"""

from __future__ import annotations

import os

# Before anything imports mujoco — see `studio/render.py`, which explains why the timing matters.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from openroboxing.paths import COMBINATION_DIR, OPENROBOXING_ROOT
from openroboxing.runtime.arena import ArenaConfig
from openroboxing.runtime.contact import ContactTracker
from openroboxing.runtime.fight import FIGHTERS, FightWorld, ScriptedPilot
from openroboxing.studio import combination_record as cr

#: Rendered size before cropping. 16:9, and large enough that the crops below still resample down.
WIDTH, HEIGHT = 720, 405

#: Every Nth control tick becomes a frame. 3 gives 50/3 = 16.7 fps, which reads as motion without
#: tripling the file size.
STRIDE = 3

#: How long a hit marker stays on screen after its episode ends, in control ticks. A punch lands in
#: a handful of ticks and would otherwise flash by inside a single GIF frame.
FLASH_TICKS = 18

#: Marker radius in metres, and its colour: the same orange the client uses for a rejected ghost, so
#: the two never read as the same thing.
MARKER_RADIUS_M = 0.075
MARKER_RGBA = (1.0, 0.42, 0.12, 0.9)

#: How far each fighter's ghost sits from the ring's centre line, and how much its lateral offset
#: alternates. They start at x = -1.2 (red) and +1.2 (blue).
#:
#: **Measured against the reach, not chosen for looks.** `spec/scoring.md` puts the G1's hand at
#: 0.38 m from its own pelvis and the torso surface roughly 0.15 m out, so a punch only lands when
#: the two pelvises are inside about 0.53 m. At 2 x 0.35 m plus a 0.22 m lateral stagger the gap was
#: 0.67 m and **nothing landed at all** — the first run of this tool reported zero hits.
CLOSE_X = 0.20
STAGGER_Y = 0.05

#: Where each fight starts, as distance from the ring's centre. ``ring.gif`` uses `ArenaConfig`'s
#: shipped 1.20 m; ``contacts.gif`` needs far less, and the number is measured rather than picked.
#:
#: `ArenaConfig` says of its 1.20 m: "two arm's reach apart, so neither can land without closing".
#: Measured here, they never close enough — the first two runs of this tool landed **zero** punches,
#: with the fighters reaching 0.777 m apart and their gloves stopping 0.535 m from the opponent's
#: pelvis. That is a forward reach of 0.24 m during shadow boxing, well under the 0.38 m of a fully
#: extended arm, because these punches are thrown from guard rather than straightened out. A glove
#: touches a torso surface roughly 0.15 m out from its pelvis only inside about 0.39 m of pelvis
#: separation. Swept: 0.30 m still lands nothing, 0.22 m lands 17 punches over 8 seconds.
RANGE_START_M = 1.20
CLOSE_START_M = 0.22

#: Shadow boxing throughout: it is the corpus family that actually punches, and a clip meant to show
#: contacts should not spend half its length on footwork. Prefixes rather than exact names, so the
#: clip survives a library rebuild renaming its records. The two fighters take different takes so
#: they do not mirror each other move for move.
RED_PREFIXES = ("shadow-boxing-r-001-a359", "shadow-boxing-r-002-a361", "shadow-boxing-r-003-a362")
BLUE_PREFIXES = ("shadow-boxing-r-003-a360", "shadow-boxing-r-001-a362", "shadow-boxing-r-002-a360")


def load_library(directory: Path) -> dict:
    records = {}
    for path in sorted(directory.glob("*.json")):
        record = cr.load(path)
        records[record.name] = record
    if not records:
        raise SystemExit(f"no combinations in {directory}; run tools.import_motions first")
    return records


def choose(library: dict, prefixes: tuple[str, ...], per_prefix: int = 2) -> list[str]:
    names: list[str] = []
    for prefix in prefixes:
        names += [n for n in sorted(library) if n.startswith(prefix)][:per_prefix]
    if not names:
        raise SystemExit(f"no combination matched any of {prefixes}")
    return names


def script(library: dict, names: list[str], side: float) -> list[tuple[int, str, tuple]]:
    """Commit each move in turn, closing to punching range but staying on this fighter's own side.

    ``side`` is -1 for red (which starts at x = -1.2) and +1 for blue. Aiming a ghost *past* centre
    walks the two through each other, which renders as one interpenetrated blob.
    """
    out, tick = [], 5
    for i, name in enumerate(names):
        out.append((tick, name, (side * CLOSE_X, STAGGER_Y * ((-1) ** i))))
        tick += library[name].duration_ticks + 6
    return out


def encode(frames: Path, out: Path, *, filters: str, fps: int) -> None:
    """Two-pass palette GIF. Raises if ffmpeg is missing rather than leaving a half-made file."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed; it encodes the GIFs")
    palette = frames / "palette.png"
    for pass_args in (
        ["-vf", f"{filters},palettegen=max_colors=96:stats_mode=diff", str(palette)],
        [
            "-i", str(palette),
            "-lavfi", f"{filters} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4",
            "-loop", "0", str(out),
        ],
    ):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(frames / "%04d.png"), *pass_args],
            check=True,
        )


def _add_marker(mujoco, scene, position, alpha: float) -> None:
    """Append one translucent sphere to an already-updated scene.

    Appended rather than added to the model: a debug annotation must not become part of the world
    the physics sees.
    """
    if scene.ngeom >= scene.maxgeom:
        return
    import numpy as np

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([MARKER_RADIUS_M] * 3),
        np.asarray(position, dtype=float),
        np.eye(3).flatten(),
        np.array([*MARKER_RGBA[:3], alpha], dtype=float),
    )
    scene.ngeom += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OPENROBOXING_ROOT.parent.parent / "docs/media")
    parser.add_argument("--library", type=Path, default=COMBINATION_DIR)
    parser.add_argument("--ticks", type=int, default=420, help="control ticks to simulate (50 Hz)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--range-start", type=float, default=RANGE_START_M,
                        help="start distance from centre for ring.gif, metres")
    parser.add_argument("--close-start", type=float, default=CLOSE_START_M,
                        help="start distance from centre for contacts.gif, metres")
    args = parser.parse_args()

    import mujoco
    import numpy as np
    from PIL import Image

    library = load_library(args.library)
    red, blue = choose(library, RED_PREFIXES), choose(library, BLUE_PREFIXES)
    print(f"library {len(library)} combinations\n  red  {red}\n  blue {blue}", flush=True)

    def simulate(separation: float):
        """One fight. Returns ``(world, snapshots, hits)`` — the qpos per frame and the real hits.

        Rendering happens afterwards, from the snapshots: a marker has to sit on a *landed punch*,
        and ``ContactTracker`` only yields one when the episode closes, which is after the punch is
        visibly over.
        """
        world = FightWorld(
            libraries={f: library for f in FIGHTERS},
            pilots={
                "red": ScriptedPilot(script(library, red, -1.0)),
                "blue": ScriptedPilot(script(library, blue, +1.0)),
            },
            require_admitted=False,
            match_seed=args.seed,
            config=ArenaConfig(start_separation=separation),
        )
        world.reset_round(0)
        tracker = ContactTracker()
        frames: list[tuple[int, object]] = []
        for tick in range(args.ticks):
            world.step(tick)
            tracker.observe(world.model, world.data, tick)
            if tick % STRIDE == 0:
                frames.append((tick, world.data.qpos.copy()))
        landed = tracker.flush()
        print(f"  start {separation:.2f} m -> {len(frames)} frames, {len(landed)} landed punches",
              flush=True)
        return world, frames, landed

    def camera(azimuth, elevation, distance, height):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        cam.lookat[:] = (0.0, 0.0, height)
        return cam

    def shoot(world, frames, hits, cam, directory: Path) -> None:
        """Render one clip from saved states, flashing a marker on each landed punch."""
        model, data = world.model, world.data
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
        model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
        renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        try:
            for index, (tick, qpos) in enumerate(frames):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                for hit in hits:
                    age = tick - hit.end_tick      # negative while the punch is still landing
                    if tick < hit.start_tick or age > FLASH_TICKS:
                        continue
                    fade = 1.0 if age <= 0 else 1.0 - age / FLASH_TICKS
                    _add_marker(mujoco, renderer.scene, hit.position, MARKER_RGBA[3] * fade)
                Image.fromarray(np.asarray(renderer.render())).save(directory / f"{index:04d}.png")
        finally:
            renderer.close()

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ring_dir, hit_dir = Path(tmp) / "ring", Path(tmp) / "hits"
        ring_dir.mkdir()
        hit_dir.mkdir()

        print("boxing at range:", flush=True)
        at_range = simulate(args.range_start)
        # Markers are drawn on this clip too — there is just almost nothing to draw. At the shipped
        # distance a punch lands about once in eight seconds (one, in the run this was tuned on),
        # which is the point `ArenaConfig.start_separation` is making about having to close.
        shoot(at_range[0], at_range[1], at_range[2], camera(90.0, -9.0, 4.5, 1.00), ring_dir)

        print("boxing in close:", flush=True)
        in_close = simulate(args.close_start)
        shoot(in_close[0], in_close[1], in_close[2], camera(90.0, -10.0, 2.2, 1.05), hit_dir)
        for hit in in_close[2][:6]:
            print(f"  t={hit.start_tick:4d} {hit.attacker} -> {hit.defender} {hit.region}"
                  f" {hit.peak_force_n:6.1f} N", flush=True)

        encode(ring_dir, args.out / "ring.gif",
               filters="crop=560:315:80:90,scale=640:-1:flags=lanczos", fps=16)
        encode(hit_dir, args.out / "contacts.gif",
               filters="crop=600:338:60:34,scale=560:-1:flags=lanczos", fps=16)

    for name in ("ring.gif", "contacts.gif"):
        path = args.out / name
        print(f"{path} — {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
