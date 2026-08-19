"""Replays: a recorded match, played back (M3-T5).

A match record plus its state trace is everything needed to see the fight again, and to argue about
it. Two things are built on that here, and they are not the same thing:

:class:`ReplayWorld`
    A ``MatchWorld`` that plays the trace back into a real arena, so :class:`~openroboxing.runtime.
    match.Match` re-runs **the rules** over a recording — no GPU, no generator, no policy. This is
    what makes ``match.py``'s claim true: a disputed knockdown is re-derivable from the trace alone.
:func:`render_round`
    The picture. Frames from the trace through MuJoCo's offscreen renderer, optionally encoded.

What replays exactly, and what does not
---------------------------------------
The trace is ``qpos``. Everything that is a *function of position* comes back exactly: where the
fighters are, how far apart, torso height, torso orientation — and therefore **every knockdown and
knockout**, which is the part a league has to be able to settle.

**Contact forces do not.** They depend on velocity and acceleration, which the trace does not carry.
Velocities are reconstructed with ``mj_differentiatePos`` (the right tool: it handles the free
joint's quaternion properly) and that gets close, but a replayed ``peak_force_n`` is a reconstruction
and must not be presented as the recorded one. The hits recorded *at simulation time* are in the
record; use those. This module never overwrites them.

Conventions
-----------
- A record is ``<name>.json`` with ``<name>.trace.npz`` beside it, as :meth:`MatchRecord.save` writes
  them. The trace is authoritative (`spec/match_record.md`).
- Frames are ``(height, width, 3)`` uint8 RGB, matching `studio/render.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any, Iterator

import numpy as np

from openroboxing.runtime.arena import ArenaConfig, build_arena
from openroboxing.runtime.match import MatchFormat
from openroboxing.spec.constants import TICK_DT

#: Cameras the arena defines. ``broadcast`` is the side-on shot; ``overhead`` looks down.
DEFAULT_CAMERA = "broadcast"

#: Frame size for a replay. 720p-ish but portrait-agnostic; the ring is wider than it is tall.
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540


class ReplayError(RuntimeError):
    """A match could not be loaded or replayed. Never recovered from silently."""


def _as_arena_config(data: dict[str, Any]) -> ArenaConfig:
    """Rebuild an :class:`ArenaConfig` from JSON, restoring the tuples JSON flattens to lists."""
    fields = {f: data[f] for f in ArenaConfig.__dataclass_fields__ if f in data}
    for name in ("rope_heights", "glove_solref"):
        if name in fields:
            fields[name] = tuple(fields[name])
    return ArenaConfig(**fields)


@dataclass
class RecordedMatch:
    """A match record and its trace, loaded back off disk."""

    record: dict[str, Any]
    traces: dict[int, np.ndarray] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | str) -> RecordedMatch:
        """Load ``<name>.json`` and the ``<name>.trace.npz`` beside it.

        A missing trace raises. The record's JSON alone is not a replay — it is a summary of one, and
        treating it as a replay would silently produce an empty fight.
        """
        path = Path(path)
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayError(f"{path}: cannot read the match record ({exc})") from exc

        trace_path = path.with_suffix(".trace.npz")
        if not trace_path.exists():
            raise ReplayError(
                f"{path}: no trace at {trace_path.name}. The trace is the authoritative record "
                "(spec/match_record.md); the JSON alone cannot be replayed"
            )
        with np.load(trace_path) as archive:
            traces = {int(name.split("_")[1]): archive[name] for name in archive.files}

        missing = [r["index"] for r in record.get("rounds", []) if r["index"] not in traces]
        if missing:
            raise ReplayError(f"{path}: rounds {missing} have no trace in {trace_path.name}")
        return cls(record=record, traces=traces, path=path)

    # -- reading it ---------------------------------------------------------------------------------
    @property
    def match_id(self) -> str:
        return self.record.get("match_id", "match")

    @property
    def round_count(self) -> int:
        return len(self.record.get("rounds", []))

    def format(self) -> MatchFormat:
        fmt = self.record.get("format")
        if fmt is None:
            raise ReplayError(f"{self.match_id}: the record carries no format")
        return MatchFormat(**fmt)

    def arena_config(self) -> ArenaConfig:
        """The ring this match was fought in, or today's defaults if the record predates `arena`.

        Falling back is safe *and* worth saying out loud: a record written before `spec/
        match_record.md` 0.2 was fought in the defaults, because that is all there was.
        """
        return _as_arena_config(self.record.get("arena", {}))

    def trace(self, index: int) -> np.ndarray:
        if index not in self.traces:
            raise ReplayError(f"no trace for round {index}; have {sorted(self.traces)}")
        return self.traces[index]

    def round(self, index: int) -> dict[str, Any]:
        for entry in self.record.get("rounds", []):
            if entry["index"] == index:
                return entry
        raise ReplayError(f"no round {index} in {self.match_id}")

    def commits(self, index: int) -> list[dict[str, Any]]:
        return list(self.round(index).get("commits", []))


class ReplayWorld:
    """A ``MatchWorld`` driven by a recording instead of by physics.

    Feeding this to :class:`~openroboxing.runtime.match.Match` re-runs the rules over a recorded
    fight. Knockdowns come back exactly; see the module docstring on why forces do not.
    """

    def __init__(self, recorded: RecordedMatch, model=None) -> None:
        import mujoco

        self._mujoco = mujoco
        self.recorded = recorded
        self.config = recorded.arena_config()
        self.model = model if model is not None else build_arena(self.config)
        self.data = mujoco.MjData(self.model)

        for index in sorted(recorded.traces):
            trace = recorded.trace(index)
            if trace.ndim != 2 or trace.shape[1] != self.model.nq:
                raise ReplayError(
                    f"round {index}'s trace is {trace.shape}; this arena has nq={self.model.nq}. "
                    "The record was made in a different ring"
                )

        self._trace = np.zeros((0, self.model.nq))
        self._index = -1

    # -- MatchWorld ----------------------------------------------------------------------------------
    def reset_round(self, index: int) -> None:
        self._trace = np.asarray(self.recorded.trace(index), dtype=np.float64)
        self._index = index
        if self._trace.shape[0] == 0:
            raise ReplayError(f"round {index} recorded no ticks")
        self._place(0)

    def step(self, tick: int) -> None:
        """Put the fighters where they were at ``tick``. No physics is integrated."""
        if self._index < 0:
            raise ReplayError("step() before reset_round(); no round is loaded")
        if tick >= self._trace.shape[0]:
            raise ReplayError(
                f"round {self._index} ran {self._trace.shape[0]} ticks; asked for tick {tick}"
            )
        self._place(tick)

    def observe(self, tracker, trace, tick: int) -> None:
        tracker.observe(self.model, self.data, tick)
        trace.observe(self.model, self.data, tick)

    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def commits(self) -> list[dict[str, Any]]:
        """The commits as recorded. A replay does not re-derive what the player did."""
        return self.recorded.commits(self._index)

    @property
    def ticks(self) -> int:
        return int(self._trace.shape[0])

    # -- placing -------------------------------------------------------------------------------------
    def _place(self, tick: int) -> None:
        """Write ``qpos`` for a tick and reconstruct ``qvel`` from the neighbouring frame.

        ``mj_differentiatePos`` rather than a plain difference: the free joint's orientation is a
        quaternion, and subtracting two quaternions is not an angular velocity.
        """
        mujoco = self._mujoco
        self.data.qpos[:] = self._trace[tick]

        if tick + 1 < self._trace.shape[0]:
            qvel = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(
                self.model, qvel, TICK_DT, self._trace[tick], self._trace[tick + 1]
            )
            self.data.qvel[:] = qvel
        else:
            self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)


# -- the picture ---------------------------------------------------------------------------------------
class ReplayRenderer:
    """An offscreen renderer held open across a whole round.

    Held open deliberately: creating a ``mujoco.Renderer`` per frame costs more than rendering one,
    and a round is three thousand of them.
    """

    def __init__(
        self,
        model,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        camera: str = DEFAULT_CAMERA,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.camera = camera
        # MuJoCo sizes its offscreen framebuffer from the model and refuses a larger request.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        try:
            self._renderer = mujoco.Renderer(model, height, width)
        except Exception as exc:
            raise ReplayError(f"could not open an offscreen renderer: {exc}") from exc

        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            self._renderer.close()
            raise ReplayError(f"the arena has no camera named {camera!r}")

    def frame(self, data) -> np.ndarray:
        self._renderer.update_scene(data, camera=self.camera)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> ReplayRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def replay_frames(
    world: ReplayWorld,
    index: int,
    *,
    stride: int = 1,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    camera: str = DEFAULT_CAMERA,
) -> Iterator[np.ndarray]:
    """Yield one rendered frame per (strided) tick of a round."""
    if stride < 1:
        raise ReplayError(f"stride must be at least 1, got {stride}")

    world.reset_round(index)
    with ReplayRenderer(world.model, width, height, camera) as renderer:
        for tick in range(0, world.ticks, stride):
            world.step(tick)
            yield renderer.frame(world.data)


def encode_video(frames: Iterator[np.ndarray], path: Path, fps: float) -> Path:
    """Pipe frames into ffmpeg. Raises if ffmpeg is missing rather than quietly writing nothing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    first = next(frames, None)
    if first is None:
        raise ReplayError("no frames to encode")
    height, width = first.shape[:2]

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        str(path),
    ]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise ReplayError(
            "ffmpeg is not installed, so a replay cannot be encoded to video. Render frames "
            "instead, or install it"
        ) from exc

    try:
        process.stdin.write(first.tobytes())
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
    except BrokenPipeError as exc:
        raise ReplayError(f"ffmpeg stopped reading: {process.stderr.read().decode()}") from exc

    if process.wait() != 0:
        raise ReplayError(f"ffmpeg failed: {process.stderr.read().decode()}")
    return path
