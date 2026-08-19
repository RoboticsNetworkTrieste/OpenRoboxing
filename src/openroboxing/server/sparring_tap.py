"""The sparring bench's recorder: per-tick capture, state derivation, the viz transform.

Implements `spec/sparring_protocol.md` 0.1. Pure numpy — no aiohttp, no mujoco — so everything here
runs in a test without a GPU or a checkpoint.

Conventions
-----------
- **Ticks are 50 Hz**, absolute since the session's last reset. The tap is a ring buffer: when it
  fills, the oldest ticks fall off and :meth:`DebugTap.window` moves.
- **The machine state is derived, never advanced.** :func:`derive_machine_state` only calls the
  commit predicates (`is_executing` / `is_scheduled`) and reads spans; ``generator_intent`` is the
  timeline's clock and only the world may call it.
- **The viz transform is an interpretation.** The reference motion lives in the generator's frame
  and the encoder never consumes its root position, so its "world position" exists only for
  drawing. :func:`viz_world_path` applies the displacement-from-now at the robot's true position —
  the coherent inverse of ``fight.to_generator_frame``'s rotation.
"""

from __future__ import annotations

from collections import deque
import io
import math

import numpy as np

from openroboxing.spec.constants import NUM_JOINTS, QPOS_DIM, TICK_HZ

__all__ = [
    "APPROACH",
    "DWELL",
    "HOLD",
    "MACHINE_STATES",
    "OPENING",
    "TapError",
    "WAITING",
    "DebugTap",
    "derive_machine_state",
    "viz_ghost",
    "viz_world_path",
    "yaw_of_quat_wxyz",
]

#: The intent state machine, `spec/sparring_protocol.md` §The state machine. Indices are what the
#: tap stores; names are what the wire carries.
MACHINE_STATES = ("OPENING", "WAITING", "APPROACH", "DWELL", "HOLD")
OPENING, WAITING, APPROACH, DWELL, HOLD = range(5)

#: Default recording cap: 10 minutes at the tick rate. Beyond it the oldest ticks fall off.
DEFAULT_MAX_TICKS = 10 * 60 * TICK_HZ


class TapError(RuntimeError):
    """The tap was misused or asked for a tick it no longer holds. Never recovered from silently."""


def derive_machine_state(commits, tick: int) -> int:
    """Which of the five states the red timeline is in at ``tick``. Read-only.

    Args:
        commits: the timeline's commit log, in issue order (``IntentTimeline.commits``).
        tick: the 50 Hz tick to classify.
    """
    for commit in commits:
        if commit.is_executing(tick):
            if commit.strike_at is None or tick < commit.strike_at:
                return APPROACH
            return DWELL
    if any(c.is_scheduled(tick) for c in commits):
        return WAITING
    for commit in reversed(commits):
        end = commit.end_tick
        if end is not None and tick >= end:
            return HOLD
    return OPENING


# -- the visualisation transform --------------------------------------------------------------------
def yaw_of_quat_wxyz(q) -> float:
    """The yaw of a ``wxyz`` quaternion."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def viz_world_path(ref_motion: np.ndarray, tick: int, robot_xy, apply_yaw: float) -> np.ndarray:
    """Generator-frame reference rows ``[tick:]``, made drawable in world coordinates.

    ``world_xy(k) = robot_xy + R(apply_yaw) · (ref_xy[k] − ref_xy[tick])`` — the displacement the
    reference intends between now and frame ``k``, applied from where the robot really is.

    Returns ``(N - tick, 2)`` world positions; row 0 is the robot's own position.
    """
    ref = np.asarray(ref_motion, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != QPOS_DIM:
        raise TapError(f"ref_motion: expected (N, {QPOS_DIM}), got {ref.shape}")
    if not 0 <= tick < ref.shape[0]:
        raise TapError(f"tick {tick} outside the motion's {ref.shape[0]} rows")

    delta = ref[tick:, 0:2] - ref[tick, 0:2]
    c, s = math.cos(apply_yaw), math.sin(apply_yaw)
    rot = np.array([[c, -s], [s, c]])
    return np.asarray(robot_xy, dtype=np.float64) + delta @ rot.T


def viz_ghost(
    ref_motion: np.ndarray,
    tick: int,
    lookahead: int,
    robot_xy,
    apply_yaw: float,
    joint_names,
) -> dict:
    """The plan ghost: the reference frame the encoder is looking at, posed for the client.

    Yaw-only — the client's shadow FK takes a heading, not a full quaternion — which is the
    documented 0.1 limitation in `spec/sparring_protocol.md`.
    """
    ref = np.asarray(ref_motion, dtype=np.float64)
    k = min(tick + lookahead, ref.shape[0] - 1)
    xy = viz_world_path(ref_motion, tick, robot_xy, apply_yaw)[k - tick]
    frame = ref[k]
    return {
        "x": float(xy[0]),
        "y": float(xy[1]),
        "z": float(frame[2]),
        "heading": float(apply_yaw + yaw_of_quat_wxyz(frame[3:7])),
        "angles": {name: float(v) for name, v in zip(joint_names, frame[7:])},
    }


# -- the recorder ------------------------------------------------------------------------------------
def _json_number(value) -> float | None:
    """A float the whole web can read: non-finite becomes ``None`` (JSON ``null``).

    Bare ``NaN`` and ``Infinity`` are Python-only extensions to JSON. Emitting one does not fail
    here — it fails in the browser, at ``JSON.parse``, taking the entire payload with it.
    """
    number = float(value)
    return number if math.isfinite(number) else None


#: Column name -> (shape per tick, dtype). Scalars are ().
_COLUMNS: dict[str, tuple[tuple[int, ...], str]] = {
    # Full arena qpos, width fixed by the first append. float64 deliberately: the scrub promise is
    # a byte-identical repack of the live frame, and a float32 round trip through mj_forward is not.
    "qpos": ((-1,), "f8"),
    "ref_red": ((QPOS_DIM,), "f4"),
    "ref_blue": ((QPOS_DIM,), "f4"),
    "err_red": ((NUM_JOINTS,), "f4"),
    "action_red": ((NUM_JOINTS,), "f4"),
    "root_h_red": ((), "f4"),
    "root_h_blue": ((), "f4"),
    "separation": ((), "f4"),
    "dist_target": ((), "f4"),  # NaN when no placement is being approached
    # The same distance, measured on the **plan** instead of the body: the reference frame the
    # encoder is chasing, against the same placement. The pair is the whole diagnosis of an
    # approach — MotionBricks is kinematic and arrives every time, the body under physics is what
    # has to get there, and only the gap between these two says which half is failing.
    "dist_plan": ((), "f4"),
    "step_ms": ((), "f4"),
    "machine": ((), "i1"),
    "commit_ordinal": ((), "i2"),
}


class DebugTap:
    """Per-tick recording of a sparring session, in memory, ring-buffered.

    Append exactly one row per stepped tick, in tick order. Events (replans, the commit log) are
    kept separately with absolute ticks, so trimming the ring never rewrites them.
    """

    def __init__(self, max_ticks: int = DEFAULT_MAX_TICKS) -> None:
        if max_ticks < 1:
            raise TapError(f"max_ticks must be positive, got {max_ticks}")
        self.max_ticks = max_ticks
        self._ticks: deque[int] = deque(maxlen=max_ticks)
        self._columns: dict[str, deque] = {
            name: deque(maxlen=max_ticks) for name in _COLUMNS
        }
        #: ``(tick, forced, plan_frames)`` per real replan of the red generator.
        self.replans: list[tuple[int, bool, int]] = []

    def clear(self) -> None:
        self._ticks.clear()
        for column in self._columns.values():
            column.clear()
        self.replans = []

    def __len__(self) -> int:
        return len(self._ticks)

    def window(self) -> tuple[int, int]:
        """``(first, last)`` tick still held. Raises while empty — an empty window is not (0, 0)."""
        if not self._ticks:
            raise TapError("the tap is empty; nothing has been recorded yet")
        return self._ticks[0], self._ticks[-1]

    def append(self, tick: int, **values) -> None:
        """Record one tick. Every column is required; a partial row would be a silent gap."""
        if self._ticks and tick != self._ticks[-1] + 1:
            raise TapError(
                f"tick {tick} does not follow {self._ticks[-1]}; the tap records every tick, in order"
            )
        missing = [name for name in _COLUMNS if name not in values]
        if missing:
            raise TapError(f"append is missing columns: {missing}")
        extra = [name for name in values if name not in _COLUMNS]
        if extra:
            raise TapError(f"append got unknown columns: {extra}")

        for name, (shape, dtype) in _COLUMNS.items():
            value = np.asarray(values[name], dtype=dtype)
            if shape == ():
                if value.shape != ():
                    raise TapError(f"{name}: expected a scalar, got shape {value.shape}")
            elif shape != (-1,) and value.shape != shape:
                raise TapError(f"{name}: expected shape {shape}, got {value.shape}")
            self._columns[name].append(value)
        self._ticks.append(tick)

    def _index(self, tick: int) -> int:
        first, last = self.window()
        if not first <= tick <= last:
            raise TapError(f"tick {tick} outside the recording window [{first}, {last}]")
        return tick - first

    def at(self, tick: int) -> dict:
        """Everything recorded for one tick."""
        index = self._index(tick)
        row = {name: self._columns[name][index] for name in _COLUMNS}
        row["tick"] = tick
        return row

    def series(self, from_tick: int, to_tick: int, stride: int = 1) -> dict:
        """Downsampled traces for the charts, as plain lists.

        The absolute-error mean/max per tick are computed here so the client never re-derives them.

        **A gap is ``None``, never ``NaN``.** ``dist_target`` is legitimately NaN whenever no
        placement is being approached, and Python's ``json`` writes that as a bare ``NaN`` token
        that JavaScript's ``JSON.parse`` refuses — one un-approached tick therefore threw away the
        whole payload, and the charts stayed blank for the entire session (2026-08-17: 970 of 1052
        samples). ``None`` becomes ``null``, which the client already draws as the gap it is.
        """
        if stride < 1:
            raise TapError(f"stride must be at least 1, got {stride}")
        first, last = self.window()
        lo, hi = max(from_tick, first), min(to_tick, last)
        if lo > hi:
            raise TapError(
                f"[{from_tick}, {to_tick}] does not intersect the window [{first}, {last}]"
            )

        indices = range(lo - first, hi - first + 1, stride)
        err = [np.abs(self._columns["err_red"][i]) for i in indices]
        pick = lambda name: [  # noqa: E731
            _json_number(self._columns[name][i]) for i in indices
        ]
        return {
            "tick": [int(self._ticks[i]) for i in indices],
            "err_mean": [_json_number(e.mean()) for e in err],
            "err_max": [_json_number(e.max()) for e in err],
            "dist": pick("dist_target"),
            "dist_plan": pick("dist_plan"),
            "root_h_red": pick("root_h_red"),
            "root_h_blue": pick("root_h_blue"),
            "step_ms": pick("step_ms"),
            "machine": [int(self._columns["machine"][i]) for i in indices],
            "replans": [
                [tick, forced, frames]
                for tick, forced, frames in self.replans
                if lo <= tick <= hi
            ],
        }

    def to_npz_bytes(self) -> bytes:
        """The whole recording as a ``numpy.savez_compressed`` archive, for offline analysis."""
        first, _ = self.window()
        arrays: dict[str, np.ndarray] = {
            "tick": np.asarray(self._ticks, dtype="i8"),
        }
        for name, (_, dtype) in _COLUMNS.items():
            arrays[name] = np.asarray(list(self._columns[name]), dtype=dtype)
        arrays["replans"] = np.asarray(
            [[t, int(f), n] for t, f, n in self.replans], dtype="i8"
        ).reshape(-1, 3)
        arrays["window_start"] = np.asarray(first, dtype="i8")

        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()
