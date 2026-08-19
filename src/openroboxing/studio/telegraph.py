"""Telegraph measurement: how long a move is readable before it lands (M2-T3).

The telegraph window is the time between the first moment a move becomes **distinguishable from the
guard baseline** and the moment it **lands**. It is the whole basis of the game being fair: the
opponent's only cue is the windup, so a move with no telegraph is unreadable and a move with too much
is useless.

This is a deterministic geometric proxy, deliberately. A learned "is this a punch yet?" classifier is
the obvious upgrade and :func:`divergence_frame` is the seam for it — but a proxy that can be
explained, argued with and recomputed is the right thing to admit a pose library on.

Conventions
-----------
- **Input** is a ``(N, 36)`` qpos stream in MuJoCo order at a stated rate (default ``TICK_HZ``).
- **Distance is measured in Cartesian body space**, not joint space, via forward kinematics. Two
  poses can differ a lot in joint angles while the fists barely move, and it is the fists the
  opponent watches.
- **The root is excluded** from the divergence signal: walking toward someone is not a telegraph.

The threshold is measured, not invented
---------------------------------------
`CLAUDE.md` forbids inventing numbers. The divergence threshold is therefore derived from the
baseline's *own* motion: a move counts as distinguishable once it exceeds the guard's natural
variation by :data:`DIVERGENCE_SIGMA` standard deviations. The only free parameter is that multiple,
and it is stated rather than buried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openroboxing.paths import G1_29DOF_XML
from openroboxing.spec.constants import QPOS_DIM, TICK_HZ

#: How many standard deviations above the baseline's own variation counts as "distinguishable".
#: The one free parameter of the proxy. 3 sigma is the conventional "not noise" bar.
DIVERGENCE_SIGMA = 3.0

#: Bodies whose motion the opponent actually reads. Hands dominate; the head and elbows carry slips
#: and covers. Names are resolved against the model and missing ones raise.
TELEGRAPH_BODIES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
)

#: Fallback if the model names its links differently — resolved by suffix match.
_BODY_SUFFIXES = ("wrist_yaw", "elbow")


class TelegraphError(RuntimeError):
    """A telegraph window could not be measured. Never recovered from silently."""


@dataclass(frozen=True)
class TelegraphResult:
    """The measurement. ``window_ms`` is the number the pose record stores."""

    window_ms: float
    divergence_frame: int
    contact_frame: int
    rate_hz: float
    threshold_m: float
    peak_displacement_m: float

    def passes(self, floor_ms: float) -> bool:
        return self.window_ms >= floor_ms


def _resolve_bodies(model, names=TELEGRAPH_BODIES) -> list[int]:
    """Body ids for the telegraph bodies, by name, raising if absent."""
    import mujoco

    ids = []
    available = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    for name in names:
        if name in available:
            ids.append(available.index(name))
            continue
        matches = [
            i
            for i, n in enumerate(available)
            if n and any(s in n for s in _BODY_SUFFIXES) and n.split("_")[0] == name.split("_")[0]
        ]
        if not matches:
            raise TelegraphError(
                f"body {name!r} not in the model; available include "
                f"{[n for n in available if n and 'wrist' in n]}"
            )
        ids.append(matches[0])
    return ids


def body_trajectories(qpos: np.ndarray, body_ids: list[int] | None = None) -> np.ndarray:
    """Forward-kinematics the qpos stream into Cartesian body positions.

    Args:
        qpos: ``(N, 36)`` MuJoCo order.

    Returns:
        ``(N, len(bodies), 3)`` world-frame positions.
    """
    import mujoco

    arr = np.asarray(qpos, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != QPOS_DIM:
        raise TelegraphError(f"expected (N, {QPOS_DIM}) qpos, got shape {arr.shape}")

    model = mujoco.MjModel.from_xml_path(str(G1_29DOF_XML))
    data = mujoco.MjData(model)
    ids = body_ids if body_ids is not None else _resolve_bodies(model)

    out = np.empty((arr.shape[0], len(ids), 3))
    for i, frame in enumerate(arr):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        out[i] = data.xpos[ids]
    return out


def _root_relative(traj: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    """Remove the root translation: walking toward someone is not a telegraph."""
    return traj - np.asarray(qpos)[:, None, 0:3]


def divergence_frame(
    motion: np.ndarray,
    baseline: np.ndarray,
    sigma: float = DIVERGENCE_SIGMA,
) -> tuple[int, float]:
    """First frame where the motion leaves the baseline's natural variation.

    This is the seam a learned classifier would replace: same signature, same contract.

    Returns:
        ``(frame, threshold_m)``. The frame is ``-1`` if the motion never diverges.
    """
    motion_traj = _root_relative(body_trajectories(motion), motion)
    baseline_traj = _root_relative(body_trajectories(baseline), baseline)

    # The guard's own wander, as a distance from its mean pose. This is what "noise" means here.
    baseline_mean = baseline_traj.mean(axis=0)
    baseline_spread = np.linalg.norm(baseline_traj - baseline_mean, axis=-1).max(axis=-1)
    threshold = float(baseline_spread.mean() + sigma * baseline_spread.std())

    distance = np.linalg.norm(motion_traj - baseline_mean, axis=-1).max(axis=-1)
    beyond = np.flatnonzero(distance > threshold)
    return (int(beyond[0]) if beyond.size else -1), threshold


def contact_frame(motion: np.ndarray) -> tuple[int, float]:
    """The frame the move lands: peak reach of the striking hand.

    A strike's contact is its furthest extension from where it started, so the apex of hand
    displacement is the landing moment. Returns ``(frame, peak_displacement_m)``.
    """
    traj = _root_relative(body_trajectories(motion), motion)
    displacement = np.linalg.norm(traj - traj[0], axis=-1).max(axis=-1)
    return int(np.argmax(displacement)), float(displacement.max())


def measure(
    motion: np.ndarray,
    baseline: np.ndarray,
    rate_hz: float = TICK_HZ,
    sigma: float = DIVERGENCE_SIGMA,
) -> TelegraphResult:
    """Measure the telegraph window of a generated motion against a guard baseline.

    Raises:
        TelegraphError: if the motion never diverges from the baseline, or lands before it becomes
            distinguishable — both mean the pose is unreadable and must not be admitted.
    """
    start, threshold = divergence_frame(motion, baseline, sigma)
    if start < 0:
        raise TelegraphError(
            "motion never becomes distinguishable from the guard baseline; "
            "it has no telegraph and cannot be admitted"
        )
    land, peak = contact_frame(motion)
    if land <= start:
        raise TelegraphError(
            f"motion lands at frame {land} but only becomes distinguishable at {start}; "
            "the strike is unreadable"
        )
    return TelegraphResult(
        window_ms=(land - start) / rate_hz * 1e3,
        divergence_frame=start,
        contact_frame=land,
        rate_hz=rate_hz,
        threshold_m=threshold,
        peak_displacement_m=peak,
    )
