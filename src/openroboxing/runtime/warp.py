"""Place a recorded combination in the ring (M5-T10).

Implements decisions D3, D4 and D5 of
``docs/superpowers/specs/2026-08-27-motion-combinations-design.md``.

A combination **starts in place** — the fighter does not travel to a start — and its **final
keyframe lands on the ghost**. Between them the recorded footwork is kept at **true size** and the
leftover travel (the residual) is added as an even drift, ramped by elapsed time.

MotionBricks only covers ``DRIFT_GAIN`` of a commanded residual (measured, M6-T1), so the residual
is divided by it before the ramp — asking for more so the fighter actually lands on the ghost. The
recorded footwork is **not** gained: it is already the right size (design D4), and gaining it too
would re-introduce exactly the distortion D4 exists to prevent.

Why not scale the recorded path proportionally: measured 2026-08-27, reaching a ghost 2 m away needs
0.6-2.1x for the travelling takes but **30-141x** for shadow boxing, whose combinations travel
1-7 cm, and seven combinations travel under 5 cm where the factor is undefined. Scaling would turn a
2 cm weight shift into a 2.8 m lurch.

Conventions
-----------
- Positions are MuJoCo world ``(x, y)`` on the ground plane; headings are radians. The same frame the
  arena, the shadow and the client use.
- A record's ``root_offset`` / ``heading_offset`` are **relative to keyframe 0** and in the take's own
  frame; they are rotated by the fighter's heading on the way out.
- ``facing_angle`` is where the fighter looks and comes from the **recording**. ``movement_angle`` is
  where it travels and comes from the **warped** displacement. They are different signals and the
  difference selects the gait (`CLAUDE.md`).
- Nothing is clamped. With a ``speed_ceiling`` set, a ghost the combination cannot reach in its
  recorded duration raises; with ``speed_ceiling=None`` there is no ceiling and no raise, and the
  fighter runs whatever drift landing on the ghost takes (owner, 2026-08-28).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from openroboxing.spec.constants import APPROACH_SPEED_M_S, DRIFT_GAIN, SECONDS_PER_TOKEN

if TYPE_CHECKING:  # `runtime` does not import `studio` at runtime - see generator.py's note.
    from openroboxing.studio.combination_record import CombinationRecord

#: A displacement below this is not a direction. Beneath it a leg inherits its facing angle rather
#: than taking ``atan2(0, 0)``, which is 0.0 and silently means "straight ahead, always".
STILL_LEG_M = 1e-3


class WarpError(RuntimeError):
    """A combination cannot be placed as asked. Never recovered from silently."""


@dataclass(frozen=True)
class Leg:
    """One pose-to-pose leg, ready to become a ``GeneratorIntent``.

    **``target_position`` is where the generator is aimed, not where the fighter ends up.** Since
    M6-T2 the residual carries the drift gain, so the final leg's target deliberately *overshoots*
    the ghost by the fraction the generator is measured to fall short of — that overshoot is what
    lands the fighter on the ghost. The two coincide only when the residual is zero.

    Anything that wants the endpoint the player chose — the client's ghost, the match record —
    must use the ghost, never this. Reading this as an endpoint renders a ghost ~25 % too far out.
    """

    joint_angles: Mapping[str, float]
    target_position: tuple[float, float]
    target_heading: float
    horizon_tokens: int
    movement_angle: float
    facing_angle: float


def warp(
    record: CombinationRecord,
    anchor_position: tuple[float, float],
    anchor_heading: float,
    ghost_position: tuple[float, float],
    *,
    speed_ceiling: float | None = APPROACH_SPEED_M_S,
) -> list[Leg]:
    """Place ``record``: start at the anchor, end on the ghost, footwork at recorded size.

    Args:
        record: the combination to place.
        anchor_position: where the fighter is now, MuJoCo world ``(x, y)``.
        anchor_heading: where the fighter faces now, radians.
        ghost_position: where the final keyframe must land.
        speed_ceiling: the fastest sustained drift the fighter can hold, m/s, checked against the
            *raw* residual (before the drift-gain correction, which is not distance the player
            asked for). ``None`` skips the check entirely: execution re-warps from wherever the
            fighter actually is and must still reach the ghost, running whatever drift that takes
            (owner, 2026-08-28). Issue-time validation of a player's placement should pass the
            default; execution should pass ``None``.

    Returns:
        One :class:`Leg` per keyframe after the first, in order.

    Raises:
        WarpError: if ``speed_ceiling`` is set and reaching the ghost within the recorded duration,
            at the raw (un-gained) residual, exceeds it.
    """
    keyframes = record.keyframes
    cos_h, sin_h = math.cos(anchor_heading), math.sin(anchor_heading)

    # Cumulative time to each keyframe, in tokens. Index 0 is the start, so its time is zero.
    elapsed: list[float] = [0.0]
    for keyframe in keyframes[1:]:
        elapsed.append(elapsed[-1] + float(keyframe.leg_tokens or 0))
    total = elapsed[-1]
    if total <= 0.0:
        raise WarpError(f"{record.name}: zero total duration")

    # The recording, rotated into the world.
    rotated = [
        (cos_h * dx - sin_h * dy, sin_h * dx + cos_h * dy)
        for dx, dy in (k.root_offset for k in keyframes)
    ]
    # Leftover travel: what the recording does not already cover.
    residual = (
        ghost_position[0] - anchor_position[0] - rotated[-1][0],
        ghost_position[1] - anchor_position[1] - rotated[-1][1],
    )
    duration_s = total * SECONDS_PER_TOKEN
    if speed_ceiling is not None:
        drift_speed = math.hypot(*residual) / duration_s
        if drift_speed > speed_ceiling:
            raise WarpError(
                f"{record.name}: reaching that placement needs {drift_speed:.2f} m/s of drift over "
                f"{duration_s:.2f} s, above the {speed_ceiling:.2f} m/s ceiling"
            )

    # The generator only covers DRIFT_GAIN of a commanded residual, so ask for more. Checked against
    # the ceiling *before* this: the gain corrects the generator's shortfall, not extra distance the
    # player asked for.
    residual = (residual[0] / DRIFT_GAIN, residual[1] / DRIFT_GAIN)

    positions: list[tuple[float, float]] = []
    for offset, time in zip(rotated, elapsed, strict=True):
        share = time / total
        positions.append(
            (
                anchor_position[0] + offset[0] + share * residual[0],
                anchor_position[1] + offset[1] + share * residual[1],
            )
        )

    legs: list[Leg] = []
    for index, (previous, current) in enumerate(pairwise(positions), start=1):
        facing = anchor_heading + keyframes[index].heading_offset
        step = (current[0] - previous[0], current[1] - previous[1])
        moving = math.hypot(*step) >= STILL_LEG_M
        legs.append(
            Leg(
                joint_angles=dict(keyframes[index].joint_angles),
                target_position=current,
                target_heading=facing,
                horizon_tokens=int(keyframes[index].leg_tokens or 0),
                movement_angle=math.atan2(step[1], step[0]) if moving else facing,
                facing_angle=facing,
            )
        )
    return legs


def ghost_heading(record: CombinationRecord, anchor_heading: float) -> float:
    """Where the ghost faces: the fighter's heading plus the combination's recorded turn.

    Derived, never chosen, and never aimed at a target (design D5). The travelling takes turn by up
    to 158 degrees, and a ghost that faced the target would discard the turn that *is* the motion.
    """
    return anchor_heading + record.recorded_heading_delta
