"""Determine the corpus's Euler composition order, by measurement (M5-T3).

The corpus writes ``root_rotate{X,Y,Z}`` in degrees without stating how they compose. Guessing is
forbidden (`CLAUDE.md`: most bugs in this project are convention bugs), so this measures.

Two criteria were tried. The first — *uprightness*, on the theory that a wrong order would tip the
pelvis away from vertical during the corpus's large turns — **does not work**, and the failure is
structural rather than bad luck: tilt is the angle between the pelvis's own Z axis and world Z, and
rotation *about* the vertical is exactly what heading is, so the metric is blind to where the heading
rotation sits in the composition. All twelve candidates score 27-31 degrees.

The criterion that does work is **yaw against the heading channel itself**. The corpus is Z-up and
writes heading in ``root_rotateZ``; under the composition where the Z rotation is applied last in
world, the yaw recovered from the resulting matrix *is* ``root_rotateZ``. Measured 2026-08-27 over
all 38 takes, exactly one convention (and its intrinsic mirror, which is the same rotation) matches
to 0.00 degrees while every other deviates by 2.1-2.9. That is :data:`~openroboxing.studio.
motion_import.EULER_ORDER`.

It also agrees with the corpus's provenance: Maya's default rotate order XYZ composes as
``Rz . Ry . Rx``, which is scipy's *extrinsic* ``"xyz"``.

The stake is small either way — the twelve conventions disagree by at most 7.8 degrees of yaw, median
3.6 — and the Euler order affects **only** the root quaternion, never the joint angles. But
``heading_offset`` is derived from it, so it is worth pinning rather than assuming.

Run: ``.venv_mb/bin/python -m openroboxing.tools.pin_euler_order``
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from openroboxing.paths import MOTIONS_DIR
from openroboxing.studio.motion_import import read_take

#: Column index of each axis within the corpus's ``root_rotate{X,Y,Z}`` block.
AXIS_COLUMN = {"x": 0, "y": 1, "z": 2}


def candidates() -> list[tuple[str, list[int]]]:
    """Every Tait-Bryan convention, each paired with the angle columns it consumes.

    The angles must follow the sequence, not the file: ``"zyx"`` composes Z first, so it is handed
    ``root_rotateZ`` first. Feeding every order the same ``(X, Y, Z)`` column block instead would
    read ``root_rotateX`` as a rotation about Z, which is not a convention — it is a bug, and it was
    one this tool had before 2026-08-27.
    """
    out: list[tuple[str, list[int]]] = []
    for sequence in itertools.permutations("xyz"):
        name = "".join(sequence)
        columns = [AXIS_COLUMN[axis] for axis in sequence]
        out.append((name, columns))  # lower case: extrinsic
        out.append((name.upper(), columns))  # upper case: intrinsic
    return out


def yaw_error_deg(euler_deg: np.ndarray, order: str, columns: list[int]) -> float:
    """Worst disagreement between the recovered yaw and the corpus's own heading channel."""
    matrices = Rotation.from_euler(order, euler_deg[:, columns], degrees=True).as_matrix()
    yaw = np.unwrap(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
    heading = np.unwrap(np.radians(euler_deg[:, AXIS_COLUMN["z"]]))
    residual = np.degrees(yaw - heading)
    # A constant offset is a frame choice, not a disagreement; the variation is what matters.
    return float(np.abs(residual - residual.mean()).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=MOTIONS_DIR)
    args = parser.parse_args()

    takes = sorted(args.corpus.glob("*.csv"))
    if not takes:
        raise SystemExit(f"no takes under {args.corpus}")
    euler = [read_take(path).frames[:, 3:6] for path in takes]

    scores = {
        order: max(yaw_error_deg(e, order, columns) for e in euler)
        for order, columns in candidates()
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1])

    print(f"{'order':>6s}  {'worst |yaw - root_rotateZ| over 38 takes (deg)':>46s}")
    for order, worst in ranked:
        print(f"{order:>6s}  {worst:46.2f}")

    winners = [order for order, worst in ranked if worst < 1e-6]
    print(f"\nexact matches: {winners or 'none'}")
    if not winners:
        raise SystemExit(
            "no convention recovers the corpus's own heading channel; the corpus is not what this "
            "tool assumes and the conversion must not proceed on a guess"
        )


if __name__ == "__main__":
    main()
