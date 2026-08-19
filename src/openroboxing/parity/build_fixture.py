"""Turn a raw sim2sim capture into the committed golden fixture (M1-T2).

Reads the CSVs produced by ``parity/capture_run.sh`` and writes a single compressed npz holding a
window of aligned arrays.

Conventions
-----------
- Every array is one row per 50 Hz control tick, and all arrays share the same row index.
- ``policy_input`` / ``encoder_input`` / ``target_motion`` come from the deploy's own dump flags and
  are already row-aligned with each other.
- The ``state_logs/*.csv`` signals carry 5 leading metadata columns
  (``index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp``) which are stripped; the
  ``index`` column is kept separately so alignment can be verified rather than assumed.
- ``q``/``dq``/``action`` are in **hardware (MuJoCo/motor) order with default offsets applied** —
  they are *not* the IsaacLab-ordered, offset-subtracted values the observation history stores. That
  transform is obs.py's job and is exactly what the parity test exercises.

Usage
-----
    python -m openroboxing.parity.build_fixture <capture_dir> --start 300 --num 400
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np

from openroboxing.paths import GOLDEN_POLICY_IO_DIR, REPO_ROOT

# Leading columns every state_logs CSV carries before its payload.
_META_COLS = 5

# signal name -> expected payload width
STATE_SIGNALS: dict[str, int] = {
    "q": 29,
    "dq": 29,
    "action": 29,
    "base_ang_vel": 3,
    "base_quat": 4,
    "token_state": 64,
    "encoder_mode": 1,
}


def _read_dump_csv(path: Path) -> np.ndarray:
    """Read a deploy dump CSV: no header, one row per tick, trailing comma on every row."""
    rows: list[list[float]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip().rstrip(",")
            if line:
                rows.append([float(x) for x in line.split(",")])
    if not rows:
        raise ValueError(f"{path} is empty")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path} has ragged rows: widths {sorted(widths)}")
    return np.asarray(rows, dtype=np.float64)


def _read_state_csv(path: Path, expected_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a state_logs CSV. Returns (index_column, payload)."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    index = raw[:, 0]
    payload = raw[:, _META_COLS:]
    if payload.shape[1] != expected_width:
        raise ValueError(
            f"{path.name}: expected {expected_width} payload columns, got {payload.shape[1]}"
        )
    return index, payload


def build(capture_dir: Path, start: int, num: int, out: Path) -> None:
    dumps = {
        "policy_input": _read_dump_csv(capture_dir / "policy_input.csv"),
        "encoder_input": _read_dump_csv(capture_dir / "encoder_input.csv"),
        "target_motion": _read_dump_csv(capture_dir / "target_motion.csv"),
    }
    n_dump = {k: v.shape[0] for k, v in dumps.items()}
    if len(set(n_dump.values())) != 1:
        raise ValueError(f"deploy dumps are not row-aligned: {n_dump}")
    total = next(iter(n_dump.values()))
    if start + num > total:
        raise ValueError(f"window [{start}, {start + num}) exceeds capture length {total}")

    arrays: dict[str, np.ndarray] = {k: v[start : start + num] for k, v in dumps.items()}

    # The state logs are written by a different code path (StateLogger) than the dump flags, so
    # their row count can differ. Align on the tail: both end when the process is killed.
    state_dir = capture_dir / "state_logs"
    for name, width in STATE_SIGNALS.items():
        path = state_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; re-capture with --enable-csv-logs")
        _, payload = _read_state_csv(path, width)
        if payload.shape[0] < total:
            raise ValueError(
                f"{name}.csv has {payload.shape[0]} rows, fewer than the {total}-tick dump; "
                "cannot align"
            )
        # Take the last `total` rows so the two streams end together, then window.
        aligned = payload[payload.shape[0] - total :]
        arrays[f"state_{name}"] = aligned[start : start + num]

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        start_tick=np.array(start),
        num_ticks=np.array(num),
        capture_length=np.array(total),
        upstream_sha=np.array(sha),
        **arrays,
    )
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    for key in sorted(arrays):
        print(f"  {key:22s} {arrays[key].shape}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_fixture",
        description="Build the golden observation-parity fixture from a sim2sim capture (M1-T2).",
    )
    parser.add_argument("capture_dir", type=Path, help="directory written by capture_run.sh")
    parser.add_argument("--start", type=int, default=300, help="first tick of the window")
    parser.add_argument("--num", type=int, default=400, help="number of ticks")
    parser.add_argument(
        "--out", type=Path, default=GOLDEN_POLICY_IO_DIR / "golden.npz", help="output npz"
    )
    args = parser.parse_args(argv)
    build(args.capture_dir, args.start, args.num, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
