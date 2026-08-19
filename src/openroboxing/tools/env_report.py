"""Environment report for OpenRoboxing (task M0-T1).

Prints the facts a match record must be traceable to: the upstream commit, the checkpoint
paths and their hashes, the GPU, and the MuJoCo / ONNX Runtime / PyTorch versions.

Conventions
-----------
- Artefact paths are resolved relative to **GR00T_ROOT**, the upstream checkout (see
  ``src/openroboxing/paths.py``), not the caller's working directory. Upstream is no longer pinned to a
  snapshot — it tracks ``main``, so this report shows both trees' HEADs rather than a drift check.
- Hashes are SHA-256 of the whole file. ``--quick`` instead hashes the first and last 8 MiB plus the
  file size, which is ~1000x faster on multi-hundred-MB checkpoints and is adequate for detecting
  an accidentally swapped file. A quick hash is prefixed ``q:`` so it can never be mistaken for a
  full one.
- Exit code is 0 if every *required* artefact is present, 1 otherwise. Optional artefacts (ONNX
  Runtime, GPU) are reported but do not fail the run.

Usage
-----
    python -m openroboxing.tools.env_report
    python -m openroboxing.tools.env_report --quick
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

# The layout lives in one place. See src/openroboxing/paths.py.
from openroboxing.paths import GR00T_ROOT, REPO_ROOT, display_path

# Artefacts the M1 runtime cannot start without.
REQUIRED_ARTEFACTS: tuple[tuple[str, str], ...] = (
    ("policy (decoder) ONNX", "gear_sonic_deploy/policy/release/model_decoder.onnx"),
    ("tokenizer (encoder) ONNX", "gear_sonic_deploy/policy/release/model_encoder.onnx"),
    ("observation config", "gear_sonic_deploy/policy/release/observation_config.yaml"),
    (
        "MotionBricks root",
        "motionbricks/out/motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt",
    ),
    (
        "MotionBricks pose",
        "motionbricks/out/motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt",
    ),
    (
        "MotionBricks vqvae",
        "motionbricks/out/motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt",
    ),
    ("MotionBricks clip cache", "motionbricks/out/G1-clip.ckpt"),
)

_QUICK_CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class Artefact:
    """One on-disk artefact and its identity."""

    label: str
    path: Path
    present: bool
    size_bytes: int
    digest: str


def _git_in(cwd: Path, *args: str) -> str:
    """Run git in `cwd`; return stripped stdout, or '<unavailable>' on failure."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return "<unavailable>"
    return result.stdout.strip() if result.returncode == 0 else "<unavailable>"


def _git(*args: str) -> str:
    """Run git in the repo root."""
    return _git_in(REPO_ROOT, *args)


def _hash_file(path: Path, quick: bool) -> str:
    """SHA-256 of the file, or a size+head+tail digest when ``quick``."""
    h = hashlib.sha256()
    if quick:
        size = path.stat().st_size
        h.update(str(size).encode())
        with path.open("rb") as fh:
            h.update(fh.read(_QUICK_CHUNK))
            if size > _QUICK_CHUNK:
                fh.seek(max(0, size - _QUICK_CHUNK))
                h.update(fh.read(_QUICK_CHUNK))
        return "q:" + h.hexdigest()[:32]
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect_artefacts(quick: bool) -> list[Artefact]:
    """Resolve, stat and hash every required artefact."""
    found: list[Artefact] = []
    for label, rel in REQUIRED_ARTEFACTS:
        path = GR00T_ROOT / rel
        if not path.exists():
            found.append(Artefact(label, path, False, 0, "<missing>"))
            continue
        found.append(Artefact(label, path, True, path.stat().st_size, _hash_file(path, quick)))
    return found


def _module_version(name: str) -> str:
    try:
        return str(getattr(importlib.import_module(name), "__version__", "<unknown>"))
    except ImportError:
        return "<not installed>"


def _gpu_lines() -> list[str]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return ["<nvidia-smi unavailable>"]
    return [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="env_report",
        description="Report upstream SHA, checkpoint hashes, GPU and library versions (M0-T1).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="hash size+head+tail instead of the whole file (much faster; digests prefixed 'q:')",
    )
    args = parser.parse_args(argv)

    print("OpenRoboxing environment report")
    print(f"repo root: {REPO_ROOT}")
    print(f"  branch            : {_git('rev-parse', '--abbrev-ref', 'HEAD')}")
    print(f"  HEAD              : {_git('rev-parse', 'HEAD')}")
    print(f"upstream: {GR00T_ROOT}")
    print(f"  HEAD              : {_git_in(GR00T_ROOT, 'rev-parse', 'HEAD')}")
    print(f"  behind origin/main: {_git_in(GR00T_ROOT, 'rev-list', '--count', 'HEAD..origin/main')}")
    print()

    print("Artefacts")
    ok = True
    for a in collect_artefacts(args.quick):
        if not a.present:
            ok = False
            print(f"  [X] {a.label}")
            print(f"        expected at {a.path}")
            continue
        print(f"  [+] {a.label}  ({a.size_bytes / 1e6:.1f} MB)")
        print(f"        {display_path(a.path)}")
        print(f"        sha256 {a.digest}")
    print()

    print("Libraries")
    for mod in ("mujoco", "onnxruntime", "torch", "numpy"):
        print(f"  {mod:12s} {_module_version(mod)}")
    print(f"  {'python':12s} {sys.version.split()[0]}")
    print()

    print("GPU")
    for line in _gpu_lines():
        print(f"  {line}")
    print()

    if _module_version("onnxruntime") == "<not installed>":
        print("[!] onnxruntime is not installed — required by runtime/policy.py (M1-T4)")
    print("OK" if ok else "INCOMPLETE — missing required artefacts above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
