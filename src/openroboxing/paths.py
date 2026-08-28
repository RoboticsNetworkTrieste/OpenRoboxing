"""Filesystem locations for OpenRoboxing.

One place that knows where things live, so no module hard-codes a relative path and no behaviour
depends on the caller's working directory.

Conventions
-----------
- ``REPO_ROOT`` is the repository root — the parent of ``src/``.
- ``GR00T_ROOT`` is the upstream GR00T-WholeBodyControl checkout. It is the ``external/gr00t-wbc``
  submodule unless ``OPENROBOXING_GR00T_ROOT`` names another checkout, which is how a machine that
  already has one avoids cloning 4.2 GB a second time.
- Every path constant is absolute. Nothing here checks existence; callers that require a file should
  raise a specific error naming it (``CLAUDE.md`` invariant 5: fail loudly).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: The upstream checkout. See the module docstring for why it is overridable.
GR00T_ROOT: Path = Path(
    os.environ.get("OPENROBOXING_GR00T_ROOT", str(REPO_ROOT / "external/gr00t-wbc"))
).resolve()

# --- Robot model -------------------------------------------------------------------------------
# The 29-DOF G1 used by the shipped policy. `scene_29dof.xml` wraps this with a floor and lights.
G1_29DOF_XML: Path = GR00T_ROOT / "gear_sonic_deploy/g1/g1_29dof.xml"
G1_29DOF_SCENE_XML: Path = GR00T_ROOT / "gear_sonic_deploy/g1/scene_29dof.xml"

#: The **simulation-ready** 29-DOF G1, and the one to compose a scene from. ``scene_29dof.xml``
#: includes this file, so it is the model every M1 measurement was made against.
#:
#: Its badly-named sibling ``g1_29dof.xml`` is kinematics and meshes only: **zero rotor armature,
#: zero joint damping, zero friction loss**. Masses, joint ranges and torque limits are identical, so
#: it looks interchangeable and is not — a stiff PD controller on zero-armature joints is unstable,
#: and a fighter built from it collapses inside a second while appearing to be driven correctly.
#: Measured 2026-08-08 while building the arena; see `spec/upstream_notes.md`.
G1_29DOF_SIM_XML: Path = GR00T_ROOT / "gear_sonic_deploy/g1/g1_29dof_old.xml"

#: An empty scene, for benchmarking the world without a robot in it.
SCENE_EMPTY_XML: Path = GR00T_ROOT / "gear_sonic_deploy/g1/scene_empty.xml"

# --- Policy artefacts --------------------------------------------------------------------------
POLICY_DIR: Path = GR00T_ROOT / "gear_sonic_deploy/policy/release"
POLICY_DECODER_ONNX: Path = POLICY_DIR / "model_decoder.onnx"
POLICY_ENCODER_ONNX: Path = POLICY_DIR / "model_encoder.onnx"

# The C++-loadable observation config. NOT `observation_config_sonic_release.yaml`, whose term names
# do not exist in the deploy registry — see spec/upstream_notes.md Q2.
OBSERVATION_CONFIG_YAML: Path = POLICY_DIR / "observation_config.yaml"

# --- Reference motions (golden-capture inputs) --------------------------------------------------
REFERENCE_MOTION_DIR: Path = GR00T_ROOT / "gear_sonic_deploy/reference/example"

# --- Upstream sources we read ------------------------------------------------------------------
MOTIONBRICKS_ROOT: Path = GR00T_ROOT / "motionbricks"

# The robot model the *generator* was trained on, and the rest-pose offsets of its `g1skel34`
# skeleton. Deliberately **not** `G1_29DOF_XML`: the two are different revisions of the G1 (the
# waist and shoulder offsets differ by 9-19 mm — see spec/upstream_notes.md §Skeleton). Anything
# whose output the generator consumes must use this model; anything describing the physical robot
# must use `G1_29DOF_XML`.
GENERATOR_SKELETON_XML: Path = MOTIONBRICKS_ROOT / "assets/skeletons/g1/g1.xml"
GENERATOR_SKELETON_DIR: Path = MOTIONBRICKS_ROOT / "out/motionbricks_root/version_1/skeleton"

# --- Our own trees -----------------------------------------------------------------------------
OPENROBOXING_ROOT: Path = Path(__file__).resolve().parent
FIXTURES_DIR: Path = REPO_ROOT / "tests/fixtures"
GOLDEN_POLICY_IO_DIR: Path = FIXTURES_DIR / "golden_policy_io"

#: Authored poses, one directory per library version (``v0.1/``, ...).
POSE_DIR: Path = OPENROBOXING_ROOT / "poses"

MOTIONS_DIR: Path = REPO_ROOT / "motions"
"""The mocap corpus: Maya-style CSV exports, one per take. See spec/combination.md."""

COMBINATION_DIR: Path = POSE_DIR / "v0.2/combinations"
"""Built combination records. Produced by tools/import_motions.py, not authored by hand."""


def display_path(path: Path) -> str:
    """How a path should be *named*, not where it happens to sit.

    Upstream artefacts are named by their position inside the GR00T-WBC tree and OpenRoboxing's own
    by their position in this repository, so a season manifest reads identically whether upstream is
    the ``external/gr00t-wbc`` submodule or a checkout named by ``OPENROBOXING_GR00T_ROOT``. That
    matters because a manifest is a release record: the same asset must produce the same string on
    every machine.

    ``GR00T_ROOT`` is tried first because with the default submodule it sits *inside* ``REPO_ROOT``,
    and an upstream artefact must not be named ``external/gr00t-wbc/...`` on one machine and
    ``gear_sonic_deploy/...`` on another. Falls back to the absolute path for anything under neither
    root, rather than raising — naming a stray file is not an error worth stopping a freeze for.
    """
    for root in (GR00T_ROOT, REPO_ROOT):
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def locate(name: str) -> Path:
    """Turn a name from :func:`display_path` back into a path.

    A manifest records what an asset *is called*, not where this machine keeps it, so verifying one
    means resolving names again. The search order mirrors ``display_path`` — upstream first, then
    this repository — which is what makes the round trip hold in both configurations.

    A name that resolves nowhere is reported under ``GR00T_ROOT``, because the assets that can
    legitimately be absent are exactly the upstream checkpoints ``install.sh`` downloads; our own
    poses and specs are in git and are never missing from a working checkout.
    """
    for root in (GR00T_ROOT, REPO_ROOT):
        candidate = root / name
        if candidate.exists():
            return candidate
    return GR00T_ROOT / name
