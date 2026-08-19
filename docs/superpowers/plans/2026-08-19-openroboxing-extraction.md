# OpenRoboxing Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `/home/hpc-dev/OpenRoboxing` as a self-contained repository that reaches GR00T-WholeBodyControl through a submodule, passes its own test suite, and serves the sparring bench — ready for the owner to push to `https://github.com/TriesteOpenRoboticsCommunity/OpenRoboxing.git`.

**Architecture:** The package moves under `src/openroboxing/` unchanged. Every path into upstream re-roots onto a new `GR00T_ROOT` constant in `paths.py`, defaulting to the `external/gr00t-wbc` submodule and overridable by `OPENROBOXING_GR00T_ROOT`. Patch P0 stops being a diff inside upstream's `full_agent.py` and becomes runtime code in `runtime/generator.py`, so the submodule stays pristine and can track NVlabs `main`.

**Tech Stack:** Python 3.10+, git submodules, git-lfs (submodule only), pytest, uv-managed venvs.

**Spec:** `docs/superpowers/specs/2026-08-19-openroboxing-extraction-design.md` (in the source repo at `openroboxing/docs/superpowers/specs/`).

---

## Conventions used throughout

**Source repo** (never modified by this plan): `/home/hpc-dev/GR00T-WholeBodyControl`
**New repo**: `/home/hpc-dev/OpenRoboxing`

Until Task 6 installs the package, run Python from the new repo root as:

```bash
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python
```

This borrows the existing venv's dependencies without installing anything into it and without
creating a new one. Task 10 replaces it with the real `install.sh` run.

Set this in every shell that works on the new repo, so the runtime finds upstream without waiting
for the submodule's LFS content:

```bash
export OPENROBOXING_GR00T_ROOT=/home/hpc-dev/GR00T-WholeBodyControl
```

**Commits.** Tasks 1-9 commit normally on `main`. Task 10 collapses all of them into the single
initial commit the owner asked for. Do not skip the intermediate commits — they are what makes a
mistake recoverable.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `.gitmodules` | records the submodule URL and `branch = main` | 1 |
| `external/gr00t-wbc/` | the upstream checkout | 1 |
| `.gitignore`, `LICENSE` | repo hygiene; Apache-2.0 matching upstream's source licence | 1 |
| `src/openroboxing/**` | the package, moved verbatim | 2 |
| `tests/**`, `docs/**` | hoisted out of the package | 2 |
| `src/openroboxing/paths.py` | **the only file that knows the layout**: `REPO_ROOT`, `GR00T_ROOT`, `display_path` | 3 |
| `src/openroboxing/tools/env_report.py` | stops computing its own root; artefacts re-root onto `GR00T_ROOT` | 4 |
| `src/openroboxing/tools/bench_world.py` | `SCENE_EMPTY` re-roots onto `GR00T_ROOT` | 4 |
| `src/openroboxing/league/manifest.py` | asset paths via `display_path`, not `relative_to(REPO_ROOT)` | 4 |
| `src/openroboxing/runtime/generator.py` | hosts P0's body and installs its call site | 5 |
| `tests/test_generator_pose_override.py` | proves P0 works against unpatched upstream | 5 |
| `pyproject.toml` | src layout, pytest config, metadata | 6 |
| `install.sh` | submodule init, LFS, ONNX download, smoke test | 7 |
| `src/openroboxing/parity/capture_run.sh` | deploy dir by env var, not hard-coded | 8 |
| `CLAUDE.md`, `README.md`, `docs/**`, `src/openroboxing/spec/upstream_patches.md` | path rewrites and the P0 restatement | 9 |

---

### Task 1: Repository skeleton and the submodule

**Files:**
- Create: `/home/hpc-dev/OpenRoboxing/.gitignore`
- Create: `/home/hpc-dev/OpenRoboxing/LICENSE`
- Create: `/home/hpc-dev/OpenRoboxing/.gitmodules` (written by `git submodule add`)

- [ ] **Step 1: Refuse to clobber an existing directory**

```bash
test ! -e /home/hpc-dev/OpenRoboxing && echo "clear to create" || echo "STOP: already exists"
```

Expected: `clear to create`. If it prints STOP, halt and ask the owner — do not delete anything.

- [ ] **Step 2: Create the repo and its first two files**

```bash
mkdir -p /home/hpc-dev/OpenRoboxing
cd /home/hpc-dev/OpenRoboxing
git init -b main
```

Write `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/

# Environments — the install script builds .venv_mb; it is never committed.
.venv*/

# Runtime output
matches/
seasons/
captures/
renders/
*.log

# The policy checkpoints are fetched by install.sh from nvidia/GEAR-SONIC.
# They live in the submodule, but guard against a stray copy landing here.
*.onnx
*.ckpt
```

Copy the licence from upstream, which is the same Apache-2.0 text this project's source is under:

```bash
cp /home/hpc-dev/GR00T-WholeBodyControl/LICENSE /home/hpc-dev/OpenRoboxing/LICENSE
```

- [ ] **Step 3: Add the submodule, borrowing objects from the existing checkout**

`GIT_LFS_SKIP_SMUDGE=1` leaves the 3.8 GB of STL meshes as pointer files — they are only needed to
render, and `install.sh` pulls them in Task 7. `--reference` avoids re-downloading 415 MB of git
objects that already exist locally.

```bash
cd /home/hpc-dev/OpenRoboxing
GIT_LFS_SKIP_SMUDGE=1 git submodule add \
  -b main \
  --reference /home/hpc-dev/GR00T-WholeBodyControl \
  https://github.com/NVlabs/GR00T-WholeBodyControl.git \
  external/gr00t-wbc
```

Expected: clones quickly (seconds, not minutes) because of `--reference`.

- [ ] **Step 4: Cut the alternates link so the new repo stands alone**

`--reference` leaves the submodule borrowing objects from the old checkout. Repack to copy them in,
then delete the link, so deleting `/home/hpc-dev/GR00T-WholeBodyControl` later cannot corrupt this
repo.

```bash
cd /home/hpc-dev/OpenRoboxing/external/gr00t-wbc
ALT="$(git rev-parse --git-dir)/objects/info/alternates"
git repack -a -d
rm -f "$ALT"
git fsck --connectivity-only --no-dangling 2>&1 | tail -5
```

Expected: `git fsck` reports no missing objects. It prints progress lines; any line containing
`missing` or `broken` means the repack did not complete — re-run `git repack -a -d` before removing
the alternates file.

- [ ] **Step 5: Verify the submodule is configured to track `main`**

```bash
cd /home/hpc-dev/OpenRoboxing
cat .gitmodules
git config -f .gitmodules submodule.external/gr00t-wbc.branch
```

Expected: `.gitmodules` contains `path = external/gr00t-wbc`, the NVlabs URL, and
`branch = main`; the second command prints `main`.

If `branch` is absent, add it:

```bash
git config -f .gitmodules submodule.external/gr00t-wbc.branch main
```

- [ ] **Step 6: Confirm upstream is pristine — P0 must NOT be present**

This is the premise the whole extraction rests on. The submodule tracks NVlabs `main`, which has
never had patch P0.

```bash
cd /home/hpc-dev/OpenRoboxing
grep -c "_override_target_joint_transforms" \
  external/gr00t-wbc/motionbricks/motionbricks/motion_backbone/demo/full_agent.py
```

Expected: `0`. A non-zero count means the submodule checked out something other than NVlabs `main` —
stop and investigate before continuing.

- [ ] **Step 7: Commit**

```bash
cd /home/hpc-dev/OpenRoboxing
git add .gitignore LICENSE .gitmodules external/gr00t-wbc
git commit -m "chore: repository skeleton and the GR00T-WBC submodule"
```

---

### Task 2: Move the tree into the src/ layout

No edits in this task — only relocation. Editing and moving in one step makes a mistake impossible
to isolate.

**Files:**
- Create: `/home/hpc-dev/OpenRoboxing/src/openroboxing/**` (from `openroboxing/`, minus `tests/` and `docs/`)
- Create: `/home/hpc-dev/OpenRoboxing/tests/**`, `/home/hpc-dev/OpenRoboxing/docs/**`
- Create: repo-root `README.md`, `LICENSING.md`, `install.sh`, `requirements-runtime.txt`

- [ ] **Step 1: Copy the package, excluding caches**

Every exclude below the caches is **anchored with a leading `/`**, which in rsync means "only at the
transfer root". Unanchored, `--exclude='README.md'` would also drop
`spec/ui-design-guide/README.md`, and `--exclude='tests/'` would drop any nested `tests/`.

```bash
cd /home/hpc-dev/OpenRoboxing
mkdir -p src
rsync -a \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='/tests/' \
  --exclude='/docs/' \
  --exclude='/README.md' \
  --exclude='/LICENSING.md' \
  --exclude='/install.sh' \
  --exclude='/requirements-runtime.txt' \
  --exclude='/pytest.ini' \
  /home/hpc-dev/GR00T-WholeBodyControl/openroboxing/ \
  src/openroboxing/
```

- [ ] **Step 2: Copy tests, docs and the root-level files**

```bash
cd /home/hpc-dev/OpenRoboxing
rsync -a --exclude='__pycache__/' --exclude='.pytest_cache/' \
  /home/hpc-dev/GR00T-WholeBodyControl/openroboxing/tests/ tests/
rsync -a --exclude='__pycache__/' \
  /home/hpc-dev/GR00T-WholeBodyControl/openroboxing/docs/ docs/
cp /home/hpc-dev/GR00T-WholeBodyControl/openroboxing/README.md .
cp /home/hpc-dev/GR00T-WholeBodyControl/openroboxing/LICENSING.md .
cp /home/hpc-dev/GR00T-WholeBodyControl/openroboxing/install.sh .
cp /home/hpc-dev/GR00T-WholeBodyControl/openroboxing/requirements-runtime.txt .
```

- [ ] **Step 3: Verify nothing was dropped**

```bash
cd /home/hpc-dev/OpenRoboxing
echo "package modules : $(find src/openroboxing -name '*.py' | wc -l)"
echo "source modules  : $(find /home/hpc-dev/GR00T-WholeBodyControl/openroboxing -name '*.py' -not -path '*__pycache__*' -not -path '*/tests/*' | wc -l)"
echo "tests           : $(find tests -name 'test_*.py' | wc -l)"
echo "fixture present : $(test -f tests/fixtures/golden_policy_io/golden.npz && echo yes || echo NO)"
echo "ui-design README: $(test -f src/openroboxing/spec/ui-design-guide/README.md && echo yes || echo NO)"
echo "client assets   : $(find src/openroboxing/client -type f | wc -l)"
echo "poses           : $(find src/openroboxing/poses -name '*.json' | wc -l)"
```

Expected: the two module counts match, both `present` lines say `yes`, and the client and pose counts
are non-zero. The `ui-design README` line is the canary for an unanchored exclude — if it says `NO`,
the excludes lost their leading `/`.

A thorough alternative, comparing the two trees file by file:

```bash
cd /home/hpc-dev/OpenRoboxing
diff <(cd /home/hpc-dev/GR00T-WholeBodyControl/openroboxing && find . -type f \
        -not -path '*__pycache__*' -not -path './.pytest_cache/*' | sed \
        -e 's#^\./tests/#TESTS/#' -e 's#^\./docs/#DOCS/#' \
        -e 's#^\./\(README\.md\|LICENSING\.md\|install\.sh\|requirements-runtime\.txt\|pytest\.ini\)$#ROOT/\1#' \
        -e 's#^\./#PKG/#' | sort) \
     <( { (cd src/openroboxing && find . -type f -not -path '*__pycache__*' | sed 's#^\./#PKG/#')
          (cd tests && find . -type f -not -path '*__pycache__*' | sed 's#^\./#TESTS/#')
          (cd docs && find . -type f | sed 's#^\./#DOCS/#')
          for f in README.md LICENSING.md install.sh requirements-runtime.txt; do echo "ROOT/$f"; done
        } | sort)
```

Expected: the only difference is `ROOT/pytest.ini`, deliberately not copied — Task 6 replaces it
with `[tool.pytest.ini_options]` in `pyproject.toml`.

- [ ] **Step 4: Confirm the layout is right and imports are still broken in the expected way**

```bash
cd /home/hpc-dev/OpenRoboxing
ls src/openroboxing/ | head -20
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python \
  -c "from openroboxing.paths import REPO_ROOT; print(REPO_ROOT)"
```

Expected: the listing shows `runtime client league parity paths.py poses server spec studio tools`
and **no** `tests` or `docs`. The Python command prints
`/home/hpc-dev/OpenRoboxing/src` — wrong, and Task 3 fixes it. Seeing the wrong value here confirms
the layout moved and the fix is needed.

- [ ] **Step 5: Commit**

```bash
cd /home/hpc-dev/OpenRoboxing
git add -A
git commit -m "chore: move the package to src/, hoist tests and docs"
```

---

### Task 3: paths.py — one file that knows the layout

**Files:**
- Modify: `src/openroboxing/paths.py` (whole file replaced)
- Test: `tests/test_paths.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_paths.py`:

```python
"""The layout contract.

`paths.py` is the only module that knows where anything lives, so these are the assertions that
make the boundary between OpenRoboxing and upstream checkable rather than assumed.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


def _reload_paths(monkeypatch, gr00t_root: str | None):
    """Re-import `paths` with a chosen OPENROBOXING_GR00T_ROOT."""
    if gr00t_root is None:
        monkeypatch.delenv("OPENROBOXING_GR00T_ROOT", raising=False)
    else:
        monkeypatch.setenv("OPENROBOXING_GR00T_ROOT", gr00t_root)
    import openroboxing.paths as paths

    return importlib.reload(paths)


def test_repo_root_is_the_repository_not_src(monkeypatch):
    paths = _reload_paths(monkeypatch, None)
    assert paths.REPO_ROOT.name != "src"
    assert (paths.REPO_ROOT / "src" / "openroboxing").is_dir()


def test_gr00t_root_defaults_to_the_submodule(monkeypatch):
    paths = _reload_paths(monkeypatch, None)
    assert paths.GR00T_ROOT == paths.REPO_ROOT / "external/gr00t-wbc"


def test_gr00t_root_honours_the_environment(monkeypatch, tmp_path):
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    paths = _reload_paths(monkeypatch, str(elsewhere))
    assert paths.GR00T_ROOT == elsewhere
    assert paths.POLICY_DECODER_ONNX == elsewhere / "gear_sonic_deploy/policy/release/model_decoder.onnx"
    assert paths.MOTIONBRICKS_ROOT == elsewhere / "motionbricks"


def test_openroboxing_paths_do_not_follow_gr00t_root(monkeypatch, tmp_path):
    paths = _reload_paths(monkeypatch, str(tmp_path))
    assert paths.OPENROBOXING_ROOT == paths.REPO_ROOT / "src/openroboxing"
    assert paths.POSE_DIR == paths.OPENROBOXING_ROOT / "poses"
    assert paths.LOADOUT_DIR == paths.POSE_DIR / "loadouts"
    assert paths.FIXTURES_DIR == paths.REPO_ROOT / "tests/fixtures"


def test_display_path_names_upstream_artefacts_by_their_upstream_position(monkeypatch, tmp_path):
    """A manifest must read the same whether upstream is the submodule or a checkout elsewhere."""
    paths = _reload_paths(monkeypatch, str(tmp_path))
    assert (
        paths.display_path(paths.POLICY_DECODER_ONNX)
        == "gear_sonic_deploy/policy/release/model_decoder.onnx"
    )


def test_display_path_names_our_own_files_by_their_repo_position(monkeypatch):
    paths = _reload_paths(monkeypatch, None)
    assert paths.display_path(paths.LOADOUT_DIR) == "src/openroboxing/poses/loadouts"


def test_display_path_prefers_upstream_when_the_submodule_is_nested(monkeypatch):
    """With the default submodule, GR00T_ROOT is inside REPO_ROOT — upstream naming must still win."""
    paths = _reload_paths(monkeypatch, None)
    assert (
        paths.display_path(paths.POLICY_ENCODER_ONNX)
        == "gear_sonic_deploy/policy/release/model_encoder.onnx"
    )


def test_display_path_falls_back_to_an_absolute_path(monkeypatch):
    paths = _reload_paths(monkeypatch, None)
    stranger = Path("/etc/hostname")
    assert paths.display_path(stranger) == "/etc/hostname"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/hpc-dev/OpenRoboxing
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python \
  -m pytest tests/test_paths.py -v -p no:cacheprovider
```

Expected: FAIL. `test_repo_root_is_the_repository_not_src` fails on `REPO_ROOT.name != "src"`, and
every `GR00T_ROOT` / `display_path` test fails with `AttributeError`.

- [ ] **Step 3: Replace `src/openroboxing/paths.py` entirely**

```python
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
#: Loadouts sit *beside* the libraries, not among them: ``load_library`` reads every JSON file in a
#: directory, so a loadout filed with the poses is parsed as a malformed pose.
LOADOUT_DIR: Path = POSE_DIR / "loadouts"


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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/hpc-dev/OpenRoboxing
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python \
  -m pytest tests/test_paths.py -v -p no:cacheprovider
```

Expected: 8 passed.

- [ ] **Step 5: Verify the artefacts actually resolve against the real checkout**

```bash
cd /home/hpc-dev/OpenRoboxing
OPENROBOXING_GR00T_ROOT=/home/hpc-dev/GR00T-WholeBodyControl \
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python - <<'PY'
from openroboxing import paths
for name in ("G1_29DOF_SIM_XML", "POLICY_DECODER_ONNX", "OBSERVATION_CONFIG_YAML",
             "GENERATOR_SKELETON_XML", "GENERATOR_SKELETON_DIR", "REFERENCE_MOTION_DIR",
             "LOADOUT_DIR", "GOLDEN_POLICY_IO_DIR"):
    p = getattr(paths, name)
    print(f"{'OK ' if p.exists() else 'MISSING'} {name}: {p}")
PY
```

Expected: every line starts `OK`. A `MISSING` line names a path that re-rooted wrongly.

- [ ] **Step 6: Commit**

```bash
cd /home/hpc-dev/OpenRoboxing
git add src/openroboxing/paths.py tests/test_paths.py
git commit -m "feat(paths): GR00T_ROOT, an env override, and stable artefact naming"
```

---

### Task 4: The three modules that computed paths for themselves

`env_report.py` computes its own `REPO_ROOT`, and it plus `manifest.py` call
`relative_to(REPO_ROOT)` on artefacts that now live under `GR00T_ROOT` — which raises `ValueError`
the moment `OPENROBOXING_GR00T_ROOT` points outside the repository. `bench_world.py` builds an
upstream path from `REPO_ROOT`. All three are corrected here.

**Files:**
- Modify: `src/openroboxing/tools/env_report.py:34`, `:39-57`, `:174`
- Modify: `src/openroboxing/tools/bench_world.py:33`, `:36`
- Modify: `src/openroboxing/league/manifest.py:171-226`
- Test: `tests/test_manifest_paths.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_manifest_paths.py`:

```python
"""A manifest names assets identically wherever upstream lives.

`freeze` used to compute `path.relative_to(REPO_ROOT)`, which raises ValueError once upstream sits
outside the repository — the normal case when OPENROBOXING_GR00T_ROOT names another checkout.
"""

from __future__ import annotations

import importlib

import pytest


def test_upstream_assets_are_named_relative_to_upstream(monkeypatch):
    monkeypatch.setenv("OPENROBOXING_GR00T_ROOT", "/home/hpc-dev/GR00T-WholeBodyControl")
    import openroboxing.paths as paths

    importlib.reload(paths)

    assert paths.display_path(paths.POLICY_ENCODER_ONNX) == (
        "gear_sonic_deploy/policy/release/model_encoder.onnx"
    )
    assert paths.display_path(paths.G1_29DOF_SIM_XML) == "gear_sonic_deploy/g1/g1_29dof_old.xml"


def test_freeze_does_not_raise_when_upstream_is_outside_the_repo(monkeypatch, tmp_path):
    """The regression this task exists for: relative_to() on a path under another root."""
    monkeypatch.setenv("OPENROBOXING_GR00T_ROOT", "/home/hpc-dev/GR00T-WholeBodyControl")
    import openroboxing.paths as paths

    importlib.reload(paths)
    if not paths.POLICY_ENCODER_ONNX.exists():
        pytest.skip("policy checkpoints are not present on this machine")

    import openroboxing.league.manifest as manifest

    importlib.reload(manifest)
    result = manifest.freeze(
        "test-season", timestamp="2026-08-19T00:00:00Z", pose_library="v0.1"
    )
    assert result.by_name("policy_encoder").path == (
        "gear_sonic_deploy/policy/release/model_encoder.onnx"
    )
    assert result.by_name("robot_model").path == "gear_sonic_deploy/g1/g1_29dof_old.xml"
    assert result.by_name("pose_library").path == "src/openroboxing/poses/v0.1"
```

`freeze` is `freeze(season, *, timestamp, pose_library="v0.1", release_acknowledgement=None)` and
returns a `SeasonManifest` with a `by_name` lookup — `timestamp` is keyword-only and there is no
`at` parameter.

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /home/hpc-dev/OpenRoboxing
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python \
  -m pytest tests/test_manifest_paths.py -v -p no:cacheprovider
```

Expected: `test_freeze_does_not_raise_when_upstream_is_outside_the_repo` FAILS with
`ValueError: '/home/hpc-dev/GR00T-WholeBodyControl/gear_sonic_deploy/...' is not in the subpath of
'/home/hpc-dev/OpenRoboxing'`.

If `freeze`'s signature differs from `freeze(season=..., at=..., pose_library=...)`, read
`src/openroboxing/league/manifest.py` and adjust the call — the assertion on `names` is the point,
not the call shape.

- [ ] **Step 3: Fix `league/manifest.py`**

Replace the import block at `:171-178`:

```python
    from openroboxing.paths import (
        G1_29DOF_SIM_XML,
        OPENROBOXING_ROOT,
        POLICY_DECODER_ONNX,
        POLICY_ENCODER_ONNX,
        POSE_DIR,
        display_path,
    )
```

Then replace each of the three `str(... .relative_to(REPO_ROOT))` expressions:

- `:193` → `path=display_path(path),`
- `:207` → `path=display_path(library),`
- `:222` → `path=display_path(spec),`

`REPO_ROOT` remains imported at `:272` for `_git_sha(REPO_ROOT)` — that call is correct and stays.

- [ ] **Step 4: Fix `tools/env_report.py`**

Replace `:33-34`:

```python
# The layout lives in one place. See openroboxing/paths.py.
from openroboxing.paths import GR00T_ROOT, REPO_ROOT, display_path
```

Delete the `PINNED_UPSTREAM_SHA` constant at `:37` and re-root the artefact table at `:39-57` — the
paths themselves are unchanged strings, they are simply resolved against `GR00T_ROOT` at `:106`:

```python
# Artefacts the M1 runtime cannot start without. Relative to GR00T_ROOT, not this repository.
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
```

At `:106`, resolve against upstream:

```python
        path = GR00T_ROOT / rel
```

At `:174`, name the artefact the way a manifest would:

```python
        print(f"        {display_path(a.path)}")
```

The git block at `:149-162` reports on the wrong repository now — `REPO_ROOT` is OpenRoboxing, and
the pinned-snapshot comparison no longer exists. Replace `:149-162` with a report on both:

```python
    print(f"repo root: {REPO_ROOT}")
    print(f"  branch            : {_git('rev-parse', '--abbrev-ref', 'HEAD')}")
    print(f"  HEAD              : {_git('rev-parse', 'HEAD')}")
    print(f"upstream: {GR00T_ROOT}")
    print(f"  HEAD              : {_git_in(GR00T_ROOT, 'rev-parse', 'HEAD')}")
    print(f"  behind origin/main: {_git_in(GR00T_ROOT, 'rev-list', '--count', 'HEAD..origin/main')}")
```

and generalise `_git` at `:73-78` so it can run in either tree:

```python
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
```

- [ ] **Step 5: Fix `tools/bench_world.py`**

At `:33`, import the constant instead of building the path:

```python
from openroboxing.paths import G1_29DOF_XML, SCENE_EMPTY_XML
```

At `:36`, use it:

```python
SCENE_EMPTY = SCENE_EMPTY_XML
```

If `REPO_ROOT` is unused elsewhere in the file, remove it from the import; if it is used, keep it.
Check with:

```bash
grep -n "REPO_ROOT" /home/hpc-dev/OpenRoboxing/src/openroboxing/tools/bench_world.py
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd /home/hpc-dev/OpenRoboxing
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python \
  -m pytest tests/test_manifest_paths.py tests/test_paths.py -v -p no:cacheprovider
```

Expected: all pass (or the freeze test skips if the ONNX files are absent — on this machine they are
present, so it should pass).

- [ ] **Step 7: Verify env_report runs end to end**

```bash
cd /home/hpc-dev/OpenRoboxing
OPENROBOXING_GR00T_ROOT=/home/hpc-dev/GR00T-WholeBodyControl \
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python \
  -m openroboxing.tools.env_report --quick
```

Expected: it prints both roots, and every required artefact is found. No traceback.

- [ ] **Step 8: Commit**

```bash
cd /home/hpc-dev/OpenRoboxing
git add src/openroboxing/tools/env_report.py src/openroboxing/tools/bench_world.py \
        src/openroboxing/league/manifest.py tests/test_manifest_paths.py
git commit -m "fix: artefacts under GR00T_ROOT no longer break relative_to(REPO_ROOT)"
```

---

### Task 5: Patch P0 becomes runtime code

The only behaviour change in the extraction. Upstream in the submodule has no
`_override_target_joint_transforms` **method** and no **call site** for it; `generator.py` supplies
both.

One compatibility requirement drives the design: on this machine `OPENROBOXING_GR00T_ROOT` points at
a checkout that **does** have P0 applied, so `generate_new_frames` there will call the upstream
method *after* our wrapper has already applied the override. Applying the override twice must
therefore be harmless. It is — the second application swaps in tensors that are already in place —
and a test pins that down rather than leaving it to luck.

**Files:**
- Modify: `src/openroboxing/runtime/generator.py:241-280` (`_install_pose_override`), plus a new module-level function
- Test: `tests/test_generator_pose_override.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_generator_pose_override.py`:

```python
"""Patch P0, installed at runtime instead of patched into upstream.

Upstream in the submodule is pristine: it has neither `_override_target_joint_transforms` nor a call
site for it. `MotionBricksGenerator._install_pose_override` supplies both by wrapping
`_generate_target_joint_transforms` on the agent instance.

These tests use a stand-in agent rather than the real one, so they run in under a second and, more
importantly, so the pristine case can be tested on a machine whose checkout happens to carry P0.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openroboxing.runtime.generator import _apply_target_pose_override


class _PristineAgent:
    """Upstream without P0: it can build a target, and knows nothing about overrides."""

    def __init__(self):
        self.calls = 0

    def _generate_target_joint_transforms(self, input: dict):
        self.calls += 1
        positions = torch.zeros(1, 4, 34, 3)
        rotations = torch.eye(3).expand(1, 4, 34, 3, 3).clone()
        root_positions = torch.zeros(1, 4, 3)
        return positions, rotations, root_positions


def _install(agent, armed_positions=None, armed_rotations=None):
    """The wrapper under test, driven without building a real generator.

    Stands in for `MotionBricksGenerator._install_pose_override`: the generator's `_arm` runs skeleton
    FK to produce the tensors, and here they are simply handed in. The generator's own version is
    exercised by the slow end-to-end tests.
    """
    from openroboxing.runtime.generator import _wrap_target_transforms

    def _arm(input: dict) -> None:
        if armed_positions is None:
            return
        input["specific_target_joint_positions"] = armed_positions
        input["specific_target_joint_rotations"] = armed_rotations

    _wrap_target_transforms(agent, _arm)


def test_wrapper_fills_the_three_keys_upstream_assigns():
    agent = _PristineAgent()
    _install(agent)
    input: dict = {}
    positions, rotations, root_positions = agent._generate_target_joint_transforms(input)

    assert input["target_global_joint_positions"] is positions
    assert input["target_global_joint_rotations"] is rotations
    assert input["target_global_root_positions"] is root_positions


def test_no_armed_pose_leaves_the_target_untouched():
    agent = _PristineAgent()
    _install(agent)
    input: dict = {}
    positions, _, _ = agent._generate_target_joint_transforms(input)
    assert torch.equal(positions, torch.zeros(1, 4, 34, 3))


def test_an_armed_pose_replaces_the_target():
    agent = _PristineAgent()
    armed_positions = torch.full((1, 4, 34, 3), 0.5)
    armed_rotations = torch.eye(3).expand(1, 4, 34, 3, 3).clone()
    _install(agent, armed_positions, armed_rotations)

    input: dict = {}
    positions, _, _ = agent._generate_target_joint_transforms(input)

    assert torch.equal(positions, armed_positions)
    assert torch.equal(input["target_global_joint_positions"], armed_positions)


def test_the_returned_tuple_matches_the_input_dict():
    """upstream re-assigns the tuple into the same keys, so the two must not diverge."""
    agent = _PristineAgent()
    armed_positions = torch.full((1, 4, 34, 3), 0.5)
    armed_rotations = torch.eye(3).expand(1, 4, 34, 3, 3).clone()
    _install(agent, armed_positions, armed_rotations)

    input: dict = {}
    positions, rotations, root_positions = agent._generate_target_joint_transforms(input)

    assert torch.equal(positions, input["target_global_joint_positions"])
    assert torch.equal(rotations, input["target_global_joint_rotations"])
    assert torch.equal(root_positions, input["target_global_root_positions"])


def test_applying_the_override_twice_is_idempotent():
    """A checkout that still carries P0 calls the upstream method after our wrapper has run."""
    input = {
        "target_global_joint_positions": torch.zeros(1, 4, 34, 3),
        "target_global_joint_rotations": torch.eye(3).expand(1, 4, 34, 3, 3).clone(),
        "specific_target_joint_positions": torch.full((1, 4, 34, 3), 0.5),
        "specific_target_joint_rotations": torch.eye(3).expand(1, 4, 34, 3, 3).clone(),
    }
    _apply_target_pose_override(input)
    once = input["target_global_joint_positions"].clone()
    _apply_target_pose_override(input)

    assert torch.equal(input["target_global_joint_positions"], once)


def test_a_shape_mismatch_raises_rather_than_broadcasting():
    input = {
        "target_global_joint_positions": torch.zeros(1, 4, 34, 3),
        "target_global_joint_rotations": torch.eye(3).expand(1, 4, 34, 3, 3).clone(),
        "specific_target_joint_positions": torch.zeros(1, 4, 29, 3),
    }
    with pytest.raises(ValueError, match="specific_target_joint_positions"):
        _apply_target_pose_override(input)


def test_absent_keys_are_inert():
    positions = torch.zeros(1, 4, 34, 3)
    input = {"target_global_joint_positions": positions}
    _apply_target_pose_override(input)
    assert input["target_global_joint_positions"] is positions


def test_the_mask_blends_per_batch_element():
    current = torch.zeros(2, 4, 34, 3)
    override = torch.ones(2, 4, 34, 3)
    input = {
        "target_global_joint_positions": current,
        "specific_target_joint_positions": override,
        "has_specific_target_pose": torch.tensor([[1], [0]]),
    }
    _apply_target_pose_override(input)
    result = input["target_global_joint_positions"]
    assert torch.equal(result[0], torch.ones(4, 34, 3))
    assert torch.equal(result[1], torch.zeros(4, 34, 3))
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /home/hpc-dev/OpenRoboxing
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python \
  -m pytest tests/test_generator_pose_override.py -v -p no:cacheprovider
```

Expected: collection fails with
`ImportError: cannot import name '_apply_target_pose_override' from 'openroboxing.runtime.generator'`.

- [ ] **Step 3: Add P0's body and the wrapper to `runtime/generator.py`**

Add both functions at module level, immediately above the `GeneratorConfig` dataclass:

```python
def _apply_target_pose_override(input: dict) -> dict:
    """Patch P0's body, installed at runtime rather than patched into upstream.

    Deliberately dumb: it swaps two tensors and validates their shapes. It cannot *compute* them,
    because the authored pose has to be re-rooted onto the placement the spring model chose, and
    that placement only exists inside ``generate_new_frames``. Everything that decides *what* the
    tensors are lives in :meth:`MotionBricksGenerator._install_pose_override`.

    Optional input keys, both ignored when absent:
        specific_target_joint_positions: [batch, NUM_FRAMES_PER_TOKEN, num_joints, 3]
        specific_target_joint_rotations: [batch, NUM_FRAMES_PER_TOKEN, num_joints, 3, 3]

    ``has_specific_target_pose`` ([batch, 1], int) blends per batch element when given; without it
    an override that is present applies fully.

    Applying this twice is a no-op the second time — the tensors it writes are the tensors it would
    write again. That matters because a checkout carrying the original patch still calls upstream's
    own copy after this one has run.
    """
    positions = input.get("specific_target_joint_positions", None)
    rotations = input.get("specific_target_joint_rotations", None)
    if positions is None and rotations is None:
        return input

    mask = input.get("has_specific_target_pose", None)

    if positions is not None:
        current = input["target_global_joint_positions"]
        if positions.shape != current.shape:
            raise ValueError(
                f"specific_target_joint_positions has shape {tuple(positions.shape)}, "
                f"expected {tuple(current.shape)}"
            )
        if mask is None:
            input["target_global_joint_positions"] = positions
        else:
            blend = mask.view([-1, 1, 1, 1]).float()
            input["target_global_joint_positions"] = positions * blend + current * (1.0 - blend)

    if rotations is not None:
        current = input["target_global_joint_rotations"]
        if rotations.shape != current.shape:
            raise ValueError(
                f"specific_target_joint_rotations has shape {tuple(rotations.shape)}, "
                f"expected {tuple(current.shape)}"
            )
        if mask is None:
            input["target_global_joint_rotations"] = rotations
        else:
            blend = mask.view([-1, 1, 1, 1, 1]).float()
            input["target_global_joint_rotations"] = rotations * blend + current * (1.0 - blend)

    return input


def _wrap_target_transforms(agent, arm) -> None:
    """Install P0's call site on an agent instance.

    Upstream's ``generate_new_frames`` does::

        input['target_global_joint_positions'], input['target_global_joint_rotations'], \\
            input['target_global_root_positions'] = self._generate_target_joint_transforms(input)

    The original patch added a call to the override on the next line. With a pristine upstream there
    is no next line, so the wrapper does the assignment itself, runs ``arm`` (which writes the
    ``specific_target_joint_*`` keys when a pose is armed) and then the override, and finally returns
    the three values re-read from ``input``. Upstream's own assignment then writes back what is
    already there — which is what makes the two paths equivalent.

    ``agent`` is the ``full_navigation_agent`` instance; ``arm`` is called with ``input`` after the
    target exists and before the override runs.
    """
    original = agent._generate_target_joint_transforms

    def _with_authored_pose(input: dict):
        positions, rotations, root_positions = original(input)
        input["target_global_joint_positions"] = positions
        input["target_global_joint_rotations"] = rotations
        input["target_global_root_positions"] = root_positions

        arm(input)
        _apply_target_pose_override(input)

        return (
            input["target_global_joint_positions"],
            input["target_global_joint_rotations"],
            input["target_global_root_positions"],
        )

    agent._generate_target_joint_transforms = _with_authored_pose
```

- [ ] **Step 4: Rewrite `_install_pose_override` (`:241-280`) to use them**

Replace the whole method:

```python
    def _install_pose_override(self) -> None:
        """Install patch P0 on the agent, and feed it its tensors.

        P0 is no longer a diff inside ``full_agent.py``: the submodule tracks NVlabs ``main`` and is
        pristine, so both the override and its call site are installed here at runtime — see
        :func:`_wrap_target_transforms` and :func:`_apply_target_pose_override`, and
        ``spec/upstream_patches.md``.

        P0 itself cannot compute its tensors, because the authored pose has to be re-rooted onto the
        placement the spring model chose and that placement only exists inside
        ``generate_new_frames``. So when a pose is armed we read the freshly generated target out of
        the input dict, re-root the pose onto it, and write the result back under the keys P0 reads.

        Every OpenRoboxing decision — heading-only placement, which model to run FK on — therefore
        lives on this side of the boundary, and upstream stays untouched.
        """

        def _arm(input: dict) -> None:
            if self._armed_pose is None:
                return
            if self._skeleton_fk is None:
                from openroboxing.studio.skeleton_fk import skeleton_fk

                self._skeleton_fk = skeleton_fk()
            positions, rotations = self._skeleton_fk.target_transforms(
                self._armed_pose,
                input["target_global_joint_positions"],
                input["target_global_joint_rotations"],
            )
            input["specific_target_joint_positions"] = positions
            input["specific_target_joint_rotations"] = rotations

        _wrap_target_transforms(self.agent, _arm)
```

The old `GeneratorError` guard for missing `target_global_joint_*` keys is gone on purpose: the
wrapper writes those keys itself one line earlier, so the condition it checked is now structurally
impossible.

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd /home/hpc-dev/OpenRoboxing
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python \
  -m pytest tests/test_generator_pose_override.py -v -p no:cacheprovider
```

Expected: 8 passed.

- [ ] **Step 6: Update the module docstring**

In `src/openroboxing/runtime/generator.py`, the "Authored poses" bullet says *"Patch P0 upstream only
consumes two tensors"*. Replace that bullet with:

```
- **Authored poses** ride on :attr:`GeneratorIntent.pose`. Patch P0 is installed here at runtime
  rather than patched into upstream — :func:`_wrap_target_transforms` adds its call site and
  :func:`_apply_target_pose_override` is its body — so the GR00T-WBC submodule stays pristine and
  can track NVlabs ``main``. Everything that decides *what* the tensors are lives in
  :meth:`MotionBricksGenerator._install_pose_override`.
```

Also correct the second line of the docstring: "Upstream and unmodified (`CLAUDE.md` invariant 3)"
is now literally true of the submodule, and worth saying so:

```
Upstream is unmodified — genuinely, now that patch P0 is installed at runtime (`CLAUDE.md`
invariant 3). The only things this module adds are a headless, scripted way to drive the generator
and that runtime patch.
```

- [ ] **Step 7: Prove it against the real generator**

This builds the actual MotionBricks agent, so it is slow and needs the checkpoints.

```bash
cd /home/hpc-dev/OpenRoboxing
OPENROBOXING_GR00T_ROOT=/home/hpc-dev/GR00T-WholeBodyControl \
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python - <<'PY'
from openroboxing.runtime.generator import MotionBricksGenerator
g = MotionBricksGenerator()
fn = g.agent._generate_target_joint_transforms
print("wrapped:", fn.__name__ == "_with_authored_pose")
print("upstream P0 present:", hasattr(g.agent, "_override_target_joint_transforms"))
PY
```

Expected: `wrapped: True`. `upstream P0 present` prints `True` on this machine (the checkout carries
the old patch) and `False` against the submodule — both are fine, which is what
`test_applying_the_override_twice_is_idempotent` exists to guarantee.

- [ ] **Step 8: Commit**

```bash
cd /home/hpc-dev/OpenRoboxing
git add src/openroboxing/runtime/generator.py tests/test_generator_pose_override.py
git commit -m "feat(generator): install patch P0 at runtime so upstream stays pristine"
```

---

### Task 6: pyproject.toml and the pytest configuration

**Files:**
- Create: `pyproject.toml`
- Delete: nothing — `pytest.ini` was deliberately not copied in Task 2

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "openroboxing"
version = "0.1.0"
description = "A boxing game for the Unitree G1, built on GEAR-SONIC and MotionBricks"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
# Runtime dependencies are measured, not guessed: see requirements-runtime.txt and its header.
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = { file = ["requirements-runtime.txt"] }

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
openroboxing = [
    "client/**/*",
    "poses/**/*.json",
    "poses/**/*.png",
    "spec/**/*.md",
    "spec/**/*.yaml",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
# `slow` covers end-to-end runs that build the generator; deselected by default.
markers = ["slow: end-to-end runs that build the generator (deselect with -m 'not slow')"]
addopts = "-m 'not slow'"

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

- [ ] **Step 2: Verify pytest picks up the configuration**

```bash
cd /home/hpc-dev/OpenRoboxing
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python \
  -m pytest --collect-only -q 2>&1 | tail -5
```

Expected: it collects from `tests/` without being told where to look, and the summary shows tests
deselected by the `not slow` marker.

- [ ] **Step 3: Run the whole suite**

```bash
cd /home/hpc-dev/OpenRoboxing
OPENROBOXING_GR00T_ROOT=/home/hpc-dev/GR00T-WholeBodyControl \
PYTHONPATH=src /home/hpc-dev/GR00T-WholeBodyControl/.venv_mb/bin/python \
  -m pytest -q 2>&1 | tail -20
```

Expected: the same pass/skip counts the source repo produces. Record the numbers — Task 10 compares
against them. If anything fails, it is a path that Tasks 3-4 missed; fix it here rather than
deferring.

- [ ] **Step 4: Commit**

```bash
cd /home/hpc-dev/OpenRoboxing
git add pyproject.toml
git commit -m "build: pyproject with the src layout and pytest configuration"
```

---

### Task 7: install.sh

**Files:**
- Modify: `install.sh`

- [ ] **Step 1: Re-root the paths at the top of the script**

Replace the header block:

```bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Overridable so this script can be tested against a throwaway venv without touching a working one.
VENV="${OPENROBOXING_VENV:-${REPO_ROOT}/.venv_mb}"
PYTHON_MIN="3.10"

# Upstream. The submodule unless the caller already has a GR00T-WholeBodyControl checkout.
GR00T_ROOT="${OPENROBOXING_GR00T_ROOT:-${REPO_ROOT}/external/gr00t-wbc}"

# The dependency list lives in one place and was measured, not guessed. See its header.
REQUIREMENTS="${REPO_ROOT}/requirements-runtime.txt"
```

Note `dirname` no longer has `/..` appended: the script now sits at the repository root.

- [ ] **Step 2: Add upstream acquisition**

Insert this function after `make_venv`:

```bash
fetch_upstream() {
  if [ -n "${OPENROBOXING_GR00T_ROOT:-}" ]; then
    say "using the GR00T-WholeBodyControl checkout at ${GR00T_ROOT}"
    [ -d "${GR00T_ROOT}/motionbricks" ] \
      || die "OPENROBOXING_GR00T_ROOT=${GR00T_ROOT} has no motionbricks/ — is that a GR00T-WBC checkout?"
    return
  fi

  say "initialising the GR00T-WholeBodyControl submodule"
  git -C "$REPO_ROOT" submodule update --init --recursive external/gr00t-wbc \
    || die "could not initialise the submodule. Is this a git checkout?"

  say "fetching LFS content (meshes; several GB on a first run)"
  if command -v git-lfs >/dev/null; then
    git -C "$GR00T_ROOT" lfs pull || warn "git lfs pull failed; meshes may be pointer files"
  else
    warn "git-lfs is not installed — the robot meshes will be pointer files and rendering will fail."
    warn "  Install it: sudo apt install git-lfs && git lfs install"
  fi
}
```

- [ ] **Step 3: Teach `check_checkpoints` to download them**

Replace the function:

```bash
check_checkpoints() {
  local policy="${GR00T_ROOT}/gear_sonic_deploy/policy/release"
  if [ -f "${policy}/model_encoder.onnx" ] && [ -f "${policy}/model_decoder.onnx" ]; then
    say "policy checkpoints found"
    return 0
  fi

  say "fetching the GEAR-SONIC policy from nvidia/GEAR-SONIC (174 MB)"
  # Upstream's own downloader, run in upstream's tree. Nothing NVIDIA-licensed is redistributed by
  # this repository — see LICENSING.md.
  if (cd "$GR00T_ROOT" && "${VENV}/bin/python" download_from_hf.py \
        --output-dir "${GR00T_ROOT}/gear_sonic_deploy" --no-planner); then
    say "policy checkpoints downloaded"
    return 0
  fi

  warn "could not download the policy checkpoints to ${policy}"
  warn "  Run it by hand:  cd ${GR00T_ROOT} && python download_from_hf.py --output-dir gear_sonic_deploy"
  warn "  They are NVIDIA-licensed; see LICENSING.md for the terms."
  return 1
}
```

- [ ] **Step 4: Point the smoke test and `--play` at the new layout**

```bash
smoke_test() {
  say "smoke test"
  cd "$REPO_ROOT"
  "${VENV}/bin/python" -m openroboxing.tools.env_report --quick || {
    warn "env_report reported problems; see above"
    return 1
  }
  "${VENV}/bin/python" -m pytest -q -x
}

play() {
  cd "$REPO_ROOT"
  say "starting a hotseat match on http://localhost:8080/"
  say "  red plays 1-6 and SPACE; blue plays U I O J K L and ENTER"
  exec "${VENV}/bin/python" -m openroboxing.tools.serve_match
}
```

- [ ] **Step 5: Install the package itself, so `src/` is importable**

In `install_packages`, after the requirements install, add:

```bash
  say "installing openroboxing (editable)"
  if command -v uv >/dev/null; then
    VIRTUAL_ENV="$VENV" uv pip install -e "$REPO_ROOT"
  else
    "${VENV}/bin/pip" install -e "$REPO_ROOT"
  fi
```

- [ ] **Step 6: Call `fetch_upstream` from `main`**

In `main`, insert the call between `make_venv` and `install_packages`:

```bash
  check_platform
  check_python
  make_venv
  fetch_upstream
  install_packages
```

And update the two acceptance-criterion comments in the header block, which still say
`bash openroboxing/install.sh`:

```bash
#   bash install.sh && bash install.sh --play
```

- [ ] **Step 7: Verify the script parses and its help path works**

```bash
cd /home/hpc-dev/OpenRoboxing
bash -n install.sh && echo "syntax OK"
```

Expected: `syntax OK`. Do not run the installer yet — Task 10 does that as the acceptance check.

- [ ] **Step 8: Commit**

```bash
cd /home/hpc-dev/OpenRoboxing
git add install.sh
git commit -m "build(install): submodule init, LFS pull, policy download, editable install"
```

---

### Task 8: capture_run.sh stops naming another machine's path

**Files:**
- Modify: `src/openroboxing/parity/capture_run.sh:16`

- [ ] **Step 1: Replace the hard-coded deploy directory**

At `:16`, replace:

```bash
DEPLOY_DIR=/home/hpc-dev/GR00T-WholeBodyControl/gear_sonic_deploy
```

with:

```bash
# The deploy tree with patches P1/P2 applied and g1_deploy_onnx_ref built. That is NOT the
# submodule — the submodule is pristine on purpose (see spec/upstream_patches.md), and a capture
# needs a working copy where the dump patches are applied. Point this at it.
DEPLOY_DIR="${OPENROBOXING_DEPLOY_DIR:-${OPENROBOXING_GR00T_ROOT:-/home/hpc-dev/GR00T-WholeBodyControl}/gear_sonic_deploy}"

if [ ! -x "$DEPLOY_DIR/target/release/g1_deploy_onnx_ref" ]; then
  echo "no g1_deploy_onnx_ref in $DEPLOY_DIR/target/release" >&2
  echo "set OPENROBOXING_DEPLOY_DIR to a GR00T-WBC deploy tree with P1/P2 applied and built" >&2
  exit 1
fi
```

- [ ] **Step 2: Verify it parses**

```bash
cd /home/hpc-dev/OpenRoboxing
bash -n src/openroboxing/parity/capture_run.sh && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
cd /home/hpc-dev/OpenRoboxing
git add src/openroboxing/parity/capture_run.sh
git commit -m "fix(parity): name the deploy tree by env var, not by absolute path"
```

---

### Task 9: Documentation

131 lines across 13 files name paths that have moved.

**Files:**
- Create: `CLAUDE.md` (moved from `docs/CLAUDE.md`)
- Modify: `README.md`, `docs/**/*.md`, `src/openroboxing/spec/*.md`

- [ ] **Step 1: Move CLAUDE.md to the root**

```bash
cd /home/hpc-dev/OpenRoboxing
git mv docs/CLAUDE.md CLAUDE.md 2>/dev/null || mv docs/CLAUDE.md CLAUDE.md
```

At the repository root it is loaded automatically by agent sessions instead of having to be found.

- [ ] **Step 2: Rewrite the paths mechanically**

```bash
cd /home/hpc-dev/OpenRoboxing
FILES=$(grep -rl "openroboxing/\|gear_sonic_deploy/\|motionbricks/" \
  --include='*.md' --include='*.html' . \
  --exclude-dir=external --exclude-dir=.git --exclude-dir='.venv*')

for f in $FILES; do
  sed -i \
    -e 's#openroboxing/tests#tests#g' \
    -e 's#openroboxing/docs#docs#g' \
    -e 's#openroboxing/install\.sh#install.sh#g' \
    -e 's#openroboxing/requirements-runtime\.txt#requirements-runtime.txt#g' \
    -e 's#openroboxing/README\.md#README.md#g' \
    -e 's#openroboxing/LICENSING\.md#LICENSING.md#g' \
    -e 's#\bopenroboxing/#src/openroboxing/#g' \
    -e 's#src/src/openroboxing/#src/openroboxing/#g' \
    "$f"
done
```

The last expression is a guard against double-application if the loop is re-run.

**Do not** rewrite `gear_sonic_deploy/` and `motionbricks/` blindly — many occurrences are prose
about upstream's own layout, where the bare path is correct. Handle them by hand in Step 3.

- [ ] **Step 3: Check what the rewrite did, and fix upstream references by hand**

```bash
cd /home/hpc-dev/OpenRoboxing
git diff --stat
grep -rn "gear_sonic_deploy/\|motionbricks/" --include='*.md' . \
  --exclude-dir=external --exclude-dir=.git --exclude-dir='.venv*' | head -40
```

For each hit, decide: a reference to a **file on disk that a reader is meant to open** gains the
`external/gr00t-wbc/` prefix; a reference **naming upstream's internal structure** stays bare. When
in doubt, leave it bare and add "(in the submodule)".

Also check nothing over-matched:

```bash
grep -rn "src/openroboxing\.\|src/src/\|python -m src\." --include='*.md' . \
  --exclude-dir=external --exclude-dir=.git --exclude-dir='.venv*'
```

Expected: no output. `python -m openroboxing.tools.X` invocations must be untouched — the package
name did not change, only its directory.

- [ ] **Step 4: Rewrite `spec/upstream_patches.md`**

Replace the "Pinned upstream commit" section with:

```markdown
## Upstream tracking

| Field | Value |
|---|---|
| Submodule | `external/gr00t-wbc` → `https://github.com/NVlabs/GR00T-WholeBodyControl.git` |
| Branch tracked | `main` |
| Override | `OPENROBOXING_GR00T_ROOT` names another checkout, bypassing the submodule |

The submodule is **pristine**. Nothing in it is edited, ever — that is what lets it track `main`.

A submodule always pins a SHA in the superproject; git has no follow-the-branch checkout mode. So
"tracking `main`" means the bump is a deliberate command:

    git submodule update --remote external/gr00t-wbc

**After every bump, re-verify:** run the test suite (`test_generator_pose_override.py` is the one
that catches a changed hook signature), and re-check the observation-registry offsets recorded in
`upstream_notes.md` — a rebase invalidates every line number in it.
```

Then restate the three patch statuses:

- **P0** — `| Status | **installed at runtime** since 2026-08-19 — see `runtime/generator.py`,
  `_wrap_target_transforms` and `_apply_target_pose_override` |`. Add a line: *"No longer a diff.
  The hook's body and its call site are installed on the agent instance at construction, so upstream
  needs no modification. `tests/test_generator_pose_override.py` proves it against a pristine
  agent."*
- **P1**, **P2** — `| Status | **upstream-side, fixture capture only** |`. Add: *"Applied in a
  GR00T-WBC working copy used to capture golden fixtures, never in the submodule. The fixture they
  produced (`tests/fixtures/golden_policy_io/golden.npz`) is committed, so nothing here needs them
  to run. Point `capture_run.sh` at that working copy with `OPENROBOXING_DEPLOY_DIR`."*

Finally, replace the "Open action (blocks `M0-T1` acceptance)" section with:

```markdown
## Closed: the fork is not needed

`M0-T1` required this checkout to become a fork of NVlabs, so that P0 could live in git. Installing
P0 at runtime removes the reason. There is no fork, and no plan for one: OpenRoboxing consumes
upstream and never modifies it.
```

- [ ] **Step 5: Write the root README**

Prepend to `README.md`:

```markdown
# OpenRoboxing

A boxing game for the Unitree G1 — one player fighter against another, driven through NVIDIA's
GEAR-SONIC whole-body policy and the MotionBricks motion generator, in MuJoCo, from a browser.

## Install

```bash
git clone --recurse-submodules https://github.com/TriesteOpenRoboticsCommunity/OpenRoboxing.git
cd OpenRoboxing
bash install.sh
```

The installer creates `.venv_mb`, initialises the `external/gr00t-wbc` submodule, pulls its LFS
meshes, and downloads the GEAR-SONIC policy from `nvidia/GEAR-SONIC`. If you already have a
GR00T-WholeBodyControl checkout, point at it and skip the clone:

```bash
export OPENROBOXING_GR00T_ROOT=/path/to/GR00T-WholeBodyControl
bash install.sh
```

## Run

```bash
.venv_mb/bin/python -m openroboxing.tools.serve_sparring   # the debug bench, http://localhost:8081/
.venv_mb/bin/python -m openroboxing.tools.serve_match      # a hotseat match, http://localhost:8080/
```

## Upstream

`external/gr00t-wbc` is NVlabs/GR00T-WholeBodyControl, tracking `main`, and is never modified.
The one behaviour OpenRoboxing needs from it — an authored key pose replacing the clip-sampled
target — is installed at runtime; see `src/openroboxing/spec/upstream_patches.md`.

Source is Apache-2.0. Model weights are NVIDIA's and are downloaded, not redistributed; see
`LICENSING.md`.

---
```

- [ ] **Step 6: Verify no stale path survives**

```bash
cd /home/hpc-dev/OpenRoboxing
grep -rn "openroboxing/tests\|openroboxing/docs\|bash openroboxing/install.sh" \
  --include='*.md' --include='*.py' --include='*.sh' . \
  --exclude-dir=external --exclude-dir=.git --exclude-dir='.venv*'
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
cd /home/hpc-dev/OpenRoboxing
git add -A
git commit -m "docs: re-point every path at the new layout, restate the patch registry"
```

---

### Task 10: Acceptance, and collapse to a single commit

**Files:** none modified — this task verifies and rewrites history.

- [ ] **Step 1: Run the real installer**

```bash
cd /home/hpc-dev/OpenRoboxing
OPENROBOXING_GR00T_ROOT=/home/hpc-dev/GR00T-WholeBodyControl bash install.sh
```

Expected: it ends with `==> ready. Start a match with:`. This creates `.venv_mb` in the new repo and
installs the package editable, so `PYTHONPATH=src` is no longer needed from here on.

- [ ] **Step 2: Run the suite from the installed package**

```bash
cd /home/hpc-dev/OpenRoboxing
OPENROBOXING_GR00T_ROOT=/home/hpc-dev/GR00T-WholeBodyControl \
  .venv_mb/bin/python -m pytest -q 2>&1 | tail -10
```

Expected: the pass/skip counts recorded in Task 6 Step 3. Any regression is a path the earlier tasks
missed.

- [ ] **Step 3: Run env_report in full**

```bash
cd /home/hpc-dev/OpenRoboxing
OPENROBOXING_GR00T_ROOT=/home/hpc-dev/GR00T-WholeBodyControl \
  .venv_mb/bin/python -m openroboxing.tools.env_report
```

Expected: every required artefact found, both repository roots reported, no traceback.

- [ ] **Step 4: Serve the sparring bench**

```bash
cd /home/hpc-dev/OpenRoboxing
OPENROBOXING_GR00T_ROOT=/home/hpc-dev/GR00T-WholeBodyControl \
  .venv_mb/bin/python -m openroboxing.tools.serve_sparring
```

Expected: `loadout ...`, then `ready: nq=..., ... substeps per tick`, then the server on :8081.
Open `http://localhost:8081/`, press `1` to stage a pose and `SPACE` to commit it, and watch the
fighter move. That is the acceptance check from `spec/sparring_protocol.md`. Stop it with Ctrl-C.

**This is the step that proves patch P0 still works.** If the fighter ignores the pose, P0's install
is wrong — go back to Task 5, do not paper over it.

- [ ] **Step 5: Collapse to the single initial commit**

```bash
cd /home/hpc-dev/OpenRoboxing
git status --porcelain
```

Expected: empty. Commit anything outstanding first, then:

```bash
cd /home/hpc-dev/OpenRoboxing
git checkout --orphan initial-import
git add -A
git commit -m "$(cat <<'MSG'
OpenRoboxing — extracted from NVlabs/GR00T-WholeBodyControl

A boxing game for the Unitree G1: intent timeline → MotionBricks → GEAR-SONIC →
MuJoCo, played from a browser.

Imported from the openroboxing branch of a GR00T-WholeBodyControl checkout at
bf9eaa2, where it was built over 75 commits. Upstream is now a submodule at
external/gr00t-wbc tracking NVlabs main, and it is pristine: the one modification
this project needs — patch P0, an authored key pose replacing the clip-sampled
target — is installed on the agent instance at runtime instead of patched into
full_agent.py. tests/test_generator_pose_override.py holds that line.

Design and plan: docs/superpowers/specs/2026-08-19-openroboxing-extraction-design.md
MSG
)"
git branch -D main
git branch -m main
git log --oneline
```

Expected: `git log --oneline` shows exactly one commit.

- [ ] **Step 6: Set the remote, and stop**

```bash
cd /home/hpc-dev/OpenRoboxing
git remote add origin https://github.com/TriesteOpenRoboticsCommunity/OpenRoboxing.git
git remote -v
git status
```

Expected: `origin` points at the target URL, the tree is clean, one commit on `main`.

**Do not push.** The owner pushes:

```bash
cd /home/hpc-dev/OpenRoboxing
git push -u origin main
```

- [ ] **Step 7: Confirm the source repo was never touched**

```bash
cd /home/hpc-dev/GR00T-WholeBodyControl
git status --porcelain
git log --oneline -1
```

Expected: the only untracked entry is the pre-existing
`motionbricks/motionbricks/motion_backbone/models/__pycache__/`, and HEAD is the spec/plan commit.
The `openroboxing/` tree is unchanged, as the design requires.

---

## Notes for whoever executes this

**The one thing that can silently break.** If patch P0's install is wrong, nothing errors — the
fighter simply ignores every authored pose and boxes as though no pose were commanded. Task 5's
tests and Task 10 Step 4 exist for exactly that failure, which is why neither is optional.

**Why the intermediate commits then get squashed.** The owner asked for a single initial commit.
Working straight to one commit would mean hours of uncommitted edits; the commits in Tasks 1-9 are
the safety net, and Task 10 Step 5 delivers what was asked for.

**If a test fails that has nothing to do with paths.** It was probably already failing in the source
repo. Check before chasing it:

```bash
cd /home/hpc-dev/GR00T-WholeBodyControl
.venv_mb/bin/python -m pytest openroboxing/tests -q 2>&1 | tail -5
```
