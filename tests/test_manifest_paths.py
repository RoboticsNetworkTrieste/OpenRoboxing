"""A manifest names assets identically wherever upstream lives.

`freeze` used to compute `path.relative_to(REPO_ROOT)`, which raises ValueError once upstream sits
outside the repository — the normal case when OPENROBOXING_GR00T_ROOT names another checkout.
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_reloaded_paths():
    """Undo the reload below.

    `importlib.reload` mutates the shared module object, and monkeypatch restores only the
    environment variable — not the constants already recomputed from it. Without this, a test here
    leaves `openroboxing.paths` pointing at whichever root ran last, and every later test in the
    suite that reads a path at run time resolves against the wrong tree — silently, against the
    LFS-pointer-only submodule instead of the real checkout. Takes no `monkeypatch` parameter on
    purpose: pytest tears autouse fixtures down *after* explicitly-requested ones, so the
    environment is already restored when this reload runs.

    Deliberately reloads `openroboxing.paths` ONLY, even though tests below reload `paths` and then
    call into `openroboxing.league.manifest`. `importlib.reload` re-executes a module in its
    *existing* namespace dict, rebinding every name there — including exception classes. This used
    to also reload `manifest`, which rebound `ManifestError` to a new class object in that dict while
    `tests/test_manifest.py` (imported at collection time, before any fixture runs) kept the old one
    — so `pytest.raises(ManifestError, ...)` there silently stopped matching what `manifest.py`'s own
    functions raised, purely because this file happened to run first in the session. Reloading
    `manifest` is also unnecessary: `freeze()` and `verify()` both import from `openroboxing.paths`
    *inside* the function body, not at module scope, so a stale `manifest` function object still
    reads `GR00T_ROOT` / `REPO_ROOT` / `locate` fresh off the live `paths` module on every call — its
    `__globals__` is the very dict this reload refreshes. If a future change makes `manifest.py`
    cache anything from `paths` at import time, that is what would need reloading here — reloading
    `manifest` itself only reintroduces the class-identity hazard this comment is warning about.
    """
    yield
    importlib.reload(sys.modules["openroboxing.paths"])


def test_upstream_assets_are_named_relative_to_upstream(monkeypatch):
    monkeypatch.setenv("OPENROBOXING_GR00T_ROOT", "/home/hpc-dev/GR00T-WholeBodyControl")
    from openroboxing import paths

    importlib.reload(paths)

    assert paths.display_path(paths.POLICY_ENCODER_ONNX) == (
        "gear_sonic_deploy/policy/release/model_encoder.onnx"
    )
    assert paths.display_path(paths.G1_29DOF_SIM_XML) == "gear_sonic_deploy/g1/g1_29dof_old.xml"


def test_freeze_does_not_raise_when_upstream_is_outside_the_repo(monkeypatch):
    """The regression this task exists for: relative_to() on a path under another root."""
    monkeypatch.setenv("OPENROBOXING_GR00T_ROOT", "/home/hpc-dev/GR00T-WholeBodyControl")
    from openroboxing import paths

    importlib.reload(paths)
    if not paths.POLICY_ENCODER_ONNX.exists():
        pytest.skip("policy checkpoints are not present on this machine")

    from openroboxing.league import manifest

    result = manifest.freeze(
        "test-season", timestamp="2026-08-19T00:00:00Z", pose_library="v0.1"
    )
    assert result.by_name("policy_encoder").path == (
        "gear_sonic_deploy/policy/release/model_encoder.onnx"
    )
    assert result.by_name("robot_model").path == "gear_sonic_deploy/g1/g1_29dof_old.xml"
    assert result.by_name("pose_library").path == "src/openroboxing/poses/v0.1"


# --- locate(), display_path()'s inverse --------------------------------------------------------
#
# freeze() names an asset with display_path() — root-agnostic on purpose, so a manifest reads the
# same on every machine. verify() has to turn that name back into a path on THIS machine, which
# only works if locate() actually inverts display_path(): locate(display_path(p)) == p. That
# identity, not either function alone, is what makes the freeze/verify round trip hold.


def test_locate_inverts_display_path_in_the_default_configuration(monkeypatch):
    """Upstream constant and one of ours, with upstream as the submodule."""
    monkeypatch.delenv("OPENROBOXING_GR00T_ROOT", raising=False)
    from openroboxing import paths

    importlib.reload(paths)

    assert paths.locate(paths.display_path(paths.POLICY_ENCODER_ONNX)) == paths.POLICY_ENCODER_ONNX
    assert paths.locate(paths.display_path(paths.COMBINATION_DIR)) == paths.COMBINATION_DIR


def test_locate_inverts_display_path_when_upstream_is_outside_the_repo(monkeypatch):
    """Same identity, with upstream a checkout OPENROBOXING_GR00T_ROOT names elsewhere."""
    monkeypatch.setenv("OPENROBOXING_GR00T_ROOT", "/home/hpc-dev/GR00T-WholeBodyControl")
    from openroboxing import paths

    importlib.reload(paths)

    assert paths.locate(paths.display_path(paths.POLICY_ENCODER_ONNX)) == paths.POLICY_ENCODER_ONNX
    assert paths.locate(paths.display_path(paths.COMBINATION_DIR)) == paths.COMBINATION_DIR


def test_verify_finds_upstream_assets_with_no_explicit_root(monkeypatch):
    """The regression this round trip exists for.

    freeze() names policy_encoder/policy_decoder/robot_model by their position under GR00T_ROOT.
    verify(manifest), called with no root the way `freeze_season.py --verify` calls it, used to
    resolve every asset under REPO_ROOT regardless — so as soon as OPENROBOXING_GR00T_ROOT pointed
    outside the repository, those three came back "missing" even though nothing had changed on
    disk. Freezing and immediately verifying the same season must report zero discrepancies.
    """
    monkeypatch.setenv("OPENROBOXING_GR00T_ROOT", "/home/hpc-dev/GR00T-WholeBodyControl")
    from openroboxing import paths

    importlib.reload(paths)
    if not paths.POLICY_ENCODER_ONNX.exists():
        pytest.skip("policy checkpoints are not present on this machine")

    from openroboxing.league import manifest

    result = manifest.freeze(
        "verify-round-trip", timestamp="2026-08-19T00:00:00Z", pose_library="v0.1"
    )
    assert manifest.verify(result) == []
