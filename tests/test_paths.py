"""The layout contract.

`paths.py` is the only module that knows where anything lives, so these are the assertions that
make the boundary between OpenRoboxing and upstream checkable rather than assumed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _reload_paths(monkeypatch, gr00t_root: str | None):
    """Re-import `paths` with a chosen OPENROBOXING_GR00T_ROOT."""
    if gr00t_root is None:
        monkeypatch.delenv("OPENROBOXING_GR00T_ROOT", raising=False)
    else:
        monkeypatch.setenv("OPENROBOXING_GR00T_ROOT", gr00t_root)
    from openroboxing import paths

    return importlib.reload(paths)


@pytest.fixture(autouse=True)
def _restore_paths_module():
    """Undo `_reload_paths`.

    `importlib.reload` mutates the shared module object, and monkeypatch only restores the
    environment variable — not the constants already recomputed from it. Without this, a test
    here leaves `openroboxing.paths` pointing at whichever root ran last, and every later test
    in the suite that reads a path at run time silently resolves against the wrong tree.
    """
    yield
    importlib.reload(sys.modules["openroboxing.paths"])


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
    assert (
        paths.POLICY_DECODER_ONNX
        == elsewhere / "gear_sonic_deploy/policy/release/model_decoder.onnx"
    )
    assert paths.MOTIONBRICKS_ROOT == elsewhere / "motionbricks"


def test_gr00t_root_is_resolved_not_merely_joined(monkeypatch, tmp_path):
    """A `..` in OPENROBOXING_GR00T_ROOT must not leak into every path derived from it."""
    (tmp_path / "b").mkdir()
    messy = tmp_path / "a" / ".." / "b"
    paths = _reload_paths(monkeypatch, str(messy))
    assert paths.GR00T_ROOT == (tmp_path / "b").resolve()


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
    """With the default submodule, GR00T_ROOT sits inside REPO_ROOT — upstream naming must win."""
    paths = _reload_paths(monkeypatch, None)
    assert (
        paths.display_path(paths.POLICY_ENCODER_ONNX)
        == "gear_sonic_deploy/policy/release/model_encoder.onnx"
    )


def test_display_path_falls_back_to_an_absolute_path(monkeypatch):
    paths = _reload_paths(monkeypatch, None)
    stranger = Path("/etc/hostname")
    assert paths.display_path(stranger) == "/etc/hostname"
