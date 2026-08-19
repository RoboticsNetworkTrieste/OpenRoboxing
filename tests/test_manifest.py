"""M6-T1 acceptance: the season freeze.

Acceptance criterion from WORKPLAN.md M6-T1:
  a released season manifest pins every asset by version and hash; a match record can be traced to
  the exact assets that produced it.

The point of hashing rather than versioning is that a version string is a claim and a hash is a
fact. The tests that matter here are the ones where an asset changes underneath a manifest and the
manifest notices.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_manifest.py -v
"""

from __future__ import annotations

import json

import pytest

from openroboxing.league.manifest import (
    LICENCE_ACKNOWLEDGEMENT,
    Asset,
    ManifestError,
    SeasonManifest,
    file_hash,
    format_manifest,
    freeze,
    trace_record,
    tree_hash,
    verify,
)

FROZEN_AT = "2026-08-08T01:00:00Z"


@pytest.fixture(scope="module")
def manifest() -> SeasonManifest:
    return freeze("season-0", timestamp=FROZEN_AT)


# --- hashing --------------------------------------------------------------------------------------------
def test_a_files_hash_follows_its_bytes(tmp_path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("one")
    first = file_hash(path)
    path.write_text("two")
    assert file_hash(path) != first


def test_a_tree_hash_ignores_filesystem_order(tmp_path) -> None:
    for name in ("b.json", "a.json", "c.json"):
        (tmp_path / name).write_text(f'{{"n": "{name}"}}')
    first, count = tree_hash(tmp_path, "*.json")

    (tmp_path / "a.json").touch()  # mtime changes, contents do not
    second, _ = tree_hash(tmp_path, "*.json")

    assert first == second
    assert count == 3


def test_a_tree_hash_notices_a_renamed_file(tmp_path) -> None:
    """Contents alone are not enough: a pose library where two poses swapped names is different."""
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("[]")
    before, _ = tree_hash(tmp_path, "*.json")

    (tmp_path / "a.json").rename(tmp_path / "c.json")
    assert tree_hash(tmp_path, "*.json")[0] != before


def test_hashing_a_missing_directory_raises(tmp_path) -> None:
    with pytest.raises(ManifestError, match="not a directory"):
        tree_hash(tmp_path / "nope")


# --- freezing --------------------------------------------------------------------------------------------
def test_a_freeze_pins_the_policy_the_robot_and_the_library(manifest) -> None:
    names = {a.name for a in manifest.assets}
    assert {"policy_encoder", "policy_decoder", "robot_model", "pose_library"} <= names
    assert all(len(a.sha256) == 64 for a in manifest.assets)


def test_a_freeze_pins_the_rules_themselves(manifest) -> None:
    """"Rules v1.0" is only a checkable claim if the specs are pinned too."""
    specs = {a.name for a in manifest.assets if a.kind == "spec"}
    assert "spec/match_record" in specs
    assert "spec/scoring" in specs
    assert "spec/season" in specs


def test_the_frozen_robot_is_the_simulation_model(manifest) -> None:
    """Not its similarly-named sibling; see spec/upstream_notes.md."""
    assert manifest.by_name("robot_model").path.endswith("g1_29dof_old.xml")


def test_a_freeze_is_reproducible(manifest) -> None:
    again = freeze("season-0", timestamp=FROZEN_AT)
    assert [a.sha256 for a in again.assets] == [a.sha256 for a in manifest.assets]
    assert again.frozen_at == manifest.frozen_at


def test_a_missing_pose_library_refuses_to_freeze() -> None:
    with pytest.raises(ManifestError, match="pose library"):
        freeze("season-0", timestamp=FROZEN_AT, pose_library="v9.9")


def test_an_unknown_asset_raises_rather_than_returning_nothing(manifest) -> None:
    with pytest.raises(ManifestError, match="no asset"):
        manifest.by_name("weights_of_the_future")


# --- the licence gate ---------------------------------------------------------------------------------------
def test_a_manifest_is_not_released_by_default(manifest) -> None:
    """The terms permit publishing (LICENSING.md); the gate is what keeps it *deliberate*, so a
    manifest is never released as a side effect of freezing one."""
    assert manifest.released is False
    assert "no" in format_manifest(manifest).lower()
    assert "LICENSING.md" in manifest.notes["licensing"]


def test_releasing_needs_the_exact_acknowledgement() -> None:
    with pytest.raises(ManifestError, match="LICENSING.md"):
        freeze("season-0", timestamp=FROZEN_AT, release_acknowledgement="sure why not")


def test_the_acknowledgement_releases_it() -> None:
    released = freeze(
        "season-0", timestamp=FROZEN_AT, release_acknowledgement=LICENCE_ACKNOWLEDGEMENT
    )
    assert released.released is True
    assert "Released" in released.notes["licensing"]


def test_a_released_manifest_carries_the_notice_the_licence_requires() -> None:
    """NVIDIA Open Model License §3(b) requires this text verbatim wherever the Model is
    distributed. Carrying it in the release record means it cannot be forgotten at release time."""
    released = freeze(
        "season-0", timestamp=FROZEN_AT, release_acknowledgement=LICENCE_ACKNOWLEDGEMENT
    )
    assert (
        "Licensed by NVIDIA Corporation under the NVIDIA Open Model License."
        in released.notes["licensing"]
    )


def test_licensing_md_exists_and_answers_the_question() -> None:
    """M6-T2 asks for a page that states the position. A release gate with no stated position
    behind it is just a locked door."""
    from openroboxing.paths import REPO_ROOT

    page = (REPO_ROOT / "LICENSING.md").read_text()
    assert "NVIDIA Open Model License" in page, "the page must name the licence it relies on"
    assert (
        "Licensed by NVIDIA Corporation under the NVIDIA Open Model License." in page
    ), "the page must carry the verbatim notice §3(b) requires"
    assert "sign-off" in page.lower(), "M6-T2 wants a human on the record, by name and date"
    assert "Nothing has been published" in page, (
        "the page must be clear that stating what MAY be published is not publishing it"
    )


# --- verification ----------------------------------------------------------------------------------------
def test_an_untouched_season_verifies_clean(manifest) -> None:
    assert verify(manifest) == []


def test_a_changed_asset_is_caught(tmp_path) -> None:
    """The reason for hashing at all: a version string would still say v0.1."""
    library = tmp_path / "poses" / "v0.1"
    library.mkdir(parents=True)
    (library / "jab.json").write_text('{"name": "jab"}')
    digest, _ = tree_hash(library, "*.json")

    small = SeasonManifest(
        season="test",
        assets=[Asset(name="pose_library", kind="tree", path="poses/v0.1", sha256=digest)],
    )
    assert verify(small, sandbox_root=tmp_path) == []

    (library / "jab.json").write_text('{"name": "jab", "sneaky": true}')
    discrepancies = verify(small, sandbox_root=tmp_path)
    assert len(discrepancies) == 1
    assert discrepancies[0].reason == "changed"
    assert discrepancies[0].name == "pose_library"


def test_a_missing_asset_is_caught(tmp_path) -> None:
    small = SeasonManifest(
        season="test",
        assets=[Asset(name="gone", kind="file", path="nowhere.onnx", sha256="0" * 64)],
    )
    discrepancies = verify(small, sandbox_root=tmp_path)
    assert [d.reason for d in discrepancies] == ["missing"]
    assert discrepancies[0].actual is None


def test_a_manifest_round_trips_through_disk(manifest, tmp_path) -> None:
    path = manifest.save(tmp_path / "season-0.json")
    loaded = SeasonManifest.load(path)

    assert loaded.season == manifest.season
    assert loaded.frozen_at == FROZEN_AT
    assert [a.sha256 for a in loaded.assets] == [a.sha256 for a in manifest.assets]
    assert verify(loaded) == []


def test_an_unreadable_manifest_raises(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    with pytest.raises(ManifestError, match="cannot read the manifest"):
        SeasonManifest.load(bad)


# --- tracing a record ------------------------------------------------------------------------------------
def _record(**versions) -> dict:
    base = {
        "robot_model": "g1_29dof_old.xml",
        "pose_library": "v0.1",
        "openroboxing_sha": "deadbee",
    }
    base.update(versions)
    return {"match_id": "m1", "versions": base}


def test_a_record_traces_to_the_manifest_that_produced_it(manifest) -> None:
    """M6-T1's second half."""
    findings = trace_record(_record(), manifest)
    assert findings["traced"] is True
    assert findings["match_id"] == "m1"


def test_a_record_naming_the_wrong_robot_does_not_trace(manifest) -> None:
    findings = trace_record(_record(robot_model="g1_29dof.xml"), manifest)
    assert findings["traced"] is False
    assert any(c["result"] == "MISMATCH" for c in findings["checks"])


def test_a_record_naming_the_wrong_library_does_not_trace(manifest) -> None:
    assert trace_record(_record(pose_library="v0.2"), manifest)["traced"] is False


def test_a_record_that_says_nothing_does_not_trace(manifest) -> None:
    """Silence is not a pass. A record with no versions cannot be traced to anything."""
    findings = trace_record({"match_id": "m1", "versions": {}}, manifest)
    assert findings["traced"] is False
    assert any(c["result"] == "record does not say" for c in findings["checks"])


def test_a_differing_git_sha_is_not_fatal(manifest) -> None:
    """Code moves between the freeze and a match; the *assets* are what is pinned."""
    findings = trace_record(_record(openroboxing_sha="0000000"), manifest)
    assert findings["traced"] is True


def test_a_real_record_traces(manifest, tmp_path) -> None:
    """End to end against the shape `tools/run_match.py` actually writes."""
    from openroboxing.tools.run_match import _versions

    record = {"match_id": "real", "versions": _versions("v0.1")}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(record))

    assert trace_record(json.loads(path.read_text()), manifest)["traced"] is True
