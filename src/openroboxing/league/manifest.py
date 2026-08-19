"""The season manifest: what a season was fought with (M6-T1).

`WORKPLAN` M6-T1's acceptance: *a released season manifest pins every asset by version and hash; a
match record can be traced to the exact assets that produced it.*

Hashes, not versions
--------------------
A version string is a claim; a hash is a fact. `spec/pose_record.md` can say ``v0.1`` while somebody
edits a pose in place, and every record written afterwards would still say ``v0.1``. The manifest
therefore pins **SHA-256 of the bytes** of every asset, and :func:`verify` compares what is on disk
against what was frozen.

What this does **not** promise
------------------------------
`CLAUDE.md` invariant 6: determinism is recorded, not assumed. A manifest lets a match be *traced to
the assets that produced it*. It does not promise those assets re-derive the match — MuJoCo is not
bit-identical across machines and GPU inference adds noise, which is exactly why
`spec/match_record.md` makes the state trace authoritative.

Licensing: answered, but still deliberate
-----------------------------------------
`WORKPLAN` M6-T2 was blocking. It is **resolved** — the weights are under the NVIDIA Open Model
License, which grants the right to create and distribute derivative models (§2.2, §2.4), subject to
shipping the agreement and an attribution notice (§3). The full reading, the attribution text and the
one remaining question are in `LICENSING.md`, signed off 2026-08-08.

The gate below stays anyway. Knowing the answer is what makes publishing *possible*; requiring an
explicit acknowledgement is what keeps it *deliberate*, so a manifest is never released as a side
effect of freezing one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

SPEC_VERSION = "0.1"

#: Read in this many bytes at a time. The policy ONNX files are tens of megabytes.
_CHUNK = 1 << 20

#: What a caller must pass to mark a manifest released, so it cannot happen by accident.
LICENCE_ACKNOWLEDGEMENT = "M6-T2-signed-off"


class ManifestError(RuntimeError):
    """A season could not be frozen or verified. Never recovered from silently."""


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(directory: Path, pattern: str = "*") -> tuple[str, int]:
    """``(hash, file count)`` for a directory, order-independent.

    Each file contributes ``relative-path + its hash``, sorted, so the result depends on the
    contents and the names and not on the filesystem's ordering.
    """
    if not directory.is_dir():
        raise ManifestError(f"not a directory: {directory}")
    entries = sorted(p for p in directory.rglob(pattern) if p.is_file())
    digest = hashlib.sha256()
    for path in entries:
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(file_hash(path).encode())
    return digest.hexdigest(), len(entries)


@dataclass
class Asset:
    """One pinned thing."""

    name: str
    kind: str  # "file" | "tree" | "spec"
    path: str
    sha256: str
    version: str | None = None
    files: int | None = None
    bytes: int | None = None


@dataclass
class SeasonManifest:
    """Everything a season was fought with, pinned."""

    season: str
    spec_version: str = SPEC_VERSION
    frozen_at: str | None = None
    released: bool = False
    git_sha: str = "unknown"
    assets: list[Asset] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def by_name(self, name: str) -> Asset:
        for asset in self.assets:
            if asset.name == name:
                return asset
        raise ManifestError(f"no asset {name!r} in {self.season}; it has {sorted(a.name for a in self.assets)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "season": self.season,
            "frozen_at": self.frozen_at,
            "released": self.released,
            "git_sha": self.git_sha,
            "assets": [asdict(a) for a in self.assets],
            "notes": self.notes,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> SeasonManifest:
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"{path}: cannot read the manifest ({exc})") from exc
        return cls(
            season=data["season"],
            spec_version=data.get("spec_version", SPEC_VERSION),
            frozen_at=data.get("frozen_at"),
            released=bool(data.get("released", False)),
            git_sha=data.get("git_sha", "unknown"),
            assets=[Asset(**a) for a in data.get("assets", [])],
            notes=data.get("notes", {}),
        )


def _git_sha(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def freeze(
    season: str,
    *,
    timestamp: str,
    pose_library: str = "v0.1",
    release_acknowledgement: str | None = None,
) -> SeasonManifest:
    """Pin every asset a season is fought with.

    Args:
        timestamp: passed in rather than read from the clock, so a manifest is reproducible and a
            test can assert its contents (`CLAUDE.md` standing rule 3 — no invented values).
        release_acknowledgement: must equal :data:`LICENCE_ACKNOWLEDGEMENT` to mark the manifest
            released. `WORKPLAN` M6-T2 makes a human sign-off **blocking** before weights are
            published, and a flag that defaults to off is how that is enforced here.
    """
    from openroboxing.paths import (
        G1_29DOF_SIM_XML,
        OPENROBOXING_ROOT,
        POLICY_DECODER_ONNX,
        POLICY_ENCODER_ONNX,
        POSE_DIR,
        REPO_ROOT,
        display_path,
    )

    assets: list[Asset] = []

    for name, path in (
        ("policy_encoder", POLICY_ENCODER_ONNX),
        ("policy_decoder", POLICY_DECODER_ONNX),
        ("robot_model", G1_29DOF_SIM_XML),
    ):
        if not path.exists():
            raise ManifestError(f"cannot freeze {name}: {path} does not exist")
        assets.append(
            Asset(
                name=name,
                kind="file",
                path=display_path(path),
                sha256=file_hash(path),
                bytes=path.stat().st_size,
            )
        )

    library = POSE_DIR / pose_library
    if not library.is_dir():
        raise ManifestError(f"cannot freeze the pose library: {library} does not exist")
    digest, count = tree_hash(library, "*.json")
    assets.append(
        Asset(
            name="pose_library",
            kind="tree",
            path=display_path(library),
            sha256=digest,
            version=pose_library,
            files=count,
        )
    )

    # The rules are the specs. Freezing them is what makes "rules v1.0" a checkable claim rather
    # than a heading in a document.
    spec_dir = OPENROBOXING_ROOT / "spec"
    for spec in sorted(spec_dir.glob("*.md")):
        assets.append(
            Asset(
                name=f"spec/{spec.stem}",
                kind="spec",
                path=display_path(spec),
                sha256=file_hash(spec),
                bytes=spec.stat().st_size,
            )
        )

    released = release_acknowledgement == LICENCE_ACKNOWLEDGEMENT
    if release_acknowledgement is not None and not released:
        raise ManifestError(
            f"release acknowledgement {release_acknowledgement!r} is not "
            f"{LICENCE_ACKNOWLEDGEMENT!r}; see LICENSING.md for what releasing commits you to"
        )

    return SeasonManifest(
        season=season,
        frozen_at=timestamp,
        released=released,
        git_sha=_git_sha(REPO_ROOT),
        assets=assets,
        notes={
            "determinism": (
                "A manifest traces a match to the assets that produced it. It does not promise "
                "re-derivation: see CLAUDE.md invariant 6 and spec/match_record.md."
            ),
            "licensing": (
                "Not released. The terms permit it (NVIDIA Open Model License §2.2, §3 — see "
                "LICENSING.md); this manifest simply has not been marked for release."
            )
            if not released
            else (
                "Released. NVIDIA Open Model License: ship a copy of the agreement and the "
                'notice "Licensed by NVIDIA Corporation under the NVIDIA Open Model License." '
                "See LICENSING.md, signed off 2026-08-08."
            ),
        },
    )


@dataclass
class Discrepancy:
    """One asset that no longer matches what was frozen."""

    name: str
    expected: str
    actual: str | None
    reason: str


def verify(manifest: SeasonManifest, sandbox_root: Path | None = None) -> list[Discrepancy]:
    """Check what is on disk against what was frozen. Empty means the season is intact.

    Each asset is relocated by name via :func:`openroboxing.paths.locate` — the inverse of the
    naming `freeze` gives every asset through `display_path`, so an upstream asset is found under
    `GR00T_ROOT` and one of ours under `REPO_ROOT`, regardless of which machine this runs on or
    whether `OPENROBOXING_GR00T_ROOT` is set. **That is the only correct way to verify a real
    manifest**, and it is what happens by default.

    `sandbox_root` resolves every asset under one directory instead, for tests that build a
    throwaway manifest over a `tmp_path` belonging to neither root. It is deliberately not named
    `root`: aiming it at a real root — `REPO_ROOT` especially, as the plausible-looking way to "be
    explicit" — silently reinstates the bug this function was fixed for, reporting every upstream
    asset as missing because they are named relative to `GR00T_ROOT`.
    """
    from openroboxing.paths import locate

    out: list[Discrepancy] = []

    for asset in manifest.assets:
        path = (
            Path(sandbox_root) / asset.path if sandbox_root is not None else locate(asset.path)
        )
        if not path.exists():
            out.append(Discrepancy(asset.name, asset.sha256, None, "missing"))
            continue
        actual = (
            tree_hash(path, "*.json")[0] if asset.kind == "tree" else file_hash(path)
        )
        if actual != asset.sha256:
            out.append(Discrepancy(asset.name, asset.sha256, actual, "changed"))
    return out


def trace_record(record: Mapping[str, Any], manifest: SeasonManifest) -> dict[str, Any]:
    """Can this match record be traced to this manifest?

    `WORKPLAN` M6-T1's second half. A record carries `versions` naming its policy, robot model and
    pose library; the manifest carries their hashes. This reports what lines up and what does not —
    it never guesses.
    """
    versions = dict(record.get("versions", {}))
    findings: dict[str, Any] = {"match_id": record.get("match_id"), "checks": [], "traced": True}

    def check(label: str, claimed: Any, asset_name: str, compare) -> None:
        try:
            asset = manifest.by_name(asset_name)
        except ManifestError:
            findings["checks"].append({"asset": asset_name, "result": "not in manifest"})
            findings["traced"] = False
            return
        if claimed is None:
            findings["checks"].append({"asset": asset_name, "result": "record does not say"})
            findings["traced"] = False
            return
        ok = compare(claimed, asset)
        findings["checks"].append(
            {
                "asset": asset_name,
                "record": claimed,
                "manifest": asset.version or Path(asset.path).name,
                "sha256": asset.sha256[:12],
                "result": "match" if ok else "MISMATCH",
            }
        )
        findings["traced"] = findings["traced"] and ok

    check(label="robot", claimed=versions.get("robot_model"), asset_name="robot_model",
          compare=lambda claimed, asset: claimed == Path(asset.path).name)
    check(label="library", claimed=versions.get("pose_library"), asset_name="pose_library",
          compare=lambda claimed, asset: claimed == asset.version)

    sha = versions.get("openroboxing_sha")
    findings["checks"].append(
        {
            "asset": "git",
            "record": sha,
            "manifest": manifest.git_sha[:12] if manifest.git_sha != "unknown" else "unknown",
            "result": (
                "match"
                if sha and manifest.git_sha.startswith(sha)
                else "differs (not fatal: code moves, assets are what is pinned)"
            ),
        }
    )
    return findings


def format_manifest(manifest: SeasonManifest, discrepancies: Sequence[Discrepancy] = ()) -> str:
    lines = [
        f"{manifest.season}   (manifest spec {manifest.spec_version})",
        f"  frozen   : {manifest.frozen_at}",
        f"  git      : {manifest.git_sha[:12]}",
        f"  released : {'yes' if manifest.released else 'no (permitted, but not marked)'}",
        "",
        f"  {'asset':<24} {'kind':<6} {'sha256':<16} detail",
    ]
    for asset in manifest.assets:
        detail = asset.version or (f"{asset.files} files" if asset.files else f"{asset.bytes or 0:,} B")
        lines.append(f"  {asset.name:<24} {asset.kind:<6} {asset.sha256[:16]} {detail}")
    if discrepancies:
        lines.append("")
        lines.append("  DISCREPANCIES:")
        for d in discrepancies:
            lines.append(f"    {d.name:<24} {d.reason} (frozen {d.expected[:12]})")
    return "\n".join(lines)
