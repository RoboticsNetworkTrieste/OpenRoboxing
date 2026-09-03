"""The ring client's orbit camera, exercised in Node against the shipped `ring.js`.

Why a Node subprocess rather than pytest assertions
----------------------------------------------------
`client/` is deliberately vanilla — no build step, no `package.json`, no npm (`CLAUDE.md`) — so there
is no JavaScript test runner to add this to, and adding one for a camera would be a large dependency
for a small feature. Node is already the one JS runtime present on a dev box, and `tests/client/
orbit.test.mjs` is a plain script that exits non-zero on failure. This module runs it and surfaces
its output.

The test drives **the shipped file**, not a copy of its logic: it calls `Ring.prototype.frameRing`,
`_bindOrbit`, `_applyOrbit` and `resetView` directly, firing real listener callbacks. `Ring`'s
constructor needs a WebGL context, which Node has no way to give it, so the instance is built with
``Object.create`` and only the handful of fields those methods touch.

Why the temporary directory
----------------------------
Node decides ESM-vs-CommonJS by file extension and the nearest ``package.json``. `client/` has
neither a ``package.json`` nor ``.mjs`` extensions — it is loaded by a browser, which needs no such
marker — so importing `ring.js` directly makes Node parse it as CommonJS and fail on `import`. The
fixture assembles a throwaway directory that *is* marked ESM, with the real `ring.js` and the real
vendored three.js copied in, so what runs is byte-for-byte what ships.

Skipped, never silently passed, when Node is absent (`CLAUDE.md` invariant 5).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from openroboxing.paths import OPENROBOXING_ROOT

CLIENT = OPENROBOXING_ROOT / "client"
HARNESS = Path(__file__).parent / "client" / "orbit.test.mjs"

#: The camera the ring was framed with before it could orbit, for a default 4.9 m ring:
#: ``position (0, -(half + 3.4), 2.6)`` looking at ``(0, 0, 0.9)``. The harness asserts the orbit's
#: home reproduces it exactly — a silent change here would move every player's default view.
HOME_POSITION = (0.0, -5.85, 2.6)


@pytest.fixture(scope="module")
def node() -> str:
    found = shutil.which("node")
    if found is None:
        pytest.skip("node is not installed; the client's orbit camera cannot be exercised")
    return found


def test_the_orbit_camera_behaves(node: str, tmp_path_factory) -> None:
    """Home framing, drag, zoom, the clamps, and the double-click reset.

    One test rather than one per behaviour: the harness is a single Node process and splitting it
    would pay that process's startup cost per assertion. Its own output names each check, and pytest
    prints all of it on failure.
    """
    work = tmp_path_factory.mktemp("orbit")
    (work / "package.json").write_text('{"type":"module"}')
    (work / "vendor").mkdir()
    shutil.copy(CLIENT / "ring.js", work / "ring.js")
    shutil.copy(CLIENT / "vendor" / "three.module.min.js", work / "vendor")
    shutil.copy(HARNESS, work / "orbit.test.mjs")

    result = subprocess.run(
        [node, "orbit.test.mjs"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # the assertion below reports the harness's own output, which check= would hide
    )
    assert result.returncode == 0, (
        f"the client's orbit camera misbehaved:\n{result.stdout}\n{result.stderr}"
    )
    assert "ALL PASS" in result.stdout, result.stdout


def test_the_harness_actually_checks_the_home_framing() -> None:
    """Guard against the harness being weakened into a no-op.

    The one regression that would go unnoticed is the *default* view moving: everything else about
    orbiting is visible the moment you drag. So the numbers the harness pins are asserted here too,
    from the Python side, where they cannot be edited out of the JavaScript without this failing.
    """
    source = HARNESS.read_text()
    assert "home framing reproduces the old fixed camera" in source
    assert "3.4" in source and "2.6" in source, "the home framing's own constants are gone"
    assert "double-click restores home exactly" in source
