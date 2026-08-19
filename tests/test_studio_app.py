"""S-T1: the Pose Studio, served.

Acceptance criterion from WORKPLAN.md S-T1:
  an author with no repo access creates a valid, admitted pose record through the browser.

What is tested here is that the browser gets **the same rules the command line has** — the Studio
reimplements nothing, so a pose the page accepts is a pose a match would accept. The half that is
deliberately not here is admission: it needs a measured ``generator_error_rad``, which means asking
MotionBricks to reach the pose, and `tools/build_library.py` already owns that.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_studio_app.py -v
    .venv_mb/bin/python -m pytest tests/test_studio_app.py -v -m slow   # renders
"""

from __future__ import annotations

import json

import pytest

from openroboxing.runtime.conventions import G1
from openroboxing.runtime.obs import default_angles
from openroboxing.server.studio_app import (
    DRAFT_LIBRARY_VERSION,
    _defaults,
    _joint_limits,
    _record_from,
    _validate,
)
from openroboxing.studio.pose_record import PoseRecordError

pytest.importorskip("aiohttp")


def _stance() -> dict[str, float]:
    return dict(zip(G1.mujoco_joint_names, (float(v) for v in default_angles(G1, "mujoco"))))


def _payload(**overrides) -> dict:
    base = {"name": "studio-test", "joint_angles": _stance(), "horizon_tokens": 8}
    base.update(overrides)
    return base


# --- the joints the browser is offered -------------------------------------------------------------------
def test_every_joint_has_a_limit_read_from_the_model() -> None:
    """A browser that decided its own limits would let an author build a pose a match rejects."""
    limits = _joint_limits()
    assert set(limits) == set(G1.mujoco_joint_names)
    assert all(v["low"] < v["high"] for v in limits.values())


def test_the_default_stance_is_inside_every_limit() -> None:
    limits, defaults = _joint_limits(), _defaults()
    for name, angle in defaults.items():
        assert limits[name]["low"] <= angle <= limits[name]["high"], f"{name} starts out of range"


# --- validation is the real one ---------------------------------------------------------------------------
def test_a_full_stance_validates() -> None:
    result = _validate(_payload())
    assert result == {"ok": True, "name": "studio-test", "admission": "draft"}


def test_a_partial_pose_is_refused_with_something_readable() -> None:
    """`to_array` indexes all 29 by name; without this the failure surfaces inside the renderer."""
    result = _validate(_payload(joint_angles={"left_elbow_joint": 0.0}))
    assert result["ok"] is False
    assert "28 joint(s) missing" in result["error"]


def test_a_joint_outside_its_limit_is_refused() -> None:
    angles = _stance()
    angles["left_elbow_joint"] = 99.0
    result = _validate(_payload(joint_angles=angles))
    assert result["ok"] is False


def test_a_nonsense_horizon_is_refused() -> None:
    assert _validate(_payload(horizon_tokens=99))["ok"] is False


def test_a_saved_pose_is_never_admitted() -> None:
    """Admission needs a measured generator_error_rad. The Studio cannot mint one."""
    assert _record_from(_payload()).admission == "draft"
    assert _record_from(_payload()).is_admitted() is False


def test_a_draft_belongs_to_no_library() -> None:
    """Stamping a real version here would let an unmeasured pose claim membership of a library it
    has not been admitted to."""
    assert _record_from(_payload()).library_version == DRAFT_LIBRARY_VERSION


def test_an_unknown_joint_name_is_refused() -> None:
    angles = _stance()
    angles["left_tentacle_joint"] = 0.3
    assert _validate(_payload(joint_angles=angles))["ok"] is False


def test_building_a_record_raises_rather_than_returning_none() -> None:
    with pytest.raises(PoseRecordError, match="missing"):
        _record_from(_payload(joint_angles={}))


# --- the HTTP surface --------------------------------------------------------------------------------------
async def _client(aiohttp_client, tmp_path):
    from openroboxing.server.studio_app import build_studio_app

    return await aiohttp_client(build_studio_app(draft_dir=tmp_path))


def test_the_api_saves_a_draft_that_loads_back(tmp_path) -> None:
    """The round trip that matters: what the browser writes must be what the library reads."""
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.server.studio_app import build_studio_app
    from openroboxing.studio.pose_record import load

    async def run() -> dict:
        client = TestClient(TestServer(build_studio_app(draft_dir=tmp_path)))
        await client.start_server()
        response = await client.post("/api/save", json=_payload(name="round-trip"))
        body = await response.json()
        await client.close()
        return body

    body = asyncio.run(run())
    assert body["ok"] is True
    assert body["admission"] == "draft"

    record = load(tmp_path / "round-trip.json")
    assert record.name == "round-trip"
    assert len(record.joint_angles) == len(G1.mujoco_joint_names)
    assert record.is_admitted() is False


def test_the_api_refuses_to_save_an_invalid_pose(tmp_path) -> None:
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.server.studio_app import build_studio_app

    async def run() -> tuple[int, dict]:
        client = TestClient(TestServer(build_studio_app(draft_dir=tmp_path)))
        await client.start_server()
        response = await client.post(
            "/api/save", json=_payload(name="bad", joint_angles={"left_elbow_joint": 0.0})
        )
        body = await response.json()
        await client.close()
        return response.status, body

    status, body = asyncio.run(run())
    assert status == 400
    assert body["ok"] is False
    assert not list(tmp_path.glob("*.json")), "an invalid pose was written to disk"


def test_the_api_serves_the_joints(tmp_path) -> None:
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.server.studio_app import build_studio_app

    async def run() -> dict:
        client = TestClient(TestServer(build_studio_app(draft_dir=tmp_path)))
        await client.start_server()
        body = await (await client.get("/api/joints")).json()
        await client.close()
        return body

    body = asyncio.run(run())
    assert set(body["limits"]) == set(G1.mujoco_joint_names)
    assert set(body["defaults"]) == set(G1.mujoco_joint_names)


# --- rendering ------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_the_studio_renders_a_pose(tmp_path) -> None:
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.server.studio_app import build_studio_app

    async def run() -> dict:
        client = TestClient(TestServer(build_studio_app(draft_dir=tmp_path)))
        await client.start_server()
        body = await (
            await client.post("/api/render", json={"joint_angles": _stance()})
        ).json()
        reach = await (
            await client.post("/api/reach", json={"joint_angles": _stance(), "side": "left"})
        ).json()
        await client.close()
        return {"render": body, "reach": reach}

    result = asyncio.run(run())
    assert result["render"]["ok"] is True
    assert result["render"]["png"].startswith("data:image/png;base64,")
    assert len(result["render"]["png"]) > 1000

    reach = result["reach"]
    assert reach["ok"] is True
    # Reach is reported **relative to the pelvis**, not in world coordinates: at rest the arms hang,
    # so the left hand sits just below the pelvis and out to its left. Asserting "above the floor"
    # here would be asserting the wrong frame.
    assert reach["left_m"] > 0.0, "the left hand should be on the left"
    assert -0.2 < reach["up_m"] < 0.05, f"a hanging arm, got up_m={reach['up_m']}"
    assert abs(reach["forward_m"]) < 0.4


@pytest.mark.slow
def test_a_partial_pose_does_not_reach_the_renderer(tmp_path) -> None:
    """It must fail where the message can say what is wrong, not deep inside MuJoCo."""
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.server.studio_app import build_studio_app

    async def run() -> tuple[int, dict]:
        client = TestClient(TestServer(build_studio_app(draft_dir=tmp_path)))
        await client.start_server()
        response = await client.post("/api/render", json={"joint_angles": {}})
        body = await response.json()
        await client.close()
        return response.status, body

    status, body = asyncio.run(run())
    assert status == 400
    assert "missing" in body["error"]


def test_the_client_page_exists() -> None:
    """S-T1 is a browser tool; the page is part of the deliverable."""
    from openroboxing.paths import OPENROBOXING_ROOT

    for name in ("studio.html", "studio.js", "studio.css"):
        path = OPENROBOXING_ROOT / "client" / name
        assert path.exists(), f"{name} is missing"
        assert path.stat().st_size > 200

    page = (OPENROBOXING_ROOT / "client" / "studio.html").read_text()
    assert "draft" in page, "the page must say a saved pose is not admitted"


def test_the_studio_never_writes_into_a_versioned_library() -> None:
    """Promoting a draft into poses/v0.1 is a deliberate act with a review."""
    from openroboxing.server.studio_app import DRAFT_DIR

    assert DRAFT_DIR.name == "dev"
    assert not json.dumps(str(DRAFT_DIR)).count("v0.1")
