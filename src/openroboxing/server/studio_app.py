"""The Pose Studio, served (S-T1).

Acceptance criterion from WORKPLAN.md S-T1:
  an author with no repo access creates a valid, admitted pose record through the browser.

A thin HTTP surface over the Studio that already exists — `studio/pose_record.py`,
`studio/render.py`, `studio/pose_ik.py`, `studio/rehearsal.py`. Nothing is reimplemented here; the
browser gets the same functions the command line has, which is what stops the two from disagreeing
about what a valid pose is.

Two halves, and the split is honest
-----------------------------------
**Authoring** — edit joint angles, see the pose, validate it — is instant and needs no GPU beyond
rendering. **Admission** is not: `spec/pose_record.md` requires a measured ``generator_error_rad``,
which means actually asking MotionBricks to reach the pose. That takes seconds and loads a
checkpoint, so it is a separate, explicit request and the UI says what it is doing.

A pose saved without admission is a **draft**, and drafts are exactly what a match refuses to use
(``Loadout.validate``). That is the intended path, not a limitation.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from openroboxing.paths import OPENROBOXING_ROOT, POSE_DIR

CLIENT_DIR = OPENROBOXING_ROOT / "client"

#: Where the browser saves. Drafts land here, not in a versioned library: promoting a draft into
#: `poses/v0.1/` is a deliberate act with a review, not a side effect of somebody clicking save.
DRAFT_DIR = POSE_DIR / "dev"

#: Rendered preview size. Small enough to feel live while a slider is dragged.
PREVIEW_WIDTH = 360
PREVIEW_HEIGHT = 480


class StudioError(RuntimeError):
    """The Studio could not serve a request. Never recovered from silently."""


def _render_png(payload: dict[str, Any]) -> bytes:
    """Render a pose to PNG bytes."""
    from PIL import Image

    from openroboxing.studio.render import render_pose

    frame = render_pose(_record_from(payload), width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def _joint_limits() -> dict[str, dict[str, float]]:
    """Every joint's range, read from the model so the browser cannot offer an illegal pose."""
    import mujoco

    from openroboxing.paths import G1_29DOF_SIM_XML
    from openroboxing.runtime.conventions import G1

    model = mujoco.MjModel.from_xml_path(str(G1_29DOF_SIM_XML))
    limits: dict[str, dict[str, float]] = {}
    for name in G1.mujoco_joint_names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[joint]
        limits[name] = {"low": float(low), "high": float(high)}
    return limits


def _defaults() -> dict[str, float]:
    from openroboxing.runtime.conventions import G1
    from openroboxing.runtime.obs import default_angles

    return dict(zip(G1.mujoco_joint_names, (float(v) for v in default_angles(G1, "mujoco"))))


#: Drafts carry this instead of a library version. A draft belongs to no library — promoting one
#: into `poses/v0.1/` is a deliberate act, and stamping it with a real version here would let an
#: unmeasured pose claim membership of a library it has not been admitted to.
DRAFT_LIBRARY_VERSION = "dev"


def _record_from(payload: dict[str, Any]):
    """Build a :class:`PoseRecord` from a browser payload, or raise with something readable."""
    from openroboxing.runtime.conventions import G1
    from openroboxing.studio.pose_record import PoseRecord, PoseRecordError

    angles = {k: float(v) for k, v in payload.get("joint_angles", {}).items()}
    missing = [n for n in G1.mujoco_joint_names if n not in angles]
    if missing:
        # A partial pose is not a pose: `to_array` indexes all 29 by name and would KeyError deep
        # inside the renderer instead of here, where the message can say what is wrong.
        raise PoseRecordError(f"{len(missing)} joint(s) missing, starting with {missing[:3]}")

    return PoseRecord(
        name=str(payload.get("name", "untitled")),
        joint_angles=angles,
        horizon_tokens=int(payload.get("horizon_tokens", 8)),
        library_version=DRAFT_LIBRARY_VERSION,
        adjustment_envelope={
            k: float(v) for k, v in payload.get("adjustment_envelope", {}).items()
        },
    )


def _validate(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a candidate pose without measuring it. Instant; no generator involved."""
    from openroboxing.studio.pose_record import PoseRecordError, validate

    try:
        record = _record_from(payload)
        validate(record)
    except PoseRecordError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "name": record.name, "admission": record.admission}


def build_studio_app(client_dir: Path = CLIENT_DIR, draft_dir: Path = DRAFT_DIR):
    """An aiohttp application serving the Pose Studio."""
    from aiohttp import web

    app = web.Application()

    async def index(request):
        return web.FileResponse(client_dir / "studio.html")

    async def joints(request):
        """Names, limits and the default stance. Everything the sliders need."""
        return web.json_response({"limits": _joint_limits(), "defaults": _defaults()})

    async def render(request):
        try:
            png = _render_png(await request.json())
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response(
            {"ok": True, "png": "data:image/png;base64," + base64.b64encode(png).decode()}
        )

    async def check(request):
        return web.json_response(_validate(await request.json()))

    async def reach(request):
        """Where the hand ends up. The one number that tells an author if a strike can land."""
        from openroboxing.studio.pose_ik import hand_position

        payload = await request.json()
        angles = {k: float(v) for k, v in payload.get("joint_angles", {}).items()}
        side = payload.get("side", "left")
        try:
            position = hand_position(angles, side)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response(
            {
                "ok": True,
                "side": side,
                "forward_m": round(float(position[0]), 4),
                "left_m": round(float(position[1]), 4),
                "up_m": round(float(position[2]), 4),
            }
        )

    async def save(request):
        """Write a draft. Admission is a separate act; this never marks a pose admitted."""
        payload = await request.json()
        result = _validate(payload)
        if not result["ok"]:
            return web.json_response(result, status=400)

        from openroboxing.studio.pose_record import save as save_record

        record = _record_from(payload)
        draft_dir.mkdir(parents=True, exist_ok=True)
        path = draft_dir / f"{record.name}.json"
        save_record(record, path)
        return web.json_response(
            {
                "ok": True,
                "path": str(path),
                "admission": record.admission,
                "note": (
                    "Saved as a draft. A match refuses drafts (Loadout.validate); admission needs a "
                    "measured generator_error_rad — see spec/pose_record.md."
                ),
            }
        )

    app.router.add_get("/", index)
    app.router.add_get("/api/joints", joints)
    app.router.add_post("/api/render", render)
    app.router.add_post("/api/check", check)
    app.router.add_post("/api/reach", reach)
    app.router.add_post("/api/save", save)
    app.router.add_static("/static/", path=client_dir, name="static")
    return app
