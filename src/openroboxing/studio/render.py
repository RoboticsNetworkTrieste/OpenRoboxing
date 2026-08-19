"""Render poses to images, so a human can judge them (M2-T5).

A pose library is a set of aesthetic judgments — whether a configuration reads as a jab is not
something joint angles answer. This module exists so those judgments can be made by looking.

Rendering uses the **deploy** model, not the generator's: this is a picture of the physical robot.
See ``spec/upstream_notes.md`` §"The generator and the policy are different G1 revisions".

Conventions
-----------
- Images are ``(height, width, 3)`` ``uint8`` RGB.
- Rendering is offscreen via EGL and needs a GPU, but no display.

``MUJOCO_GL`` is set **when this module is imported**, and that timing is the whole trick: MuJoCo
binds its GL platform library the first time ``mujoco`` is imported, and setting the variable
afterwards does nothing but produce "an OpenGL platform library has not been loaded". Importing
``openroboxing.studio.render`` before ``mujoco`` is therefore sufficient, and is why the assignment
sits at module scope rather than inside the render call. An existing value is never overridden, so a
developer at a real display keeps their own backend.
"""

from __future__ import annotations

import os

# Before any `import mujoco` in the process — see the module docstring.
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from openroboxing.paths import G1_29DOF_SCENE_XML
from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.spec.constants import QPOS_DIM
from openroboxing.studio.pose_record import PoseRecord

#: Where the camera stands. A three-quarter view: a straight-on shot hides the reach of a punch, and
#: a pure side view hides which arm threw it.
DEFAULT_AZIMUTH = 135.0
DEFAULT_ELEVATION = -12.0
DEFAULT_DISTANCE = 3.2

#: The pelvis height a rendered pose stands at. Poses carry no placement, so one is supplied.
STANDING_HEIGHT = 0.793


class RenderError(RuntimeError):
    """A pose could not be rendered. Never recovered from silently."""


def _model_and_data():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RenderError(f"mujoco is required to render poses: {exc}") from exc

    model = mujoco.MjModel.from_xml_path(str(G1_29DOF_SCENE_XML))
    return mujoco, model, mujoco.MjData(model)


def pose_qpos(
    pose: PoseRecord, *, height: float = STANDING_HEIGHT, conventions: G1Conventions = G1
) -> np.ndarray:
    """A ``(36,)`` qpos standing the pose upright at the origin."""
    qpos = np.zeros(QPOS_DIM)
    qpos[2] = height
    qpos[3] = 1.0
    qpos[7:] = pose.to_array(conventions)
    return qpos


def render_qpos(
    qpos: np.ndarray,
    *,
    width: int = 480,
    height: int = 640,
    azimuth: float = DEFAULT_AZIMUTH,
    elevation: float = DEFAULT_ELEVATION,
    distance: float = DEFAULT_DISTANCE,
) -> np.ndarray:
    """Render one qpos frame. Returns ``(height, width, 3)`` uint8 RGB."""
    arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if arr.shape != (QPOS_DIM,):
        raise RenderError(f"expected a ({QPOS_DIM},) qpos, got {arr.shape}")

    mujoco, model, data = _model_and_data()
    data.qpos[:] = arr
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
    camera.lookat[:] = (arr[0], arr[1], arr[2] * 0.75)

    # MuJoCo sizes its offscreen framebuffer from the model, defaulting to 640x480, and refuses a
    # larger request. Raising it here beats editing the upstream scene XML, which is read-only.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    try:
        renderer = mujoco.Renderer(model, height, width)
    except Exception as exc:
        raise RenderError(
            f"could not open an offscreen renderer (MUJOCO_GL="
            f"{os.environ.get('MUJOCO_GL')!r}): {exc}. If this says no OpenGL platform library "
            "was loaded, something imported mujoco before openroboxing.studio.render"
        ) from exc
    try:
        renderer.update_scene(data, camera=camera)
        return renderer.render()
    finally:
        renderer.close()


def render_pose(pose: PoseRecord, **kwargs) -> np.ndarray:
    """Render a pose record standing upright."""
    return render_qpos(pose_qpos(pose), **kwargs)


def contact_sheet(
    images: list[np.ndarray], labels: list[str], *, columns: int = 4, pad: int = 6
) -> np.ndarray:
    """Tile rendered poses into one image, each with its label burned in.

    Labels are drawn as a coarse bitmap rather than with a font library, so the sheet has no
    dependency beyond numpy and is readable at a glance — which is all it has to be.
    """
    if len(images) != len(labels):
        raise RenderError(f"{len(images)} images but {len(labels)} labels")
    if not images:
        raise RenderError("nothing to tile")
    if columns < 1:
        raise RenderError(f"columns must be at least 1, got {columns}")

    height, width = images[0].shape[:2]
    for index, image in enumerate(images):
        if image.shape[:2] != (height, width):
            raise RenderError(
                f"image {index} is {image.shape[:2]}, expected {(height, width)}; tiles must match"
            )

    rows = int(np.ceil(len(images) / columns))
    label_height = 14
    tile_h, tile_w = height + label_height + pad, width + pad
    sheet = np.full((rows * tile_h + pad, columns * tile_w + pad, 3), 24, dtype=np.uint8)

    for index, (image, label) in enumerate(zip(images, labels)):
        row, column = divmod(index, columns)
        top = pad + row * tile_h
        left = pad + column * tile_w
        sheet[top : top + height, left : left + width] = image
        _draw_label(sheet, label, top + height + 3, left, width)
    return sheet


#: A 5x7 bitmap font, enough for the labels a contact sheet carries.
_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00110", "01000", "10000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "#": ("01010", "01010", "11111", "01010", "11111", "01010", "01010"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    " ": ("00000",) * 7,
    "?": ("01110", "10001", "00001", "00110", "00100", "00000", "00100"),
    "a": ('00000', '01110', '00001', '01111', '10001', '10011', '01101'),
    "b": ('10000', '10000', '11110', '10001', '10001', '10001', '11110'),
    "c": ('00000', '00000', '01110', '10001', '10000', '10001', '01110'),
    "d": ('00001', '00001', '01111', '10001', '10001', '10001', '01111'),
    "e": ('00000', '00000', '01110', '10001', '11111', '10000', '01110'),
    "f": ('00110', '01001', '01000', '11100', '01000', '01000', '01000'),
    "g": ('00000', '01111', '10001', '10001', '01111', '00001', '01110'),
    "h": ('10000', '10000', '10110', '11001', '10001', '10001', '10001'),
    "i": ('00100', '00000', '01100', '00100', '00100', '00100', '01110'),
    "j": ('00010', '00000', '00110', '00010', '00010', '10010', '01100'),
    "k": ('10000', '10000', '10010', '10100', '11000', '10100', '10010'),
    "l": ('01100', '00100', '00100', '00100', '00100', '00100', '01110'),
    "m": ('00000', '00000', '11010', '10101', '10101', '10101', '10101'),
    "n": ('00000', '00000', '10110', '11001', '10001', '10001', '10001'),
    "o": ('00000', '00000', '01110', '10001', '10001', '10001', '01110'),
    "p": ('00000', '11110', '10001', '10001', '11110', '10000', '10000'),
    "q": ('00000', '01111', '10001', '10001', '01111', '00001', '00001'),
    "r": ('00000', '00000', '10110', '11001', '10000', '10000', '10000'),
    "s": ('00000', '00000', '01111', '10000', '01110', '00001', '11110'),
    "t": ('01000', '01000', '11100', '01000', '01000', '01001', '00110'),
    "u": ('00000', '00000', '10001', '10001', '10001', '10011', '01101'),
    "v": ('00000', '00000', '10001', '10001', '10001', '01010', '00100'),
    "w": ('00000', '00000', '10001', '10001', '10101', '10101', '01010'),
    "x": ('00000', '00000', '10001', '01010', '00100', '01010', '10001'),
    "y": ('00000', '10001', '10001', '10001', '01111', '00001', '01110'),
    "z": ('00000', '00000', '11111', '00010', '00100', '01000', '11111'),
}


def _draw_label(sheet: np.ndarray, text: str, top: int, left: int, max_width: int) -> None:
    """Burn a label into the sheet. Unknown characters render as '?', never as a crash."""
    x = left
    for character in text[: max_width // 6]:
        glyph = _GLYPHS.get(character.lower(), _GLYPHS["?"])
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1" and top + row < sheet.shape[0] and x + column < sheet.shape[1]:
                    sheet[top + row, x + column] = 235
        x += 6


def save_png(image: np.ndarray, path: Path) -> Path:
    """Write an RGB array as a PNG, without adding an image-library dependency."""
    import struct
    import zlib

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RenderError(f"expected (H, W, 3) RGB, got {array.shape}")

    height, width = array.shape[:2]
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    return path
