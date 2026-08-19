"""M1-T6: the single-fighter runtime.

The full acceptance criterion is the 30 s run in `tools/run_single.py`; building the generator costs
~3 s and loading it into pytest on every run is not worth it, so the heavy end-to-end check is marked
`slow` and deselected by default.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_world.py -v            # fast checks
    .venv_mb/bin/python -m pytest tests/test_world.py -v -m slow    # end to end
    .venv_mb/bin/python -m openroboxing.tools.run_single --style walk_boxing --seconds 30
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import G1_29DOF_SCENE_XML
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.world import LOOKAHEAD_TICKS, WorldError
from openroboxing.spec.constants import HISTORY_LEN, NUM_JOINTS, TICK_DT


def test_lookahead_matches_the_encoder_stride() -> None:
    """The world must pull the generator far enough ahead to feed the encoder."""
    from openroboxing.runtime.bridge import ENCODER_FRAME_STRIDE

    assert LOOKAHEAD_TICKS == (HISTORY_LEN - 1) * ENCODER_FRAME_STRIDE == 45


def test_scene_is_the_29dof_g1() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(G1_29DOF_SCENE_XML))
    assert model.nq == 7 + NUM_JOINTS
    assert model.nv == 6 + NUM_JOINTS
    assert model.nu == NUM_JOINTS


def test_actuators_map_to_joints_by_name() -> None:
    """Every actuated joint must be reachable, and the mapping derived by name (invariant 4)."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(G1_29DOF_SCENE_XML))
    joint_to_actuator = {}
    for a in range(model.nu):
        joint_id = int(model.actuator_trnid[a, 0])
        joint_to_actuator[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)] = a
    assert set(joint_to_actuator) == set(G1.mujoco_joint_names)


def test_substep_count_divides_the_control_tick() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(G1_29DOF_SCENE_XML))
    substeps = TICK_DT / model.opt.timestep
    assert (
        abs(substeps - round(substeps)) < 1e-9
    ), "physics timestep does not divide the control tick"


def test_missing_scene_raises() -> None:
    pytest.importorskip("mujoco")
    from pathlib import Path

    from openroboxing.runtime.world import SingleFighterWorld

    with pytest.raises(WorldError, match="scene not found"):
        SingleFighterWorld(scene_xml=Path("/nonexistent/scene.xml"))


@pytest.mark.slow
def test_thirty_second_run_does_not_fall() -> None:
    """M1-T6 acceptance: 30 s of walk_boxing under physics without falling."""
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    from openroboxing.runtime.world import SingleFighterWorld

    world = SingleFighterWorld(style="walk_boxing", seed=1234)
    world.reset(seed=1234)
    log = world.run(seconds=30.0)

    assert not log.fell, f"fighter fell at tick {log.fell_at_tick}"
    assert len(log.tick) == 30 * 50
    assert min(log.root_height) > 0.5, "root dipped further than a walking gait should"

    per_joint = log.per_joint_error()
    assert per_joint.shape == (NUM_JOINTS,)
    assert np.isfinite(per_joint).all()
    # A stationary or diverged run would show either no travel or absurd error.
    travelled = float(np.linalg.norm(log.root_position[-1][:2] - log.root_position[0][:2]))
    assert travelled > 1.0, f"fighter barely moved ({travelled:.2f} m)"
    assert per_joint.mean() < 0.5, f"mean tracking error {per_joint.mean():.3f} rad is too large"
