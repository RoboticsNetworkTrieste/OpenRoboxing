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

    Stands in for `MotionBricksGenerator._install_pose_override`: the generator's `_arm` runs
    skeleton FK to produce the tensors, and here they are simply handed in. The generator's own
    version is exercised by the slow end-to-end tests.
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
