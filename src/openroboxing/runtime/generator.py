"""MotionBricks wrapper: intents → 36-dim MuJoCo qpos at 30 Hz.

Upstream is unmodified — genuinely, now that patch P0 is installed at runtime (`CLAUDE.md`
invariant 3). The only things this module adds are a headless, scripted way to drive the generator
and that runtime patch.

Why the import shim
-------------------
``motionbricks...demo.utils`` imports ``demo.controllers`` at module scope, which imports ``pynput``,
which opens an X connection **at import time**. That makes the whole module unimportable on a
headless machine even when no keyboard is wanted. :func:`_stub_pynput_if_unavailable` installs a stub
that satisfies the import and **raises on any actual use**, so a keyboard read can never silently
return "no keys pressed" — it fails loudly, per invariant 5.

We never construct a controller: :class:`_HeadlessDemo` overrides ``_initialize_controller``, and
control signals are built here instead.

Conventions
-----------
- **Output**: ``(36,)`` MuJoCo ``qpos`` per frame — 3 root position, 4 root quaternion ``wxyz``,
  29 joints in **MuJoCo order** — at ``GENERATOR_HZ``.
- **Directions** are unit vectors in the horizontal plane, ``[cos, sin, 0]``, as the upstream
  controller builds them.
- **Placement** is ``(x, y)`` in **MuJoCo world coordinates** plus a heading in radians, and it is
  free: the spring model reads ``specific_target_positions[:, -1, [1, 0]]`` directly, gated by
  ``has_specific_target`` (``full_agent.py:244-247``). Measured — a fighter commanded to ``(3, 2)``
  ends 0.12 m away, and 2.3 m from where it goes uncommanded.

  The ``[1, 0]`` index pair is the MuJoCo→motion change of basis (MuJoCo ``y`` is motion ``x``,
  MuJoCo ``x`` is motion ``z``), not an arbitrary quirk. Positions must be full 3-vectors:
  canonicalisation subtracts a 3-D origin and rotates by a 3x3 (``:613-615``). The height component
  is carried through and never read.

  ``_override_target_transforms`` is a *second*, alternative override gated on
  ``BYPASS_SPRING_MODEL``, which ``navigation_demo`` never enables. It is not used here, and it
  would need ``specific_target_*`` shaped over ``NUM_FRAMES_PER_TOKEN`` rather than one frame.
- **Authored poses** ride on :attr:`GeneratorIntent.pose`. Patch P0 is installed here at runtime
  rather than patched into upstream — :func:`_wrap_target_transforms` adds its call site and
  :func:`_apply_target_pose_override` is its body — so the GR00T-WBC submodule stays pristine and
  can track NVlabs ``main``. Everything that decides *what* the tensors are lives in
  :meth:`MotionBricksGenerator._install_pose_override`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np

from openroboxing.paths import MOTIONBRICKS_ROOT
from openroboxing.spec.constants import MAX_TOKENS, MIN_TOKENS, NUM_TIME_TOKENS, QPOS_DIM


class GeneratorError(RuntimeError):
    """The generator could not be built or driven. Never recovered from silently."""


def narrow_allowed_tokens(
    clip_mask: list[int], horizon_tokens: int | None, *, style: str = "?"
) -> list[int]:
    """Narrow a clip's token-length mask to a single requested horizon.

    Upstream stores an :data:`NUM_TIME_TOKENS`-long mask over token counts ``MIN_TOKENS..MAX_TOKENS``
    (``clips.py``), so a horizon of *n* tokens is the mask with only index ``n - MIN_TOKENS`` set.

    Asking for a length the clip does not permit **raises** rather than falling back to the clip's
    own range: a move that runs for a different length than the commit promised is a scoring bug, and
    silently substituting one is exactly the kind of fallback `CLAUDE.md` invariant 5 forbids.
    """
    allowed = list(clip_mask)
    if len(allowed) != NUM_TIME_TOKENS:
        raise GeneratorError(
            f"clip {style!r} declares {len(allowed)} token slots, expected "
            f"{NUM_TIME_TOKENS} ({MIN_TOKENS}..{MAX_TOKENS})"
        )
    if horizon_tokens is None:
        return allowed

    if not MIN_TOKENS <= horizon_tokens <= MAX_TOKENS:
        raise GeneratorError(
            f"horizon_tokens {horizon_tokens} outside [{MIN_TOKENS}, {MAX_TOKENS}]"
        )
    index = horizon_tokens - MIN_TOKENS
    if not allowed[index]:
        permitted = [MIN_TOKENS + i for i, ok in enumerate(allowed) if ok]
        raise GeneratorError(
            f"clip {style!r} does not allow a {horizon_tokens}-token move; it permits {permitted}"
        )
    return [1 if i == index else 0 for i in range(NUM_TIME_TOKENS)]


def _stub_pynput_if_unavailable() -> bool:
    """Make ``import pynput`` succeed headlessly, without making a keyboard silently work.

    Returns True if a stub was installed. Any attribute access on the stub raises.
    """
    try:  # a real pynput (with a display) is always preferred
        import pynput  # noqa: F401

        return False
    except Exception:
        pass

    def _reject(name: str):
        # Dunders must resolve normally: `inspect` probes __file__, __name__, __loader__ and friends
        # while walking frames, and raising there breaks unrelated machinery (torch does this during
        # custom-op registration). Only real API access is refused.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise GeneratorError(
            f"pynput.keyboard.{name} was used, but pynput is unavailable on this machine "
            "(no X display). The headless generator must not read a keyboard; build control "
            "signals explicitly instead."
        )

    module = types.ModuleType("pynput")
    keyboard = types.ModuleType("pynput.keyboard")
    keyboard.__getattr__ = _reject  # type: ignore[attr-defined]
    module.keyboard = keyboard  # type: ignore[attr-defined]
    sys.modules.setdefault("pynput", module)
    sys.modules.setdefault("pynput.keyboard", keyboard)
    return True


@contextlib.contextmanager
def _chdir(path: Path):
    """Temporarily change the working directory.

    Upstream's checkpoint config stores **relative** paths (``folder:
    out/motionbricks_root/version_1/skeleton``), so the agent can only be constructed with the
    working directory at the motionbricks package root. Scoped to construction and always restored,
    including on error, so it cannot leak into a caller's process.
    """
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _apply_target_pose_override(input: dict) -> dict:
    """Patch P0's body, installed at runtime rather than patched into upstream.

    Deliberately dumb: it swaps two tensors and validates their shapes. It cannot *compute* them,
    because the authored pose has to be re-rooted onto the placement the spring model chose, and
    that placement only exists inside ``generate_new_frames``. Everything that decides *what* the
    tensors are lives in :meth:`MotionBricksGenerator._install_pose_override`.

    Optional input keys, both ignored when absent:
        specific_target_joint_positions: [batch, NUM_FRAMES_PER_TOKEN, num_joints, 3]
        specific_target_joint_rotations: [batch, NUM_FRAMES_PER_TOKEN, num_joints, 3, 3]

    ``has_specific_target_pose`` ([batch, 1], int) blends per batch element when given; without it
    an override that is present applies fully.

    Applying this twice is a no-op the second time — the tensors it writes are the tensors it would
    write again. That matters because a checkout carrying the original patch still calls upstream's
    own copy after this one has run.
    """
    positions = input.get("specific_target_joint_positions", None)
    rotations = input.get("specific_target_joint_rotations", None)
    if positions is None and rotations is None:
        return input

    mask = input.get("has_specific_target_pose", None)

    if positions is not None:
        current = input["target_global_joint_positions"]
        if positions.shape != current.shape:
            raise ValueError(
                f"specific_target_joint_positions has shape {tuple(positions.shape)}, "
                f"expected {tuple(current.shape)}"
            )
        if mask is None:
            input["target_global_joint_positions"] = positions
        else:
            blend = mask.view([-1, 1, 1, 1]).float()
            input["target_global_joint_positions"] = positions * blend + current * (1.0 - blend)

    if rotations is not None:
        current = input["target_global_joint_rotations"]
        if rotations.shape != current.shape:
            raise ValueError(
                f"specific_target_joint_rotations has shape {tuple(rotations.shape)}, "
                f"expected {tuple(current.shape)}"
            )
        if mask is None:
            input["target_global_joint_rotations"] = rotations
        else:
            blend = mask.view([-1, 1, 1, 1, 1]).float()
            input["target_global_joint_rotations"] = rotations * blend + current * (1.0 - blend)

    return input


def _wrap_target_transforms(agent, arm) -> None:
    """Install P0's call site on an agent instance.

    Upstream's ``generate_new_frames`` does::

        input['target_global_joint_positions'], input['target_global_joint_rotations'], \\
            input['target_global_root_positions'] = self._generate_target_joint_transforms(input)

    The original patch added a call to the override on the next line. With a pristine upstream there
    is no next line, so the wrapper does the assignment itself, runs ``arm`` (which writes the
    ``specific_target_joint_*`` keys when a pose is armed) and then the override, and finally returns
    the three values re-read from ``input``. Upstream's own assignment then writes back what is
    already there — which is what makes the two paths equivalent.

    ``agent`` is the ``full_navigation_agent`` instance; ``arm`` is called with ``input`` after the
    target exists and before the override runs.
    """
    original = agent._generate_target_joint_transforms

    def _with_authored_pose(input: dict):
        positions, rotations, root_positions = original(input)
        input["target_global_joint_positions"] = positions
        input["target_global_joint_rotations"] = rotations
        input["target_global_root_positions"] = root_positions

        arm(input)
        _apply_target_pose_override(input)

        return (
            input["target_global_joint_positions"],
            input["target_global_joint_rotations"],
            input["target_global_root_positions"],
        )

    agent._generate_target_joint_transforms = _with_authored_pose


@dataclass
class GeneratorConfig:
    """How the generator is built. Mirrors the upstream demo's defaults."""

    clips: str = "G1"
    planner: str = "default"
    generate_dt: float = 2.0
    random_seed: int = 1234
    source_root_realignment: int = 1
    target_root_realignment: int = 1
    force_canonicalization: int = 1
    pre_filter_qpos: int = 1
    skip_ending_target_cond: int = 0
    random_speed_scale: int = 0
    speed_scale: tuple[float, float] = (0.8, 1.2)
    use_qpos: int = 1
    lookat_movement_direction: int = 0
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorIntent:
    """One tick of scripted control, the headless equivalent of a keypress.

    Attributes:
        style: clip name, e.g. ``"walk_boxing"``. Resolved to the one-hot ``mode``.
        movement_angle: radians; direction of travel in the horizontal plane.
        facing_angle: radians; direction the fighter faces.
        target_position: optional ``(x, z)`` placement — see ``spec/intent.md`` step 3.
        target_heading: optional heading in radians to accompany ``target_position``.
        pose: optional authored key pose the generation should reach. ``None`` leaves the target
            pose to the clip library, which is upstream's behaviour.
        horizon_tokens: how long the generated move may run, in tokens. ``None`` leaves the clip's
            own allowance untouched, which is what the runtime asks for since ``spec/intent.md``
            2.0 — the model picks its own length by argmax.
    """

    style: str = "walk_boxing"
    movement_angle: float = 0.0
    facing_angle: float = 0.0
    target_position: tuple[float, float] | None = None
    target_heading: float | None = None
    pose: object | None = None  # PoseRecord; typed loosely to keep studio out of the runtime import
    horizon_tokens: int | None = None


class MotionBricksGenerator:
    """Headless MotionBricks, driven by :class:`GeneratorIntent` instead of a keyboard."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self._stubbed_pynput = _stub_pynput_if_unavailable()

        if str(MOTIONBRICKS_ROOT) not in sys.path:
            sys.path.insert(0, str(MOTIONBRICKS_ROOT))

        try:
            from motionbricks.motion_backbone.demo.clips import clip_holder_G1
            from motionbricks.motion_backbone.demo.utils import navigation_demo
            import torch
        except ImportError as exc:  # pragma: no cover - environment problem
            raise GeneratorError(
                f"cannot import MotionBricks from {MOTIONBRICKS_ROOT}: {exc}"
            ) from exc

        self._torch = torch
        self._clip_holder_class = clip_holder_G1
        self.clip_names: tuple[str, ...] = tuple(clip_holder_G1.CLIPS.keys())

        class _HeadlessDemo(navigation_demo):
            """navigation_demo without the keyboard controller."""

            def _initialize_controller(self) -> None:  # noqa: D401 - upstream hook
                self.controller = None

        args = self._build_args()
        try:
            with _chdir(MOTIONBRICKS_ROOT):
                self._demo = _HeadlessDemo(args)
        except Exception as exc:
            raise GeneratorError(f"failed to build the MotionBricks agent: {exc}") from exc

        self.agent = self._demo.full_agent
        self.mj_model = self._demo.mj_model
        self.mj_data = self._demo.mj_data

        # Its own stream, never the global one. Upstream falls back to `t.randint` drawn from the
        # *global* torch RNG when no `random_seed` is given (full_agent.py:220, :326), so two
        # generators in one process sample the clip library from a shared sequence and each one's
        # output depends on how often the other has been driven. Measured: red's plan changed when
        # blue was generated first. A per-generator stream makes isolation structural.
        self._rng = np.random.default_rng(self.config.random_seed)
        self._armed_pose: object | None = None
        self._skeleton_fk = None
        self._install_pose_override()

    def _install_pose_override(self) -> None:
        """Install patch P0 on the agent, and feed it its tensors.

        P0 is no longer a diff inside ``full_agent.py``: the submodule tracks NVlabs ``main`` and is
        pristine, so both the override and its call site are installed here at runtime — see
        :func:`_wrap_target_transforms` and :func:`_apply_target_pose_override`, and
        ``spec/upstream_patches.md``.

        P0 itself cannot compute its tensors, because the authored pose has to be re-rooted onto the
        placement the spring model chose and that placement only exists inside
        ``generate_new_frames``. So when a pose is armed we read the freshly generated target out of
        the input dict, re-root the pose onto it, and write the result back under the keys P0 reads.

        Every OpenRoboxing decision — heading-only placement, which model to run FK on — therefore
        lives on this side of the boundary, and upstream stays untouched.
        """

        def _arm(input: dict) -> None:
            if self._armed_pose is None:
                return
            if self._skeleton_fk is None:
                from openroboxing.studio.skeleton_fk import skeleton_fk

                self._skeleton_fk = skeleton_fk()
            positions, rotations = self._skeleton_fk.target_transforms(
                self._armed_pose,
                input["target_global_joint_positions"],
                input["target_global_joint_rotations"],
            )
            input["specific_target_joint_positions"] = positions
            input["specific_target_joint_rotations"] = rotations

        _wrap_target_transforms(self.agent, _arm)

    def _build_args(self) -> SimpleNamespace:
        """Assemble the argparse-shaped object upstream expects.

        Path attributes are deliberately omitted so upstream's ``_parse_args`` fills in absolute
        defaults relative to the motionbricks package, rather than inheriting our working directory.
        """
        cfg = self.config
        args = SimpleNamespace(
            clips=cfg.clips,
            planner=cfg.planner,
            EXP=cfg.planner,
            generate_dt=cfg.generate_dt,
            random_seed=cfg.random_seed,
            source_root_realignment=cfg.source_root_realignment,
            target_root_realignment=cfg.target_root_realignment,
            force_canonicalization=cfg.force_canonicalization,
            pre_filter_qpos=cfg.pre_filter_qpos,
            skip_ending_target_cond=cfg.skip_ending_target_cond,
            random_speed_scale=cfg.random_speed_scale,
            speed_scale=list(cfg.speed_scale),
            use_qpos=cfg.use_qpos,
            lookat_movement_direction=cfg.lookat_movement_direction,
            reprocess_clips=0,
            has_viewer=0,
            allowed_mode=None,
            recording_dir=None,
            return_model_configs=True,
            return_dataloader=True,
            num_runs=1,
            max_steps=10_000,
            controller="random",
        )
        for key, value in cfg.extra.items():
            setattr(args, key, value)
        return args

    # -- driving --------------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Reset the agent and reseed. Call before each run so a run is reproducible.

        Reseeds this generator's own stream. The global seeds are set too, because upstream's model
        may draw from them, but nothing this generator controls depends on them any more.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            np.random.seed(seed)
            self._torch.manual_seed(seed)
        self.agent.reset()

    def mode_index(self, style: str) -> int:
        if style not in self.clip_names:
            raise GeneratorError(
                f"unknown style {style!r}; available: {', '.join(sorted(self.clip_names))}"
            )
        return self.clip_names.index(style)

    def _allowed_tokens(self, intent: GeneratorIntent) -> list[int]:
        clip = self._clip_holder_class.CLIPS[intent.style]["allowed_pred_num_tokens"]
        return narrow_allowed_tokens(clip, intent.horizon_tokens, style=intent.style)

    def control_signals(self, intent: GeneratorIntent, context_qpos: np.ndarray) -> dict:
        """Build the input dict ``generate_new_frames`` expects.

        Args:
            context_qpos: ``(K, 36)`` recent qpos frames giving the agent its context.
        """
        t = self._torch
        mode = t.tensor([self.mode_index(intent.style)]).view([1, -1])

        signals = {
            "movement_direction": t.tensor(
                [np.cos(intent.movement_angle), np.sin(intent.movement_angle), 0.0]
            )
            .float()
            .view([1, -1]),
            "facing_direction": t.tensor(
                [np.cos(intent.facing_angle), np.sin(intent.facing_angle), 0.0]
            )
            .float()
            .view([1, -1]),
            "mode": mode,
            "allowed_pred_num_tokens": t.tensor(
                self._allowed_tokens(intent)
            ).view([1, -1]),
            "context_mujoco_qpos": t.from_numpy(np.asarray(context_qpos, dtype=np.float32)).view(
                [1, -1, QPOS_DIM]
            ),
            # Always explicit. Left absent, upstream draws it from the global RNG — see __init__.
            "random_seed": t.tensor([int(self._rng.integers(0, 10_000))]).int(),
        }

        if intent.target_position is not None:
            if intent.target_heading is None:
                raise GeneratorError(
                    "target_position requires target_heading; upstream's "
                    "_override_target_transforms returns early unless both are present "
                    "(full_agent.py:301)"
                )
            x, y = intent.target_position
            # A full 3-vector in MuJoCo world coordinates, not a ground-plane pair: canonicalisation
            # subtracts `first_frame_position` (3-D) and rotates by a 3x3 (full_agent.py:558-561).
            # The height component is carried through untouched and never read.
            signals["specific_target_positions"] = (
                t.tensor([x, y, 0.0]).float().view([1, 1, 3])
            )
            signals["specific_target_headings"] = (
                t.tensor([intent.target_heading]).float().view([1, 1])
            )
            signals["has_specific_target"] = t.tensor([[True]]).int()

        return signals

    def generate(
        self,
        intent: GeneratorIntent,
        context_qpos: np.ndarray,
        dt: float,
        *,
        force: bool = False,
    ) -> None:
        """Ask the agent to (re)plan.

        Normally a no-op until the plan cursor passes ``dt × GENERATOR_HZ`` frames
        (``full_agent.py:122-124``); ``force`` overrides that and plans immediately.

        **The replan cadence decides whether a committed move completes.** A plan is an in-between
        from the current context to the target pose, and the target is its *last* frame. Replanning
        every ``dt`` seconds discards the plan's tail, so a move whose plan is longer than ``dt``
        never arrives — see :func:`~openroboxing.studio.rehearsal.rehearse_commit`.
        """
        signals = self.control_signals(intent, context_qpos)
        # Armed for the duration of the call only, so a pose can never outlive the intent that
        # carried it and silently steer a later replan.
        self._armed_pose = intent.pose
        try:
            with self._torch.no_grad():
                self.agent.generate_new_frames(signals, dt, force_generation=force)
        finally:
            self._armed_pose = None

    def plan(self) -> np.ndarray:
        """The agent's current plan in full, as ``(N, 36)``.

        Reads the buffer directly rather than stepping :meth:`next_frame`, because the plan's tail —
        where the target pose lands — is exactly what a replanning loop never reaches.
        """
        frames = self.agent.frames["mujoco_qpos"][0]
        arr = np.asarray(
            frames.detach().cpu().numpy() if hasattr(frames, "detach") else frames,
            dtype=np.float64,
        )
        if arr.ndim != 2 or arr.shape[1] != QPOS_DIM:
            raise GeneratorError(f"expected an (N, {QPOS_DIM}) plan, got {arr.shape}")
        return arr

    def next_frame(self) -> np.ndarray:
        """Pop the next generated frame. Returns ``(36,)`` MuJoCo qpos."""
        qpos = self.agent.get_next_frame()
        arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if arr.shape != (QPOS_DIM,):
            raise GeneratorError(f"expected a ({QPOS_DIM},) qpos frame, got {arr.shape}")
        return arr

    def context_qpos(self) -> np.ndarray:
        """The agent's own recent-context qpos buffer, as ``(K, 36)``."""
        ctx = self.agent.get_context_mujoco_qpos()
        return np.asarray(ctx.detach().cpu().numpy() if hasattr(ctx, "detach") else ctx).reshape(
            -1, QPOS_DIM
        )
