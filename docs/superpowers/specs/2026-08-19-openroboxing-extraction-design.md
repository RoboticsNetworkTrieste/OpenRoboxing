# Extraction — OpenRoboxing leaves GR00T-WholeBodyControl

Date: 2026-08-19 · approved by the project owner in brainstorming (five decisions recorded below).

## What it is

OpenRoboxing becomes its own repository —
`https://github.com/TriesteOpenRoboticsCommunity/OpenRoboxing.git`, which already exists and is
private — and reaches its upstream through a **git submodule** pointing at
`NVlabs/GR00T-WholeBodyControl`. Development continues there. The `openroboxing/` tree inside
`GR00T-WholeBodyControl` is left untouched as a frozen record.

The extraction is possible cheaply because of a property the project already has: **one file knows
where upstream lives**. `paths.py` holds every path into `gear_sonic_deploy/` and `motionbricks/`,
and no module hard-codes a relative path. Moving the boundary is therefore a change to `paths.py`
plus one genuine behaviour change (§5), not a change spread across the codebase.

## Owner's decisions

| Question | Decision |
|---|---|
| What does the submodule point at | **Pristine `NVlabs/GR00T-WholeBodyControl`**, no fork. Patch P0 is applied at runtime instead |
| Which commit | **Latest `main`**, not a pin |
| History | **Fresh start** — one initial commit, no rewritten history |
| Layout | **`src/` layout** |
| Reaching GitHub | Built locally, committed, smoke-tested; **the owner pushes** |
| Old copy | **Left untouched** in `GR00T-WholeBodyControl` |

## The coupling, as measured

Measured on 2026-08-19 against branch `openroboxing` at `bf9eaa2`.

| | |
|---|---|
| `openroboxing/` on disk | 8.8 MB |
| Commits touching it | 75 of the 77 the branch is ahead of `origin/main` |
| Python imports from outside the package | **none at module scope** — `motionbricks` and `gear_sonic` are imported lazily inside functions |
| Coupling that remains | file paths, all of them via `paths.py`, plus `env_report.py:34` which computes its own `REPO_ROOT` |

Artifacts the runtime reads, and where they come from:

| Artifact | Size | Source |
|---|---|---|
| `gear_sonic_deploy/g1/` (MJCF + STL meshes) | 44 MB | tracked in upstream git, LFS |
| `gear_sonic_deploy/reference/example/` | 17 MB | tracked in upstream git |
| `motionbricks/` source + `assets/skeletons/g1/` | 167 MB | tracked in upstream git |
| `motionbricks/out/**` (3 checkpoints + `G1-clip.ckpt`) | 2.2 GB | tracked in upstream git (`.gitignore` carries an explicit `!motionbricks/out/**/*.ckpt`) |
| `gear_sonic_deploy/policy/release/model_{encoder,decoder}.onnx` | 174 MB | **not in git anywhere** — fetched from `nvidia/GEAR-SONIC` by upstream's own `download_from_hf.py` |

So the submodule delivers everything except the two ONNX files, and those have a documented
download path. **Nothing NVIDIA-licensed is redistributed by this repository.** `LICENSING.md`
travels unchanged and remains accurate.

## 1. Target layout

```
OpenRoboxing/
├── src/openroboxing/            the package, internally unchanged
│   ├── __init__.py  paths.py
│   ├── runtime/  server/  studio/  tools/  league/  parity/
│   ├── spec/                    stays in-package: spec.constants is an imported module
│   ├── client/                  stays in-package: served by aiohttp from OPENROBOXING_ROOT
│   └── poses/                   stays in-package: runtime data (libraries + loadouts/)
├── external/gr00t-wbc/          submodule → NVlabs/GR00T-WholeBodyControl, branch main
├── tests/                       hoisted, including fixtures/ and conftest.py
├── docs/                        hoisted, including superpowers/{specs,plans}
├── CLAUDE.md                    was docs/CLAUDE.md — at the root it auto-loads for agent sessions
├── pyproject.toml               new: src layout, pytest config, project metadata
├── install.sh  requirements-runtime.txt  LICENSING.md  README.md
├── LICENSE                      new: Apache-2.0, matching upstream's source licence
├── .gitmodules  .gitignore
```

No git-LFS. The whole tree is 8.8 MB and its largest file is the 2.4 MB
`tests/fixtures/golden_policy_io/golden.npz`; the two PNGs under `poses/` are below 0.5 MB. Plain
git handles all of it, and every LFS-sized asset lives in the submodule where upstream's own
`.gitattributes` already governs it.

`spec/`, `client/` and `poses/` stay inside the package deliberately. `spec/` because
`openroboxing.spec.constants` is imported code; `client/` and `poses/` because they are data the
running server resolves from `OPENROBOXING_ROOT`, and hoisting them would add path indirection for
no gain. `tests/` and `docs/` hoist because nothing imports them.

`docs/CLAUDE.md` becomes the root `CLAUDE.md`. It is the live instruction file — 199 lines, and the
most recent commit on the branch edits it — and at the repository root it is picked up
automatically instead of having to be found.

## 2. What travels, and what stays

**Travels:** everything under `openroboxing/`, minus `__pycache__/` and `.pytest_cache/`.

**Stays behind in `GR00T-WholeBodyControl`:** `planner_pilot/`, `bricklayer/`, `rokoko_converter/`,
and the 23-DOF deploy work. Verified: no module under `openroboxing/` imports or reads any of them.

**Patches P1 and P2** — the policy-input and encoder-input dumps in
`gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp` (commit `4cf41a9`) — also stay behind. They are
C++ and **fixture-only**: they exist to capture `tests/fixtures/golden_policy_io/golden.npz`
against the real deploy binary. That fixture is already captured and travels with the tests, so
nothing in the new repository needs the patches to run. Re-capturing does, and re-capturing needs
something the submodule cannot provide anyway: a built `g1_deploy_onnx_ref` and a robot. The
existing `GR00T-WholeBodyControl` checkout is exactly that working copy and is being kept.

`parity/capture_run.sh` travels, with one fix — it currently hard-codes
`DEPLOY_DIR=/home/hpc-dev/GR00T-WholeBodyControl/gear_sonic_deploy`. That becomes
`${OPENROBOXING_DEPLOY_DIR:-$GR00T_ROOT/gear_sonic_deploy}`, so it points at whichever checkout has
P1/P2 applied and the binary built. Its header gains a line saying so.

## 3. The GR00T-WBC boundary

`paths.py` gains one constant and re-roots the rest:

```python
REPO_ROOT: Path = Path(__file__).resolve().parents[2]      # was parents[1]

#: Where the upstream checkout lives. The submodule is the reproducible answer; the environment
#: variable lets a machine that already has a GR00T-WholeBodyControl checkout use it instead of
#: cloning 4.2 GB (3.8 GB of it LFS meshes) a second time.
GR00T_ROOT: Path = Path(
    os.environ.get("OPENROBOXING_GR00T_ROOT", REPO_ROOT / "external/gr00t-wbc")
)
```

Every upstream path becomes `GR00T_ROOT / ...`; every OpenRoboxing path stays
`OPENROBOXING_ROOT / ...` except `FIXTURES_DIR`, which follows `tests/` to `REPO_ROOT /
"tests/fixtures"`.

`tools/env_report.py:34` computes `REPO_ROOT = Path(__file__).resolve().parents[2]` for itself.
It stops doing that and imports `REPO_ROOT` and `GR00T_ROOT` from `paths`, so there is again exactly
one place that knows the layout. Its artifact checklist re-roots to `GR00T_ROOT`.

**Tracking `main`.** `.gitmodules` records `branch = main`. A submodule always pins a SHA in the
superproject — that is git's data model and there is no follow-the-branch checkout mode — so
"latest main" means the bump is one deliberate command:

```bash
git submodule update --remote external/gr00t-wbc
```

`spec/upstream_notes.md` already warns that moving off the recorded snapshot invalidates its line
numbers. The bump therefore carries a documented re-verify step in the README: run the test suite,
and re-check the observation-registry offsets that `upstream_notes.md` records. The initial commit
records whatever `main` is on the day of extraction.

## 4. Bootstrap

`install.sh` keeps its shape — platform check, python check, venv, requirements, checkpoint check,
smoke test — and gains upstream acquisition before the smoke test:

1. `git submodule update --init --recursive external/gr00t-wbc` — skipped when
   `OPENROBOXING_GR00T_ROOT` is set.
2. `git lfs pull` inside the submodule, for the STL meshes.
3. Policy ONNX: if `$GR00T_ROOT/gear_sonic_deploy/policy/release/model_{encoder,decoder}.onnx` are
   absent, run upstream's `download_from_hf.py --output-dir gear_sonic_deploy` from inside the
   submodule. It reports the failure and the manual command rather than dying silently, matching
   the script's existing `check_checkpoints` behaviour.

Paths inside the script move: `REQUIREMENTS` to the repo root, the smoke test to `pytest tests`,
and `--play` to `serve_match` unchanged.

## 5. Patch P0 becomes runtime code

This is the only place where behaviour, not location, changes, and the only part of the extraction
that can be wrong in a way tests must catch.

**Today.** `motionbricks/motionbricks/motion_backbone/demo/full_agent.py` carries patch P0
(commit `e18f80b`, +55 lines): a new method `_override_target_joint_transforms`, plus a call to it
inside `generate_new_frames` immediately after `_generate_target_joint_transforms` fills three keys
into `input`. `runtime/generator.py:241-280` then wraps that method **on the agent instance**, so
that an armed pose is re-rooted onto the placement the spring model chose before P0 swaps it in.

**After.** The submodule is pristine, so neither the method nor its call site exists.
`_install_pose_override` supplies both:

- **The body.** P0's logic — shape validation against the current tensors, and the optional
  per-batch blend via `has_specific_target_pose` — moves into `runtime/generator.py` as a plain
  module-level function. It is 35 lines of tensor swapping with no upstream dependency.
- **The call site.** `agent._generate_target_joint_transforms` is wrapped on the instance. The
  wrapper calls the original, writes its three-tuple into `input` exactly as `generate_new_frames`
  does, runs the override, and returns the three values re-read from `input`. The outer assignment
  in `generate_new_frames` then re-assigns the same, now-overridden, values.

That last step is what makes the two equivalent: upstream assigns, then P0 mutates `input`. The
wrapper mutates `input` first and returns what it wrote, so the assignment is idempotent. The
override reads `input['target_global_joint_positions']`, which the wrapper has already set.

Everything OpenRoboxing decides — heading-only placement, which model FK runs on — stays where it
is, on this side of the boundary. The change is strictly about who installs the hook.

**The test that makes this safe.** `tests/test_generator_pose_override.py` asserts against an
agent whose `full_agent.py` has **no P0**: that `_generate_target_joint_transforms` is wrapped, that
an armed pose reaches `specific_target_joint_positions`/`_rotations`, that a shape mismatch raises,
and that with no pose armed the tensors are byte-identical to the unwrapped call. A submodule bump
that changes upstream's signature or call order fails here rather than silently disarming the pose
system — which would otherwise present as a fighter that simply ignores every authored pose.

`spec/upstream_patches.md` keeps all three entries and is restated:

- **P0** — status becomes *installed at runtime*, with a pointer to `_install_pose_override`.
- **P1, P2** — status becomes *upstream-side, fixture capture only*; they are applied in a
  GR00T-WBC working copy when a capture is run, never in the submodule.
- The pinned-snapshot table is replaced by the tracking-`main` policy of §3.
- The "Open action" requiring a fork is closed as **not needed** — installing P0 at runtime is
  what makes the fork unnecessary.

## 6. Documentation

131 lines across 13 files name `openroboxing/`, `gear_sonic_deploy/` or `motionbricks/` paths.
Each is rewritten to the new layout: `openroboxing/x` → `src/openroboxing/x`,
`openroboxing/tests` → `tests`, `openroboxing/docs` → `docs`, and upstream paths gain the
`external/gr00t-wbc/` prefix. `README.md`'s command table moves to the repo root and its commands
drop the `openroboxing/` prefix where it was a path rather than a module name — module invocations
(`python -m openroboxing.tools.serve_sparring`) are unchanged, because the package name does not
change.

A new root `README.md` opens with what the project is, the one-command install, and the
`serve_sparring` line; the existing `openroboxing/README.md` content follows it.

## 7. Acceptance

The extraction is done when, in the new repository:

1. `bash install.sh` completes and reports ready, on this machine via
   `OPENROBOXING_GR00T_ROOT=/home/hpc-dev/GR00T-WholeBodyControl`.
2. `.venv_mb/bin/python -m pytest tests -q` passes, including the new P0 test.
3. `.venv_mb/bin/python -m openroboxing.tools.env_report` finds every artifact.
4. `.venv_mb/bin/python -m openroboxing.tools.serve_sparring` serves the bench on :8081 and a
   commit runs — the manual check from `spec/sparring_protocol.md`.
5. `git status` is clean, one commit exists, `origin` is set to the target URL, and nothing has
   been pushed.

The owner then pushes.

## 8. Risks

| Risk | Handling |
|---|---|
| A pristine submodule silently disarms the pose system | The §5 test runs against unpatched upstream and fails loudly |
| `main` drifts and invalidates `upstream_notes.md` line numbers | Bump is a deliberate command with a documented re-verify step; never automatic |
| A fresh clone costs 4.2 GB incl. 3.8 GB LFS | `OPENROBOXING_GR00T_ROOT` lets an existing checkout be reused; the submodule is for fresh machines |
| Something needed was missed in the move | The old tree is left untouched and remains the reference |
| `git submodule update` reverts a hand-edit in the submodule | Nothing the runtime needs is hand-edited there — that is the point of §5. P1/P2 are applied only in the separate GR00T-WBC working copy used for fixture capture, never in the submodule |
| A parity re-capture is needed later | The old checkout keeps P1/P2 and the built binary; `capture_run.sh` is pointed at it by env var (§2) |
