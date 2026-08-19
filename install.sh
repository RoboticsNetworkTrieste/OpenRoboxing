#!/usr/bin/env bash
# One-command install for OpenRoboxing (M4-T3).
#
# Acceptance criterion from WORKPLAN.md M4-T3:
#   on a clean Windows machine with WSL, one documented command reaches a running hotseat match.
#
#   bash install.sh && bash install.sh --play
#
# It acquires upstream itself: the GR00T-WholeBodyControl submodule (or the checkout named by
# OPENROBOXING_GR00T_ROOT), that checkout's Git-LFS meshes, and the GEAR-SONIC policy checkpoints.
# The checkpoints are *fetched* from nvidia/GEAR-SONIC with upstream's own downloader, never
# redistributed by this repository — which is the distinction LICENSING.md draws (WORKPLAN M6-T2).
# If the fetch fails the script says so and prints the command to run by hand.
#
# Linux and WSL. It refuses to run anywhere it has not been tested rather than guessing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Overridable so this script can be tested against a throwaway venv without touching a working one.
VENV="${OPENROBOXING_VENV:-${REPO_ROOT}/.venv_mb}"
PYTHON_MIN="3.10"

# Upstream. The submodule unless the caller already has a GR00T-WholeBodyControl checkout.
GR00T_ROOT="${OPENROBOXING_GR00T_ROOT:-${REPO_ROOT}/external/gr00t-wbc}"

# The dependency list lives in one place and was measured, not guessed. See its header.
REQUIREMENTS="${REPO_ROOT}/requirements-runtime.txt"

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

check_platform() {
  case "$(uname -s)" in
    Linux) ;;
    *) die "only Linux and WSL are supported. On Windows, install WSL first: wsl --install" ;;
  esac
  if grep -qi microsoft /proc/version 2>/dev/null; then
    say "WSL detected"
  fi
}

check_python() {
  command -v python3 >/dev/null || die "python3 not found. On WSL: sudo apt install python3 python3-venv"
  local version
  version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "$(printf '%s\n%s\n' "$PYTHON_MIN" "$version" | sort -V | head -1)" != "$PYTHON_MIN" ]; then
    die "python ${version} is too old; ${PYTHON_MIN}+ is required"
  fi
  say "python ${version}"
}

make_venv() {
  if [ -d "$VENV" ]; then
    say "using the existing venv at ${VENV}"
    return
  fi
  say "creating ${VENV}"
  python3 -m venv "$VENV" || die "could not create a venv. On WSL: sudo apt install python3-venv"
}

fetch_upstream() {
  if [ -n "${OPENROBOXING_GR00T_ROOT:-}" ]; then
    say "using the GR00T-WholeBodyControl checkout at ${GR00T_ROOT}"
    [ -d "${GR00T_ROOT}/motionbricks" ] \
      || die "OPENROBOXING_GR00T_ROOT=${GR00T_ROOT} has no motionbricks/ — is that a GR00T-WBC checkout?"
    return
  fi

  say "initialising the GR00T-WholeBodyControl submodule"
  git -C "$REPO_ROOT" submodule update --init --recursive external/gr00t-wbc \
    || die "could not initialise the submodule. Is this a git checkout?"

  # Running a match imports upstream's Python, which scatters __pycache__ through the submodule and
  # leaves the superproject reporting ` M external/gr00t-wbc` forever. That reads as "the pristine
  # submodule has been modified" — the one thing this layout promises never happens — and invites
  # someone to helpfully commit it.
  #
  # Writing `__pycache__/` into the submodule's .git/info/exclude does NOT fix this, measured
  # 2026-08-19. Upstream's own .gitignore already carries `__pycache__/` and `*.py[cod]` on lines
  # 2-3, so 21 of the 22 directories are ignored before we do anything; the only one that surfaces
  # is motionbricks/motion_backbone/models/, force-re-included by a negation in
  # motionbricks/.gitignore (`!motionbricks/motion_backbone/models/**`, line 34). A tracked
  # .gitignore outranks $GIT_DIR/info/exclude, so nothing written there can ever win.
  #
  # Tell the *superproject* not to count untracked files in the submodule instead. `untracked`, not
  # `dirty`/`all`: a genuine edit to tracked content, or a moved HEAD, must still be reported —
  # verified by editing a tracked file and watching ` M external/gr00t-wbc` come back. Local to
  # .git/config, never committed, and skipped if the caller has already stated a policy of their own.
  if ! git -C "$REPO_ROOT" config --get submodule.external/gr00t-wbc.ignore >/dev/null 2>&1; then
    git -C "$REPO_ROOT" config submodule.external/gr00t-wbc.ignore untracked \
      || warn "could not set submodule.external/gr00t-wbc.ignore; expect a stray ' M external/gr00t-wbc'"
  fi

  say "fetching LFS content (meshes and generator checkpoints; several GB on a first run)"
  if command -v git-lfs >/dev/null; then
    # `--exclude=` clears upstream's own `fetchexclude = motionbricks/out/**` (see its .lfsconfig).
    # Without it, `git lfs pull` leaves the four MotionBricks .ckpt files as 132-byte pointer stubs;
    # the generator then hands that pointer *text* to torch.load and dies with "Unsupported operand
    # 118" — byte 118 being the 'v' of "version https://git-lfs.github.com/spec/v1". Upstream can
    # afford to skip those weights. A match is nothing but the generator, so we cannot: measured
    # 2026-08-19, the fast suite passes without them and all 20 generator-backed tests fail.
    git -C "$GR00T_ROOT" lfs pull --exclude= \
      || warn "git lfs pull failed; meshes and generator checkpoints may be pointer files"
  else
    warn "git-lfs is not installed — the robot meshes will be pointer files and rendering will fail."
    warn "  Install it: sudo apt install git-lfs && git lfs install"
  fi
}

# uv when it is there (this repo's venvs are uv-managed and have no pip), the venv's pip otherwise.
venv_install() {
  if command -v uv >/dev/null; then
    VIRTUAL_ENV="$VENV" uv pip install "$@"
  elif [ -x "${VENV}/bin/pip" ]; then
    "${VENV}/bin/pip" install "$@"
  else
    die "neither uv nor pip is available in ${VENV}. Install uv: https://docs.astral.sh/uv/"
  fi
}

install_packages() {
  [ -f "$REQUIREMENTS" ] || die "missing ${REQUIREMENTS}"
  if ! command -v uv >/dev/null && [ -x "${VENV}/bin/pip" ]; then
    "${VENV}/bin/pip" install --upgrade pip >/dev/null
  fi

  say "installing the match runtime from $(basename "$REQUIREMENTS")"
  venv_install -r "$REQUIREMENTS"

  say "installing openroboxing (editable)"
  venv_install -e "$REPO_ROOT"

  # pytest is the *smoke test's* tool, not the match's, so it is deliberately absent from
  # requirements-runtime.txt — that file's header commits it to "what a match needs, and nothing
  # else". It still has to be installed here, or `bash install.sh` can never reach its own
  # acceptance criterion. pytest-timeout is not installed and not wanted; see smoke_test().
  say "installing pytest (the smoke test's own tool)"
  venv_install pytest
}

check_checkpoints() {
  local policy="${GR00T_ROOT}/gear_sonic_deploy/policy/release"
  if [ -f "${policy}/model_encoder.onnx" ] && [ -f "${policy}/model_decoder.onnx" ]; then
    say "policy checkpoints found"
    return 0
  fi

  say "fetching the GEAR-SONIC policy from nvidia/GEAR-SONIC (174 MB)"
  # Upstream's own downloader, run in upstream's tree. Nothing NVIDIA-licensed is redistributed by
  # this repository — see LICENSING.md.
  if (cd "$GR00T_ROOT" && "${VENV}/bin/python" download_from_hf.py \
        --output-dir "${GR00T_ROOT}/gear_sonic_deploy" --no-planner); then
    say "policy checkpoints downloaded"
    return 0
  fi

  warn "could not download the policy checkpoints to ${policy}"
  warn "  Run it by hand:  cd ${GR00T_ROOT} && python download_from_hf.py --output-dir gear_sonic_deploy"
  warn "  They are NVIDIA-licensed; see LICENSING.md for the terms."
  return 1
}

smoke_test() {
  say "smoke test"
  cd "$REPO_ROOT"
  "${VENV}/bin/python" -m openroboxing.tools.env_report --quick || {
    warn "env_report reported problems; see above"
    return 1
  }
  # No path argument: pyproject.toml sets testpaths = ["tests"]. No --timeout either — pytest-timeout
  # is not a dependency, so that flag only ever bought a silent second run of the whole suite.
  "${VENV}/bin/python" -m pytest -q -x
}

play() {
  cd "$REPO_ROOT"
  say "starting a hotseat match on http://localhost:8080/"
  say "  red plays 1-6 and SPACE; blue plays U I O J K L and ENTER"
  exec "${VENV}/bin/python" -m openroboxing.tools.serve_match
}

main() {
  if [ "${1:-}" = "--play" ]; then
    play
  fi

  check_platform
  check_python
  make_venv
  fetch_upstream
  install_packages

  local ready=0
  check_checkpoints || ready=1

  if [ "$ready" -eq 0 ]; then
    smoke_test || ready=1
  fi

  echo
  if [ "$ready" -eq 0 ]; then
    say "ready. Start a match with:"
    echo "      bash install.sh --play"
  else
    warn "install finished, but the smoke test did not pass. See the messages above."
    exit 1
  fi
}

main "$@"
