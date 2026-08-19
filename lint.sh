#!/usr/bin/env bash
# Lint gate for OpenRoboxing.
#
# CLAUDE.md's definition of done, item 3, is "lint.sh clean". That used to mean upstream's
# lint.sh, which cannot run here for two independent reasons:
#
#   1. it calls `python -m ruff` and `python -m black`, and there is no `python` on the PATH in
#      this environment — only `python3` and the venv's interpreter;
#   2. it never propagates a non-zero exit, so it could not gate anything even when it ran.
#
# It also ran `pip install black ruff` into whatever environment it found, which these uv-managed
# venvs do not accept. This script does none of that: it resolves an interpreter and a linter, or
# it says exactly what is missing and how to get it.
#
#   bash lint.sh
#
# Checks `src/`, `tests/`, and the repository's own shell scripts. Exits non-zero if any check
# reports a finding, so it can gate a commit or a CI job.
#
# To apply what ruff can fix, do it deliberately and review the diff — this script will not do it
# for you:  uvx ruff@<pinned version> check --fix src tests
#
# The ruff version is pinned on purpose. An unpinned linter answers differently on different
# machines and on different days; a gate that moves under you is worse than no gate at all.

# NOT `set -e`, unlike install.sh: every check must run even after one fails, or the summary at the
# end is a lie about which checks were actually performed.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Overridable so this script can be pointed at a throwaway venv, exactly as install.sh allows.
VENV="${OPENROBOXING_VENV:-${REPO_ROOT}/.venv_mb}"

# Pinned. Bump deliberately, and expect the finding count to move when you do.
RUFF_VERSION="0.16.2"

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

FAILED=0
SKIPPED=0

# ---------------------------------------------------------------------------------------------
# What to lint: the package, the tests, and any script that grows at the repository root.
# ---------------------------------------------------------------------------------------------
PY_TARGETS=()
for d in src tests; do
  [ -d "${REPO_ROOT}/${d}" ] && PY_TARGETS+=("${REPO_ROOT}/${d}")
done
while IFS= read -r f; do
  PY_TARGETS+=("$f")
done < <(find "$REPO_ROOT" -maxdepth 1 -name '*.py' -type f 2>/dev/null)

SH_TARGETS=()
while IFS= read -r f; do
  SH_TARGETS+=("$f")
done < <(find "$REPO_ROOT" -maxdepth 1 -name '*.sh' -type f 2>/dev/null)

[ ${#PY_TARGETS[@]} -gt 0 ] || die "no Python to lint. Is ${REPO_ROOT} an OpenRoboxing checkout?"

# ---------------------------------------------------------------------------------------------
# Find a ruff. Preference order, most self-contained first.
# ---------------------------------------------------------------------------------------------
RUFF=()
RUFF_SOURCE=""

resolve_ruff() {
  if [ -x "${VENV}/bin/python" ] && "${VENV}/bin/python" -c 'import ruff' >/dev/null 2>&1; then
    RUFF=("${VENV}/bin/python" -m ruff)
    RUFF_SOURCE="${VENV}/bin/python -m ruff"
    return 0
  fi

  if [ ! -x "${VENV}/bin/python" ]; then
    warn "no interpreter at ${VENV}/bin/python — run 'bash install.sh' to build the venv."
    warn "  continuing: ruff lints source files and does not need the project's venv."
  fi

  if command -v uvx >/dev/null 2>&1; then
    RUFF=(uvx "ruff@${RUFF_VERSION}")
    RUFF_SOURCE="uvx ruff@${RUFF_VERSION}"
    return 0
  fi

  if command -v ruff >/dev/null 2>&1; then
    RUFF=(ruff)
    RUFF_SOURCE="ruff on PATH ($(ruff --version 2>/dev/null)) — NOT the pinned ${RUFF_VERSION}"
    warn "using an unpinned ruff; findings may differ from the pinned ${RUFF_VERSION}."
    return 0
  fi

  die "no ruff. Install it into the venv:
      VIRTUAL_ENV=${VENV} uv pip install ruff==${RUFF_VERSION}
    or install uv, which can run it without installing:
      curl -LsSf https://astral.sh/uv/install.sh | sh"
}

# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------
check_ruff() {
  say "ruff check  (${RUFF_SOURCE})"
  if "${RUFF[@]}" check "${PY_TARGETS[@]}"; then
    say "  no findings"
  else
    FAILED=$((FAILED + 1))
  fi
}

check_format() {
  # black is the formatter of record (CLAUDE.md, Conventions). ruff's formatter is *not* a drop-in
  # substitute here — it disagrees with the formatting already in the tree — so this reports
  # honestly as skipped rather than substituting a different tool and calling the answer the same.
  local black=""
  if [ -x "${VENV}/bin/black" ]; then
    black="${VENV}/bin/black"
  elif command -v black >/dev/null 2>&1; then
    black="black"
  fi

  if [ -z "$black" ]; then
    warn "black not found — formatting NOT checked (this check is skipped, not passed)"
    warn "  install it with:  VIRTUAL_ENV=${VENV} uv pip install black"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  say "black --check  (${black})"
  if "$black" --check --quiet "${PY_TARGETS[@]}"; then
    say "  no findings"
  else
    FAILED=$((FAILED + 1))
  fi
}

check_shell() {
  [ ${#SH_TARGETS[@]} -gt 0 ] || return 0

  say "shell syntax  (bash -n)"
  local bad=0
  for f in "${SH_TARGETS[@]}"; do
    bash -n "$f" || bad=1
  done
  if [ "$bad" -eq 0 ]; then
    say "  no findings"
  else
    FAILED=$((FAILED + 1))
  fi

  if command -v shellcheck >/dev/null 2>&1; then
    say "shellcheck"
    if shellcheck "${SH_TARGETS[@]}"; then
      say "  no findings"
    else
      FAILED=$((FAILED + 1))
    fi
  else
    warn "shellcheck not installed — shell scripts checked for syntax only (skipped, not passed)"
    SKIPPED=$((SKIPPED + 1))
  fi
}

main() {
  resolve_ruff
  check_ruff
  check_format
  check_shell

  echo
  [ "$SKIPPED" -gt 0 ] && warn "${SKIPPED} check(s) skipped for a missing tool — see above"
  if [ "$FAILED" -gt 0 ]; then
    die "${FAILED} check(s) reported findings"
  fi
  say "clean"
}

main "$@"
