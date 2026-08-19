#!/bin/bash
# Drive the G1 deploy through its keyboard state machine to capture golden fixtures (M1-T2).
#
# The deploy's keyboard handler reads STDIN_FILENO with non-blocking reads, so operator keys can be
# fed on stdin from a FIFO:
#   ']'  -> start_control  (ProgramState::WAIT_FOR_CONTROL -> CONTROL)
#   'T'  -> play_motion    (play the reference motion to its end)
#
# Usage: capture_run.sh <capture_dir> <seconds_to_record>
#
# NOTE: no `set -u` here. setup_env.sh dereferences unset variables (LD_LIBRARY_PATH etc.), which
# under `set -u` aborts this script the moment it is sourced.

CAP="$1"
SECONDS_TO_RECORD="${2:-25}"
# The deploy tree with patches P1/P2 applied and g1_deploy_onnx_ref built. That is NOT the
# submodule — the submodule is pristine on purpose (see spec/upstream_patches.md), and a capture
# needs a working copy where the dump patches are applied. Point this at it.
DEPLOY_DIR="${OPENROBOXING_DEPLOY_DIR:-${OPENROBOXING_GR00T_ROOT:-/home/hpc-dev/GR00T-WholeBodyControl}/gear_sonic_deploy}"

if [ ! -x "$DEPLOY_DIR/target/release/g1_deploy_onnx_ref" ]; then
  echo "no g1_deploy_onnx_ref in $DEPLOY_DIR/target/release" >&2
  echo "set OPENROBOXING_DEPLOY_DIR to a GR00T-WBC deploy tree with P1/P2 applied and built" >&2
  exit 1
fi
FIFO="$CAP/keys.fifo"

mkdir -p "$CAP"
rm -f "$FIFO"
mkfifo "$FIFO"

cd "$DEPLOY_DIR" || exit 1
echo "sourcing setup_env.sh"
# shellcheck disable=SC1091
source scripts/setup_env.sh > "$CAP/setup_env.log" 2>&1 || true
echo "setup_env sourced"

# Hold the FIFO open for the whole run so the deploy never sees EOF on stdin.
# Must be read-write (<>): opening a FIFO write-only blocks until a reader appears, and our reader
# (the deploy) is started below -- that would deadlock.
echo "opening fifo"
exec 3<>"$FIFO"
echo "fifo open"

./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --input-type manager \
  --disable-crc-check \
  --policy-input-logfile "$CAP/policy_input.csv" \
  --encoder-input-logfile "$CAP/encoder_input.csv" \
  --target-motion-logfile "$CAP/target_motion.csv" \
  --logs-dir "$CAP/state_logs" \
  --enable-csv-logs \
  < "$FIFO" > "$CAP/deploy.log" 2>&1 &
DEPLOY_PID=$!

# Wait for the constructor to finish (model load + TensorRT). "Init Done" is the marker.
for _ in $(seq 1 240); do
  grep -q "Init Done" "$CAP/deploy.log" 2>/dev/null && break
  kill -0 "$DEPLOY_PID" 2>/dev/null || { echo "DEPLOY DIED DURING INIT"; exit 1; }
  sleep 1
done
grep -q "Init Done" "$CAP/deploy.log" || { echo "TIMEOUT waiting for Init Done"; kill "$DEPLOY_PID"; exit 1; }
echo "init complete; entering control"

printf ']' >&3          # start control system
sleep 3                  # let it settle into CONTROL / reach the default pose
printf 'T' >&3          # play the reference motion
echo "playing for ${SECONDS_TO_RECORD}s"
sleep "$SECONDS_TO_RECORD"

# Stop cleanly so the ofstreams flush and close.
kill -INT "$DEPLOY_PID" 2>/dev/null
sleep 3
kill -TERM "$DEPLOY_PID" 2>/dev/null
sleep 1
kill -KILL "$DEPLOY_PID" 2>/dev/null
exec 3>&-
rm -f "$FIFO"
echo "capture finished"
wc -l "$CAP"/*.csv 2>/dev/null
