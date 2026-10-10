#!/usr/bin/env bash
# rlbench pre-setup hook for `--feature timeslice`:
# 1) Stages timeslice.py and verl-timeslice.patch into the rendered run-config directory
#    so they are bundled into ConfigMap/rlbench-run-config-${JOB_ID} and uploaded to
#    the Ray driver working directory by `ray job submit --working-dir /etc/rlbench/config`.
# 2) Yields any stale orchestrator locks held by ${JOB_ID:-job1} on the trainers
#    or samplers groups before starting a new time-sliced run.
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
JOB_ID="${TIMESLICE_JOB_ID:-${JOB_ID:-job1}}"
TRAINER_GROUP="${TIMESLICE_TRAINER_GROUP:-trainers}"
SAMPLER_GROUP="${TIMESLICE_SAMPLER_GROUP:-samplers}"
TRAINER_ENABLED="${TIMESLICE_TRAINER_ENABLED:-1}"
SAMPLER_ENABLED="${TIMESLICE_SAMPLER_ENABLED:-0}"
RLTS_BIN="${RLTS_BIN:-rlts}"

if [ -n "${RUN_FOLDER:-}" ] && [ -d "${RUN_FOLDER}/config" ]; then
  for cfg_dir in "${RUN_FOLDER}"/config/*/; do
    [ -d "$cfg_dir" ] || continue
    [ "$(basename "$cfg_dir")" = "rendered" ] && continue
    cp -f "$HERE/timeslice.py" "$cfg_dir/timeslice.py"
    cp -f "$HERE/verl-timeslice.patch" "$cfg_dir/verl-timeslice.patch"
  done
fi

if command -v "$RLTS_BIN" >/dev/null 2>&1; then
  echo "timeslice pre-setup: checking/yielding stale locks for job=$JOB_ID (trainer_ts=$TRAINER_ENABLED:$TRAINER_GROUP, sampler_ts=$SAMPLER_ENABLED:$SAMPLER_GROUP)"
  if [ "$SAMPLER_ENABLED" = "1" ] || [ "$SAMPLER_ENABLED" = "true" ]; then
    "$RLTS_BIN" orchestrator yield "$SAMPLER_GROUP" "$JOB_ID" >/dev/null 2>&1 || true
  fi
  if [ "$TRAINER_ENABLED" = "1" ] || [ "$TRAINER_ENABLED" = "true" ]; then
    "$RLTS_BIN" orchestrator yield "$TRAINER_GROUP" "$JOB_ID" >/dev/null 2>&1 || true
  fi
else
  echo "timeslice pre-setup: rlts CLI not in PATH; skipping host-side lock yield for job=$JOB_ID"
fi
