#!/usr/bin/env bash
# rlbench post-run hook for `--feature timeslice`:
# Collects time-slicing orchestrator and snapshot-agent logs into the run folder
# and releases any leftover locks held by ${JOB_ID:-job1}.
set -uo pipefail

OUT_DIR="${RUN_FOLDER:?}/logs"
mkdir -p "$OUT_DIR"

TS_NS="${TIMESLICE_SYSTEM_NAMESPACE:-timeslice-system}"
JOB_ID="${TIMESLICE_JOB_ID:-${JOB_ID:-job1}}"
TRAINER_GROUP="${TIMESLICE_TRAINER_GROUP:-trainers}"
SAMPLER_GROUP="${TIMESLICE_SAMPLER_GROUP:-samplers}"
TRAINER_ENABLED="${TIMESLICE_TRAINER_ENABLED:-1}"
SAMPLER_ENABLED="${TIMESLICE_SAMPLER_ENABLED:-0}"
RLTS_BIN="${RLTS_BIN:-rlts}"

# Capture orchestrator logs
orch_pod="$(kubectl get pods -n "$TS_NS" -l app.kubernetes.io/name=timesliceorchestrator -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [ -n "$orch_pod" ]; then
  kubectl logs -n "$TS_NS" "$orch_pod" --timestamps > "$OUT_DIR/orchestrator.log" 2>/dev/null || true
fi

# Capture snapshot-agent logs across GPU nodes
agent_pods="$(kubectl get pods -n "$TS_NS" -l app.kubernetes.io/name=snapshot-agent -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)"
if [ -n "$agent_pods" ]; then
  : > "$OUT_DIR/snapshot-agent.log"
  for apod in $agent_pods; do
    echo "=== $apod ===" >> "$OUT_DIR/snapshot-agent.log"
    kubectl logs -n "$TS_NS" "$apod" --timestamps >> "$OUT_DIR/snapshot-agent.log" 2>/dev/null || true
  done
fi

# Ensure locks are yielded on teardown
if command -v "$RLTS_BIN" >/dev/null 2>&1; then
  if [ "$SAMPLER_ENABLED" = "1" ] || [ "$SAMPLER_ENABLED" = "true" ]; then
    "$RLTS_BIN" orchestrator yield "$SAMPLER_GROUP" "$JOB_ID" >/dev/null 2>&1 || true
  fi
  if [ "$TRAINER_ENABLED" = "1" ] || [ "$TRAINER_ENABLED" = "true" ]; then
    "$RLTS_BIN" orchestrator yield "$TRAINER_GROUP" "$JOB_ID" >/dev/null 2>&1 || true
  fi
fi
echo "timeslice post-run: collected orchestrator/snapshot-agent logs and cleaned up locks for job=$JOB_ID"
