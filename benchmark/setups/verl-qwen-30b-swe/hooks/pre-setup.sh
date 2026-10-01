#!/usr/bin/env bash
# rlbench pre-setup hook: a previous run's rollout sandboxes can outlive it
# (the Ray job ends before every uni-agent session reaches provider.stop();
# they'd self-delete via shutdownTime, but until then they occupy the sandbox
# pool and can starve the next run of capacity). Start clean.
#   env from rlbench: RUN_ID, NAMESPACE
set -uo pipefail
n="$(kubectl get sandboxes -n "${NAMESPACE:?}" -l rlbench/component=rollout-sandbox --no-headers 2>/dev/null | wc -l | tr -d ' ')"
if [ "${n:-0}" -gt 0 ]; then
  echo "pre-setup: deleting $n leftover rollout sandboxes in $NAMESPACE"
  kubectl delete sandboxes -n "$NAMESPACE" -l rlbench/component=rollout-sandbox --wait=false >/dev/null 2>&1 || true
else
  echo "pre-setup: no leftover sandboxes"
fi
