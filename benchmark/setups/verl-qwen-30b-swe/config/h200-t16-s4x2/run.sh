#!/usr/bin/env bash
# h200-t16-s4x2: trainer 2x8 H200 (DP=2 across nodes, EP on-node), sampler 8 H200 as
# 4 vLLM replicas x TP2. Same trainer/batch/session shape as h200-t16-s8; the barrier
# is the multi-replica sampler: every policy request is now *routed* to one of four
# engines, which is what request-routing features act on (a single replica makes
# routing a no-op). Naming: s<R>x<G> = R replicas of G GPUs.
set -euo pipefail

export TRAIN_NNODES=2
export TRAIN_GPUS=8
export TRAIN_TP=2
export TRAIN_CP=1
export TRAIN_EP=4
export ROLLOUT_GPUS=8
export ROLLOUT_TP=2          # verl makes ROLLOUT_GPUS / ROLLOUT_TP = 4 standalone replicas
export TRAIN_BATCH=64
export ROLLOUT_N=8
export SYNC_STEP=4
export MINI_BATCH=16
export SESSIONS=128
export GATEWAYS=8
export AGENT_WORKERS=8
# standard benchmark = 2 steps (baked at render time); longer validation only on explicit request: `rlbench run ... --var STEPS=10`
export TOTAL_STEPS=${STEPS:-2}
# validation: SWE-bench Verified subset at start + end; `--var VAL_TASKS=500` for the full set
export VAL_TASKS=${VAL_TASKS:-100}

exec bash "$(dirname "$0")/common.sh"
