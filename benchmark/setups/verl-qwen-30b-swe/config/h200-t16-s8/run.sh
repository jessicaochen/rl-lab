#!/usr/bin/env bash
# h200-t16-s8: trainer 2x8 H200, sampler 8 H200 — multi-node trainer (DP=2 across nodes; EP stays on-node)
set -euo pipefail

export TRAIN_NNODES=2
export TRAIN_GPUS=8
export TRAIN_TP=2
export TRAIN_CP=1
export TRAIN_EP=4
export ROLLOUT_GPUS=8
export ROLLOUT_TP=8
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
