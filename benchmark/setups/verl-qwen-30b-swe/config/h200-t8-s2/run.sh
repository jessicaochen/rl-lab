#!/usr/bin/env bash
# h200-t8-s2: trainer 1x8 H200, sampler 2 H200 — multi-GPU sampler (TP2 across NVLink, 2x KV)
set -euo pipefail

export TRAIN_NNODES=1
export TRAIN_GPUS=8
export TRAIN_TP=2
export TRAIN_CP=1
export TRAIN_EP=4
export ROLLOUT_GPUS=2
export ROLLOUT_TP=2
export TRAIN_BATCH=32
export ROLLOUT_N=8
export SYNC_STEP=4
export MINI_BATCH=8
export SESSIONS=48
export GATEWAYS=4
export AGENT_WORKERS=4
# standard benchmark = 2 steps (baked at render time); longer validation only on explicit request: `rlbench run ... --var STEPS=10`
export TOTAL_STEPS=${STEPS:-2}
# validation: SWE-bench Verified subset at start + end; `--var VAL_TASKS=500` for the full set
export VAL_TASKS=${VAL_TASKS:-100}

exec bash "$(dirname "$0")/common.sh"
