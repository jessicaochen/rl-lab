# rlbench feature fragment: sourced by config/common.sh after its own overrides.
# Turns the always-on RlbenchRolloutAdapter into scheduler mode and gives the
# scheduler its config.
# ${SCHEDULER_CONFIG} is an rlbench render-time variable (features/inference-scheduler/vars.env).
set -euo pipefail
_pyis_cfg="$(dirname "$0")/${SCHEDULER_CONFIG:-scheduler.yaml}"
[ -f "$_pyis_cfg" ] || { echo "feature-inference-scheduler: $_pyis_cfg not found" >&2; exit 2; }
# the gateway actors run on the head with /data mounted; an absolute path there
# avoids depending on each actor's working directory
mkdir -p "$RUN_OUT"
cp "$_pyis_cfg" "$RUN_OUT/scheduler.yaml"
# Deliberately NO worker_process_setup_hook and NO PROMETHEUS_MULTIPROC_DIR: the
# scheduler's own vLLM patch would have to run in every Ray worker, and importing
# vLLM there initialized CUDA before Ray assigned GPUs (every trainer rank on GPU 0,
# OOM at model build; run 20261005-144210). The SchedulerClient scrapes /metrics
# itself instead, so the server processes stay untouched.
EXTRA_OVERRIDES+=(
  '+ray_kwargs.ray_init.runtime_env.env_vars.RLBENCH_ROUTER="inference-scheduler"'
  "+ray_kwargs.ray_init.runtime_env.env_vars.ROUTER_CONFIG_PATH=\"$RUN_OUT/scheduler.yaml\""
)
echo "feature-inference-scheduler: router=inference-scheduler config=$RUN_OUT/scheduler.yaml"
