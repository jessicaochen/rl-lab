# rlbench feature fragment: sourced by config/common.sh when `--feature timeslice` is active.
# Configures llm-d-rl-time-slicing dual-pool lock coordination (`trainers` & `samplers`)
# and multi-GPU trainer/sampler offload/restore lifecycle.
set -euo pipefail

export RLBENCH_NO_HYBRID_ROLLOUT=1
export NO_HYBRID_ROLLOUT=1
export TIMESLICE_ENABLED="${TIMESLICE_ENABLED:-1}"
export TIMESLICE_TRAINER_ENABLED="${TIMESLICE_TRAINER_ENABLED:-1}"
export TIMESLICE_SAMPLER_ENABLED="${TIMESLICE_SAMPLER_ENABLED:-1}"
export VERL_TRAINER_POST_SYNC_OFFLOAD="${VERL_TRAINER_POST_SYNC_OFFLOAD:-$TIMESLICE_TRAINER_ENABLED}"
export VERL_SAMPLER_SLEEP_OFFLOAD="${VERL_SAMPLER_SLEEP_OFFLOAD:-$TIMESLICE_SAMPLER_ENABLED}"
export TIMESLICE_JOB_ID="${TIMESLICE_JOB_ID:-${JOB_ID:-job1}}"
export TIMESLICE_ORCHESTRATOR_ADDR="${TIMESLICE_ORCHESTRATOR_ADDR:-timeslice-timesliceorchestrator.timeslice-system.svc.cluster.local:50051}"
export TIMESLICE_TRAINER_GROUP="${TIMESLICE_TRAINER_GROUP:-trainers}"
export TIMESLICE_SAMPLER_GROUP="${TIMESLICE_SAMPLER_GROUP:-samplers}"
export TIMESLICE_MODE="${TIMESLICE_MODE:-hybrid}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.80}"
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=21600
export NCCL_WATCHDOG_TIMEOUT_SEC=21600
export TORCH_DISTRIBUTED_TIMEOUT=21600

# Optional batch/session overrides via `--var TS_TRAIN_BATCH=...` etc.
if [ -n "${TS_TRAIN_BATCH:-}" ]; then export TRAIN_BATCH="${TS_TRAIN_BATCH:-}"; fi
if [ -n "${TS_ROLLOUT_N:-}" ]; then export ROLLOUT_N="${TS_ROLLOUT_N:-}"; fi
if [ -n "${TS_SYNC_STEP:-}" ]; then export SYNC_STEP="${TS_SYNC_STEP:-}"; fi
if [ -n "${TS_MINI_BATCH:-}" ]; then export MINI_BATCH="${TS_MINI_BATCH:-}"; fi
if [ -n "${TS_SESSIONS:-}" ]; then export SESSIONS="${TS_SESSIONS:-}"; fi

mkdir -p "$RUN_OUT" /data/timeslice

_TS_OVERRIDES=(
  '++ray_kwargs.ray_init.runtime_env.env_vars.RLBENCH_NO_HYBRID_ROLLOUT="1"'
  "++ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_ENABLED=\"$TIMESLICE_ENABLED\""
  "++ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_TRAINER_ENABLED=\"$TIMESLICE_TRAINER_ENABLED\""
  "++ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_SAMPLER_ENABLED=\"$TIMESLICE_SAMPLER_ENABLED\""
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_TRAINER_POST_SYNC_OFFLOAD=\"$VERL_TRAINER_POST_SYNC_OFFLOAD\""
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_SAMPLER_SLEEP_OFFLOAD=\"$VERL_SAMPLER_SLEEP_OFFLOAD\""
  "++ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_JOB_ID=\"$TIMESLICE_JOB_ID\""
  "++ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_ORCHESTRATOR_ADDR=\"$TIMESLICE_ORCHESTRATOR_ADDR\""
  "++ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_TRAINER_GROUP=\"$TIMESLICE_TRAINER_GROUP\""
  "++ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_SAMPLER_GROUP=\"$TIMESLICE_SAMPLER_GROUP\""
  "++ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_MODE=\"$TIMESLICE_MODE\""
  '++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="/opt/verl:/data/timeslice"'
  '++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NVLS_ENABLE="0"'
  '++ray_kwargs.ray_init.runtime_env.env_vars.TORCH_NCCL_ENABLE_MONITORING="0"'
  '++ray_kwargs.ray_init.runtime_env.env_vars.TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="21600"'
  '++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_WATCHDOG_TIMEOUT_SEC="21600"'
  '++ray_kwargs.ray_init.runtime_env.env_vars.TORCH_DISTRIBUTED_TIMEOUT="21600"'
  "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL"
)

if [ "${TIMESLICE_SHIM_ENABLED:-0}" = "1" ] && [ -n "${TIMESLICE_SHIM_PATH:-}" ]; then
  _TS_OVERRIDES+=(
    "++ray_kwargs.ray_init.runtime_env.env_vars.LD_PRELOAD=\"${TIMESLICE_SHIM_PATH:-}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_NCCL_SO_PATH=\"${TIMESLICE_SHIM_PATH:-}\""
  )
fi

EXTRA_OVERRIDES+=("${_TS_OVERRIDES[@]}")

echo "feature-timeslice: enabled=$TIMESLICE_ENABLED trainer_ts=$TIMESLICE_TRAINER_ENABLED sampler_ts=$TIMESLICE_SAMPLER_ENABLED trainer_offload=$VERL_TRAINER_POST_SYNC_OFFLOAD sampler_offload=$VERL_SAMPLER_SLEEP_OFFLOAD job=$TIMESLICE_JOB_ID mode=$TIMESLICE_MODE addr=$TIMESLICE_ORCHESTRATOR_ADDR groups=($TIMESLICE_TRAINER_GROUP,$TIMESLICE_SAMPLER_GROUP) gpu_mem_util=$ROLLOUT_GPU_MEM_UTIL"

# Locate timeslice.py and verl-timeslice.patch (either alongside feature-timeslice.sh in the
# uploaded working-dir or in ../features/timeslice) and apply them across all nodes in this RayCluster.
_FRAG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_TS_PY=""
_TS_PATCH=""
for _cand in "$_FRAG_DIR" "$(cd "$_FRAG_DIR/.." 2>/dev/null && pwd)" "/data/timeslice"; do
  if [ -f "$_cand/timeslice.py" ] && [ -f "$_cand/verl-timeslice.patch" ]; then
    _TS_PY="$_cand/timeslice.py"
    _TS_PATCH="$_cand/verl-timeslice.patch"
    break
  fi
done

if [ -n "$_TS_PY" ] && [ -n "$_TS_PATCH" ]; then
  TS_STAGE_DIR="$RUN_OUT/timeslice"
  mkdir -p "$TS_STAGE_DIR" /data/timeslice
  cp -f "$_TS_PY" "$TS_STAGE_DIR/timeslice.py"
  cp -f "$_TS_PATCH" "$TS_STAGE_DIR/verl-timeslice.patch"
  cp -f "$_TS_PY" /data/timeslice/timeslice.py 2>/dev/null || true
  cp -f "$_TS_PATCH" /data/timeslice/verl-timeslice.patch 2>/dev/null || true
  export PYTHONPATH="/opt/verl:$TS_STAGE_DIR:/data/timeslice"
  EXPECTED_GPUS=$(( TRAIN_NNODES * TRAIN_GPUS + ROLLOUT_GPUS ))
  python3 - "$EXPECTED_GPUS" "$TS_STAGE_DIR" <<'PY'
import os
import shutil
import subprocess
import sys
import time
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

want_gpus = int(sys.argv[1])
stage_dir = sys.argv[2]
ray.init(address="auto", ignore_reinit_error=True)

deadline = time.time() + 300
while time.time() < deadline:
    alive = [n for n in ray.nodes() if n.get("Alive")]
    have_gpus = int(sum(n.get("Resources", {}).get("GPU", 0) for n in alive))
    if have_gpus >= want_gpus and len(alive) >= 3:
        break
    print(f"feature-timeslice: waiting for Ray nodes (have {len(alive)} nodes, {have_gpus}/{want_gpus} GPUs)...", flush=True)
    time.sleep(5)
else:
    raise SystemExit(f"feature-timeslice: timed out waiting for {want_gpus} GPUs in RayCluster")

@ray.remote(num_cpus=0)
def _patch_node(src_dir: str):
    import socket
    if not os.path.isdir("/opt/verl"):
        return f"{socket.gethostname()}: no /opt/verl"
    shutil.copy2(os.path.join(src_dir, "timeslice.py"), "/opt/verl/timeslice.py")
    shutil.copy2(os.path.join(src_dir, "timeslice.py"), "/opt/verl/verl/utils/timeslice.py")
    subprocess.run(["git", "config", "--global", "--add", "safe.directory", "/opt/verl"], check=False)
    subprocess.run(["git", "-C", "/opt/verl", "checkout", "--", "."], check=True)
    if os.path.exists("/opt/verl/verl/utils/gpu_mem_log.py"):
        os.remove("/opt/verl/verl/utils/gpu_mem_log.py")
    patch_file = os.path.join(src_dir, "verl-timeslice.patch")
    subprocess.run(["git", "-C", "/opt/verl", "apply", "--check", patch_file], check=True)
    subprocess.run(["git", "-C", "/opt/verl", "apply", patch_file], check=True)
    return f"{socket.gethostname()}: patched /opt/verl OK"

alive = [n for n in ray.nodes() if n.get("Alive")]
refs = [
    _patch_node.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
    ).remote(stage_dir)
    for n in alive
]
for res in ray.get(refs):
    print(f"feature-timeslice: {res}", flush=True)
ray.shutdown()
PY
fi
