# `--feature timeslice` — Multi-GPU Trainer & Sampler Time-Slicing Integration

Integrates [`llm-d-rl-time-slicing`](https://github.com/llm-d-incubation/llm-d-rl-time-slicing) into the `verl-qwen-30b-swe` benchmark setup so multiple concurrent RL jobs (`job1`, `job2`) can time-slice multi-GPU Trainer (`>= 2` GPUs, e.g. 8x H200 Megatron-LM TP2·EP4) and/or multi-GPU Sampler (`>= 2` GPUs, e.g. 2x H200 standalone vLLM TP2) pools with exclusive GPU VRAM access during active slices.

## Usage

### 1. Feature OFF (Baseline)
```bash
rlbench run setups/verl-qwen-30b-swe \
  --config setups/verl-qwen-30b-swe/config/h200-t8-s2 \
  --var STEPS=2 --var VAL_TASKS=0
```
When `--feature timeslice` is omitted, `TIMESLICE_ENABLED` is unset and the baseline benchmark executes with zero lock or offload overhead.

### 2. Feature ON — Dual-Pool Time-Slicing (Both Trainer & Sampler, Default)
```bash
# Launch Job 1
rlbench run setups/verl-qwen-30b-swe \
  --config setups/verl-qwen-30b-swe/config/h200-t8-s2 \
  --feature timeslice \
  --var JOB_ID=job1 --var STEPS=2 --var VAL_TASKS=0

# Launch Job 2 concurrently on the shared GPU pools
rlbench run setups/verl-qwen-30b-swe \
  --config setups/verl-qwen-30b-swe/config/h200-t8-s2 \
  --feature timeslice \
  --var JOB_ID=job2 --var STEPS=2 --var VAL_TASKS=0
```

### 3. Toggling Trainer-Only or Sampler-Only Time-Slicing

You can independently toggle whether the **trainers** and/or **samplers** participate in time-slicing lock orchestration and explicit `verl` GPU offload/reload calls:

| Variable | Default (`--feature timeslice`) | Description |
| --- | --- | --- |
| `TIMESLICE_ENABLED` | `1` | Master switch for the `timeslice` feature |
| `TIMESLICE_TRAINER_ENABLED` | `1` | Acquires/releases the `"trainers"` orchestrator lock around training & weight sync |
| `TIMESLICE_SAMPLER_ENABLED` | `1` | Acquires/releases the `"samplers"` orchestrator lock around rollout generation & weight sync |
| `VERL_TRAINER_POST_SYNC_OFFLOAD` | `${TIMESLICE_TRAINER_ENABLED}` | Explicit `engine.to("cpu", point="post_sync")` + `empty_cache` in `ActorRolloutRefWorker.update_weights()` after NCCL weight push |
| `VERL_SAMPLER_SLEEP_OFFLOAD` | `${TIMESLICE_SAMPLER_ENABLED}` | Explicit `vLLM.sleep(level=1)` and `vLLM.wake_up(tags=["weights", "kv_cache"])` in standalone `vLLMHttpServer` and `PPOTrainerSeparateAsync` |
| `TIMESLICE_JOB_ID` | `${JOB_ID:-job1}` | Job identifier for orchestrator RPCs and `[timeslice]` / `[gpu-mem]` telemetry |

- **Trainer-Only Time-Slicing** (samplers stay awake and unshared):
  ```bash
  rlbench run setups/verl-qwen-30b-swe \
    --config setups/verl-qwen-30b-swe/config/h200-t8-s2 \
    --feature timeslice \
    --var JOB_ID=job1 --var TIMESLICE_SAMPLER_ENABLED=0
  ```
- **Sampler-Only Time-Slicing** (trainers stay unshared):
  ```bash
  rlbench run setups/verl-qwen-30b-swe \
    --config setups/verl-qwen-30b-swe/config/h200-t8-s2 \
    --feature timeslice \
    --var JOB_ID=job1 --var TIMESLICE_TRAINER_ENABLED=0
  ```

## Feature Components

- `vars.env`: Default render-time variables (`TIMESLICE_ENABLED=1`, `TIMESLICE_TRAINER_ENABLED=1`, `TIMESLICE_SAMPLER_ENABLED=1`, `VERL_TRAINER_POST_SYNC_OFFLOAD=1`, `VERL_SAMPLER_SLEEP_OFFLOAD=1`, `TIMESLICE_ORCHESTRATOR_ADDR=timeslice-timesliceorchestrator.timeslice-system.svc.cluster.local:50051`, `TIMESLICE_TRAINER_GROUP=trainers`, `TIMESLICE_SAMPLER_GROUP=samplers`, `TIMESLICE_MODE=hybrid`, `JOB_ID=job1`, `TIMESLICE_JOB_ID=job1`, `ROLLOUT_GPU_MEM_UTIL=0.80`, `NO_HYBRID_ROLLOUT=1`, plus per-job DRA `ResourceClaim` bindings).
- `config/feature-timeslice.sh`: Sourced automatically by `config/common.sh` when `--feature timeslice` is active. Exports `TIMESLICE_*` and `VERL_*_OFFLOAD` variables, sets `NCCL_NVLS_ENABLE=0`, forwards Ray `runtime_env.env_vars`, and hot-patches `/opt/verl` across all RayCluster nodes prior to training start.
- `setup/30-resource-claims.yaml`: Shared DRA `ResourceClaim` manifests (`shared-trainers-gpu-claim` and `shared-samplers-gpu-claim`) carrying `rlbench.timeslice.io/group` labels so multiple concurrent jobs can co-schedule onto the same physical multi-GPU nodes.
- `hooks/pre-setup.sh` & `hooks/post-run.sh`: Stage `timeslice.py` and `verl-timeslice.patch` into the run config, yield stale orchestrator locks (`rlts orchestrator yield`) for enabled role groups before and after runs, and collect `orchestrator.log` and `snapshot-agent.log` into `${RUN_FOLDER}/logs/`.
- `timeslice.py`: Python module implementing `TimeSliceOrchestratorClient`, `DualPoolRoleLocks` (`RoleLocks`) with role-level `trainer_enabled` / `sampler_enabled` gating and global lock order **`TRAINER` before `SAMPLER`**, multi-GPU Megatron trainer offload/restore (`MegatronEngine.to("cpu")` + `point="post_sync"` + `aggressive_empty_cache`), multi-GPU vLLM sampler request draining + `sleep(level=1)` / `wake_up(tags=["weights", "kv_cache"])`, `cuda-checkpoint` + `universal_cr_shim_v2.c` (`SIG35`/`SIG36`) helpers, and `[timeslice]` / `[gpu-mem]` structured telemetry formatters.
- `verl-timeslice.patch`: Unified diff patch against `/opt/verl` wiring disaggregated 3-phase dual-pool locking, trainer post-sync offload, standalone sampler sleep/wake, and `[gpu-mem]` / `[timeslice]` telemetry into `PPOTrainerSeparateAsync`, `MegatronEngine`, `ActorRolloutRefWorker`, and `vLLMHttpServer`.
- `test_timeslice.py`: Unit test suite for the `timeslice` feature module.

## GPU Lock Acquire/Release & Offload/Restore Execution Flow

When `--feature timeslice` is active (`TIMESLICE_ENABLED=1`), GPU locks are managed via `DualPoolRoleLocks` (`timeslice.py`) across the `"trainers"` and `"samplers"` lock groups (`TRAINER` before `SAMPLER`) in `PPOTrainerSeparateAsync` (`verl-timeslice.patch`):

1. **Pre-Setup / Post-Run Hooks (`hooks/pre-setup.sh`, `hooks/post-run.sh`)**: Force-release any leftover locks on startup and teardown for enabled role groups via `rlts orchestrator yield`.
2. **Job Initialization (`PPOTrainerSeparateAsync._setup()` & `on_init_end()`)**:
   - `_setup()` **acquires `"trainer"`** (if `TIMESLICE_TRAINER_ENABLED=1`), initializes the Megatron actor workers (`init:to_cpu` offloads initial model shards to pinned CPU memory), and then **acquires `"sampler"`** (if `TIMESLICE_SAMPLER_ENABLED=1`) before launching the standalone vLLM sampler replicas.
   - `on_init_end()` runs the initial NCCL weight sync (`update_weights`), which offloads the reloaded trainer export buffer via `post_sync:to_cpu` (`VERL_TRAINER_POST_SYNC_OFFLOAD=1`), and **releases `"trainer"`** while retaining `"sampler"` for Step 1 rollout generation.
3. **Per-Step 3-Phase Interleaving (`PPOTrainerSeparateAsync.step()` & `on_step_end()`)**:
   - **Phase 1 — Sampling (`step()`)**: Holding only `"sampler"` (with `"trainer"` released so another job can train simultaneously), queues the step's rollout batch and runs `replay_buffer.sample()`. Once the batch finishes, if `VERL_SAMPLER_SLEEP_OFFLOAD=1`, drains in-flight requests (`abort_replicas()`) and calls `sleep_replicas()` (`vLLM.sleep(level=1)`, freeing ~107.4 GiB/GPU), then **releases `"sampler"`**.
   - **Phase 2 — Training (`step()`)**: **Acquires `"trainer"`** (with `"sampler"` released so another job can sample simultaneously) and executes `_compute_old_log_prob`, `_compute_advantage`, and `_update_actor` (`MegatronEngine` reloads to GPU for forward/backward/step and offloads back to CPU via `eval_end:to_cpu` and `update_actor` `init:to_cpu`).
   - **Phase 3 — Weight Sync (`on_step_end()`)**: While still holding `"trainer"`, **acquires `"sampler"`** (preserving `TRAINER -> SAMPLER` global lock order), wakes standalone vLLM (`wake_up_replicas()` -> `wake_up(tags=["weights", "kv_cache"])` + `reset_prefix_cache`), pushes updated weights over NCCL (`update_weights()`), offloads the trainer export buffer (`post_sync:to_cpu`), and **releases `"trainer"`** while keeping `"sampler"` held directly into the next step's Phase 1 (or sleeps vLLM and releases both locks on the final step).
