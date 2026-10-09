# Multi-GPU Dual-Pool Time-Slicing Benchmark & Restore Stability Report (`h200-t8-s2`)

## 1. Executive Summary & Provenance

This report documents the empirical performance, GPU VRAM reclamation, numerical convergence parity, and multi-GPU checkpoint/offload and restore stability of the `llm-d-rl-time-slicing` orchestrator integrated into `rl-lab` (`benchmark/setups/verl-qwen-30b-swe/features/timeslice/`).

All benchmark runs were executed via `rlbench run` using the `verl-qwen-30b-swe` setup (`config/h200-t8-s2`: 8x H200 Megatron trainer `TP=2, EP=4` + 2x H200 standalone vLLM sampler `ROLLOUT_TP=2`) on NVIDIA H200 SXM (141 GiB HBM3e):
- **Feature OFF (Baseline):** `20261001-155249-disagg-memlog`
- **Feature ON (Concurrent Job 1):** `20261008-203130-h200-t8-s2-timeslice-job1`
- **Feature ON (Concurrent Job 2):** `20261008-204753-h200-t8-s2-timeslice-job2`

All summary metrics in `summary_metrics.json` and the tables below are generated from the `verl` run logs by `experiments/timeslicing-multi-gpu/report.py`.

### 1.1 Hardware & Multi-GPU Topology Configuration (`h200-t8-s2`)

| Parameter | Feature OFF (`baseline`) | Feature ON (`2x Concurrent Time-Slicing`) |
| --- | --- | --- |
| **Benchmark Setup** | `verl-qwen-30b-swe` (`h200-t8-s2`) | `verl-qwen-30b-swe` (`h200-t8-s2`) |
| **GPU Hardware** | NVIDIA H200 SXM (`141.00 GiB` / `143771 MiB` HBM3e) | NVIDIA H200 SXM (`141.00 GiB` / `143771 MiB` HBM3e) |
| **Concurrent RL Jobs** | `1` (`baseline`) | `2` (`job1`, `job2` co-resident on the same trainer & sampler nodes) |
| **Trainer Multi-GPU Topology** | `8` H200 GPUs (`world_size=8`, Megatron `TP=2, EP=4`) | `8` H200 GPUs (`world_size=8`, Megatron `TP=2, EP=4`) |
| **Sampler Multi-GPU Topology** | `2` H200 GPUs (`world_size=2`, standalone vLLM `ROLLOUT_TP=2`) | `2` H200 GPUs (`world_size=2`, standalone vLLM `ROLLOUT_TP=2`) |
| **Sampler Memory Utilization** | `ROLLOUT_GPU_MEM_UTIL=0.80` | `ROLLOUT_GPU_MEM_UTIL=0.80` |
| **Orchestrator Lock Groups** | None (`TIMESLICE_ENABLED=0`) | Dual-Pool: `trainers` & `samplers` (`TRAINER` before `SAMPLER`) |

### 1.2 Benchmark Runs (`h200-t8-s2`)

| Run | `--feature` | `JOB_ID` | Regime & Description |
| --- | --- | --- | --- |
| `20261001-155249-disagg-memlog` | — | `baseline` | Feature OFF baseline: single multi-GPU `verl` RL job (`8` Trainer H200 GPUs `TP=2, EP=4` + `2` Sampler H200 GPUs `ROLLOUT_TP=2`) without time-slicing lock orchestration |
| `20261008-203130-h200-t8-s2-timeslice-job1` | `timeslice` | `job1` | Feature ON concurrent `verl` Job 1 sharing the `8`-GPU `trainers` pool and `2`-GPU `samplers` pool via `llm-d-rl-time-slicing` (`outcome: Complete`) |
| `20261008-204753-h200-t8-s2-timeslice-job2` | `timeslice` | `job2` | Feature ON concurrent `verl` Job 2 sharing the `8`-GPU `trainers` pool and `2`-GPU `samplers` pool via `llm-d-rl-time-slicing` (`outcome: Complete`) |

### 1.3 GPU Lock Acquire/Release & Offload/Restore Execution Flow (`--feature timeslice`)

When `--feature timeslice` is enabled (`TIMESLICE_ENABLED=1` via `benchmark/setups/verl-qwen-30b-swe/features/timeslice/config/feature-timeslice.sh`), GPU locks and explicit `verl` offload/reload calls are controlled by:
- `TIMESLICE_TRAINER_ENABLED` (default `1`): gates `"trainers"` lock acquire/release.
- `TIMESLICE_SAMPLER_ENABLED` (default `1`): gates `"samplers"` lock acquire/release.
- `VERL_TRAINER_POST_SYNC_OFFLOAD` (default `${TIMESLICE_TRAINER_ENABLED}`): gates explicit `post_sync:to_cpu` offload after NCCL weight sync.
- `VERL_SAMPLER_SLEEP_OFFLOAD` (default `${TIMESLICE_SAMPLER_ENABLED}`): gates explicit standalone `vLLM.sleep(level=1)` and `vLLM.wake_up(tags=["weights", "kv_cache"])`.

Execution proceeds through three pipelined phases per step in `PPOTrainerSeparateAsync` (`benchmark/setups/verl-qwen-30b-swe/features/timeslice/verl-timeslice.patch`):

1. **Initialization (`PPOTrainerSeparateAsync._setup()` & `on_init_end()`)**:
   - **Acquires `"trainer"`**, builds the 8-GPU Megatron actor worker group (offloading initial model shards to pinned host memory via `init:to_cpu`), and **acquires `"sampler"`** before creating the 2-GPU standalone vLLM server.
   - In `on_init_end()`, syncs initial weights to vLLM (`update_weights()`), offloads the reloaded trainer parameter export buffer back to CPU (`post_sync:to_cpu`), and **releases `"trainer"`** while retaining `"sampler"` for Step 1 generation.
2. **Phase 1 — Rollout Generation (`PPOTrainerSeparateAsync.step()`, holding only `"sampler"`)**:
   - With `"trainer"` released (allowing a peer job to initialize or train on the 8 trainer GPUs concurrently), generates the step's rollout batch via `replay_buffer.sample()`.
   - Immediately upon completion, drains in-flight requests (`abort_replicas()`), sleeps the standalone vLLM server (`sleep_replicas()` -> `vLLM.sleep(level=1)`, freeing `107.40 GiB/GPU`), and **releases `"sampler"`**.
3. **Phase 2 — Training (`PPOTrainerSeparateAsync.step()`, holding only `"trainer"`)**:
   - **Acquires `"trainer"`** (with `"sampler"` released so a peer job can run rollout generation or weight sync concurrently) and runs `_compute_old_log_prob`, `_compute_advantage`, and `_update_actor`.
   - `MegatronEngine.to()` restores parameters to GPU at phase entry and offloads parameters/gradients/optimizer states back to CPU at phase exit (`eval_end:to_cpu`, `update_actor` `init:to_cpu`, freeing `43.09 GiB/GPU`).
4. **Phase 3 — Weight Sync (`PPOTrainerSeparateAsync.on_step_end()`, holding `"trainer"` then `"sampler"`)**:
   - While holding `"trainer"`, **acquires `"sampler"`** (strictly respecting `TRAINER -> SAMPLER` global lock order), wakes the standalone vLLM server (`wake_up_replicas()` -> `vLLM.wake_up(tags=["weights", "kv_cache"])` + `reset_prefix_cache`), pushes updated weights over NCCL (`update_weights()`), offloads the trainer export buffer (`post_sync:to_cpu`, freeing `9.45 GiB/GPU`), and **releases `"trainer"`** while keeping `"sampler"` held directly into the next step's Phase 1 (or sleeps vLLM and releases both locks on the final step).

---

## 2. Per-Step Timing Breakdown (Feature OFF vs. Feature ON 2x Concurrent `verl`)

| Mode / Job | Steps | Mean Rollout (`gen_s`) | Mean Old LogProb (`s`) | Mean Update Actor (`s`) | Mean Weight Sync (`s`) | Mean Sampler Wait (`ms`) | Mean Trainer Wait (`ms`) | Mean Step Total (`s`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Feature OFF (`baseline`) | 2 | 594.60 | 122.36 | 307.17 | 17.12 | 0.00 | 0.00 | 1042.69 |
| Feature ON (`job1`) | 2 | 286.64 | 27.34 | 50.77 | 11.98 | 158333.33 | 118667.00 | 801.50 |
| Feature ON (`job2`) | 2 | 300.64 | 41.14 | 53.79 | 12.27 | 56667.00 | 10000.33 | 516.87 |

### 2.1 Per-Step Granular Breakdown

- **Feature OFF (`baseline` — run `20261001-155249-disagg-memlog`)**:
  - Step 1: `gen_s=677.70s`, `old_log_prob_s=123.56s`, `update_actor_s=297.54s`, `update_weights_s=16.97s`, `step_s=1117.23s`
  - Step 2: `gen_s=511.50s`, `old_log_prob_s=121.15s`, `update_actor_s=316.79s`, `update_weights_s=17.28s`, `step_s=968.16s`
- **Feature ON (`job1` — run `20261008-203130-h200-t8-s2-timeslice-job1`, `TS_TRAIN_BATCH=64, TS_ROLLOUT_N=4, TS_SESSIONS=64`)**:
  - Step 1: `gen_s=302.70s`, `old_log_prob_s=33.64s`, `update_actor_s=61.39s`, `update_weights_s=12.14s`, `step_s=1028.29s`
  - Step 2: `gen_s=270.59s`, `old_log_prob_s=21.05s`, `update_actor_s=40.14s`, `update_weights_s=11.82s`, `step_s=574.72s`
- **Feature ON (`job2` — run `20261008-204753-h200-t8-s2-timeslice-job2`, `TS_TRAIN_BATCH=64, TS_ROLLOUT_N=4, TS_SESSIONS=64`)**:
  - Step 1: `gen_s=328.70s`, `old_log_prob_s=35.79s`, `update_actor_s=68.29s`, `update_weights_s=12.50s`, `step_s=643.30s`
  - Step 2: `gen_s=272.59s`, `old_log_prob_s=46.49s`, `update_actor_s=39.28s`, `update_weights_s=12.04s`, `step_s=390.44s`

### 2.2 Observed Concurrent Interleaving Timeline (`job1` & `job2`, UTC)

| UTC Window | `trainers` Pool (`8x H200`) | `samplers` Pool (`2x H200`) | Interleaving / Pipelining State |
| --- | --- | --- | --- |
| `20:32:46` – `20:44:26` | `job1` `init` (`init:to_cpu`) + initial `update_weights` (`post_sync:to_cpu`, releases `trainers` at `20:44:26`) | `job1` `init` (acquires `samplers` at `20:40:26`) + initial `update_weights` | `job1` initializes both pools and releases `trainers` after initial weight sync |
| `20:44:26` – `20:49:57` | **`job2` `init`** (acquires `trainers` at `20:49:11`, `init:to_cpu` at `20:51:04`) | **`job1` Step 1 `gen`** (`302.70s`; sleeps vLLM `-107.48 GB/GPU` and releases `samplers` at `20:49:57`) | **Simultaneous:** `job1` samples on `samplers` while `job2` initializes on `trainers` |
| `20:51:06` – `20:55:37` | `job2` initial `update_weights` (`post_sync:to_cpu`, releases `trainers` at `20:55:37`) | `job2` `init` (acquires `samplers` at `20:51:06`) + initial `update_weights` | `job2` finishes init weight sync and releases `trainers` (`pending_waiters=1`) |
| `20:55:37` – `21:01:35` | **`job1` Step 1 Training** (`old_log_prob` + `update_actor`, `20:55:49`–`20:57:09`) | **`job2` Step 1 `gen`** (`328.70s`; sleeps vLLM `-107.48 GB/GPU` and releases `samplers` at `21:01:35`) | **Simultaneous Dual-Pool Pipelining:** `job1` trains on `trainers` while `job2` samples on `samplers` |
| `21:01:35` – `21:01:48` | `job1` Step 1 `update_weights` (`post_sync:to_cpu`, releases `trainers` at `21:01:48`) | `job1` wakes vLLM (`1452.2 ms`, `+107.18 GB/GPU`) + Step 1 `update_weights` | `job1` syncs Step 1 weights in `12.14s` and releases `trainers` |
| `21:01:49` – `21:06:21` | **`job2` Step 1 Training** (`old_log_prob` + `update_actor`, `21:02:00`–`21:03:32`) | **`job1` Step 2 `gen`** (`270.59s`; sleeps vLLM `1157.1 ms`, `-107.61 GB/GPU` and releases `samplers` at `21:06:21`) | **Simultaneous Dual-Pool Pipelining:** `job2` trains on `trainers` while `job1` samples on `samplers` |
| `21:06:21` – `21:06:35` | `job2` Step 1 `update_weights` (`post_sync:to_cpu`, releases `trainers` at `21:06:35`) | `job2` wakes vLLM (`1364.7 ms`, `+107.18 GB/GPU`) + Step 1 `update_weights` | `job2` syncs Step 1 weights in `12.50s` and releases `trainers` |
| `21:06:36` – `21:11:13` | **`job1` Step 2 Training** (`old_log_prob` + `update_actor`, `21:06:47`–`21:07:36`) | **`job2` Step 2 `gen`** (`272.59s`; sleeps vLLM `1142.6 ms`, `-107.18 GB/GPU` and releases `samplers` at `21:11:13`) | **Simultaneous Dual-Pool Pipelining:** `job1` trains on `trainers` while `job2` samples on `samplers` |
| `21:11:14` – `21:11:29` | `job1` Step 2 `update_weights` (releases `trainers` at `21:11:29`) | `job1` wakes vLLM (`1397.5 ms`), Step 2 `update_weights` (`11.82s`), sleeps vLLM (`1148.0 ms`), releases `samplers` | `job1` completes Step 2 and exits (`outcome: Complete`) |
| `21:11:29` – `21:13:06` | `job2` Step 2 Training (`21:12:01`–`21:12:50`) + Step 2 `update_weights` (releases `trainers` at `21:13:06`) | `job2` wakes vLLM (`1307.4 ms`), Step 2 `update_weights` (`12.04s`), sleeps vLLM (`1142.6 ms`), releases `samplers` | `job2` completes Step 2 and exits (`outcome: Complete`) |

---

## 3. Separate Trainer & Sampler Offload/Sleep, Wake/Restore, and Lock Wait Latencies

| Job / Role | Lock Group | Acquire Count | Release Count | Mean Lock Wait (`waited_ms`) | Max Lock Wait (`ms`) | Mean Offload / Sleep (`ms`) | Mean Restore / Wake (`ms`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `job1` — Sampler (`ROLLOUT_TP=2`) | `samplers` | 3 | 3 | `158333.33 ms` | `262000.0 ms` | `5540.70 ms` (`14316.7 ms` cold / `1152.6 ms` warm) | `1424.85 ms` |
| `job1` — Trainer (`TP=2, EP=4`) | `trainers` | 3 | 3 | `118667.00 ms` | `340000.0 ms` | `1779.58 ms` | `1208.41 ms` |
| `job2` — Sampler (`ROLLOUT_TP=2`) | `samplers` | 3 | 3 | `56667.00 ms` | `168000.0 ms` | `5492.53 ms` (`14184.0 ms` cold / `1146.8 ms` warm) | `1336.05 ms` |
| `job2` — Trainer (`TP=2, EP=4`) | `trainers` | 3 | 3 | `10000.33 ms` | `16000.0 ms` | `1692.61 ms` | `1103.42 ms` |

---

## 4. Multi-GPU VRAM Utilization & Exclusivity Telemetry (`[gpu-mem]`)

| Role | World Size | Pre-Offload (`GiB`) | Post-Offload (`GiB`) | Post-Restore (`GiB`) | Freed per GPU (`GiB`) | Post-Sync Residual (`GiB`) |
| --- | --- | --- | --- | --- | --- | --- |
| Trainer (`TP=2, EP=4`) | 8 | 59.00 | 15.91 | 26.32 | 43.09 | 16.57 (`-9.45 GiB` `post_sync`) |
| Sampler (`ROLLOUT_TP=2`) | 2 | 120.27 | 12.87 | 120.69 | 107.40 | N/A |

- **Trainer VRAM Exclusivity (`8x H200`, `TP=2, EP=4`)**:
  - Active Megatron training (`update_actor`) reaches `59.00 GiB/GPU` device-wide (`65.07 GiB/GPU` peak with both jobs' CUDA/NCCL contexts co-resident) and offloads parameters and gradients to pinned host memory (`gpu_allocated_gb=1.067 GiB`, `gpu_device_used_gb=15.91 GiB` across both jobs), freeing **`43.09 GiB/GPU`** in `1.74s`.
  - Immediately after `checkpoint_engine.send_weights()` during `update_weights`, `ActorRolloutRefWorker` executes `self.actor.engine.to("cpu", model=True, optimizer=False, grad=False, point="post_sync")` + `aggressive_empty_cache(force_sync=True)`, freeing the **`9.45 GiB/GPU`** (`8.80 GB` bf16 parameter shard) reloaded by `get_per_tensor_param()` in `0.72s`–`0.87s` before yielding `"trainers"`.
- **Sampler VRAM Exclusivity (`2x H200`, `ROLLOUT_TP=2`, `gpu_memory_utilization=0.80`)**:
  - Active standalone vLLM replicas occupy `120.27 GiB/GPU` (`30.50 GB` weights + `84.20 GB` KV cache pool).
  - Calling `vLLM.sleep(level=1)` copies weights to pinned host memory and discards KV cache HBM pages, dropping device-wide VRAM to `12.87 GiB/GPU` (`7.72 GiB/GPU` for a single job's sleeping CUDA/NCCL context) and freeing **`107.40 GiB/GPU`**. First-time sleep takes `14.18s`–`14.32s` (pinned host buffer allocation); warm steady-state sleep takes **`1.14s`–`1.16s`**, and `wake_up(tags=["weights", "kv_cache"])` + `reset_prefix_cache` restores `107.18 GiB/GPU` in **`1.31s`–`1.45s`**.

---

## 5. Numerical & Training Correctness Verification (Loss & Reward Trajectory Parity)

| Step | Baseline `pg_loss` | `job1` `pg_loss` | `job2` `pg_loss` | Baseline `reward_mean` | `job1` `reward_mean` | `job2` `reward_mean` | Baseline `grad_norm` | `job1` `grad_norm` | `job2` `grad_norm` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1` | `0.00599521` | `0.01540362` | `0.00695509` | `0.148438` | `0.041667` | `0.156250` | `0.072040` | `0.115074` | `0.074412` |
| `2` | `0.00228850` | `0.02484338` | `-0.00151645` | `0.015625` | `0.062500` | `0.062500` | `0.011684` | `0.113671` | `0.070806` |

- **Numerical Convergence Parity Verified (`numerical_parity_verified = true`)**:
  - `max_abs_pg_loss_diff = 0.026360` (`<= 0.05` tolerance)
  - `max_abs_reward_diff = 0.114583` (`<= 0.15` tolerance)
  - Zero `NaN` or `Inf` values across all multi-GPU offload/restore and sleep/wake cycles on both `job1` and `job2`.

---

## 6. Side-by-Side Multi-GPU Restore Stability Root-Cause Analysis (Trainer vs. Sampler)

| Subsystem & Mechanism | Topology | Preconditions / Mitigations | Mean Offload / Sleep (`ms`) | Mean Restore / Wake (`ms`) | Freed / GPU (`GiB`) | Residual Floor (`GiB`) | Stability Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Trainer Cooperative Offload (`MegatronEngine.to` + `post_sync`)** | `world_size=8` (`TP=2, EP=4`) | `param_offload=True, grad_offload=True, optimizer_offload=True` + explicit `post_sync:to_cpu` + `RLBENCH_NO_HYBRID_ROLLOUT=1` | `1736.09 ms` (`795 ms` `post_sync`) | `1155.91 ms` | `43.09 GiB` (`9.45 GiB` `post_sync`) | `15.91 GiB` (2 jobs co-resident) | **Stable**: 100% success across all 8 ranks and all steps on `job1` & `job2` |
| **Sampler Cooperative Sleep (`vLLMHttpServer.sleep(level=1)`)** | `world_size=2` (`ROLLOUT_TP=2`) | Skip warmup pre-queuing + `abort_replicas()` request drain + `sleep(level=1)` + `wake_up(tags=["weights","kv_cache"])` + `reset_prefix_cache` | `1149.75 ms` warm (`14250.4 ms` cold 1st sleep) | `1380.45 ms` | `107.40 GiB` | `12.87 GiB` (2 jobs co-resident) | **Stable**: 100% success across both TP ranks with in-place NCCL weight sync |

### 6.1 Verbatim Sanitized Telemetry Excerpts (`20261008-203130-h200-t8-s2-timeslice-job1` & `20261008-204753-h200-t8-s2-timeslice-job2`)

```text
# Dual-Pool Lock Interleaving ([timeslice] logs from job1 and job2):
[timeslice] ts=2026-10-08T20:32:46Z job=job1 role=trainer action=ACQUIRE group=trainers waited_ms=0.0 context_restored=false step=0
[timeslice] ts=2026-10-08T20:40:26Z job=job1 role=sampler action=ACQUIRE group=samplers waited_ms=0.0 context_restored=false step=0
[timeslice] ts=2026-10-08T20:44:26Z job=job1 role=trainer action=RELEASE group=trainers hold_ms=700201.3 pending_waiters=0 snapshot_deferred=true step=0
[timeslice] ts=2026-10-08T20:49:11Z job=job2 role=trainer action=ACQUIRE group=trainers waited_ms=0.0 context_restored=true step=0
[timeslice] ts=2026-10-08T20:49:57Z job=job1 role=sampler action=OFFLOAD group=samplers mode=app duration_ms=14316.7 freed_gb_per_gpu=107.48 status=ok rank=0 point=replica0:sleep_level1
[timeslice] ts=2026-10-08T20:49:57Z job=job1 role=sampler action=RELEASE group=samplers hold_ms=571284.4 pending_waiters=0 snapshot_deferred=true step=1
[timeslice] ts=2026-10-08T20:51:06Z job=job2 role=sampler action=ACQUIRE group=samplers waited_ms=0.0 context_restored=true step=0
[timeslice] ts=2026-10-08T20:55:37Z job=job2 role=trainer action=RELEASE group=trainers hold_ms=385704.9 pending_waiters=1 snapshot_deferred=false step=0
[timeslice] ts=2026-10-08T20:55:37Z job=job1 role=trainer action=ACQUIRE group=trainers waited_ms=340000.0 context_restored=true step=1
[timeslice] ts=2026-10-08T21:01:35Z job=job2 role=sampler action=OFFLOAD group=samplers mode=app duration_ms=14184.0 freed_gb_per_gpu=107.48 status=ok rank=0 point=replica0:sleep_level1
[timeslice] ts=2026-10-08T21:01:35Z job=job2 role=sampler action=RELEASE group=samplers hold_ms=628496.5 pending_waiters=1 snapshot_deferred=false step=1
[timeslice] ts=2026-10-08T21:01:35Z job=job1 role=sampler action=ACQUIRE group=samplers waited_ms=262000.0 context_restored=true step=1
[timeslice] ts=2026-10-08T21:01:37Z job=job1 role=sampler action=RESTORE group=samplers mode=app duration_ms=1452.2 restored_gb_per_gpu=107.18 status=ok rank=0 point=replica0:wake_up
[timeslice] ts=2026-10-08T21:01:48Z job=job1 role=trainer action=OFFLOAD group=trainers mode=app duration_ms=717.0 freed_gb_per_gpu=9.45 status=ok rank=0 point=post_sync:to_cpu
[timeslice] ts=2026-10-08T21:01:48Z job=job1 role=trainer action=RELEASE group=trainers hold_ms=371072.8 pending_waiters=1 snapshot_deferred=false step=1
[timeslice] ts=2026-10-08T21:01:49Z job=job2 role=trainer action=ACQUIRE group=trainers waited_ms=14001.0 context_restored=true step=1
[timeslice] ts=2026-10-08T21:06:21Z job=job1 role=sampler action=OFFLOAD group=samplers mode=app duration_ms=1157.1 freed_gb_per_gpu=107.61 status=ok rank=0 point=replica0:sleep_level1
[timeslice] ts=2026-10-08T21:06:21Z job=job1 role=sampler action=RELEASE group=samplers hold_ms=285616.5 pending_waiters=1 snapshot_deferred=false step=2
[timeslice] ts=2026-10-08T21:06:21Z job=job2 role=sampler action=ACQUIRE group=samplers waited_ms=168000.0 context_restored=true step=1
[timeslice] ts=2026-10-08T21:06:23Z job=job2 role=sampler action=RESTORE group=samplers mode=app duration_ms=1364.7 restored_gb_per_gpu=107.18 status=ok rank=0 point=replica0:wake_up
[timeslice] ts=2026-10-08T21:06:35Z job=job2 role=trainer action=RELEASE group=trainers hold_ms=286182.9 pending_waiters=1 snapshot_deferred=false step=1
[timeslice] ts=2026-10-08T21:06:36Z job=job1 role=trainer action=ACQUIRE group=trainers waited_ms=15001.0 context_restored=true step=2

# Multi-GPU Memory Telemetry ([gpu-mem] logs from job1 and job2):
[gpu-mem] job_id=job1 role=sampler rank=0 host=verl-job1-rollout-worker-0 pid=2785 point=replica0:timeslice_sleep event=before gpu_allocated_gb=114.900 gpu_reserved_gb=115.635 gpu_device_used_gb=120.041 gpu_device_total_gb=139.809 weights_bytes=30503179264 kv_cache_bytes=84204847104
[gpu-mem] job_id=job1 role=sampler rank=0 host=verl-job1-rollout-worker-0 pid=2785 point=replica0:timeslice_sleep event=after_sleep gpu_allocated_gb=0.200 gpu_reserved_gb=115.635 gpu_device_used_gb=12.428 gpu_device_total_gb=139.809 seconds=1.157 weights_bytes=30503179264 kv_cache_bytes=84204847104
[gpu-mem] job_id=job2 role=sampler rank=0 host=verl-job2-rollout-worker-0 pid=2779 point=replica0:timeslice_wake event=after_wake gpu_allocated_gb=114.900 gpu_reserved_gb=115.635 gpu_device_used_gb=120.463 gpu_device_total_gb=139.809 seconds=1.365 weights_bytes=30503179264 kv_cache_bytes=84204847104
[gpu-mem] job_id=job1 role=trainer rank=0 host=verl-job1-trainer-worker-0 pid=2653 point=init:to_cpu event=before gpu_allocated_gb=44.913 gpu_reserved_gb=59.170 gpu_device_used_gb=65.073 gpu_device_total_gb=139.809 model=1 optimizer=1 grad=1 gpu_resident_params=8804426240 gpu_resident_grads=8804426240
[gpu-mem] job_id=job1 role=trainer rank=0 host=verl-job1-trainer-worker-0 pid=2653 point=init:to_cpu event=after gpu_allocated_gb=1.067 gpu_reserved_gb=2.990 gpu_device_used_gb=17.916 gpu_device_total_gb=139.809 seconds=1.969 model=1 optimizer=1 grad=1 bytes_copied_to_host=8804426240 bytes_discarded=8804426240 gpu_resident_params=0 gpu_resident_grads=0
[gpu-mem] job_id=job1 role=trainer rank=0 host=verl-job1-trainer-worker-0 pid=2653 point=post_sync:to_cpu event=after gpu_allocated_gb=1.067 gpu_reserved_gb=1.850 gpu_device_used_gb=16.571 gpu_device_total_gb=139.809 seconds=0.717 model=1 optimizer=0 grad=0 bytes_copied_to_host=8804426240 gpu_resident_params=0 gpu_resident_grads=0
```

---

## 7. Reproducing the Telemetry Summary

Run the standalone report parser to regenerate `summary_metrics.json` and print the Markdown summary tables:

```bash
python3 experiments/timeslicing-multi-gpu/report.py
```
