# Experiment: how much GPU memory moves to host at the end of training and sampling

**Question.** When an RL step's training phase finishes, and when its sampling phase
finishes, how much GPU memory does verl move to host memory, and how much does it merely
free? Measured on a clean disaggregated layout: trainer GPUs hold only the Megatron state,
sampler GPUs only vLLM.

**TL;DR** (measured, Qwen3-30B-A3B on H200s with 139.8 GiB usable per GPU; details in [Results](#results)):

| per GPU | moved to host | freed, not copied | left on the GPU |
|---|---|---|---|
| trainer, end of training (after the weight push) | 8.2 GiB bf16 params, 0.9 s | — | 14.9 GiB (11 %) |
| sampler, end of sampling (vLLM sleep level 1 at the weight-sync pause) | 28.4 GiB weights, 1.1 s | 78.4 GiB KV cache | 7–8 GiB (5–6 %) |

**Rung.** `h200-t8-s2` from the scalability ladder
(`benchmark/setups/verl-qwen-30b-swe/config/h200-t8-s2`): Qwen3-30B-A3B-Thinking-2507,
trainer 1×8 H200 (Megatron TP2·CP1·EP4, DP1; `param_offload`, `optimizer_offload`,
CPU-resident Adam via Megatron's HybridDeviceOptimizer), sampler 2 H200 (vLLM TP2,
`gpu_memory_utilization=0.8`), batch 32 prompts × 8 rollouts, 48 concurrent agent
sessions, 2 steps, validation off.

## Where the moves happen

| phase end | verl (release/v0.9.0) | what moves |
|---|---|---|
| training: end of `update_actor` | train-mode context exit → `MegatronEngine.to("cpu")` | bf16 params → pinned host; grad buffers freed, never copied |
| training: end of `compute_log_prob` | eval-mode context exit, same hook | bf16 params → pinned host |
| training: after the weight push | **added here**: verl reloads the params to export them and, on the NCCL path, leaves them on the GPU; we call the hook it uses on the colocated path | bf16 params → pinned host |
| sampling: the weight-sync pause | **added here**: the standalone sampler never offloads, so inside the pause (requests aborted and drained) we run `sleep(level=1)` → measure → `wake_up`, then the normal in-place weight update and cache drop proceed | weights → pinned host; KV cache freed, never copied |

Megatron's hybrid-device optimizer keeps the Adam state and fp32 master params on the CPU
for the whole run. That is placement, not a phase-end move; it appears only as "resident on
CPU". Nothing else offloadable exists on this path (no reference model, no critic).

## Mechanism

One temporary patch of verl in the live Ray pods (`verl-disagg-memlog.patch`;
`patch-pods.sh apply | check | revert`). verl is an editable install at `/opt/verl` and Ray
starts fresh actor processes per job, so no pod restarts are needed. Reverted after the run;
`check` reports a clean tree on all 4 pods.

- **A. No hybrid vLLM.** `separate_async` also starts "hybrid" vLLM replicas on the trainer
  GPUs (used only for validation; the switch-to-rollout strategy is a stub).
  `--var NO_HYBRID_ROLLOUT=1` sets `RLBENCH_NO_HYBRID_ROLLOUT=1` through the Ray runtime env
  (a shell export does not reach verl's `TaskRunnerV1` actor): workers get role `actor`, and
  no server or checkpoint manager is created for the trainer pool.
- **B. Trainer logging and the post-sync offload.** Every `MegatronEngine.to()` prints one
  `[gpu-mem]` line per rank before and after the move: PyTorch allocated and reserved, device
  used (`cudaMemGetInfo`, what nvidia-smi and DCGM see), process RSS, system used, seconds,
  and exact byte counts taken from the buffers the offload helpers touch (params, grads, fp32
  main params, optimizer state), including what is still resident per category.
- **C. Sampler cycle.** `vLLMHttpServer.release_kv_cache`, an empty slot in the sync
  sequence, runs measure → `sleep(level=1)` → measure → `wake_up(weights, kv_cache)` →
  `reset_prefix_cache` → measure; the TP worker processes report via `collective_rpc`.

`--var RAY_DEDUP_LOGS=0` is required because Ray collapses worker lines that differ only in
numbers. `report.py` joins DCGM samples on node roles from `events/placement.json` (Ray
places verl's pools by GPU count, not by pod) and relabels every `init:to_cpu` after the
first per rank as a train exit (nested train-mode contexts reset the engine's mode flag
before the outer offload runs).

## Commands

```
# from the repo root
experiments/gpu-host-offload/patch-pods.sh apply && experiments/gpu-host-offload/patch-pods.sh check
# from benchmark/
V=setups/verl-qwen-30b-swe
rlbench run $V --config $V/config/h200-t8-s2 --keep --timeout 3h --name disagg-memlog \
  --var VAL_TASKS=0 --var RAY_DEDUP_LOGS=0 --var NO_HYBRID_ROLLOUT=1
# from the repo root
python3 experiments/gpu-host-offload/report.py benchmark/runs/<id> \
  --ref benchmark/runs/20261001-094845-nccl-nvls-p2p-off --baseline benchmark/runs/20260930-093817-h200-t8-s2-smoke
experiments/gpu-host-offload/patch-pods.sh revert && experiments/gpu-host-offload/patch-pods.sh check
```

## Results

Run `20261001-155249-disagg-memlog` (2026-10-01): Complete, 2 steps (8 update and log-prob
passes, 3 weight syncs), 594 `[gpu-mem]` lines from 8 trainer ranks and 2 sampler TP
workers. GiB throughout; "GB" only where the source is psutil. Every individual move is in
the run folder's `report.md`.

### Trainer, per GPU, steady state (mean over 8 ranks and over occurrences after the first)

| move | occurrences | GPU used before → after (GiB) | copied → host | grads discarded | copied → GPU | grads re-allocated | torch cache released | host RSS Δ (GB) | s | GB/s (copied) | ×8 GPUs copied |
|---|---|---|---|---|---|---|---|---|---|---|---|
| compute_log_prob enter: load | 7 | 11.5 → 19.7 | 0.0 | 0.0 | 8.2 | 0.0 | 0.0 | +0.0 | 0.61 | 13.5 | 65.6 |
| compute_log_prob exit: offload | 7 | 60.4 → 10.0 | 8.2 | 0.0 | 0.0 | 0.0 | 42.3 | +0.0 | 0.91 | 9.0 | 65.6 |
| update_actor enter: load | 7 | 10.0 → 26.4 | 0.0 | 0.0 | 8.2 | 8.2 | 0.0 | +0.0 | 1.67 | 4.9 | 65.6 |
| update_actor exit: offload | 7 | 61.1 → 11.7 | 8.2 | 8.2 | 0.0 | 0.0 | 32.6 | +0.0 | 1.91 | 4.3 | 65.6 |
| after weight sync: offload | 2 | 24.2 → 14.9 | 8.2 | 0.0 | 0.0 | 0.0 | 1.1 | +0.0 | 0.86 | 9.6 | 65.6 |

- Each phase end copies the 8.2 GiB bf16 param shard into its pinned host buffer (65.6 GiB
  across the node); the update exit also discards the 8.2 GiB grad buffer. The much larger
  drop in device memory (≈61 → 12 GiB) is PyTorch's cached activations released by
  `empty_cache`, not a move.
- Offloads run at 4–9 GB/s, loads at ≈13.5 GB/s; the update enter also re-allocates and
  zeroes the grads (1.7 s in total).
- Host memory is flat in steady state: the pinned buffers are allocated at init (+7.9 GB RSS
  per rank). From the first update on, the optimizer holds 42.7 GiB per rank (342 GiB per
  node) of Adam state and fp32 master params on the CPU, which is why the "fp32 main" counter
  stays empty.

### Sampler: sleep/wake cycle at each weight-sync pause (per TP worker; both ranks identical)

| sync pause | GPU used before → asleep → awake (GiB) | freed by sleep | host RSS before → asleep (GB) | weights | KV cache | sleep s | wake s |
|---|---|---|---|---|---|---|---|
| 1 | 114.7 → 6.9 → 114.0 | 107.8 | 3.3 → 41.9 | 28.4 | 78.4 | 13.6 | 1.4 |
| 2 | 115.8 → 7.8 → 116.1 | 108.0 | 42.0 → 42.0 | 28.4 | 78.4 | 1.1 | 1.4 |
| 3 | 116.1 → 7.8 → 116.1 | 108.3 | 42.0 → 42.0 | 28.4 | 78.4 | 1.1 | 1.4 |

Level 1 is the most vLLM can offload: the 28.4 GiB of weights go to pinned host memory
(vLLM backs up everything it tags as weights, hence 38.6 GB of RSS growth), the 78.4 GiB KV
cache is discarded, and ≈1 GiB of activation and graph pools is freed. The first sleep pays
for allocating and pinning the host buffers (13.6 s); later ones reuse them (1.1 s, ≈34 GB/s).
PyTorch's own counters stay at 109.8 GiB throughout because sleep mode works below the
PyTorch allocator; device-used is the real number.

### Left on each GPU after the offload

| machine | RL stage | moment measured | left on GPU | % of the 139.8 GiB H200 | PyTorch allocated | PyTorch reserved (cached, free) | outside PyTorch (CUDA context, NCCL) |
|---|---|---|---|---|---|---|---|
| trainer (8× H200) | start-up: policy built, weights parked on host before anything runs | after the init offload, before NCCL communicators exist | 2.1 GiB | 1.5% | 0.0 | 0.0 | 2.1 GiB |
| trainer | scoring the sampled trajectories with the current policy (old log-probs), before the update | end of `compute_log_prob` | 10.0 GiB | 7.2% | 1.2 GiB | 2.6 GiB | ≈7 GiB |
| trainer | policy update: forward/backward/optimizer step on the batch | end of `update_actor` | 11.7 GiB | 8.4% | 1.1 GiB | 4.5 GiB | ≈7 GiB |
| trainer | pushing the new policy weights to the sampler | after the weight sync (two 2 GiB NCCL send buckets still allocated, freed right after) | 14.9 GiB | 10.7% | 4.5 GiB | 7.3 GiB | ≈7 GiB |
| sampler (2× H200) | generation paused to receive new weights (the only point where sampling "ends") | asleep at vLLM sleep level 1 | 7–8 GiB | 5–6% | n/a (below PyTorch) | n/a | all of it |

Nothing offloadable remains in any row: the per-category counters (params, grads, fp32 main
params, optimizer state) are zero after every offload. The ≈7 GiB outside PyTorch was
1.9 GiB right after init and grew when the TP/EP and weight-sync NCCL communicators were
created.

### DCGM view and step timings

| run | role | DCGM FB_USED per busy GPU, later 75% of the job (GiB): median / p10 / p90 / max | samples |
|---|---|---|---|
| this `20261001-155249-disagg-memlog` | sampler | 116.1 / 115.8 / 116.1 / 122.1 | 102 |
| this `20261001-155249-disagg-memlog` | trainer | 14.6 / 9.7 / 66.2 / 79.3 | 408 |
| with hybrid `20261001-094845-nccl-nvls-p2p-off` | sampler | 114.7 / 114.7 / 116.1 / 122.1 | 220 |
| with hybrid `20261001-094845-nccl-nvls-p2p-off` | trainer | 45.7 / 41.4 / 52.7 / 63.2 | 880 |

| run | step | gen | old_log_prob | update_actor | update_weights | step |
|---|---|---|---|---|---|---|
| this `20261001-155249-disagg-memlog` | 1 | 677.7 | 123.6 | 297.5 | 17.0 | 1117.2 |
| this `20261001-155249-disagg-memlog` | 2 | 511.5 | 121.1 | 316.8 | 17.3 | 968.2 |
| baseline `20260930-093817-h200-t8-s2-smoke` | 1 | 572.3 | 144.5 | 320.4 | 11.7 | 1050.4 |
| baseline `20260930-093817-h200-t8-s2-smoke` | 2 | 563.8 | 136.3 | 323.9 | 12.1 | 1038.1 |

Between phases a trainer GPU now sits at ~14.6 GiB instead of ~45.7 GiB in the earlier run,
which carried the sleeping hybrid vLLM processes plus the params verl left resident after the
sync. Step time is within noise of the unpatched baseline (`gen` differs by sampling noise);
`update_weights` grows by ≈5 s per step: the sampler cycle (≈2.6 s), the post-sync offload
(0.9 s) and the extra RPCs.

## Caveats

- Megatron's hybrid-device optimizer (CPU Adam, `optimizer_offload_fraction=1.0`) was left
  as configured; its per-step grad/param traffic inside the optimizer step is not counted.
- One 2-step run; the baseline timings come from an earlier unpatched 2-step run of the same
  rung.

## Theoretical extrapolation to larger models — NOT measured

Everything above this heading is measured. This section is arithmetic only: it scales the
measured per-parameter constants of the 30B run to 397B and 1.6T parameters under the same
recipe (bf16 params and grads on the GPU during the update, Adam state on the CPU, bf16 vLLM
weights at `gpu_memory_utilization=0.8`). No run of either size exists; treat the numbers as
order-of-magnitude expectations.

**Constants taken from the measured run** (30.5 B params):

| quantity | measured | rule used below |
|---|---|---|
| bf16 params copied per phase end, all trainer GPUs | 65.6 GiB | 2 bytes/param × 1.15 (attention/dense parts are replicated across the expert-parallel group) |
| grads discarded per phase end | same as params | 2 bytes/param × 1.15 |
| optimizer state on CPU (fp32 master + Adam m, v) | 342 GiB | 12 bytes/param |
| pinned host copy of the params | 65.6 GiB | 2 bytes/param × 1.15 |
| trainer peak above the parked state during the update (activations + allocator cache) | ≈45 GiB/GPU | constant (same sequence lengths, full recompute) |
| non-offloadable floor per GPU | 10–12 GiB | ≈12 GiB, broken down further below |
| trainer offload / load throughput per GPU | 4–9 GB/s / 13.5 GB/s | unchanged (PCIe Gen5, 8 GPUs sharing host memory) |
| sampler bf16 weights per GPU | 28.4 GiB at TP2 | 2 bytes/param ÷ TP |
| sampler sleep throughput | ≈34 GB/s steady; first cycle 13.6 s for 38.6 GB of pinning | unchanged |

**Scaling rules.** Bytes moved at a phase end are linear in parameter count: the trainer
copies 2 bytes/param and discards 2 bytes/param, the sampler copies 2 bytes/param and discards
its KV cache. Per GPU the amount is divided by the model-parallel degree, so per-GPU move
times stay flat only if GPU count grows with the model. Host memory has no such relief:
≈14 bytes/param on the trainer (pinned bf16 + optimizer) plus 2 bytes/param pinned on the
sampler during a sleep. Minimum trainer GPUs: params + grads (4.6 bytes/param ÷ G) + 45 GiB
headroom + 12 GiB floor ≤ 139.8 GiB, so the bf16 shard must stay ≤ ≈41 GiB per GPU.

| | 30.5 B (measured) | 397 B (theoretical) | 1.6 T (theoretical) |
|---|---|---|---|
| bf16 weights | 56.8 GiB | 740 GiB | 2.98 TiB |
| trainer GPUs needed (shard ≤ 41 GiB) | 8 (1 node) | ≥ 21 → 32 GPUs, 4 nodes | ≥ 84 → 128 GPUs, 16 nodes |
| bf16 param shard copied per GPU per phase end | 8.2 GiB | ≈ 27 GiB | ≈ 27 GiB |
| copied per phase end, whole trainer | 65.6 GiB | ≈ 0.85 TiB | ≈ 3.4 TiB |
| grads discarded per phase end, whole trainer | 65.6 GiB | ≈ 0.85 TiB | ≈ 3.4 TiB |
| offload time per phase end (per GPU, concurrent) | 0.9–1.9 s | ≈ 3 s (log-prob exit), ≈ 6 s (update exit) | same per GPU |
| load time before each pass | 0.6–1.7 s | ≈ 2 s, ≈ 5.5 s with grad re-alloc | same per GPU |
| host RAM for optimizer state + pinned params | 0.4 TiB | ≈ 5.2 TiB (≈ 1.3 TiB per node over 4) | ≈ 21 TiB (≈ 1.3 TiB per node over 16) |
| sampler layout for bf16 weights | 2 GPUs, TP2 | 1 node, TP8: 92 GiB weights/GPU | ≥ 4 nodes (e.g. TP8 × PP4): 93 GiB weights/GPU |
| KV cache left per sampler GPU at 0.8 utilisation | 78 GiB | ≈ 19 GiB (4× less concurrency) | ≈ 19 GiB |
| sampler sleep: copied to host per GPU / per node | 28 GiB / 57 GiB | ≈ 92 GiB / 740 GiB | ≈ 93 GiB / 740 GiB per node, 3 TiB total |
| sampler sleep time (steady / first cycle) | 1.1 s / 13.6 s | ≈ 3 s / ≈ 35 s | ≈ 3 s / ≈ 35 s |

**Residual GPU memory after an offload.** The measured floor is four components that scale
differently:

| component (per GPU) | measured, 30.5 B (8 GPUs, 1 node) | 397 B (32 GPUs, 4 nodes, cross-node TP/EP/PP) | 1.6 T (128 GPUs, 16 nodes) | scaling rule |
|---|---|---|---|---|
| CUDA context, cuBLAS/TE handles | ≈2 GiB (seen right after init) | ≈2 GiB | ≈2 GiB | per process, independent of model size |
| NCCL communicators (TP/EP/CP, DP, weight-sync groups) | ≈5–6 GiB (floor grew from 2.1 to ≈8 GiB when they were created) | ≈8–12 GiB | ≈10–15 GiB | grows with the number of communicators and peers; cross-node groups add per-peer network buffers (`NCCL_BUFFSIZE` × channels × peers) |
| small persistent PyTorch tensors (non-DDP buffers, rotary/router caches, export-task cache) | ≈1 GiB | ≈1–3 GiB | ≈2–4 GiB | grows with layer count and hidden size, not with the sharded parameter count |
| allocator cache left after `empty_cache` | 2.5–4.5 GiB | ≈3–5 GiB | ≈3–5 GiB | tied to the per-GPU activation working set, assumed constant |
| **trainer total after a phase-end offload** | **10–12 GiB (7–8 %)** | **≈14–22 GiB (10–16 %)** | **≈17–26 GiB (12–19 %)** | |
| transient extra right after the weight sync (checkpoint-engine send buckets) | +4 GiB (2 × 2 GiB) | +4 GiB | +4 GiB | set by `update_weights_bucket_megabytes` |
| **sampler GPU asleep** (CUDA context, NCCL, CUDA-graph pools, untagged buffers) | **7–8 GiB (5–6 %)** | **≈10–15 GiB (7–11 %)** | **≈10–15 GiB (7–11 %)** | graph pools and custom all-reduce buffers scale with hidden size and TP degree; sleep level 1 never touches them |

The residue does not grow with the sharded parameter count, because everything that does
(params, grads, optimizer state, KV cache) is exactly what the offload removes; only the
NCCL share grows, with communicator and peer count. The per-component split of the measured
floor is inferred from when the memory appeared (before versus after communicator
creation), not from a per-allocation profile; `torch.cuda.memory._dump_snapshot` plus the
nvidia-smi process list would be needed to pin it down.

**What the arithmetic says.**
- Once GPU count scales with the model, the per-GPU picture barely changes: ~27 GiB copied
  per phase end, a few seconds each way, a floor near a tenth of the card. The aggregate is
  what grows: a 1.6 T model moves ~3.4 TiB to host and back at every phase end, 4–5 times
  per RL step, and 128 trainer GPUs at a ≈20 GiB floor hold ≈2.5 TiB of GPU memory while
  waiting for samples.
- Host RAM, not GPU memory, becomes the binding constraint: ≈14 bytes/param. An
  a3-ultragpu-8g node has about 2.7 TiB, so 397 B needs at least 2 nodes and 1.6 T at least 8
  for host RAM alone, before Decoupled-PPO snapshots (another 2 bytes/param) and the measured
  ≈50 GB per rank of process overhead.
- The sampler's KV budget collapses: in bf16 at 0.8 utilisation a 397 B model leaves 19 GiB
  of KV cache per GPU instead of 78, so the same 48 agent sessions need ~4× the sampler GPUs
  or FP8/INT4 weights (halving or quartering every sampler row above).
- The first sleep cycle's pinning cost (≈35 s) argues for pre-pinning the host buffers at
  start-up if the offload is used every step.
- Both sizes need cross-node model parallelism on the trainer, which this cluster cannot do
  efficiently (no RDMA; `TP·CP·EP ≤ 8` rule), so none of this is runnable here as-is.

**Not captured by the linear rules:** the 1.15 replication factor depends on the
expert/dense split of the architecture; the activation headroom depends on sequence length
and recompute settings; PCIe bandwidth is shared among the 8 GPUs of a node, so simultaneous
offloads may not sustain the rates measured here.
