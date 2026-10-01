# Scaling parity: how the verl/Megatron setup matches prime-rl's memory recipe

prime-rl trained Qwen3-30B-A3B with **FSDP2 + ep=8 + cp=2, activation
checkpointing with CPU offload (bounded in-flight activations), and CPU-offloaded
optimizer state**. This doc records how the verl/Megatron configuration covers
each of those levers today, and what to change when we scale up (larger models,
longer context, more nodes).

| prime-rl lever | verl/Megatron equivalent | config |
|---|---|---|
| FSDP2 full param sharding | TP/EP/PP shard params structurally; the **distributed optimizer** shards optimizer state across data-parallel ranks (ZeRO-1-like); `param_offload` additionally parks inactive params in CPU and releases grad buffers | `actor.megatron.param_offload=True` |
| ep=8 (expert parallel) | Megatron expert parallelism, identical degree | `actor.megatron.expert_model_parallel_size=8` (`expert_tensor_parallel_size=1`) |
| cp=2 (context parallel) | Megatron context parallelism; plus **dynamic CP** (per-micro-batch CP sizing) for long-context scaling | `actor.megatron.context_parallel_size=2`; dynamic CP needs Megatron-Core ≥ PR #5154 |
| CPU-offloaded optimizer states | Two levels: coarse offload, plus Megatron's hybrid optimizer with fractional CPU residency | `actor.megatron.optimizer_offload=True`, `+actor.optim.override_optimizer_config.optimizer_cpu_offload=True`, `optimizer_offload_fraction=1.0` |
| Activation offload to CPU (`ac_offloading`, bounded inflight) | **No direct equivalent** — Megatron path uses full activation *recomputation* instead (compute-for-memory rather than PCIe traffic) | `+actor.megatron.override_transformer_config.recompute_granularity=full`, `recompute_method=uniform`, `recompute_num_layers=1` |

Reference recipe (in verl tree): `verl/experimental/fully_async_policy/shell/
grpo_30b_a3b_base_math_megatron_96_32.sh` — this exact model, TP2 × CP2 × EP8,
full optimizer CPU offload, disaggregated trainer/rollout pools.

## EP group placement — the first measured lesson (2026-09-29)

Megatron orders ranks (tp, cp, ep, dp), so the expert-parallel group spans
`TP × CP × EP` consecutive ranks. The in-tree 30B recipe's TP2 × CP2 × EP8 =
16 ranks — **both trainer nodes in one EP group**, i.e. every MoE all-to-all
crosses the node boundary. On our pod network (no RDMA) that measured ~60 Gbps
sustained on a single node NIC and turned a 64×8-sample training pass into
a >75-minute step (generation for the same batch: ~19 min, fully overlapped).
prime-rl's FSDP2 + ep=8 had the same shape, but its per-node NCCL layout kept
the EP all-to-all mostly intra-node.

Rule: keep `TP × CP × EP ≤ 8` (one NVLink island) unless the trainer pool has
RDMA. Default now: **TP2 × CP1 × EP4** (EP group = one node; DP=2 across
nodes carries only gradient all-reduce, which is much lighter). Long-context
scaling then comes from CP within the node (TP1 × CP2 × EP4) or from
recompute/dynamic-CP — not from a cross-node EP group.

## Measured on this cluster (scalability ladder, 2026-09-30)

All runs: Qwen3-30B-A3B bf16, GRPO, prompt 8k / response 32k, Megatron
TP2·CP1·EP4 (EP on-node), full optimizer CPU offload + full recompute, vLLM
standalone, NCCL checkpoint-engine weight sync, no RDMA (pod TCP).

| trainer GPUs | sampler | batch × n | update_actor (s / sample) | weight sync | gen per batch | step |
|---|---|---|---|---|---|---|
| 8 (1 node, DP1)  | 1 GPU TP1 | 16 × 8  | 150 s / 128 = 1.17 | 12 s | 874 s | 1102 s (gen-bound, trainer busy 21%) |
| 8 (1 node, DP1)  | 2 GPU TP2 | 32 × 8  | 286 s / 256 = 1.12 | 13 s | 498 s | 922 s (trainer busy 46%) |
| 16 (2 nodes, DP2) | 8 GPU TP8 | 64 × 8 | 322 s / 512 = 0.63 | 12.6 s | 305 s | 776 s (trainer busy 60%) |
| *16, TP2·CP2·EP8 (EP across nodes)* | 8 GPU TP8 | 64 × 8 | **4660 s / 512 = 9.1** | 23 s | 296 s | 6559 s |

Takeaways for scaling further:
- **Cross-node DP is cheap, cross-node EP is ruinous** on this network: DP2
  across nodes halved per-sample cost (near-linear); EP across nodes cost 14×.
  Grow the trainer by adding DP replicas of an on-node TP·CP·EP ≤ 8 island.
- **Weight sync is flat at ~12-13 s** regardless of trainer size (16→8 GPU
  transfer of ~60 GB bf16 over TCP); with `parameter_sync_step=4` it is <1% of
  a step. No need for the kimi/delta backends at this scale.
- **The balance point moved**: from generation-bound (21% trainer busy) to
  ~60% busy at 8 sampler GPUs. The next lever is more sampler capacity (a
  second TP8 replica), not more trainer GPUs.
- MFU stays ~5-6% across the ladder — expected for RL post-training with full
  recompute + CPU-offloaded optimizer at these micro-batch sizes; raising it
  means fewer recompute layers or veomni (FSDP2+EP) once memory allows.

## When we scale up

- **Longer context (toward 131k)**: raise `context_parallel_size` first (it
  divides activation memory linearly), enable dynamic CP so short samples don't
  pay the large-CP tax, keep full recompute. No in-tree 131k config exists for
  30B — treat every doubling (32k → 64k → 131k) as a measured experiment
  (`ppo_max_token_len_per_gpu` governs packing).
- **Bigger models**: add PP (`pipeline_model_parallel_size`) before growing TP
  beyond a node; EP scales with expert count (divisor of 128 for Qwen3-MoE).
- **If recompute cost dominates step time** (it replays the forward pass):
  switch the engine to **veomni** (`FSDP2 + expert_parallel_size +
  ulysses_parallel_size`) — the closest semantic match to prime-rl's recipe —
  which also unlocks the `delta_sharded` weight-sync backend (Megatron isn't
  supported by delta sync yet).

## Weight sync (trainer → standalone vLLM)

Backend table for ~30B (published verl measurements, IB clusters):

| backend | 30B-A3B sync time | notes |
|---|---|---|
| `nccl` (default here) | ~7s | all_gather + broadcast; fixed cluster |
| `kimi_ckpt_engine` | ~4.4s | needs checkpoint-engine ≥0.4.0 |
| `delta_sharded` | ~7-8s (1-3% of bytes/step) | FSDP-family producers only — pairs with veomni, not Megatron |

Amortize with `parameter_sync_step` (sync every K trainer steps) — unlike
prime-rl, which synced after every step. Our TCP-only pod network (no RDMA on
these pools) will run slower than the published IB numbers; measure in the
bring-up smoke and tune K to keep sync under a few percent of step time.
