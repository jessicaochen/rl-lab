# Scalability ladder — results

Three variants run **in order**. The **standard benchmark is a 2-step run**
(identical prompt 8k / response 32k, GRPO, bf16 everywhere): a barrier is
proven or broken within two steps and step 2 is the comparable data point;
longer validations happen only when explicitly requested. Rungs 1-2 also got
10-step runs before that rule was set — kept below as supplementary rows. A
rung must pass before the next starts, so any blocker is attributable to the
barrier that rung introduces. Numbers come from the rlbench run folders
(`runs/<id>/logs/` driver `step:` lines, `runs/<id>/metrics/` DCGM scrapes).

| variant | trainer | sampler | batch × n | sessions | barrier |
|---|---|---|---|---|---|
| h200-t8-s1  | 1×8 (TP2·CP1·EP4, DP1) | 1 GPU, TP1 | 16 × 8 | 24  | single-node trainer, single-GPU sampler |
| h200-t8-s2  | 1×8 (same)             | 2 GPU, TP2 | 32 × 8 | 48  | multi-GPU sampler |
| h200-t16-s8 | 2×8 (TP2·CP1·EP4, DP2) | 8 GPU, TP8 | 64 × 8 | 128 | multi-node trainer (DP only crosses nodes) |
| h200-t16-s4x2 | 2×8 (same) | 4 replicas × 2 GPU (TP2) | 64 × 8 | 128 | multi-replica sampler: every policy request is routed to one of 4 engines (the rung request-routing features are measured on; baseline = verl's session-sticky balancer) |

## Per-step timings (seconds; 2-step runs show step1 / step2; supplementary 10-step rows show mean over steps 2..10)

| variant | run | gen | old_log_prob | update_actor | update_weights | step | actor MFU | score mean | staleness | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| *baseline: cross-node EP8* (TP2·CP2·EP8, t16-s8 topology, batch 64×8, step 1 only) | `20260929-*-verl-full` | 296 | — | **4660** | 23 | 6559 | 0.35% | 0.039 | 0 | EP group spanned both nodes → ~60 Gbps all-to-all on pod TCP; the reason the ladder pins TP·CP·EP ≤ 8 |
| **h200-t8-s1** (2-step, canonical) | `20260929-161234-h200-t8-s1-smoke` | 674 / 720 | 84 / 65 | 178 / 151 | 12 | 949 / 948 | 5.0% / 5.3% | 0.086 / 0.135 | 0 / 0.13 | EP on-node: 1.4 s/sample vs baseline 9.1; generation-bound (trainer busy ~24% of step; 1-GPU vLLM ≈ 2.1k tok/s) |
| h200-t8-s1 (10-step, supplementary) | `20260929-183139-h200-t8-s1-bench` | 874 | 65 | 150 | 12 | 1102 | 6.2% | 0.065 | 0.16 | 10/10 steps in 3h12m; trainer busy ≈21% of step (65+150+12 of 1102s) — the single vLLM GPU is the bottleneck by construction; response mean 12.2k tok, ~10 turns |
| **h200-t8-s2** (2-step, canonical) | `20260930-093817-h200-t8-s2-smoke` | 572 / 564 | 145 / 136 | 320 / 324 | 12 | 1050 / 1038 | 5.3% / 6.7% | 0.148 / 0.008 | 0 / 0.22 | TP2 sampler: 256 episodes/batch in ~570s (≈2.4× rung-1 throughput); training 1.25 s/sample; trainer busy ≈45% of step |
| h200-t8-s2 (10-step, supplementary) | `20260930-103431-h200-t8-s2-bench` | 498 | 124 | 286 | 13 | 922 | 6.3% | 0.072 | 0.207 | 10/10 steps in 2h47m; trainer busy ≈46% of step; TP2 sampler ≈2× rung-1 batch in less time; response mean 11899 tok, ~10 turns; ~2 tasks/batch lost to the litellm cwd bug (fixed in tool v2 for rung 3) |
| **h200-t16-s8** (2-step, canonical) | `20260930-132837-h200-t16-s8-smoke` | 458 / 305 | 139 / 133 | 324 / 322 | 12.6 | 937 / 776 | 5.6% / 5.9% | 0.096 / 0.061 | 0 / 0.28 | 512/512 sessions both batches, **0 failures** (tool v2); 16-GPU DP2 trainer: 0.63 s/sample (half of the 8-GPU rungs) → near-linear across nodes with EP on-node; 14× faster than the cross-node-EP baseline on identical hardware; weight sync 16→8 GPUs unchanged at ~12.6s; 128 concurrent sandboxes sustained |
| **h200-t16-s4x2** (2-step, canonical, VAL_TASKS=0, image v9) | `20261005-161446-h200-t16-s4x2` | 340 / 90 | 148 / 137 | 327 / 337 | 13 | 839 / 581 | 5.4% / 5.4% | 0.088 / 0.100 | 0 / 0.36 | sampler as 4 replicas × TP2 (same GPUs as rung 3's TP8): 512 episodes/batch over 4 engines, per-turn latency p50 9.1 s / p95 49 s / p99 78 s at the gateway, TTFT 0.21 s, queue time 0.02 s, prefix-cache hit 0.94, replica request share CV 0.013 — engines never queue at 128 sessions (KV 9 % used); step 2 581 s vs 776 s for TP8 (smaller TP = less collective overhead per token, more concurrent engines); a cold-start repeat on image v7 gave 632 s (`20261002-150931-h200-t16-s4x2`), i.e. ±8 % run-to-run |
| h200-t16-s8 (2-step **+ SWE-bench Verified validation**, metrics e2e) | `20260930-152518-h200-t16-s8-val` | 438 / 243 | 146 / 135 | 321 / 327 | 13 | 922 / 722 | 5.8% / 6.0% | 0.070 / 0.078 | 0 / 0.29 | same topology + 100-instance Verified val at step 0 and step 2 (acc **0.10 → 0.08**, 100/100 graded each pass, 0 session failures of 1736); step-2 `gen` 243s because the async sampler overlapped step-1 training; 100 min wall incl. ~32 min cold init + 2 × ~14-27 min val passes |

## Per-rung observations

- **h200-t8-s1 bench**: PASSED — Complete, 10/10 steps, no failed sessions. Steady state (steps 2-10): gen 627-1081s dominates; training (old_log_prob+update_actor+update_weights) is a flat ~227s/step; weight sync a constant 12s. MFU 5.0→6.9%. Score mean ~0.07 with per-step spread 0.00-0.12 (GRPO signal is sparse at 16×8 on r2e). Baseline for the sampler step-ups: rung 2 should cut `gen` roughly in half if TP2 scales.
- **h200-t8-s1**: smoke Complete (2/2 steps logged → verl `total_training_steps` counts sync steps as expected; the step-accounting worry is closed). 128/128 sessions succeeded, 0 failed, 24 sandboxes concurrent. Trainer memory 41 GB/GPU peak. The rung is generation-bound by design (1 sampler GPU), which makes it the clean baseline for the sampler step-ups.
- **h200-t8-s2 bench**: PASSED — Complete 10/10. Steady state gen 498s for 256 episodes (rung 1: 874s for 128) → the TP2 sampler more than doubled throughput; training cost 286s/256 samples = 1.12 s/sample (rung 1: 1.17). Multi-GPU sampling crossed cleanly; no new failure modes beyond the env bug.
- **h200-t8-s2**: smoke Batch 1: 248/256 sessions; the 8 failures were all one uid ("empty trajectories" for every rollout of a single task — a per-task env issue, group dropped by GRPO); batch 2: 0 failures. Sampler scale-up beat linear: 256 episodes generated in 572s at TP2 vs 128 in ~700s at TP1 (≈2.4× throughput on 2× GPUs; bigger KV budget → 48 concurrent sessions).
- **h200-t16-s8**: PASSED (2-step run, 35 min wall). The multi-node barrier crossed cleanly: cross-node traffic is only the DP=2 gradient all-reduce, and `update_actor` per sample dropped from 1.17-1.25 s (8 GPUs) to 0.63 s (16 GPUs). Generation at TP8 with 128 sessions: 512 episodes in 305-458s — the trainer is now busy ≈60% of the step (133+322+13 of 776s), i.e. the ladder moved the system from generation-bound (rung 1, 21%) toward balanced. Zero session failures across 1024 episodes with tool image v2. Ray placed the 16 trainer ranks on the "rollout" pod + one "trainer" pod and vLLM on the other trainer pod — harmless because the worker groups are identical.

- **h200-t16-s4x2**: PASSED (2-step, 31 min job window warm). The 4 × TP2 sampler beats
  the TP8 single replica on step time (581 vs 776 s) with the same 8 GPUs; the gateway and
  engine evidence (new in these runs) shows the sampler is far from saturated at 128
  sessions: no engine ever reports a waiting request, KV usage ~9 %, prefix-cache hit
  rate 0.94 thanks to verl's session-sticky routing. First feature measured on this
  rung: `--feature inference-scheduler` (see `experiments/inference-scheduler/`), two
  profiles, both worse than the baseline at this load: the repo's backpressure profile
  re-routes 38 % of trajectories mid-session (prefix hits 0.885, step 2 +8 %); upstream's
  prefix-only profile keeps stickiness (0.934) but balances at task granularity with a
  per-gateway load view, leaving two replicas with twice the work of the other two
  (request share CV 0.34, per-turn p99 +19 %, slowest trajectory +24 %, step 2 +10 %).
- **h200-t16-s8 + validation (metrics e2e)**: PASSED — every metric in the
  benchmark question list is produced from the run folder by
  `tools/run_report.py`. Headline numbers (step 2 unless noted):
  SWE-bench Verified acc 0.10 → 0.08 after 2 steps (n=100: ±0.03 is noise; the
  delta becomes meaningful on longer runs / `VAL_TASKS=500`); GPU duty cycle
  from GKE's managed DCGM exporter attributed by `events/placement.json` —
  trainer nodes GPU-util 29%, SM-active 13%, tensor-pipe 3%, 181 W mean vs
  sampler node 37% / 23% / 4% / 252 W (whole-job window, so init and both
  validation passes dilute the trainer; verl's own trainer-busy fraction is
  59% of a step); tokens/s/GPU 409 (verl's step-wide number), 1354 during
  `update_actor`, sampler 3272 generated tok/s/GPU; MFU 6.0% = 59 TFLOP/s/GPU;
  prompt 1405 / response 12422 tokens, ~10 turns. The 2-step run with 100-task
  validation took 100 min wall — above the ~90 min target because this was a
  cold start (Filestore re-created, model re-downloaded, SWE-bench images
  pulled for the first time: val pass 0 took 27 min vs 14 min for the final
  one). **Finding**: the DCGM join exposed a second sampler-pool node that
  hosted no Ray actor and sat at 0% / 78 W for the whole run — 8 idle H200s;
  `duty_cycle.py` now lists such nodes under `unattributed`.

## Open items carried into the ladder

- **Task-repo shadowing (env bug, found in rung 2)**: every rollout of certain
  tasks died instantly with `exit_status=AttributeError` (2 of 32 tasks per
  batch → groups dropped by GRPO). Root cause: `import litellm` inserts the
  process cwd (`/testbed`) into `sys.path`, so the *task repository* shadows
  real packages for the tool python — the tornado task's 2014 checkout broke
  `tenacity` (`tornado.gen.sleep`), numpy tasks likewise. Fixed in the vendored
  `images/run_agent.py` (import litellm first, scrub cwd entries) → tool image
  **v2**, used from rung 3 on; rung 1-2 numbers were taken with v1 (the failed
  groups cost reward signal, not throughput). Upstream candidate for uni-agent.

- ~~verl V1 `separate_async` step accounting~~ — resolved: rung-1 smoke logged
  exactly `STEPS` steps. The baseline run genuinely stopped after step 1
  (cause not investigated; it used the old cross-node config, superseded).
- **GKE auto-upgrade killed the first rung-1 bench** (2026-09-30 00:07 UTC):
  STABLE-channel auto-upgrade rolled master + every node pool
  (1.35.6-gke.1250000 → 1250001) while the RayCluster was live; the job died
  92s in with `ActorDiedError`. k8s node events vanished with the drained
  nodes — only `gcloud container operations list` showed it. Mitigations: a
  30-day `no_upgrades` maintenance exclusion (`rlbench-ladder`) and a
  `hooks/post-run.sh` that snapshots GKE operations + node inventory into
  every run folder.
- Infra: every GPU pool needs `GKE_METADATA` for the GCS FUSE mount (sampler
  pool was missing it; fixed + captured in `provision.sh`).

## Metrics collected per run (from 2026-09-30 onward)

`tools/run_report.py runs/<id>` prints, grouped by question:
- **RL convergence** — SWE-bench Verified accuracy at start and end
  (`val-core/.../acc/mean@1`, 100-instance deterministic subset by default,
  `--var VAL_TASKS=500` for the full set), per-step train score/reward.
- **RL efficiency** — GPU duty cycle per role from GKE's managed DCGM exporter
  (GPU util, graphics-engine / SM / tensor-core / DRAM activity, power)
  attributed via `events/placement.json` node roles; verl's
  trainer-busy fraction; weight-sync share; off-policy staleness.
- **RL performance** — avg prompt/response/global sequence length, total wall
  clock (k8s job and rlbench), step time + breakdown, tokens/s/GPU (verl's,
  update-phase, sampler-side), MFU and TFLOP/s/GPU.
- **Trajectories** — token-level per session (`agent-logs.tar.gz`), readable
  per-step dumps (`rollouts.tar.gz`, `val-rollouts.tar.gz`), and
  `tools/trajectories.py` to decode token-level ones to text (needs
  `transformers` + the tokenizer; works on the Ray head pod with
  `HF_HOME=/data/hf-cache HF_HUB_OFFLINE=1`).

Conventions: means are over steps ≥ 2 (step 1 carries first-batch effects), so
a 2-step run's "mean" is step 2 — per-step values are always listed too.
Duty-cycle numbers span the k8s job window (init + validation included);
compare them with verl's trainer-busy fraction, which is per training step.
