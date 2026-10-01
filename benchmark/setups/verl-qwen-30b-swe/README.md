# verl / Qwen3-30B-A3B / SWE — rlbench setup

Benchmarks **async agentic RL** with [verl](https://github.com/verl-project/verl)
(v0.9.0, V1 `separate_async` trainer) + [uni-agent](https://github.com/verl-project/uni-agent):
Qwen3-30B-A3B (MoE) learns r2e-gym SWE tasks; the real **mini-swe-agent** runs
*inside* per-task gVisor sandboxes (kubernetes-sigs/agent-sandbox, OSS install)
and calls the policy through the Uni-Agent Gateway (direct in-cluster reach —
no tunnel). Successor to the retired prime-rl setup; same cluster infra.

## Shape

| piece | what |
|---|---|
| `scrape-targets.txt` | Points rlbench at GKE's managed DCGM exporter pods (profiling counters, node-name labels). |
| `setup/20-raycluster.yaml` | One RayCluster: head (cpu pool, driver + gateways) + `trainer` group (2×8 H200, Megatron TP2·CP2·EP8) + `rollout` group (1×8, standalone vLLM TP8). verl carves pools via `trainer.nnodes`/`rollout.nnodes`. |
| `job.yaml` | Submitter Job = rlbench's completion signal: waits for the Ray dashboard, `ray job submit --working-dir <rendered config>` (blocking). |
| `provider/` | uni-agent sandbox provider `agent_sandbox`: one Sandbox CR (v1beta1) per episode, pod-exec transport, tool-image "mounts" → initContainer+emptyDir. |
| `tasks/` | uni-agent task `r2e_gym`: hide tests → agent (or gold-patch oracle) → restore → `run_tests.sh` → exact pytest-map reward. |
| `images/` | `driver.Dockerfile` (verl@v0.9.0 + uni-agent + our packages), `tool.Dockerfile` (mini-swe-agent sidecar, busybox final stage). |
| `config/common.sh` | The hydra invocation; every topology/batch knob is a shell var. Enforces `TRAIN_BATCH == SYNC_STEP × MINI_BATCH` and `TP·CP·EP ≤ 8` (EP on one node). |
| `config/h200-t<T>-s<S>/` | One dir per ladder rung (`h200-t8-s1`, `h200-t8-s2`, `h200-t16-s8`): `run.sh` sets the knobs (default `TOTAL_STEPS=2`) and execs `common.sh` (symlinked, with `task_config.yaml`). |
| `hooks/` | `pre-setup.sh` deletes leftover rollout sandboxes (previous run's sessions can outlive the Ray job); `post-run.sh` snapshots GKE operations + node inventory into the run folder. |
| `tools/run_report.py` | The metric sheet for one run folder: RL convergence (SWE-bench Verified accuracy start/end, rewards), RL efficiency (GPU duty cycle per role, trainer busy fraction), RL performance (seq lengths, wall clock, step time, tokens/s/GPU, TFLOP/s/GPU), trajectories. `tools/duty_cycle.py`, `tools/trajectories.py` (decode token-level trajectories to text), `tools/ladder_row.py` are its parts. |
| `docs/scaling-ladder.md` | Per-rung results table (filled as each rung passes). |
| `docs/scaling-parity.md` | How Megatron matches prime-rl's FSDP/EP8/CPU-offload recipe; weight-sync backends; the 131k path. |

## Usage

```sh
export CPU_POOL=... TRAINER_POOL=... SAMPLER_POOL=...      # node pool names
export DRIVER_IMAGE=<registry>/verl-driver:<tag>
export TOOL_IMAGE=<registry>/mini-swe-agent-tool:<tag>
export IMAGE_CACHE_PREFIX=<registry>/dockerhub-cache        # AR remote repo
export CKPT_BUCKET=<gcs bucket>                             # HNS bucket, WI-bound
# ladder, in order. The standard benchmark is a 2-step run (the default) with
# SWE-bench Verified validation (100-instance subset) at start and end;
# --var STEPS=10 for longer runs, --var VAL_TASKS=500 for the full Verified set.
rlbench run . --config config/h200-t8-s1 --keep --name h200-t8-s1
rlbench run . --config config/h200-t8-s2 --keep --name h200-t8-s2
rlbench run . --config config/h200-t16-s8 --keep --name h200-t16-s8
```

Templating: `${NAME}` / `${NAME:-default}` are rlbench render-time variables
(from env or `--var`), baked into the run folder copy; bare `$name` is shell-time
and never touched. Manifests render strictly, config files leniently. One-time cluster prep: `provision.sh` (OSS agent-sandbox, gVisor spot pool,
Filestore CSI, Workload Identity + GCS FUSE on every GPU pool, checkpoint
bucket + KSA binding, Artifact Registry repos incl. the Docker Hub cache).

## Wiring rules inherited from the prime-rl era (violate and it hangs)

- GKE driver injection: `LD_LIBRARY_PATH=/usr/local/nvidia/lib64` on every GPU
  container.
- **GPU metrics come from GKE's managed DCGM exporter** (`gke-managed-system`,
  declared in `scrape-targets.txt`): it exposes the profiling fields (SM /
  tensor-core / DRAM activity) with `Hostname` = node name. A second DCGM
  exporter on the same nodes gets blank profiling fields (one profiling client
  per GPU) — don't deploy one.
- Cross-node NCCL over TCP: pin `NCCL_SOCKET_IFNAME=eth0`; no RDMA on these pools.
- GPU pools carry a `timeslice.io/shared` taint — tolerate it.
- Filestore volumes are root-owned → `fix-perms` initContainers.
- SWE task images need root inside gVisor (hence the OSS agent-sandbox stack).
- Docker Hub task images go through the AR pull-through cache (`image_map`),
  `ready_timeout=1800` for multi-GB pulls.
- Thinking-model generations must be bounded (`response_length`, concurrency
  caps) or engine pause/drain windows blow up.
- **Every GPU node pool needs Workload Identity metadata** (`--workload-metadata=GKE_METADATA`): the GCS FUSE CSI mount on the Ray worker groups fails with `Workload Identity Federation is not enabled on node` otherwise — and because of the next rule, any GPU node may host any worker.
- **Ray places verl's pools by GPU count, not by k8s node pool**: the trainer
  ranks may land on the "sampler" node and vLLM on a "trainer" node. Keep all
  GPU worker groups identical (mounts, GCS FUSE, memory); don't infer roles
  from node names — map pod IPs from the driver log when reading DCGM.
- **Sandbox pool sizing**: `max_concurrent_sessions` × per-sandbox cpu must fit
  the pool (10 × n4-standard-32 ≈ 140 two-vCPU sandboxes → 128 sessions).
- **Block GKE auto-upgrades for the duration of a campaign**: STABLE-channel
  auto-upgrade rolled all node pools mid-run and killed a benchmark
  (`ActorDiedError` 92s in). Add a `no_upgrades` maintenance exclusion before
  multi-hour runs; `hooks/post-run.sh` records GKE operations in the run folder
  because node events don't survive node replacement.
- **In-sandbox tool python must not see the task repo**: litellm adds cwd to
  `sys.path`; with cwd=/testbed, repos named like real packages (tornado, numpy,
  ...) shadow them and mini-swe-agent crashes before its first call. The vendored
  `images/run_agent.py` scrubs cwd entries after importing litellm (tool image ≥ v2).
- **Validation is SWE-bench Verified, never trained on**: rows carry `name: swe_bench`
  (uni-agent's task; swebench images via the AR cache, F2P/P2P grading in-sandbox,
  conda env `testbed`); training rows stay `r2e_gym`. The driver image needs the
  `swebench` package (image ≥ v6).
- **separate_async invariant**: `train_batch_size == parameter_sync_step ×
  ppo_mini_batch_size`.
