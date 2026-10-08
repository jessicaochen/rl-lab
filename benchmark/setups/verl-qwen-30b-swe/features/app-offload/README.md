# Feature: app-offload (external offload / reload of the trainer's GPU memory through Ray)

A controller pod **outside Ray** — a plain k8s DaemonSet pod on every GPU node, one per node —
that tells the verl **trainer workers** on its own node to move their GPU memory to the host in
the idle window after each weight sync, and to bring it back. It calls into the application layer
(verl's Ray actors) and never changes verl's configuration, so the same controller runs against a
job configured with or without verl's own offload. The sampler is **not** touched by default
(`APP_OFFLOAD_SAMPLER_CYCLES_PER_STEP=0`); the sampler code path is kept for the record and is
described, with its known dead end, further down.

```sh
# arm A: verl offload on (default) + controller      arm B: verl offload off + controller
rlbench run . --config config/h200-t8-s2 --feature app-offload --var VAL_TASKS=0 --var NO_HYBRID_ROLLOUT=1 --name app-offload-on
rlbench run . --config config/h200-t8-s2 --feature app-offload --var VAL_TASKS=0 --var NO_HYBRID_ROLLOUT=1 --var MEGATRON_OFFLOAD=False --name app-offload-off
```

Experiment write-up: `experiments/trainer-app-offload/`.

## What it does

| piece | role |
|---|---|
| `setup/30-app-offload-controller.yaml` | DaemonSet on `${TRAINER_POOL}` + `${SAMPLER_POOL}` nodes, driver image, no GPU request, mounts the run ConfigMap (the script) and the shared volume (its output). Uses the `verl-driver` ServiceAccount only to list pods: Ray's node ip is the pod ip, so the pods on this k8s node tell the controller which actors are local. |
| `config/app_offload_controller.py` | The controller. Discovery through the Ray **dashboard state API** (`list_actors/list_tasks/list_nodes`), handles through **Ray Client** (`ray://verl-head-svc:10001`, namespace read from the actors), calls on methods verl already exposes. |
| `vars.env` | Hold times, cycles per step, poll/settle intervals, dry-run. |
| `report.py` | `evidence(run_folder)` from `logs/app-offload.tar.gz`, printed by `tools/compare_runs.py`. |

### Trainer (WorkerDict actors, one per rank)

- **When**: the post-weight-sync idle gap — no RUNNING task on any local WorkerDict for
  `APP_OFFLOAD_SETTLE_S`, and the two newest FINISHED verl tasks are `actor_rollout_update_weights`
  then `actor_rollout_execute_checkpoint_engine` (the finalize). This is the only window in which an
  externally submitted call cannot interleave with verl's own work: the actor is an asyncio actor
  (its `update_weights` is a coroutine), so outside that window a call could run between the awaits
  of a weight sync.
- **What**: `actor_rollout_to("cpu", model=True, optimizer=True, grad=True)` on every local rank
  (`MegatronEngine.to`, "executes irrespective of offload config"), hold
  `APP_OFFLOAD_TRAINER_HOLD_S`, then `actor_rollout_to("device")`. The gap ends with the params
  **resident**, exactly as verl leaves them after a sync, so verl's next phase finds the same state
  in both arms. During the hold the controller watches for verl activity; if any task starts it logs
  `violation` and reloads immediately.
- Stock verl leaves the bf16 param shard resident after the NCCL push (8.2 GiB/GPU at this rung);
  with verl's offload off the grads (another 8.2 GiB) are resident too and the controller's offload
  discards them (`load_grad=True` on reload re-allocates and zeroes them).

### Sampler (standalone vLLMHttpServer, node_rank 0) — off by default, kept for the record

The standalone sampler has no idle window in `separate_async` and verl's `sleep()/wake_up()` are
no-ops in standalone mode. The controller therefore **makes** a window, once per weight sync
(`APP_OFFLOAD_SAMPLER_CYCLES_PER_STEP`), `APP_OFFLOAD_SAMPLER_AFTER_SYNC_S` after the sync finished
(the next sync is a full step away): `abort_all_requests` → `wait_for_requests_to_drain` → sleep
level 1 → hold `APP_OFFLOAD_SAMPLER_HOLD_S` → wake (weights, kv_cache) + `reset_prefix_cache` →
`resume_generation`. This is verl's own sync-pause sequence with a sleep inside; aborted requests
are resumed by the agent loop. Each cycle costs ≈ hold + drain of generation time.

**How the engine is reached — status.** The first version scheduled `self.engine.sleep(...)` on
the actor's event loop from a `__ray_call__`; that **does not work** (R1, 2026-10-08): Ray runs
`__ray*` methods directly in the task thread, outside the loop (`_raylet.pyx` `function_executor`),
so there is no running loop to schedule on, and because the pause had already been issued the
engine stayed paused and the run stalled. The current code instead flips the server's
`rollout_mode` to COLOCATED (a plain attribute write through `__ray_call__`) and calls the
server's own async `sleep()` / `wake_up()` methods, which run on the loop and call
`engine.sleep(level=1)` / `engine.wake_up(tags)` + `reset_prefix_cache`; the mode is restored
afterwards, and `resume_generation` now runs unconditionally. **Untested.** The Ray-free
alternative is vLLM's dev HTTP API (`VLLM_SERVER_DEV_MODE=1` in the sampler's runtime env:
`/pause`, `/sleep`, `/wake_up`, `/resume`, `/is_sleeping`, `/is_paused` on the server's own port);
see the research section of `experiments/trainer-app-offload/README.md`.

### Measurement (needs no verl patch)

Memory is read from inside the actor processes via `__ray_call__`: pynvml device-used for the
process's visible GPUs (no CUDA context created), torch allocator stats where torch already has a
context (trainer ranks), RSS; TP-worker torch stats via `collective_rpc`. GKE's managed DCGM
exporter (setup `scrape-targets.txt`) corroborates from outside at 30 s cadence — holds are chosen
long enough to be visible there.

## Requirements

- The head Service must expose the Ray Client port (10001) and the dashboard (8265). KubeRay adds
  both by default when the head container declares no ports (it does not in
  `setup/20-raycluster.yaml`); verify with `kubectl get svc verl-head-svc -n rlbench-verl-swe`.
- Driver image ≥ v9 (Ray 2.49, pynvml via vLLM). The controller runs the same image.
- `features/app-offload/setup` creates a DaemonSet; rlbench labels, streams and deletes it like any
  other run object but does not wait for it (only Deployments are waited on) — the controller waits
  for the Ray job on its own.

## How to verify from the run folder

- `logs/app-offload-controller-*.log` (streamed stdout) and `logs/app-offload.tar.gz` (JSONL per
  node): first a `start`, then `probe` records until `ready: true` (actors found with names and
  namespace, Ray Client connected, in-actor snapshot works), then per idle gap
  `gap_start → offload → reload → gap_end`, per sync `sampler_cycle_start → sampler_pause →
  sampler_sleep → sampler_wake`.
- `python3 features/app-offload/report.py runs/<id>`: counts, durations, GiB freed per GPU,
  violations, failures, observed gap lengths (the arm-B safety margin = gap − hold). The sampler
  block stays empty unless cycles were enabled.
- `config/features.json` lists the feature and the resolved `APP_OFFLOAD_*` vars.

## Caveats

- The state API reports **qualified** class names: the trainer actors show up as
  `create_colocated_worker_cls.<locals>.WorkerDict` (verl defines the class inside a function),
  so the controller matches on the `.WorkerDict` suffix. Found on the first live probe
  (2026-10-08); the run was torn down and redeployed with the fix.

- **Arm B is safe only by timing**: with verl's offload off, verl's phase entry does not reload, so
  the controller's external reload must land before the next phase begins. The controller holds
  for a fixed time well inside the observed gap (≈500 s at `h200-t8-s2`); a robust version needs a
  handshake with the trainer. The `gap_end` records give the margin actually observed.
- The controller touches only actors on its own node; with Ray placing verl's pools by GPU count,
  any GPU node may host trainer ranks or vLLM servers, so every node runs one controller pod and
  idle nodes simply keep logging probes.
- A controller crash mid-hold leaves params offloaded (arm B: the next phase would fail) or the
  engine asleep; `terminationGracePeriodSeconds` and the shutdown handler cover SIGTERM, not kills.
