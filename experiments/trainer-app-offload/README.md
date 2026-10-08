# Experiment: trainer-app-offload — externally triggered offload / reload of the trainer's GPU memory

**Question.** Can a process *outside* the Ray cluster — a plain pod on the same node — tell
verl's trainer workers to move their GPU memory to the host while the trainer is idle in the
`separate_async` RL loop, and bring it back, regardless of verl's own offload configuration? What
does it cost, and is it safe? **Scope: the trainer.** The standalone sampler was part of the
first attempt; what was learned about it (one dead end, several candidate paths) is kept below,
but it is no longer exercised by the runs.

**Status.** Background research 2026-10-07; first run R1 (arm A) 2026-10-08, aborted after the first
idle gap — trainer path confirmed, sampler path failed (see below). Arm B not run yet. Cluster
cleaned and lease released after R1. Source facts are from the measured run
`20261001-155249-disagg-memlog` (`experiments/gpu-host-offload/`), the DCGM scrapes of the other
recorded runs, and verl source at the driver image's exact ref (`VERL_REF` in
`benchmark/setups/verl-qwen-30b-swe/images/driver.Dockerfile`, adc7eef).

## What the experiment established (so far)

- **The trainer can be offloaded and reloaded from outside the Ray cluster.** A plain k8s pod on
  the trainer's node, not a Ray node, discovers verl's worker actors through the Ray dashboard
  state API, recognises the post-weight-sync idle gap from their task history, and calls the
  existing `actor_rollout_to("cpu" | "device")` method over Ray Client. Measured once (run R1,
  arm A, verl's own offload on): gap recognised 4.7 s after the sync, all 8 ranks offloaded in
  2.05 s, 8.2 GiB per GPU freed (the param shard verl leaves resident), reloaded in 1.76 s, no
  verl activity during the 181 s hold.
- **The standalone sampler cannot be reached the same way.** verl's standalone `sleep()` /
  `wake_up()` are no-ops and Ray's generic `__ray_call__` runs outside the actor's event loop,
  so the engine coroutines cannot be scheduled from it. The attempt left the engine paused and
  stalled the run. Negative result; root cause and alternatives are documented below.
- **Not yet done:** arm B (verl's offload settings off, the controller as the only offloader), a
  steady-state gap after a training step (R1 ended before step 1 trained), and any of the
  alternative sampler paths. The runs from here on are trainer-only.

## Background

### Where the GPU is actually idle in the loop

| machine | idle window | how long (h200-t16-s4x2, step 2) |
|---|---|---|
| trainer | waiting for the sampler to finish the batch | 90 s of 581 s (at `h200-t8-s1` most of the step) |
| trainer | between old-log-prob exit and update enter | **not idle** — back-to-back GPU work, yet verl offloads and reloads here |
| sampler | the weight-sync pause (requests aborted and drained) | ≈13 s per step |
| sampler | anywhere else | never idle in `separate_async` |

### What verl already offloads (settings in `config/common.sh`)

| setting | effect | boundary |
|---|---|---|
| `actor.megatron.param_offload=True`, `grad_offload=True`, `optimizer_offload=True` | `MegatronEngine.to("cpu")` on exit of every train-mode and eval-mode context: bf16 param shard → pinned host (8.2 GiB/GPU, 0.9–1.9 s), grad buffer resized to 0 (8.2 GiB, never copied) | end of `compute_log_prob`, end of `update_actor`, init |
| `optim.override_optimizer_config.optimizer_cpu_offload=True`, `optimizer_offload_fraction=1.0` | Megatron HybridDeviceOptimizer keeps Adam state + fp32 master params on the CPU for the whole run (42.7 GiB/rank) | placement, not a move |
| `rollout.free_cache_engine=True`, `rollout.enable_sleep_mode=True` (defaults) | prerequisite for any vLLM sleep; the **hybrid** replicas on the trainer GPUs sleep at level 2 (weights and KV cache discarded) when the trainer takes over | `switch_to_trainer` |
| `rollout.checkpoint_engine.update_weights_bucket_megabytes=2048` | two send/recv buckets per sync, allocated in `prepare`, freed in `finalize` (transient 4 GiB) | weight sync |

At the exact ref, the optimizer offload path also clears Megatron's global memory buffer,
TransformerEngine's dummy wgrads and FP8 weight workspaces, so the usual persistent
workspaces are already covered. Measured after every offload: params / grads / fp32 main /
optimizer state on the GPU = 0 / 0 / 0 / 0 (`runs/20261001-155249-disagg-memlog/report.md`).

### What is left on the GPU anyway (measured, per H200, 139.8 GiB usable)

Trainer, during the sampling wait:

| object | size | stock verl | lever |
|---|---|---|---|
| bf16 param shard left resident after the NCCL weight push (`get_per_tensor_param` reloads it for export; the naive colocated path offloads afterwards, the NCCL path does not) | **8.2 GiB** | left resident | one call to the existing `engine.to("cpu", model=True, optimizer=False, grad=False)` + empty-cache after `send_weights` in `engine_workers.update_weights` — the `post_sync` hunk of `experiments/gpu-host-offload/verl-disagg-memlog.patch`; measured 0.9 s |
| sleeping hybrid vLLM replica: CUDA context, TP NCCL communicators, executables of 35 captured decode graphs | **≈7–8 GiB** (floor of the standalone replica asleep; hybrid not measured directly) | kept; woken only for validation, and `should_switch_to_rollout()` is a stub (`return False`) | no setting in v0.9.0 removes them; the `RLBENCH_NO_HYBRID_ROLLOUT` hunks of the same patch skip their creation. **Untested with validation on** — validation is the one thing that uses them |
| PyTorch reserved-but-free cache after `empty_cache` | 1.5–3.7 GiB | fragmentation inside segments that still hold live blocks | allocator config; not tried |
| live allocations with all four tracked categories at zero | 1.1–1.3 GiB | unattributed (appears after the first log-prob pass, then stable) | needs `torch.cuda.memory._snapshot()` at a phase end to name the objects |
| outside PyTorch: CUDA context (1.9 GiB right after init) + NCCL communicators (TP2, EP4, DP, CP, weight-sync group on actor rank 0, plus the hybrid vLLM's own) | **≈7 GiB**, of which ≈5 is NCCL | kept | `NCCL_NVLS_ENABLE=0` (step-time neutral at `h200-t8-s2` in `experiments/nccl-transport/`, memory effect unmeasured), `NCCL_BUFFSIZE`, `checkpoint_engine.engine_kwargs.nccl.rebuild_group=True` (destroys the weight-sync group after every sync, re-inits it each time; passes through `engine_kwargs`) |

Consistency check: the trainer idle floor was 9.7–14.6 GiB in the disaggregated run and
21.6–23.5 GiB (p05) in the hybrid runs of the same rung — the gap matches the first two rows.

Sampler (standalone vLLM):

| object | size | stock verl | lever |
|---|---|---|---|
| weights during the sync pause | 28.4 GiB (TP2) | updated in place, never moved | `sleep(level=1)` in the empty standalone `release_kv_cache` slot, `wake_up` in `resume_kv_cache` — the sampler hunk of the patch; measured 1.1 s sleep (13.6 s the first time, pinning), 1.4 s wake. Level 1 is the safe level: the NCCL sync writes into the existing weight buffers |
| KV cache block pool during the pause | 78.4 GiB | contents dropped by `reset_prefix_cache`, memory kept | same sleep call frees it (discarded, not copied) |
| CUDA graph executables | part of the 7–8 GiB floor | persist through sleep | no vLLM API short of re-capture |
| headroom from `gpu_memory_utilization=0.8` | ≈24 GiB never used | a knob, not a leftover | engines sit at 9 % KV use at 128 sessions, so no reason to move it |

### The one boundary that wants the opposite

Old-log-prob exit and update enter are back-to-back on the trainer. The engine still offloads
the shard (0.9 s) and reloads it with grad re-allocation (1.7 s) between them — ≈2.6 s per
step for no idle window. verl has the escape hatch already: the `disable_auto_offload` kwarg
on the engine contexts, which it uses between mini-batches inside `update_actor`. Passing it on
the log-prob call keeps the shard resident across that seam.

### What no call removes

≈2 GiB CUDA context, the NCCL communicators the knobs above can shrink but not eliminate, and
1–4 GiB of allocator residue: the 10–12 GiB floor of `experiments/gpu-host-offload/`.

### Why this matters (and when it does not)

At the 30B rung the trainer's peak during `update_actor` is 62–68 GiB of 139.8, so reclaimed
idle memory has no consumer: freeing the ≈15 GiB of the two trainer leftovers will not move
step time by itself. It matters when something is placed in the window (a larger micro-batch
or less recompute in the 337 s update, or work co-located on the trainer GPUs during the
sampling wait), and in the 397B / 1.6T extrapolation where the per-GPU floor becomes a real
term in the minimum GPU count.

### What `experiments/gpu-host-offload/` did and did not cover

- Covered and measured: the post-sync param offload (row 1) and the standalone sampler
  sleep/wake cycle. Two steps, step time inside noise, no loss-curve check.
- Covered indirectly: hybrid replicas were removed for a clean layout, not measured; their
  footprint is inferred from DCGM only, and removal was never run with validation.
- Not covered: NCCL communicator memory and `rebuild_group`, allocator residue and
  fragmentation, the unattributed 1.1 GiB, the wasted log-prob→update reload, and whether any
  reclaimed memory converts to step time.

If the setup ever moves to the FSDP backend the equivalents are `fsdp_config.param_offload`
and `optimizer_offload`, and the same missing calls apply.

## Design

The controller is `benchmark/setups/verl-qwen-30b-swe/features/app-offload/` (its README has the
operational detail). In one sentence: a plain k8s DaemonSet pod on every GPU node — **not a Ray
node** — discovers verl's actors through the Ray dashboard state API, gets handles over Ray Client,
and calls methods verl already exposes, on the actors of its own node only.

| | trainer (WorkerDict actors, one per rank) |
|---|---|
| idle window | post-weight-sync gap: no RUNNING task for ≥ 3 s and the two newest FINISHED verl tasks are `update_weights` then `execute_checkpoint_engine` (finalize). Outside it an external call could interleave at the awaits of the asyncio actor's weight sync. |
| calls | `actor_rollout_to("cpu", model=True, optimizer=True, grad=True)` → hold 180 s → `actor_rollout_to("device", model=True, optimizer=False, grad=False)` |
| end state | params resident, grads not allocated — what verl itself leaves after a sync, in both arms |
| measured | per action: seconds; pynvml device-used, torch allocated/reserved, RSS from inside each rank, before/after; verl activity during the hold (`violation`); observed gap length |

The sampler is not touched (`APP_OFFLOAD_SAMPLER_CYCLES_PER_STEP=0`, the feature default since
R1). It has no idle window in `separate_async` anyway; the controller-made pause that R1 tried,
why it failed, and the candidate paths are in the Results and Research sections.

**Arms.** Identical controller; the only difference is verl's own configuration, set at run setup:

| arm | `--var MEGATRON_OFFLOAD` | what the controller's offload moves | what verl does at the next phase entry |
|---|---|---|---|
| A | `True` (default) | the 8.2 GiB param shard verl left resident after the push | reloads itself (offload flags on) — safe by construction |
| B | `False` | params + grads (16.4 GiB); the first offload also creates the pinned host buffers | nothing — only the controller's reload (after the 180 s hold) makes the next phase possible; safe by timing (gap ≈ 500 s), the observed margin is reported |

**Layout.** `h200-t8-s2`, 2 steps, validation off, **no hybrid replicas**: verl hot-patched in
the live pods with `verl-no-hybrid.patch` (only the layout switch of
`experiments/gpu-host-offload/`, none of its instrumentation or extra offloads), applied with
`experiments/gpu-host-offload/patch-pods.sh` (`PATCH_FILE=…`).

Not in scope: the sampler (see above), performance (no baseline arms), the hybrid replicas, any
consumer for the freed memory.

## Commands

```
# from benchmark/. Trainer-only (sampler cycles are off by default). On a fresh namespace the pods
# must be hot-patched BEFORE the driver starts:
# PRE_SUBMIT_WAIT_FILE holds the submitter until the gate file exists on the shared volume.
V=setups/verl-qwen-30b-swe
COMMON="--config $V/config/h200-t8-s2 --timeout 3h --var VAL_TASKS=0 --var RAY_DEDUP_LOGS=0 --var NO_HYBRID_ROLLOUT=1 --feature app-offload"
rlbench run $V $COMMON --name app-offload-on --keep --var PRE_SUBMIT_WAIT_FILE=/data/gate-app-offload   # arm A (background)
#   ... once the Ray pods are Running (from the repo root):
PATCH_FILE=experiments/trainer-app-offload/verl-no-hybrid.patch PATCH_NEW_FILES="" experiments/gpu-host-offload/patch-pods.sh apply
PATCH_NEW_FILES="" experiments/gpu-host-offload/patch-pods.sh check
kubectl exec -n rlbench-verl-swe deploy/verl-head -c ray-head -- touch /data/gate-app-offload   # or the head pod by name
#   arm B on the still-patched pods (--keep above), no gate needed:
rlbench run $V $COMMON --name app-offload-off --var MEGATRON_OFFLOAD=False   # arm B
python3 setups/verl-qwen-30b-swe/features/app-offload/report.py runs/<id>
# from the repo root
python3 experiments/trainer-app-offload/report.py arm-A=benchmark/runs/<id> arm-B=benchmark/runs/<id> \
  --ref disagg=benchmark/runs/20261001-155249-disagg-memlog
PATCH_NEW_FILES="" experiments/gpu-host-offload/patch-pods.sh revert && PATCH_NEW_FILES="" experiments/gpu-host-offload/patch-pods.sh check
```

## Results (R1, 2026-10-08, run `20261008-104030-app-offload-on`, arm A, aborted after the first gap)

Generated by `report.py` from the run folder's controller records (`logs/app-offload.tar.gz`).
The run was stopped by hand once the sampler path had failed (see below); no training step
completed, so there are no step timings. Arm B was not run.

**Trainer: external offload and reload through Ray work.**

| gap | sync finished → gap detected | offload (8 ranks, parallel) | device used before → after (GiB/GPU, mean over ranks) | freed | held | reload | after reload |
|---|---|---|---|---|---|---|---|
| 1 (after the init-time weight sync) | 4.7 s | **2.05 s** | 12.2 → 4.0 | **8.2 GiB** | 181 s | **1.76 s** | 20.4 |

| point | torch allocated (GiB, mean over ranks) | torch reserved | device used | process RSS (GB) |
|---|---|---|---|---|
| before offload | 8.2 | 8.2 | 12.2 | 47.0 |
| after offload | 0.0 | 0.0 | 4.0 | 47.0 |
| after reload | 16.4 | 16.4 | 20.4 | 47.0 |

- The controller found the eight `WorkerDict` actors of its node by name through the dashboard
  state API, connected over Ray Client, and recognised the idle gap from the task history
  (`update_weights` then `execute_checkpoint_engine` finished, nothing running) 4.7 s after the
  sync. No verl task started during the 181 s hold (zero `violation` records).
- The 8.2 GiB freed per GPU is exactly the bf16 param shard verl leaves resident after the NCCL
  push (`experiments/gpu-host-offload/`). Host RSS did not move: the pinned host buffers already
  existed (arm A).
- The reload used `grad=True`, which re-allocates and zeroes the 8.2 GiB grad buffer on top of
  the params (20.4 GiB resident afterwards, versus 12.2 before the controller acted). To leave the
  rank exactly as verl does after a sync the reload must be `to("device", grad=False)`; the
  controller still does `grad=True` and should be changed before any rerun.
- Controller-side cost: the eight offload calls plus two snapshot rounds took ≈4 s of wall clock
  per gap; the Ray Client round trip for a snapshot was ≈0.3 s.

**Sampler: the external sleep through Ray's `__ray_call__` does not work, and it left the engine
paused.**

| step | result |
|---|---|
| trigger (≥ 20 s after the sync, no sync running) | correct |
| `abort_all_requests` → `wait_for_requests_to_drain` | 0.04 s (48 in-flight generations aborted, engine paused) |
| sleep: `__ray_call__(fn)` with `fn` doing `asyncio.get_running_loop().create_task(self.engine.sleep(level=1))` | **`RuntimeError: no running event loop`** |
| wake: same mechanism | same error |
| `resume_generation` | **never ran** — it followed the wake inside the same `finally` block |

Consequence: the standalone engine stayed paused, the agent sessions' re-sent requests hung, the
trainer waited for a batch that could not complete, and the run stalled (driver log silent from
18:27 UTC). The repair is a single `resume_generation()` call on the server actor; it had to be
done by hand (the session's sandbox refused the remote command), and the operator chose to stop
the run instead.

**Root cause** (Ray 2.49, `python/ray/_raylet.pyx`, `function_executor`): on an asyncio actor
Ray wraps a plain sync method with `sync_to_async` and runs it *on the event loop*, but methods
whose name starts with `__ray` are special-cased — "Just execute the method if it's ray internal
method" — and run directly in the task-execution thread. `__ray_call__` is therefore the one
sync entry point from which no running event loop is reachable, so a coroutine cannot be
scheduled from it. The assumption that it ran on the loop was the single unverified step of the
design, flagged as such in the plan, and it is wrong.

**Other findings from R1**

- The state API reports qualified class names (`create_colocated_worker_cls.<locals>.WorkerDict`);
  the first deployment matched the bare name and never found the trainers (torn down, redeployed).
- The vLLM server actor's event loop is blocked while the engine loads the model; `__ray_call__`
  snapshots on it time out until the engine is up (the probe handled this by retrying).
- `collective_rpc` refuses a Python callable (`VLLM_ALLOW_INSECURE_SERIALIZATION` unset), so
  per-TP-worker allocator stats are not available that way; device-level pynvml from the server
  actor process is.
- Ray Client's proxy on the head fails with `psutil.ZombieProcess` after a client that was killed
  mid-handshake (a `kubectl exec` without `-i`); a plain Ray driver inside the head pod is the
  robust path for one-off repairs.
- The `PRE_SUBMIT_WAIT_FILE` gate let the pods be hot-patched on a cold cluster before the driver
  imported verl; the generalised `patch-pods.sh` applied the two-file patch to all four pods.

### Caveats

- One gap, one run, arm A only. The trainer numbers are a single measurement; they match the
  per-move numbers of `experiments/gpu-host-offload/` (2.05 s vs 0.9 s there: here the call goes
  through Ray Client and includes the optimizer-offload bookkeeping and `empty_cache`).
- The controller's trainer path was never exercised in a steady-state gap (after a training
  step) — the run ended before step 1 trained — nor with verl's offload off (arm B).
- Nothing was committed; the verl hot patch died with the pods (namespace deleted), so no revert
  was needed.

## Research: alternate paths to trigger offload / reload from outside the Ray cluster

What the run established: the **trainer** is controllable from outside through Ray's own actor
interface (state API for discovery and idleness, Ray Client for the call), because verl exposes a
real sync method for it. The **sampler** is not reachable the same way: verl's standalone
`sleep()`/`wake_up()` are no-ops and the only generic hook, `__ray_call__`, cannot reach the
event loop. The candidates below are ordered by how little they depend on Ray and on verl's
cooperation. Sources: vLLM v0.24.0 (`vllm/entrypoints/serve/dev/*/api_router.py`,
`vllm/v1/engine/async_llm.py`), Ray 2.49 `_raylet.pyx`, verl adc7eef.

| # | path | reaches | needs Ray? | what makes it plausible | what could break it | cheapest test |
|---|---|---|---|---|---|---|
| 1 | **vLLM's dev HTTP API on the server's own port** (`VLLM_SERVER_DEV_MODE=1`): `POST /pause?mode=abort&wait_for_inflight_requests=false&clear_cache=true`, `POST /sleep?level=1`, `GET /is_sleeping`, `POST /wake_up?tags=weights&tags=kv_cache`, `POST /reset_prefix_cache`, `POST /resume`, `GET /is_paused` | standalone and hybrid vLLM engines | **no** (plain HTTP; port from `replica-metrics/replicas.json` on the shared volume, or `get_server_address` via the state API once) | verl builds its server with vLLM's `build_app`, which mounts `register_vllm_dev_api_routers` when the env var is set; the env var reaches the server actor through verl's runtime env (`--var RAY_ENV_VARS=VLLM_SERVER_DEV_MODE=1`); `/pause` + `/resume` are vLLM's RLHF endpoints, the same `pause_generation`/`resume_generation` verl's sync uses; all handlers run on the engine's loop | the FastAPI app is bound to the pod IP on an ephemeral port — fine from a pod on the same node; dev mode also exposes `/collective_rpc` and weight-update routes to anyone on the pod network; verl's `rollout_mode` bookkeeping is bypassed (verl never sleeps standalone engines, so nothing in verl expects a state) | a probe run with the env var set, then `curl` the four endpoints from the controller pod in a controller-made window; verify with `/is_sleeping` and the pynvml snapshot |
| 2 | **Flip the server's `rollout_mode` and call its own async methods** (`__ray_call__` sets `rollout_mode=COLOCATED`, then `h.sleep.remote()` / `h.wake_up.remote(tags)`, then restore) | standalone vLLM | yes (Ray Client) | `__ray_call__` works for a plain attribute write (it needs no loop); `sleep`/`wake_up` are `async def` and run on the loop; in COLOCATED mode they call `engine.sleep(level=1)` and `engine.wake_up(tags)` + `reset_prefix_cache` — the code path verl itself uses for colocated rollouts | a verl call that reads `rollout_mode` during the window would see COLOCATED (`wake_up`/`sleep` are the only readers; verl never calls them on standalone servers); still Ray-dependent | already implemented in the controller (untested); one probe run |
| 3 | **`collective_rpc("sleep", kwargs={"level": 1})` / `("wake_up", kwargs={"tags": [...]})` by method name** | the TP worker processes | yes | string method names are accepted (only callables are refused); vLLM 0.24's `/collective_rpc` route is the same call | bypasses the engine's `is_sleeping` bookkeeping and scheduler state; reaches one DP shard's workers only (verl's own `_sleep_hybrid` comment); a later engine-level `wake_up` may disagree about the sleep state | compare with path 1 on the same engine |
| 4 | **Trainer through Ray, as run** (`actor_rollout_to("cpu"/"device", …)`) | trainer ranks | yes | measured: works, idempotent, safe between sync calls | the actor is asyncio: must stay out of `update_weights` (state-API gap rule); arm B needs the reload before verl's next call — timing only | arm B run with `grad=False` on reload |
| 5 | **Trainer without Ray: a tiny control endpoint inside each rank** installed by verl's `worker_process_setup_hook` (stdlib `http.server` on a thread, or a `SIGUSR1`/`SIGUSR2` handler that calls `engine.to`) | trainer ranks | no at call time (HTTP or signal from a same-node pod with `hostPID`); the hook itself is Ray runtime-env config | the hook runs in every worker process before the actor is built; importing only stdlib avoids the CUDA-before-assignment failure that an earlier vLLM-importing hook caused (`features/inference-scheduler/README.md`) | a thread calling `engine.to()` concurrently with a running forward is unsafe — the handler must queue the request and let the actor's own thread execute it between tasks, which needs a hook into verl's dispatch (e.g. wrapping `WorkerDict` methods) | design only; no run |
| 6 | **Admission-style gating instead of calls**: do nothing to the processes, make verl do the moves by construction (`param_offload` on + the one-line post-sync offload) | trainer | n/a | the previous experiment measured it | not "external"; out of scope here | — |

**Recommendation for the next run.** Keep the trainer path as is with `grad=False` on reload,
and switch the sampler to path 1, with path 2 as the Ray-based fallback. Path 1 is the only one
that is external in the strict sense (no Ray at all) and uses the exact coroutines verl relies
on. Its only new requirement is one environment variable in the sampler's runtime env, which is
a setup knob, not a verl patch. The controller must also treat `resume` as unconditional (done)
and verify `/is_paused` is false before it declares a cycle finished.
