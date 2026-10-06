# Feature: inference-scheduler (py-inference-scheduler request routing)

Routes every policy request to one of the sampler's vLLM replicas with
[py-inference-scheduler](https://github.com/llm-d-incubation/py-inference-scheduler)
(llm-d's RL-oriented router: load-, KV- and prefix-aware scoring per request,
optional KV-saturation flow control) instead of verl's built-in balancer.

```sh
rlbench run . --config config/h200-t16-s4x2 --feature inference-scheduler --keep --name h200-t16-s4x2-sched
```

## Where it plugs in

sandbox agent → uni-agent gateway → verl `LLMServerClient.generate()` →
**which replica?** → `vLLMHttpServer`. The baseline answer is verl's
`GlobalRequestLoadBalancer`: sticky per session (uni-agent uses the session id
as verl's request id), least-inflight at session start. With this feature the
setup's always-on `rlbench_verl_provider.rollout_adapter` builds a
`SchedulerClient` in every gateway actor: a `FullyAsyncLLMServerClient` (so
requests aborted at weight syncs still resume) whose `_acquire_server` asks the
scheduler, with verl's balancer as fallback. The upstream verl hook
(`PyInferenceAgentLoopManager`) is not usable here: it takes over the
`agent_loop_manager_class` slot uni-agent owns and subclasses the plain client.

What the fragment does (`config/feature-inference-scheduler.sh`): sets
`RLBENCH_ROUTER=inference-scheduler`, copies `scheduler.yaml` to
`$RUN_OUT/scheduler.yaml` and points `ROUTER_CONFIG_PATH` at it. Nothing else:
the scorers' engine inputs (`num_requests_waiting/running`, `kv_cache_usage_perc`)
are scraped by the client from each replica's vLLM `/metrics` (address from
verl's own `get_server_address`), the same numbers upstream reads through a
patched `vLLMHttpServer.get_routing_stats`. That patch is not used here: verl
0.9.0 lacks the method, Ray loads actor classes by reference, and the one way
to patch every process (`worker_process_setup_hook`) imports vLLM in every Ray
worker at startup, which initializes CUDA before Ray assigns GPUs and put all
eight trainer ranks of a node on GPU 0 (OOM at model build, run
`20261005-144210-h200-t16-s4x2-sched`).

## Profiles

| file | scorers | what it is |
|---|---|---|
| `config/scheduler-prefix.yaml` (default) | `prefix_cache` 1.0 | the treatment of upstream's SWE-bench A/B: soft stickiness to the replica holding the prompt's prefix, least-loaded fallback for new prompts. Measured 2026-10-06 on h200-t16-s4x2: stickiness fully restored (0 trajectories on >1 replica, prefix hits 0.934 = baseline) but the 8 GRPO rollouts of a task share their first prompt and therefore one replica, and the no-match fallback only sees this gateway's inflight, so replicas ended up 2:1 unbalanced (request share CV 0.34 vs 0.013); per-turn p99 +19 %, slowest trajectory +24 % |
| `config/scheduler-backpressure.yaml` | `waiting_queue` 5, `least_queue` 2, `kv_cache` 1, `prefix_cache` 1 | the repo's verl example; on an unsaturated sampler the inflight term outvotes prefix affinity and moves sessions between replicas (measured 2026-10-05: 38–53 % of trajectories on >1 replica, prefix hits 0.885 vs 0.936 baseline) |

Select with `--var SCHEDULER_CONFIG=<file>`; `features.json` records the choice and
`report.py` prints the parsed profile.

## Requirements

- Driver image ≥ **v7** (py-inference-scheduler pinned in `images/driver.Dockerfile`).
- A rung with **≥ 2 vLLM replicas** (`ROLLOUT_GPUS / ROLLOUT_TP ≥ 2`), e.g.
  `config/h200-t16-s4x2`. On single-replica rungs routing is a no-op.
- `rollout.disable_log_stats=False` (set by `common.sh` for every run).

## Caveats

- Scheduler state is **per gateway actor** (GATEWAYS of them): sessions are
  sticky to a gateway, so per-session prefix affinity works, but cross-session
  prefix sharing and the `least_queue` inflight view are per gateway. The
  engine-reported `waiting_queue` / `kv_cache` inputs are global truth. A
  shared scheduler actor is the follow-up if the data says it matters.
- Endpoint membership follows verl's balancer on every decision (hybrid
  replicas on trainer GPUs join during init/validation and leave for training).
- `kv_saturation` flow control is off in the shipped profile.

## Evidence (run folder)

- `config/features.json` lists `inference-scheduler`; `result.json.features` too.
- `logs/gateway-logs.tar.gz`: records carry `router=inference-scheduler` and a
  `decisions` list (candidates with `num_waiting_reqs` / `num_running_reqs` /
  `kv` / `queue_len`, chosen server, fallback flag).
  `python3 features/inference-scheduler/report.py runs/<id>` summarizes:
  fallbacks (expect ~0 in steady state), `engine_stats_ok_fraction` (expect 1.0;
  lower means `get_routing_stats`/`/metrics` did not work in some replica),
  decisions per server, candidate-set sizes (12 → 4 after step 1 when the
  hybrid replicas leave).
- Driver log: `[rlbench-router] built SchedulerClient (router=inference-scheduler)`
  per gateway, `membership: +[...]` lines, and `decision #1` / `#100` samples.
- The comparison itself (per-turn latency, TTFT/queue time, replica balance,
  prefix-hit rate, step time) comes from the setup's always-on evidence:
  `tools/compare_runs.py baseline=runs/<b> inference-scheduler=runs/<f>`.
