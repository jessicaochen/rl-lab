#!/usr/bin/env python3
"""One report per run, grouped as the benchmark questions are asked:
RL convergence / RL efficiency / RL performance / trajectories.

    python3 tools/run_report.py runs/<run-id> [--json]

Everything is read from the run folder (driver log step metrics, result.json,
DCGM scrapes + placement snapshot, agent-logs). H200 peak = verl's own table
(989 TFLOP/s bf16 dense), so TFLOP/s/GPU = MFU x 989.
"""
import argparse, glob, json, re, statistics, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ladder_row import parse as parse_steps  # noqa: E402
from duty_cycle import report as duty_report  # noqa: E402
from gen_latency import report as gen_report  # noqa: E402

H200_PEAK_TFLOPS = 989.0

def utc(s): return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

def val_metrics(steps):
    """verl logs validation as `val-core/<source>/acc/mean@1` on the step line
    it ran after: step 0 = val_before_train, last step = final validation."""
    hits = [(n, k, v) for n in sorted(steps) for k, v in steps[n].items() if re.match(r"val-core/.+/acc/mean@\d+$", k)]
    if not hits: return None
    first, last = hits[0], hits[-1]
    return {"metric": first[1], "start": first[2], "start_step": first[0],
            "end": last[2] if len(hits) > 1 else None, "end_step": last[0] if len(hits) > 1 else None,
            "delta": round(last[2] - first[2], 4) if len(hits) > 1 else None, "n_passes": len(hits)}

def counts_from_agent_logs(rf: Path):
    t = rf / "logs" / "agent-logs.tar.gz"
    if not t.exists(): return None
    n = fin = 0; rewards = []
    with tarfile.open(t) as tar:
        for m in tar.getmembers():
            if m.name.endswith("/trajectory.json"):
                d = json.load(tar.extractfile(m)); n += 1
                for tr in d.get("trajectories", []):
                    fin += bool(tr.get("finished")); rewards.append(tr.get("reward_score"))
    return {"sessions": n, "finished": fin, "episode_reward_mean": round(statistics.mean([r for r in rewards if r is not None]), 4) if rewards else None}

def build(rf: Path, trainer_gpus=None, sampler_gpus=None):
    res = json.loads((rf / "result.json").read_text())
    log = sorted(glob.glob(str(rf / "logs" / "verl-train-*.log")))[-1]
    steps = parse_steps(Path(log))
    ns = sorted(n for n in steps if "timing_s/step" in steps[n])  # training steps (step 0 = val-only line)
    tail = [n for n in ns if n >= 2] or ns  # step 1 carries first-batch/warmup effects
    def mean(k): 
        v = [steps[n][k] for n in tail if k in steps[n]]; return statistics.mean(v) if v else None
    # topology from rendered config if present
    cfg = {}
    for f in glob.glob(str(rf / "config" / "h200-*" / "run.sh")):
        cfg = dict(re.findall(r"export (\w+)=([^\s#]+)", Path(f).read_text()))
    tg = trainer_gpus or (int(cfg.get("TRAIN_NNODES", 1)) * int(cfg.get("TRAIN_GPUS", 8)))
    sg = sampler_gpus or int(cfg.get("ROLLOUT_GPUS", 8))
    batch_eps = int(cfg.get("TRAIN_BATCH", 0)) * int(cfg.get("ROLLOUT_N", 0)) or None
    js = res.get("job_status", {})
    job_wall = (utc(js["completionTime"]) - utc(js["startTime"])).total_seconds() if js.get("completionTime") else None
    rl_start, rl_end = res.get("timestamps_utc", {}).get("start"), res.get("timestamps_utc", {}).get("collected")
    rlbench_wall = (utc(rl_end) - utc(rl_start)).total_seconds() if rl_start and rl_end else res.get("wall_clock_s")
    mfu = mean("perf/mfu/actor"); step_t = mean("timing_s/step"); upd = mean("timing_s/update_actor")
    tokens = mean("perf/total_num_tokens"); resp = mean("response_length/mean")
    duty = duty_report(rf)
    out = {
      "run": rf.name, "outcome": res.get("outcome"), "training_steps": len(ns), "steps_averaged": tail, "topology": {"trainer_gpus": tg, "sampler_gpus": sg, "episodes_per_batch": batch_eps},
      "rl_convergence": {
        "val_accuracy": val_metrics(steps),
        "train_score_mean_per_step": {n: round(steps[n].get("critic/score/mean", float("nan")), 4) for n in ns},
        "train_score_max_per_step": {n: steps[n].get("critic/score/max") for n in ns},
        "reward_mean_last_step": steps[ns[-1]].get("critic/rewards/mean") if ns else None,
      },
      "rl_efficiency": {
        "gpu_duty_cycle_by_role": duty.get("roles"), "dcgm_scrapes_in_window": duty.get("n_scrapes"),
        "trainer_busy_fraction_verl": duty.get("trainer_busy_fraction_verl"),
        "weight_sync_share_of_step": round(mean("timing_s/update_weights") / step_t, 4) if step_t and mean("timing_s/update_weights") else None,
        "off_policy_staleness_mean": mean("training/off_policy/trajectory_staleness/mean"),
      },
      "rl_performance": {
        "avg_prompt_len_tokens": mean("prompt_length/mean"), "avg_response_len_tokens": resp, "avg_global_seqlen_per_rank": mean("global_seqlen/mean"),
        "avg_turns": mean("training/num_turns/mean"),
        "wall_clock_job_s": job_wall, "wall_clock_rlbench_s": rlbench_wall,
        "step_time_s_per_step": {n: steps[n].get("timing_s/step") for n in ns},
        "step_time_s_mean": step_t, "step_breakdown_s": {k.split("/")[1]: mean(k) for k in ("timing_s/gen", "timing_s/old_log_prob", "timing_s/update_actor", "timing_s/update_weights")},
        "tokens_per_s_per_gpu_verl": mean("perf/throughput"),
        "tokens_per_s_per_gpu_update_phase": (tokens / upd / tg) if tokens and upd else None,
        "sampler_gen_tokens_per_s_per_gpu": (resp * batch_eps / mean("timing_s/gen") / sg) if resp and batch_eps and mean("timing_s/gen") else None,
        "mfu_actor": mfu, "tflops_per_gpu": (mfu * H200_PEAK_TFLOPS) if mfu else None,
      },
      "generation": _generation(gen_report(rf)),
      "trajectories": {**(counts_from_agent_logs(rf) or {}),
        "token_level": "logs/agent-logs.tar.gz (trajectory.npz per session)",
        "text_dumps": [p.name for p in (rf / "logs").glob("*rollouts.tar.gz")] or "none (run predates rollout_data_dir)"},
    }
    return out

def _generation(g):
    """Headline generation-side numbers (full detail: tools/gen_latency.py)."""
    gw = g["gateway"].get("steady") or g["gateway"].get("all") or {}
    tj = g["gateway"].get("trajectories_steady") or g["gateway"].get("trajectories_all") or {}
    rm = g["replica_metrics"].get("aggregate") or {}
    ls = g["vllm_log_stats"]
    return {
        "gateway_requests": gw.get("requests"), "per_turn_latency_p50_s": gw.get("latency_p50_s"),
        "generation_s_per_trajectory_mean": tj.get("generation_s_per_trajectory_mean"),
        "generation_s_slowest_trajectory": tj.get("generation_s_slowest_trajectory"),
        "trajectories_on_multiple_replicas_share": tj.get("trajectories_on_multiple_replicas_share"),
        "per_turn_latency_p95_s": gw.get("latency_p95_s"), "per_turn_latency_p99_s": gw.get("latency_p99_s"),
        "requests_resumed_after_abort": gw.get("resumed_requests"), "request_share_cv_across_replicas": gw.get("server_request_cv"),
        "ttft_mean_s": (rm.get("ttft") or {}).get("mean_s"), "queue_time_mean_s": (rm.get("queue") or {}).get("mean_s"),
        "prefix_cache_hit_rate": rm.get("prefix_hit_rate"), "preemptions": rm.get("preemptions"),
        "replicas_seen_in_stats_lines": ls.get("replicas"), "running_cv_across_replicas": ls.get("running_cv_mean"),
        "idle_while_queued_share": ls.get("idle_while_queued_share"),
        "notes": [b["note"] for b in g.values() if isinstance(b, dict) and b.get("note")] or None,
    }

def fmt(v):
    if isinstance(v, float): return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"
    return str(v)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_folder"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args(); r = build(Path(a.run_folder))
    if a.json: print(json.dumps(r, indent=2, default=str)); return
    print(f"# {r['run']}  outcome={r['outcome']}  training_steps={r['training_steps']}  means over steps {r['steps_averaged']}  topology={r['topology']}")
    for sec in ("rl_convergence", "rl_efficiency", "rl_performance", "generation", "trajectories"):
        print(f"\n## {sec}")
        for k, v in r[sec].items():
            if isinstance(v, dict): print(f"- {k}:"); [print(f"    {kk}: {fmt(vv)}") for kk, vv in v.items()]
            else: print(f"- {k}: {fmt(v)}")

if __name__ == "__main__":
    main()
