#!/usr/bin/env python3
"""Results table across runs of this setup, derived only from run folders.

    python3 tools/compare_runs.py baseline=runs/<id> inference-scheduler=runs/<id> [label=runs/<id> ...]

Rows = metrics labelled by the phase they measure, columns = runs, plus ratio
vs the first column for numeric rows. Generalizes experiments/nccl-transport/
compare.py: the same sampling / training / sync / end-to-end / evaluation rows
(driver-log `step:` lines, result.json job window, agent-logs.tar.gz sessions,
rendered run.sh topology) plus a *generation* block from tools/gen_latency.py
(gateway per-request records, replica /metrics snapshots, vLLM stats lines).
Runs whose config/features.json lists a feature with a report.py get that
feature's evidence printed in the per-run section.
"""
import glob, importlib.util, json, re, statistics, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SETUP = TOOLS.parent
sys.path.insert(0, str(TOOLS))
from ladder_row import parse as parse_steps  # noqa: E402
from duty_cycle import report as duty_report  # noqa: E402
from gen_latency import report as gen_report  # noqa: E402

H200_PEAK_TFLOPS = 989.0

def utc(s): return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

SESSION_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
SESSION_TOK = re.compile(r"response_tokens=(\d+) model_tokens=(\d+)")

def sampler_from_sessions(tar_path: Path) -> dict:
    """Sampler-side numbers from uni-agent's per-session framework.log; independent of
    verl's timing_s/gen, which in separate_async is the trainer's wait for a batch."""
    starts, ends, model_tok, durs = [], [], 0, []
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            if not m.name.endswith("/framework.log"): continue
            lines = tar.extractfile(m).read().decode(errors="replace").splitlines()
            ts = [SESSION_TS.match(l).group(1) for l in lines if SESSION_TS.match(l)]
            tok = SESSION_TOK.search("\n".join(lines))
            if len(ts) < 2 or not tok: continue
            t0 = datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S"); t1 = datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
            starts.append(t0); ends.append(t1); durs.append((t1 - t0).total_seconds()); model_tok += int(tok.group(2))
    if not starts: return {}
    return {"sessions": len(starts), "window_s": (max(ends) - min(starts)).total_seconds(), "model_tokens": model_tok,
            "session_s_mean": statistics.mean(durs), "session_s_p95": sorted(durs)[int(0.95 * (len(durs) - 1))],
            "session_s_max": max(durs), "tok_per_session_s": model_tok / sum(durs) if sum(durs) else None}

def feature_evidence(rf: Path) -> dict:
    fj = rf / "config" / "features.json"
    if not fj.exists(): return {}
    out = {}
    for f in json.loads(fj.read_text()):
        rep = SETUP / "features" / f["name"] / "report.py"
        if not rep.exists():
            out[f["name"]] = {"note": "enabled; no report.py"}; continue
        spec = importlib.util.spec_from_file_location(f"feature_report_{f['name'].replace('-', '_')}", rep)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        out[f["name"]] = mod.evidence(rf)
    return out

def load(rf: Path) -> dict:
    log = sorted(glob.glob(str(rf / "logs" / "verl-train-*.log")))[-1]
    all_steps = parse_steps(Path(log))
    steps = {n: v for n, v in all_steps.items() if "timing_s/step" in v}
    ns = sorted(steps); last = steps[ns[-1]] if ns else {}
    cfg = {}
    for f in glob.glob(str(rf / "config" / "h200-*" / "run.sh")):
        cfg = dict(re.findall(r"export (\w+)=([^\s#]+)", Path(f).read_text()))
    tg = int(cfg.get("TRAIN_NNODES", 1)) * int(cfg.get("TRAIN_GPUS", 8)); sg = int(cfg.get("ROLLOUT_GPUS", 8))
    replicas = sg // int(cfg.get("ROLLOUT_TP", sg) or sg)
    res = json.loads((rf / "result.json").read_text()); js = res.get("job_status", {})
    wall = (utc(js["completionTime"]) - utc(js["startTime"])).total_seconds() if js.get("completionTime") else None
    sessions = fin = 0
    t = rf / "logs" / "agent-logs.tar.gz"
    if t.exists():
        with tarfile.open(t) as tar:
            for m in tar.getmembers():
                if m.name.endswith("/trajectory.json"):
                    d = json.load(tar.extractfile(m)); sessions += 1
                    fin += sum(bool(x.get("finished")) for x in d.get("trajectories", []))
    smp = sampler_from_sessions(t) if t.exists() else {}
    gen = gen_report(rf)
    gw = gen["gateway"].get("steady") or gen["gateway"].get("all") or {}
    tj = gen["gateway"].get("trajectories_steady") or gen["gateway"].get("trajectories_all") or {}
    ls = gen["vllm_log_stats"]; rm = gen["replica_metrics"].get("aggregate", {})
    upd = last.get("timing_s/update_actor"); tok = last.get("perf/total_num_tokens"); mfu = last.get("perf/mfu/actor")
    val = [(n, v) for n in sorted(all_steps) for k, v in all_steps[n].items() if re.match(r"val-core/.+/acc/mean@\d+$", k)]
    return {
        "run": rf.name, "outcome": res.get("outcome"), "steps": ns, "features": res.get("features", []),
        "rows": {
            ("topology", "sampler replicas x GPUs"): f"{replicas} x {sg // replicas if replicas else sg}",
            ("sampling", "sampler active window, all sessions (s)"): smp.get("window_s"),
            ("sampling", "policy tokens generated, all sessions"): smp.get("model_tokens"),
            ("sampling", "policy tok/s per sampler GPU (window-based)"): (smp["model_tokens"] / smp["window_s"] / sg) if smp.get("window_s") else None,
            ("sampling", "session duration mean (s)"): smp.get("session_s_mean"),
            ("sampling", "session duration p95 (s)"): smp.get("session_s_p95"),
            ("sampling", "session duration max (s)"): smp.get("session_s_max"),
            ("sampling", "policy tok/s per running session"): smp.get("tok_per_session_s"),
            ("sampling", "response length mean, last step (tok)"): last.get("response_length/mean"),
            ("sampling", "turns mean, last step"): last.get("training/num_turns/mean"),
            ("generation (gateway, model version >= 1)", "requests"): gw.get("requests"),
            ("generation (gateway, model version >= 1)", "per-turn latency p50 (s)"): gw.get("latency_p50_s"),
            ("generation (gateway, model version >= 1)", "per-turn latency p95 (s)"): gw.get("latency_p95_s"),
            ("generation (gateway, model version >= 1)", "per-turn latency p99 (s)"): gw.get("latency_p99_s"),
            ("generation (gateway, model version >= 1)", "completion tok/s per request, mean"): gw.get("tok_per_s_per_request_mean"),
            ("generation (gateway, model version >= 1)", "requests resumed after abort"): gw.get("resumed_requests"),
            ("generation (gateway, model version >= 1)", "request share CV across replicas"): gw.get("server_request_cv"),
            ("generation (per trajectory, model version >= 1)", "trajectories"): tj.get("trajectories"),
            ("generation (per trajectory, model version >= 1)", "mean generation per trajectory (s)"): tj.get("generation_s_per_trajectory_mean"),
            ("generation (per trajectory, model version >= 1)", "p95 generation per trajectory (s)"): tj.get("generation_s_per_trajectory_p95"),
            ("generation (per trajectory, model version >= 1)", "slowest trajectory generation (s)"): tj.get("generation_s_slowest_trajectory"),
            ("generation (per trajectory, model version >= 1)", "turns per trajectory"): tj.get("turns_per_trajectory_mean"),
            ("generation (per trajectory, model version >= 1)", "generated tokens per trajectory"): tj.get("generated_tokens_per_trajectory_mean"),
            ("generation (per trajectory, model version >= 1)", "share of trajectories served by >1 replica"): tj.get("trajectories_on_multiple_replicas_share"),
            ("generation (engine /metrics)", "TTFT mean (s)"): (rm.get("ttft") or {}).get("mean_s"),
            ("generation (engine /metrics)", "TTFT p95 bucket upper bound, worst replica (s)"): (rm.get("ttft") or {}).get("p95_le_s_worst_replica"),
            ("generation (engine /metrics)", "queue time mean (s)"): (rm.get("queue") or {}).get("mean_s"),
            ("generation (engine /metrics)", "prefill time mean (s)"): (rm.get("prefill") or {}).get("mean_s"),
            ("generation (engine /metrics)", "e2e request latency mean (s)"): (rm.get("e2e") or {}).get("mean_s"),
            ("generation (engine /metrics)", "prefix cache hit rate"): rm.get("prefix_hit_rate"),
            ("generation (engine /metrics)", "preemptions"): rm.get("preemptions"),
            ("generation (engine /metrics)", "generated tokens CV across replicas"): rm.get("generation_tokens_cv_across_replicas"),
            ("generation (vLLM stats lines)", "running-requests CV across replicas, per 10 s tick"): ls.get("running_cv_mean"),
            ("generation (vLLM stats lines)", "share of ticks with an idle replica while another queues"): ls.get("idle_while_queued_share"),
            ("sampling (as waited on by trainer)", "verl timing_s/gen, last step (s) — ~0 when generation overlapped the previous update"): last.get("timing_s/gen"),
            ("training", "old_log_prob (s)"): last.get("timing_s/old_log_prob"),
            ("training", "update_actor (s)"): upd,
            ("training", "actor MFU (%)"): mfu * 100 if mfu else None,
            ("training", "update-phase tok/s per trainer GPU"): (tok / upd / tg) if tok and upd else None,
            ("training->sampling sync", "update_weights (s)"): last.get("timing_s/update_weights"),
            ("end-to-end", "step time, step 1 (s)"): steps[ns[0]].get("timing_s/step") if ns else None,
            ("end-to-end", "step time, last step (s)"): last.get("timing_s/step"),
            ("end-to-end", "verl throughput tok/s/GPU (step-wide)"): last.get("perf/throughput"),
            ("end-to-end", "off-policy staleness mean"): last.get("training/off_policy/trajectory_staleness/mean"),
            ("end-to-end", "job wall clock (s)"): wall,
            ("end-to-end", "episodes finished / sessions (incl. retries)"): f"{fin} / {sessions}" if sessions else None,
            ("evaluation", "SWE-bench Verified acc start -> end"): f"{val[0][1]:.2f} -> {val[-1][1]:.2f}" if len(val) > 1 else None,
        },
        "gen": gen, "evidence": feature_evidence(rf),
        "duty": duty_report(rf) if (rf / "events" / "placement.json").exists() else None,
    }

def fmt(v):
    if v is None: return "n/a"
    if isinstance(v, str): return v
    return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}"

def main():
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    if not runs or any(len(r) != 2 for r in runs): sys.exit(__doc__)
    data = [(label, load(Path(p))) for label, p in runs]
    base = data[0][1]
    cols = [l for l, _ in data] + [f"{l} / {data[0][0]}" for l, _ in data[1:]]
    print("| phase | metric | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for key in base["rows"]:
        phase, metric = key
        vals = [d["rows"].get(key) for _, d in data]
        ratios = [f"{v / vals[0]:.2f}x" if isinstance(v, (int, float)) and isinstance(vals[0], (int, float)) and vals[0] else ""
                  for v in vals[1:]]
        print(f"| {phase} | {metric} | " + " | ".join([fmt(v) for v in vals] + ratios) + " |")
    print("\nruns: " + ", ".join(f"{l}=`{d['run']}` ({d['outcome']}, steps {d['steps']}, features {d['features'] or '-'})" for l, d in data))
    print("\n## Per-run detail (reported where collected)\n")
    for l, d in data:
        print(f"### {l}")
        for name, ev in d["evidence"].items():
            print(f"- feature `{name}`: {json.dumps(ev, default=str)}")
        for sec in ("gateway", "vllm_log_stats", "replica_metrics"):
            body = d["gen"].get(sec, {})
            if "note" in body: print(f"- {sec}: {body['note']}")
        reps = d["gen"]["vllm_log_stats"].get("per_replica") or {}
        for k, v in reps.items():
            print(f"- replica {k}: running {v['running_mean']:.1f} waiting {v['waiting_mean']:.1f} KV {v['kv_usage_mean_pct']:.0f}% "
                  f"prefix-hit {fmt(v['prefix_hit_mean_pct'])}% gen tok/s {fmt(v['gen_tok_per_s_mean'])} ({v['samples']} samples)")
        if d["duty"]:
            for role, r in d["duty"]["roles"].items():
                print(f"- DCGM {role}: util {r['gpu_util_mean_pct']}% sm {r['sm_active_mean']} tensor {r['tensor_active_mean']} power {r['power_mean_w']}W over {d['duty']['n_scrapes']} scrapes")
        print()

if __name__ == "__main__":
    main()
