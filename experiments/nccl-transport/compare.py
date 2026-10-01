#!/usr/bin/env python3
"""Results table for the NCCL-transport experiment, derived only from run folders.

    python3 experiments/nccl-transport/compare.py baseline=benchmark/runs/<id> nvls-off=benchmark/runs/<id> nvls-p2p-off=benchmark/runs/<id>   # from the repo root

Rows = metrics (each labelled with the phase it measures), columns = runs,
plus ratio-vs-first-column for every timing. Every value comes from the run
folder: driver-log `step:` lines (verl), result.json (k8s job window),
agent-logs.tar.gz (episode counts), rendered run.sh (topology). Metrics the
baseline cannot supply (per-role DCGM duty cycle, NCCL transport evidence)
are printed in a separate section for the runs that have them.
"""
import glob, json, re, statistics, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # experiments/<name>/compare.py -> repo root
TOOLS = REPO / "benchmark" / "setups" / "verl-qwen-30b-swe" / "tools"
sys.path.insert(0, str(TOOLS))
from ladder_row import parse as parse_steps  # noqa: E402
from duty_cycle import report as duty_report  # noqa: E402

H200_PEAK_TFLOPS = 989.0
NCCL_EVIDENCE = [  # (label, regex) — counted over the driver log
    ("env overrides seen by NCCL", r"NCCL INFO (NCCL_[A-Z_0-9]+) set by environment to ([^\s.]+)"),
    ("NVLS multicast status", r"NVLS multicast support is (available|not available)"),
    ("channels via P2P (NVLink/CUMEM)", r"via P2P/(?:CUMEM|IPC|direct pointer)[^\s]*"),
    ("channels via SHM (host memory)", r"via SHM[^\s]*"),
    ("channels via NET (sockets)", r"via NET/[A-Za-z]+"),
]

def utc(s): return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

SESSION_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
SESSION_TOK = re.compile(r"response_tokens=(\d+) model_tokens=(\d+)")

def sampler_from_sessions(tar_path: Path) -> dict:
    """Sampler-side numbers from uni-agent's per-session framework.log (first
    line = session start, last timestamped line = scored; summary line carries
    model_tokens = tokens the policy generated). Independent of verl's
    `timing_s/gen`, which in separate_async only measures how long the trainer
    *waited* for a batch and reads ~0 when generation overlapped the previous
    update."""
    starts, ends, model_tok, durs = [], [], 0, []
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            if not m.name.endswith("/framework.log"): continue
            lines = tar.extractfile(m).read().decode(errors="replace").splitlines()
            ts = [SESSION_TS.match(l).group(1) for l in lines if SESSION_TS.match(l)]
            tok = SESSION_TOK.search("\n".join(lines))
            if len(ts) < 2 or not tok: continue  # never ran / failed before scoring
            t0 = datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S"); t1 = datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
            starts.append(t0); ends.append(t1); durs.append((t1 - t0).total_seconds()); model_tok += int(tok.group(2))
    if not starts: return {}
    window = (max(ends) - min(starts)).total_seconds()
    return {"sessions": len(starts), "window_s": window, "model_tokens": model_tok,
            "session_s_mean": statistics.mean(durs), "tok_per_session_s": model_tok / sum(durs) if sum(durs) else None}

def load(rf: Path) -> dict:
    log = sorted(glob.glob(str(rf / "logs" / "verl-train-*.log")))[-1]
    steps = {n: v for n, v in parse_steps(Path(log)).items() if "timing_s/step" in v}
    ns = sorted(steps); last = steps[ns[-1]] if ns else {}
    cfg = {}
    for f in glob.glob(str(rf / "config" / "h200-*" / "run.sh")):
        cfg = dict(re.findall(r"export (\w+)=([^\s#]+)", Path(f).read_text()))
    tg = int(cfg.get("TRAIN_NNODES", 1)) * int(cfg.get("TRAIN_GPUS", 8)); sg = int(cfg.get("ROLLOUT_GPUS", 8))
    eps = int(cfg.get("TRAIN_BATCH", 0)) * int(cfg.get("ROLLOUT_N", 0))
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
    text = re.sub(r"\x1b\[[0-9;]*m", "", Path(log).read_text(errors="replace"))  # strip Ray's ANSI colours
    evidence = {}
    for label, rx in NCCL_EVIDENCE:
        hits = re.findall(rx, text)
        if hits:
            if isinstance(hits[0], tuple):  # env overrides: name -> value (unique)
                evidence[label] = sorted({f"{k}={v}" for k, v in hits})
            else:
                c = {}
                for h in hits: c[h] = c.get(h, 0) + 1
                evidence[label] = c
    smp = sampler_from_sessions(t) if t.exists() else {}
    gen, upd = last.get("timing_s/gen"), last.get("timing_s/update_actor")
    resp, tok = last.get("response_length/mean"), last.get("perf/total_num_tokens")
    mfu = last.get("perf/mfu/actor")
    return {
        "run": rf.name, "outcome": res.get("outcome"), "steps": ns, "last": last,
        "rows": {
            ("sampling", "sampler active window, all sessions (s)"): smp.get("window_s"),
            ("sampling", "policy tokens generated, all sessions"): smp.get("model_tokens"),
            ("sampling", "policy tok/s per sampler GPU (window-based)"): (smp["model_tokens"] / smp["window_s"] / sg) if smp.get("window_s") else None,
            ("sampling", "session duration mean (s)"): smp.get("session_s_mean"),
            ("sampling", "policy tok/s per running session"): smp.get("tok_per_session_s"),
            ("sampling", "response length mean, step 2 (tok)"): resp,
            ("sampling", "turns mean, step 2"): last.get("training/num_turns/mean"),
            ("sampling (as waited on by trainer)", "verl timing_s/gen, step 2 (s) — ~0 when generation overlapped the previous update"): gen,
            ("training", "old_log_prob (s)"): last.get("timing_s/old_log_prob"),
            ("training", "update_actor (s)"): upd,
            ("training", "actor MFU (%)"): mfu * 100 if mfu else None,
            ("training", "TFLOP/s per trainer GPU (MFU x 989)"): mfu * H200_PEAK_TFLOPS if mfu else None,
            ("training", "update-phase tok/s per trainer GPU"): (tok / upd / tg) if tok and upd else None,
            ("training->sampling sync", "update_weights (s)"): last.get("timing_s/update_weights"),
            ("end-to-end", "step time, step 1 (s)"): steps[ns[0]].get("timing_s/step") if ns else None,
            ("end-to-end", "step time, step 2 (s)"): last.get("timing_s/step"),
            ("end-to-end", "verl throughput tok/s/GPU (step-wide)"): last.get("perf/throughput"),
            ("end-to-end", "off-policy staleness mean"): last.get("training/off_policy/trajectory_staleness/mean"),
            ("end-to-end", "job wall clock (s)"): wall,
            ("end-to-end", "episodes finished / sessions (incl. retries)"): f"{fin} / {sessions}" if sessions else None,
            ("evaluation", "SWE-bench Verified acc start -> end"): _val(parse_steps(Path(log))),
        },
        "evidence": evidence,
        "duty": duty_report(rf) if (rf / "events" / "placement.json").exists() else None,
    }

def _val(all_steps):
    hits = [(n, v) for n in sorted(all_steps) for k, v in all_steps[n].items() if re.match(r"val-core/.+/acc/mean@\d+$", k)]
    return f"{hits[0][1]:.2f} -> {hits[-1][1]:.2f}" if len(hits) > 1 else None

def fmt(v):
    if v is None: return "n/a"
    if isinstance(v, str): return v
    return f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"

def main():
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    if not runs: sys.exit(__doc__)
    data = [(label, load(Path(p))) for label, p in runs]
    base = data[0][1]
    cols = [l for l, _ in data] + [f"{l} / {data[0][0]}" for l, _ in data[1:]]
    print("| phase | metric | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for key in base["rows"]:
        phase, metric = key
        vals = [d["rows"].get(key) for _, d in data]
        ratios = []
        for v in vals[1:]:
            b = vals[0]
            ratios.append(f"{v / b:.2f}x" if isinstance(v, (int, float)) and isinstance(b, (int, float)) and b else "")
        print(f"| {phase} | {metric} | " + " | ".join([fmt(v) for v in vals] + ratios) + " |")
    print("\nruns: " + ", ".join(f"{l}=`{d['run']}` ({d['outcome']}, steps {d['steps']})" for l, d in data))
    print("\n## Not comparable across all runs (reported where collected)\n")
    for l, d in data:
        print(f"### {l}")
        if d["evidence"]:
            for k, v in d["evidence"].items(): print(f"- {k}: {json.dumps(v)}")
        else:
            print("- NCCL transport evidence: none in log (NCCL_DEBUG not raised)")
        if d["duty"]:
            for role, r in d["duty"]["roles"].items():
                print(f"- DCGM {role}: util {r['gpu_util_mean_pct']}% sm {r['sm_active_mean']} tensor {r['tensor_active_mean']} power {r['power_mean_w']}W over {d['duty']['n_scrapes']} scrapes")
        else:
            print("- per-role GPU duty cycle: n/a (no events/placement.json)")
        print()

if __name__ == "__main__":
    main()
