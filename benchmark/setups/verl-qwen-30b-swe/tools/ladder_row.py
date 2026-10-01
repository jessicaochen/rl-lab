#!/usr/bin/env python3
"""Summarize a verl rlbench run folder into a scaling-ladder table row.

Reads the streamed driver log (runs/<id>/logs/verl-train-*.log), parses the
`step:N - key:value - ...` metric lines, and prints per-step values plus the
mean over steps 2..N (step 1 includes warmup/first-batch effects).

    python3 tools/ladder_row.py runs/<run-id> [--label h200-t8-s1]
"""
import argparse, glob, json, re, statistics, sys
from pathlib import Path

KEYS = ["timing_s/gen", "timing_s/old_log_prob", "timing_s/update_actor", "timing_s/update_weights",
        "timing_s/step", "perf/mfu/actor", "critic/score/mean",
        "training/off_policy/trajectory_staleness/mean", "response_length/mean", "training/num_turns/mean"]
NUM = re.compile(r"np\.(?:float64|int64)\(([-\d.e+]+)\)|([-\d.e+]+)")

def parse(log: Path):
    steps = {}
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"step:(\d+) - (.*)$", line)
        if not m:
            continue
        n = int(m.group(1)); vals = {}
        for kv in m.group(2).split(" - "):
            if ":" not in kv: continue
            k, v = kv.split(":", 1)
            mm = NUM.search(v)
            if mm:
                vals[k] = float(mm.group(1) or mm.group(2))
        steps[n] = vals
    return steps

def fmt(k, v):
    if v is None: return "—"
    if k == "perf/mfu/actor": return f"{100*v:.1f}%"
    if k.startswith("timing_s/") or k.startswith("response_length") or k.startswith("training/num_turns"): return f"{v:.0f}"
    return f"{v:.3f}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_folder"); ap.add_argument("--label", default=None)
    a = ap.parse_args()
    rf = Path(a.run_folder)
    logs = sorted(glob.glob(str(rf / "logs" / "verl-train-*.log")))
    if not logs: sys.exit(f"no verl-train log in {rf}/logs")
    steps = {n: v for n, v in parse(Path(logs[-1])).items() if "timing_s/step" in v}  # drop val-only step 0
    if not steps: sys.exit("no step lines found")
    result = json.loads((rf / "result.json").read_text()) if (rf / "result.json").exists() else {}
    label = a.label or rf.name
    ns = sorted(steps)
    print(f"# {label}: outcome={result.get('outcome','?')} steps_logged={len(ns)} (steps {ns[0]}..{ns[-1]})")
    print("| step | " + " | ".join(k.split('/')[-1] if k.startswith('timing_s') else k for k in KEYS) + " |")
    print("|---|" + "---|" * len(KEYS))
    for n in ns:
        print(f"| {n} | " + " | ".join(fmt(k, steps[n].get(k)) for k in KEYS) + " |")
    tail = [n for n in ns if n >= 2] or ns
    means = {k: statistics.mean([steps[n][k] for n in tail if k in steps[n]]) if any(k in steps[n] for n in tail) else None for k in KEYS}
    print(f"| mean({tail[0]}..{tail[-1]}) | " + " | ".join(fmt(k, means[k]) for k in KEYS) + " |")
    # ladder-table row (same column order as docs/scaling-ladder.md)
    row = [label, f"`{rf.name}`"] + [fmt(k, means[k]) for k in KEYS[:7]] + [fmt(KEYS[7], means[KEYS[7]])]
    print("\nladder row:\n| " + " | ".join(row) + " |  |")

if __name__ == "__main__":
    main()
