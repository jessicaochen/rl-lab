#!/usr/bin/env python3
"""GPU->host offload report for one run folder of the verl setup.

usage: python3 experiments/gpu-host-offload/report.py <run-folder> [--ref <with-hybrid run>] [--baseline <timing run>]

Reads only the run folder: the `[gpu-mem]` lines the patched verl prints in the driver
log (logs/verl-train-*.log), DCGM scrapes (metrics/*.prom + events/placement.json) and
step lines. Prints markdown.
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark" / "setups" / "verl-qwen-30b-swe" / "tools"))
from ladder_row import parse as parse_steps  # noqa: E402

GB = 1024**3
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"^\S+ (\S+) .*?\[gpu-mem\] (.*)$")
POINT_NAMES = {
    "init:to_cpu": "init: offload after build",
    "train_end:to_gpu": "update_actor enter: load",
    "train_end:to_cpu": "update_actor exit: offload",
    "eval_end:to_gpu": "compute_log_prob enter: load",
    "eval_end:to_cpu": "compute_log_prob exit: offload",
    "post_sync:to_cpu": "after weight sync: offload",
}
TIMING_KEYS = ["timing_s/gen", "timing_s/old_log_prob", "timing_s/update_actor", "timing_s/update_weights", "timing_s/step"]


def num(v):
    try:
        f = float(v)
        return int(f) if re.fullmatch(r"-?\d+", v) else f
    except ValueError:
        return v


def driver_log(run: Path) -> Path:
    logs = sorted((run / "logs").glob("verl-train-*.log"), key=lambda p: p.stat().st_size)
    if not logs:
        sys.exit(f"no driver log in {run}")
    return logs[-1]


def parse_memlines(log: Path):
    rows = []
    for raw in log.read_text(errors="replace").splitlines():
        m = LINE.match(ANSI.sub("", raw))
        if not m:
            continue
        rec = {"ts": m.group(1)}
        for tok in m.group(2).split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                rec[k] = num(v)
        rows.append(rec)
    return rows


def relabel_train_exits(rows):
    """verl's Megatron train-mode context resets engine.mode to None *before* it offloads, so
    the end-of-update_actor offload is printed with the init label. Only the first
    init:to_cpu per rank is the real init; every later one is a train-mode exit."""
    seen = set()
    for r in rows:
        if r.get("role") == "trainer" and r.get("point") == "init:to_cpu":
            key = (r["rank"], r["pid"], r["event"])
            if key in seen:
                r["point"] = "train_end:to_cpu"
            else:
                seen.add(key)
    return rows


def pair_trainer(rows):
    """before/after pairs per (rank, point); the n-th pair of a point on a rank = occurrence n."""
    relabel_train_exits(rows)
    pending, pairs, counter = {}, [], defaultdict(int)
    for r in rows:
        if r.get("role") != "trainer":
            continue
        key = (r["rank"], r["point"])
        if r["event"] == "before":
            pending[key] = r
        elif r["event"] == "after" and key in pending:
            counter[key] += 1
            pairs.append((counter[key], pending.pop(key), r))
    return pairs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def trainer_table(pairs):
    out = ["| move | occurrence | ranks | GPU used before → after (GB/GPU) | GPU Δ | torch allocated before → after | torch reserved before → after | copied → host (GB/GPU) | grads discarded (GB/GPU) | copied → GPU | grads re-allocated | host RSS before → after (GB) | RSS Δ | host used Δ (GB) | s (mean/max) | GB/s (copied) | node total copied (GB) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for occ, b, a in pairs:
        groups[(b["point"], occ)].append((b, a))
    order = sorted(groups, key=lambda k: min(b["ts"] for b, _ in groups[k]))
    for key in order:
        point, occ = key
        bs, as_ = zip(*groups[key])
        n = len(bs)
        used_b, used_a = mean([b["gpu_device_used_gb"] for b in bs]), mean([a["gpu_device_used_gb"] for a in as_])
        al_b, al_a = mean([b["gpu_allocated_gb"] for b in bs]), mean([a["gpu_allocated_gb"] for a in as_])
        rss_b, rss_a = mean([b["host_rss_gb"] for b in bs]), mean([a["host_rss_gb"] for a in as_])
        hu = mean([a["host_used_gb"] - b["host_used_gb"] for b, a in zip(bs, as_)])
        rs_b, rs_a = mean([b["gpu_reserved_gb"] for b in bs]), mean([a["gpu_reserved_gb"] for a in as_])
        copied = mean([a["bytes_copied_to_host"] for a in as_]) / GB
        disc = mean([a["bytes_discarded"] for a in as_]) / GB
        dres = lambda c: mean([a[f"gpu_resident_{c}"] - b[f"gpu_resident_{c}"] for b, a in zip(bs, as_)]) / GB
        back = max(0.0, dres("params") + dres("main_params") + dres("opt_state"))
        realloc = max(0.0, dres("grads"))
        secs = [a["seconds"] for a in as_]
        moved = copied + back
        gbps = moved / mean(secs) if mean(secs) > 0 else float("nan")
        out.append(f"| {POINT_NAMES.get(point, point)} | {occ} | {n} | {f1(used_b)} → {f1(used_a)} | {used_a - used_b:+.1f} | {f1(al_b)} → {f1(al_a)} | {f1(rs_b)} → {f1(rs_a)} | {f1(copied)} | {f1(disc)} | {f1(back)} | {f1(realloc)} | {f1(rss_b)} → {f1(rss_a)} | {rss_a - rss_b:+.1f} | {hu:+.1f} | {f2(mean(secs))}/{f2(max(secs))} | {f1(gbps)} | {f1((copied + back) * n)} |")
    return out


def summary_table(pairs):
    """Steady state: per move, mean over all occurrences after the first (step 1's first
    pass carries allocator warm-up and the HDO state allocation)."""
    out = ["| move (steady state, per GPU) | occurrences | GPU used before → after (GiB) | copied → host (GiB) | grads discarded (GiB) | copied → GPU (GiB) | grads re-allocated (GiB) | torch cache released (GiB) | host RSS Δ (GB) | s | GB/s (copied) | ×8 GPUs copied (GiB) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    by_point = defaultdict(list)
    for occ, b, a in pairs:
        if occ > 1:
            by_point[b["point"]].append((b, a))
    order = ["eval_end:to_gpu", "eval_end:to_cpu", "train_end:to_gpu", "train_end:to_cpu", "post_sync:to_cpu"]
    for point in order + [k for k in by_point if k not in order]:
        if point not in by_point:
            continue
        bs, as_ = zip(*by_point[point])
        used_b, used_a = mean([b["gpu_device_used_gb"] for b in bs]), mean([a["gpu_device_used_gb"] for a in as_])
        copied = mean([a["bytes_copied_to_host"] for a in as_]) / GB
        disc = mean([a["bytes_discarded"] for a in as_]) / GB
        dres = lambda c: mean([a[f"gpu_resident_{c}"] - b[f"gpu_resident_{c}"] for b, a in zip(bs, as_)]) / GB
        back = max(0.0, dres("params") + dres("main_params") + dres("opt_state"))
        realloc = max(0.0, dres("grads"))
        cache = mean([(b["gpu_reserved_gb"] - b["gpu_allocated_gb"]) - (a["gpu_reserved_gb"] - a["gpu_allocated_gb"]) for b, a in zip(bs, as_)])
        rss = mean([a["host_rss_gb"] - b["host_rss_gb"] for b, a in zip(bs, as_)])
        secs = mean([a["seconds"] for a in as_])
        gbps = (copied + back) / secs if secs > 0 else float("nan")
        out.append(f"| {POINT_NAMES.get(point, point)} | {len(bs) // 8 if len(bs) >= 8 else len(bs)} | {f1(used_b)} → {f1(used_a)} | {f1(copied)} | {f1(disc)} | {f1(back)} | {f1(realloc)} | {f1(max(0.0, cache))} | {rss:+.1f} | {f2(secs)} | {f1(gbps)} | {f1((copied + back) * 8)} |")
    return out


def residual_table(pairs):
    out = ["| move | occurrence | still on GPU after (GB/GPU): params / grads / fp32 main / opt state | on CPU after (GB/rank): params / fp32 main / opt state |",
           "|---|---|---|---|"]
    groups = defaultdict(list)
    for occ, b, a in pairs:
        if a["point"].endswith("to_cpu"):
            groups[(a["point"], occ)].append(a)
    for key in sorted(groups, key=lambda k: min(a["ts"] for a in groups[k])):
        as_ = groups[key]
        g = [mean([a[f"gpu_resident_{c}"] for a in as_]) / GB for c in ("params", "grads", "main_params", "opt_state")]
        c = [mean([a[f"cpu_resident_{c}"] for a in as_]) / GB for c in ("params", "main_params", "opt_state")]
        out.append(f"| {POINT_NAMES.get(key[0], key[0])} | {key[1]} | {' / '.join(f2(x) for x in g)} | {' / '.join(f1(x) for x in c)} |")
    return out


def sampler_table(rows):
    out = ["| sync pause | TP rank | GPU used before → asleep → awake (GB/GPU) | freed by sleep | torch alloc before → asleep → awake | host RSS before → asleep → awake (GB) | RSS Δ (sleep) | weights (GB) | KV cache (GB) | sleep s | wake s |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    seq = defaultdict(list)
    for r in rows:
        if r.get("role") == "sampler":
            seq[r["rank"]].append(r)
    for rank in sorted(seq):
        occ = 0
        evs = seq[rank]
        for i in range(0, len(evs) - 2, 3):
            b, s, w = evs[i], evs[i + 1], evs[i + 2]
            if not (b["event"] == "before" and s["event"] == "after_sleep" and w["event"] == "after_wake"):
                continue
            occ += 1
            out.append(f"| {occ} | {rank} | {f1(b['gpu_device_used_gb'])} → {f1(s['gpu_device_used_gb'])} → {f1(w['gpu_device_used_gb'])} | {f1(b['gpu_device_used_gb'] - s['gpu_device_used_gb'])} | {f1(b['gpu_allocated_gb'])} → {f1(s['gpu_allocated_gb'])} → {f1(w['gpu_allocated_gb'])} | {f1(b['host_rss_gb'])} → {f1(s['host_rss_gb'])} → {f1(w['host_rss_gb'])} | {s['host_rss_gb'] - b['host_rss_gb']:+.1f} | {f1(b['weights_bytes'] / GB)} | {f1(b['kv_cache_bytes'] / GB)} | {f2(s['seconds'])} | {f2(w['seconds'])} |")
    return out


def utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def dcgm_fb_used(run: Path):
    """{role: [FB_USED GiB samples]} over the k8s job window, joined on node name via placement.json."""
    place = run / "events" / "placement.json"
    res = json.loads((run / "result.json").read_text())
    if not place.exists() or not res.get("job_status"):
        return {}
    # Ray places verl's pools by GPU count, not by pod name: join on Hostname (= node) via
    # node_roles, which post-run.sh derives from the actor types seen per node ip.
    node_roles = json.loads(place.read_text())["node_roles"]
    js = res["job_status"]
    t0, t1 = utc(js["startTime"]), utc(js.get("completionTime") or res["timestamps_utc"]["collected"])
    samples = defaultdict(list)
    for f in (run / "metrics").glob("*.prom"):
        m = re.search(r"(\d{8}T\d{6})Z", f.name)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        if not (t0 + (t1 - t0) * 0.25 <= ts <= t1):  # skip the first quarter (init, model load)
            continue
        for line in f.read_text(errors="replace").splitlines():
            if not line.startswith("DCGM_FI_DEV_FB_USED{"):
                continue
            hm = re.search(r'Hostname="([^"]*)"', line)
            role = node_roles.get(hm.group(1) if hm else "")
            if role not in ("trainer", "sampler"):
                continue
            v = float(line.rsplit(" ", 1)[1]) / 1024
            if v > 0.5:
                samples[role].append(v)
    return samples


def dcgm_table(runs):
    out = ["| run | role | DCGM FB_USED per busy GPU, later 75% of the job (GiB): median / p10 / p90 / max | samples |", "|---|---|---|---|"]
    for label, run in runs:
        for role, xs in sorted(dcgm_fb_used(run).items()):
            xs.sort()
            q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
            out.append(f"| {label} `{run.name}` | {role} | {f1(st.median(xs))} / {f1(q(0.1))} / {f1(q(0.9))} / {f1(xs[-1])} | {len(xs)} |")
    return out


def timing_table(runs):
    out = ["| run | step | " + " | ".join(k.split('/')[1] for k in TIMING_KEYS) + " |", "|---|---|" + "---|" * len(TIMING_KEYS)]
    for label, run in runs:
        steps = parse_steps(driver_log(run))
        for n in sorted(steps):
            out.append(f"| {label} `{run.name}` | {n} | " + " | ".join(f1(steps[n].get(k, float('nan'))) for k in TIMING_KEYS) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ref", type=Path, help="earlier run with the hybrid vLLM on the trainer GPUs (DCGM comparison)")
    ap.add_argument("--baseline", type=Path, help="unpatched run of the same rung (step timings)")
    a = ap.parse_args()
    rows = parse_memlines(driver_log(a.run))
    pairs = pair_trainer(rows)
    res = json.loads((a.run / "result.json").read_text())
    ranks = sorted({r["rank"] for r in rows if r.get("role") == "trainer"})
    sranks = sorted({r["rank"] for r in rows if r.get("role") == "sampler"})
    print(f"run `{a.run.name}`: outcome {res['outcome']}, {len(rows)} [gpu-mem] lines, trainer ranks {ranks}, sampler TP ranks {sranks}\n")
    print("### Trainer: steady state per phase end (mean over 8 ranks and over occurrences after the first)\n")
    print("\n".join(summary_table(pairs)))
    print("\n### Trainer: every GPU↔host move (mean over ranks; GB = GiB)\n")
    print("\n".join(trainer_table(pairs)))
    print("\n### Trainer: what is left on the GPU after each offload\n")
    print("\n".join(residual_table(pairs)))
    print("\n### Sampler: explicit offload cycle at the weight-sync pause (per TP worker)\n")
    print("\n".join(sampler_table(rows)))
    runs = [("this", a.run)] + ([("with hybrid", a.ref)] if a.ref else [])
    print("\n### Resident GPU memory seen by DCGM\n")
    print("\n".join(dcgm_table(runs)))
    truns = [("this", a.run)] + ([("baseline", a.baseline)] if a.baseline else [])
    print("\n### Step timings (s)\n")
    print("\n".join(timing_table(truns)))


if __name__ == "__main__":
    main()
