#!/usr/bin/env python3
"""Results tables for experiments/trainer-app-offload, from run folders only.

    python3 experiments/trainer-app-offload/report.py arm-A=benchmark/runs/<id> arm-B=benchmark/runs/<id> \
        [--ref label=benchmark/runs/<id> ...]

Per run: the controller's probe outcome, every trainer offload/reload and sampler sleep/wake with
duration and device memory before/after (from the controller's JSONL in logs/app-offload.tar.gz),
the safety audit (violations, failures, observed idle-gap margins), DCGM FB_USED per role (outside
view) and verl step timings. Reference runs (no controller) only get the DCGM and timing rows.
"""
import argparse
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETUP = REPO / "benchmark" / "setups" / "verl-qwen-30b-swe"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gho = _load("gho_report", REPO / "experiments" / "gpu-host-offload" / "report.py")   # dcgm_fb_used, timing_table, f1
feat = _load("app_offload_report", SETUP / "features" / "app-offload" / "report.py")  # records, evidence


def f1(x):
    return "n/a" if x is None else f"{x:.1f}"


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return st.mean(xs) if xs else None


def _du(d: dict):
    """mean device-used over actors (GiB/GPU)"""
    return _mean(list(d.values())) if d else None


def probe_table(runs):
    out = ["| run | controller nodes | probe passed on | actors seen (class: n) | local trainers / servers (last probe) |", "|---|---|---|---|---|"]
    for label, rf in runs:
        recs = feat.records(rf)
        probes = [r for r in recs if r.get("action") == "probe"]
        if not probes:
            out.append(f"| {label} `{rf.name}` | — | — | no controller records | — |")
            continue
        last = probes[-1]
        nodes = sorted({r.get("node") for r in recs if r.get("node")})
        passed = sorted({p["node"] for p in probes if p.get("ready")})
        seen = ", ".join(f"{k}: {v}" for k, v in sorted((last.get("actors_by_class") or {}).items()))
        out.append(f"| {label} `{rf.name}` | {len(nodes)} | {len(passed)}/{len(nodes)} | {seen} | "
                   f"{len(last.get('local_trainers') or [])} / {len(last.get('local_servers') or [])} |")
    return out


def trainer_table(runs):
    out = ["| run | gap | sync finished → gap start (s) | offload s | device used before → after (GiB/GPU, mean over ranks) | freed | held s | reload s | after reload | violation | observed gap s |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, rf in runs:
        recs = feat.records(rf)
        by_gap = {}
        for r in recs:
            g = r.get("gap_id")
            if g is not None and r.get("action") in ("gap_start", "offload", "reload", "violation", "gap_end"):
                by_gap.setdefault(g, {})[r["action"]] = r
        for i, (g, d) in enumerate(sorted(by_gap.items()), 1):
            o, rl, ge, gs = d.get("offload", {}), d.get("reload", {}), d.get("gap_end", {}), d.get("gap_start", {})
            b, a, a2 = _du(o.get("device_used_before", {})), _du(o.get("device_used_after", {})), _du(rl.get("device_used_after", {}))
            freed = (b - a) if (b is not None and a is not None) else None
            out.append(f"| {label} | {i} | {f1(gs.get('sync_finished_ago_s'))} | {f2(o.get('seconds'))} | {f1(b)} → {f1(a)} | {f1(freed)} | "
                       f"{f1(rl.get('held_s'))} | {f2(rl.get('seconds'))} | {f1(a2)} | {d.get('violation', {}).get('task', '—') if 'violation' in d else '—'} | {f1(ge.get('gap_s'))} |")
    return out


def trainer_torch_table(runs):
    """torch allocator view inside the ranks (what the offload actually moved vs what the device shows)"""
    out = ["| run | gap | point | torch allocated (GiB, mean over ranks) | torch reserved | device used | RSS (GB) |", "|---|---|---|---|---|---|---|"]
    for label, rf in runs:
        recs = [r for r in feat.records(rf) if r.get("action") in ("offload", "reload")]
        gaps = sorted({r["gap_id"] for r in recs})
        for r in recs:
            i = gaps.index(r["gap_id"]) + 1
            points = [("before offload", r.get("snapshots_before")), ("after offload", r.get("snapshots_after"))] if r["action"] == "offload" \
                else [("after reload", r.get("snapshots_after"))]
            for name, snaps in points:
                if not snaps:
                    continue
                vals = [s for s in snaps.values() if isinstance(s, dict) and "device_used_gib" in s]
                al = _mean([s.get("torch_allocated_gib") for s in vals])
                rs = _mean([s.get("torch_reserved_gib") for s in vals])
                du = _mean([_mean(list(s["device_used_gib"].values())) for s in vals])
                rss = _mean([s.get("rss_gib") for s in vals])
                out.append(f"| {label} | {i} | {name} | {f1(al)} | {f1(rs)} | {f1(du)} | {f1(rss)} |")
    return out


def sampler_table(runs):
    out = ["| run | cycle | pause+drain s | sleep s | device used before → asleep (GiB/GPU) | freed | held s | wake s | awake | resume s | generation paused s | error |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, rf in runs:
        recs = feat.records(rf)
        cycles = {}
        for r in recs:
            if r.get("action") in ("sampler_pause", "sampler_sleep", "sampler_wake", "failed") and r.get("sync_id") is not None:
                cycles.setdefault((r["sync_id"], r.get("server")), {})[r["action"]] = r
        for i, (k, d) in enumerate(sorted(cycles.items()), 1):
            p, s, w, f = d.get("sampler_pause", {}), d.get("sampler_sleep", {}), d.get("sampler_wake", {}), d.get("failed", {})
            b, a, aw = _du(s.get("device_used_before", {})), _du(s.get("device_used_asleep", {})), _du(w.get("device_used_awake", {}))
            freed = (b - a) if (b is not None and a is not None) else None
            out.append(f"| {label} | {i} | {f2(p.get('pause_s'))} | {f2((s.get('sleep') or {}).get('wall_s'))} | {f1(b)} → {f1(a)} | {f1(freed)} | "
                       f"{f1(w.get('held_s'))} | {f2((w.get('wake') or {}).get('wall_s'))} | {f1(aw)} | {f2(w.get('resume_s'))} | {f1(w.get('generation_paused_s'))} | "
                       f"{(f.get('error') or '')[:60] or '—'} |")
    return out


def audit_table(runs):
    out = ["| run | probe ready | gaps | offloads / reloads | violations | sampler cycles / failures | controller errors | observed gap min / mean (s) | hold (s) | margin min (s) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for label, rf in runs:
        e = feat.evidence(rf)
        if "trainer" not in e:
            out.append(f"| {label} | — | — | — | — | — | — | — | — | — |")
            continue
        t, s = e["trainer"], e["sampler"]
        gmin, gmean = t["observed_gap_s_min_mean"]
        held = t.get("held_s_mean")
        margin = (gmin - held) if (gmin is not None and held is not None) else None
        out.append(f"| {label} | {len(e['probe_passed_on_nodes'])}/{len(e['controller_nodes'])} nodes | {t['gaps_detected']} | {t['offloads']} / {t['reloads']} | {t['violations']} | "
                   f"{s['sleeps']} / {len(e['failures'])} | {e['controller_errors']} | {f1(gmin)} / {f1(gmean)} | {f1(held)} | {f1(margin)} |")
    return out


def gateway_resumes(rf: Path):
    """requests the gateway recorded with more than one attempt = resumed after an abort
    (verl's own sync pauses + the controller's pauses); same rule as tools/gen_latency.py"""
    import tarfile
    t = rf / "logs" / "gateway-logs.tar.gz"
    if not t.exists():
        return None
    n = total = 0
    with tarfile.open(t) as tar:
        for m in tar.getmembers():
            if m.name.endswith(".jsonl"):
                for line in tar.extractfile(m).read().decode(errors="replace").splitlines():
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    n += (r.get("attempts") or 1) > 1
    return f"{n} of {total} requests"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=path/to/run-folder (controller runs)")
    ap.add_argument("--ref", action="append", default=[], help="label=path (reference run without controller; DCGM + timings only)")
    a = ap.parse_args()
    runs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.runs]
    refs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.ref]
    print("## Controller probe\n"); print("\n".join(probe_table(runs)))
    print("\n## Trainer: external offload / reload per idle gap\n"); print("\n".join(trainer_table(runs)))
    print("\n## Trainer: inside-the-rank view\n"); print("\n".join(trainer_torch_table(runs)))
    print("\n## Sampler: controller-made pause with sleep level 1\n"); print("\n".join(sampler_table(runs)))
    print("\n## Safety audit\n"); print("\n".join(audit_table(runs)))
    print("\n## Gateway: requests resumed after an abort (verl syncs + controller pauses)\n")
    for label, rf in runs + refs:
        print(f"- {label} `{rf.name}`: {gateway_resumes(rf)}")
    for title, fn in (("Outside view: DCGM FB_USED per role", gho.dcgm_table), ("verl step timings (s)", gho.timing_table)):
        print(f"\n## {title}\n")
        try:
            print("\n".join(fn(runs + refs)))
        except Exception as e:  # noqa: BLE001  (aborted runs have no job window / step lines)
            print(f"n/a ({type(e).__name__}: {e})")


if __name__ == "__main__":
    sys.exit(main())
