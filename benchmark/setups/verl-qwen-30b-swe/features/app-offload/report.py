#!/usr/bin/env python3
"""Feature evidence for app-offload runs, from the run folder only.

    python3 features/app-offload/report.py runs/<run-id>

Reads logs/app-offload.tar.gz (one JSONL per controller pod = per GPU node, written by
config/app_offload_controller.py and collected by hooks/post-run.sh) and answers: did the
external controller find the actors and pass its startup probe, how many idle gaps did it act
in, what did each offload / reload / sampler sleep / wake cost, how much device memory moved,
and did anything go wrong (verl activity while offloaded, failed calls). Importable:
``records(run_folder) -> list[dict]`` and ``evidence(run_folder) -> dict``.
"""
import json
import statistics
import sys
import tarfile
from pathlib import Path


def records(rf) -> list:
    t = Path(rf) / "logs" / "app-offload.tar.gz"
    out = []
    if not t.exists():
        return out
    with tarfile.open(t) as tar:
        for m in tar.getmembers():
            if m.name.endswith(".jsonl"):
                for line in tar.extractfile(m).read().decode(errors="replace").splitlines():
                    if line.strip():
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    out.sort(key=lambda r: r.get("ts", 0))
    return out


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 2) if xs else None


def _max(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(max(xs), 2) if xs else None


def _freed(before: dict, after: dict):
    """mean over actors of (device-used before - after), GiB per GPU"""
    ds = [before[k] - after[k] for k in before if k in after]
    return round(statistics.mean(ds), 2) if ds else None


def evidence(rf) -> dict:
    recs = records(rf)
    if not recs:
        return {"note": "no logs/app-offload.tar.gz in run folder"}
    by = {}
    for r in recs:
        by.setdefault(r.get("action"), []).append(r)
    probes = by.get("probe", [])
    ready = [p for p in probes if p.get("ready")]
    offl, rel, gaps_end = by.get("offload", []), by.get("reload", []), by.get("gap_end", [])
    sl, wk = by.get("sampler_sleep", []), by.get("sampler_wake", [])
    out = {
        "controller_nodes": sorted({r.get("node") for r in recs if r.get("node")}),
        "probe_passed_on_nodes": sorted({p["node"] for p in ready}),
        "probe_failed_checks_last": sorted({f"{n}:{k}" for n, p in {p["node"]: p for p in probes}.items() if not p.get("ready")
                                            for k, v in (p.get("checks") or {}).items() if v is False}),
        "actors_seen": (probes[-1].get("actors_by_class") if probes else None),
        "trainer": {
            "gaps_detected": len(by.get("gap_start", [])),
            "offloads": len(offl),
            "offload_s_mean_max": [_mean([r.get("seconds") for r in offl]), _max([r.get("seconds") for r in offl])],
            "offload_freed_gib_per_gpu_mean": _mean([_freed(r.get("device_used_before", {}), r.get("device_used_after", {})) for r in offl]),
            "reloads": len(rel),
            "reload_s_mean_max": [_mean([r.get("seconds") for r in rel]), _max([r.get("seconds") for r in rel])],
            "held_s_mean": _mean([r.get("held_s") for r in rel]),
            "violations": len(by.get("violation", [])),
            "observed_gap_s_min_mean": [min((g.get("gap_s") for g in gaps_end), default=None), _mean([g.get("gap_s") for g in gaps_end])],
            "call_errors": sum(1 for r in offl + rel if r.get("error")),
        },
        "sampler": {
            "cycles_started": len(by.get("sampler_cycle_start", [])),
            "sleeps": len(sl),
            "sleep_wall_s_mean_max": [_mean([(r.get("sleep") or {}).get("wall_s") for r in sl]), _max([(r.get("sleep") or {}).get("wall_s") for r in sl])],
            "sleep_freed_gib_per_gpu_mean": _mean([_freed(r.get("device_used_before", {}), r.get("device_used_asleep", {})) for r in sl]),
            "wakes": len(wk),
            "wake_wall_s_mean_max": [_mean([(r.get("wake") or {}).get("wall_s") for r in wk]), _max([(r.get("wake") or {}).get("wall_s") for r in wk])],
            "generation_paused_s_mean": _mean([r.get("generation_paused_s") for r in wk]),
            "skipped": len(by.get("sampler_cycle_skipped", [])),
        },
        "failures": [{"stage": r.get("stage"), "error": (r.get("error") or "")[:160]} for r in by.get("failed", [])],
        "controller_errors": len(by.get("error", [])),
        "shutdown_actions": len(by.get("shutdown_reload", [])) + len(by.get("shutdown_wake", [])),
    }
    out["active"] = bool(ready) and (out["trainer"]["offloads"] + out["sampler"]["sleeps"]) > 0
    return out


if __name__ == "__main__":
    print(json.dumps(evidence(sys.argv[1]), indent=2))
