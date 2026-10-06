#!/usr/bin/env python3
"""Feature evidence for inference-scheduler runs, from the run folder only.

    python3 features/inference-scheduler/report.py runs/<run-id>

Reads logs/gateway-logs.tar.gz (records written by the setup's
RlbenchRolloutAdapter; the scheduler adds a `decisions` list per request) and
answers: was the scheduler active, how often did it fall back to verl's
balancer, did it see engine stats, how were decisions spread over replicas,
and how did the endpoint set change over the run. Importable:
``evidence(run_folder) -> dict``.
"""
import collections
import json
import statistics
import sys
import tarfile
from pathlib import Path


def _records(rf: Path):
    t = Path(rf) / "logs" / "gateway-logs.tar.gz"
    if not t.exists():
        return
    with tarfile.open(t) as tar:
        for m in tar.getmembers():
            if m.name.endswith(".jsonl"):
                for line in tar.extractfile(m).read().decode(errors="replace").splitlines():
                    if line.strip():
                        yield json.loads(line)


def _profile(rf: Path) -> dict:
    """Which scheduler profile ran: from features.json vars + the rendered config copy."""
    fj = rf / "config" / "features.json"
    if not fj.exists():
        return {}
    feats = [f for f in json.loads(fj.read_text()) if f["name"] == "inference-scheduler"]
    if not feats:
        return {}
    name = feats[0].get("vars", {}).get("SCHEDULER_CONFIG", "scheduler.yaml")
    hits = list((rf / "config").glob(f"*/{name}"))
    out = {"profile_file": name}
    if hits:
        try:
            import yaml
            cfg = yaml.safe_load(hits[0].read_text())
            out["profiles"] = {k: [f"{sc['type']}x{sc.get('weight', 1)}" for sc in v.get("scorers", [])]
                               for k, v in (cfg.get("profiles") or {}).items()}
        except Exception:  # noqa: BLE001
            out["profiles"] = "unparsed"
    return out


def evidence(rf) -> dict:
    n = n_sched = fallbacks = decisions = stats_ok = stats_err = 0
    traj_servers = collections.defaultdict(set)   # session -> replicas that served it
    traj_mv = {}
    per_server = collections.Counter()
    membership = collections.Counter()
    waiting, running, kv = [], [], []
    for r in _records(Path(rf)):
        n += 1
        if r.get("router") != "inference-scheduler":
            continue
        n_sched += 1
        if r.get("server_id"):
            traj_servers[r["session_id"]].add(r["server_id"])
            traj_mv.setdefault(r["session_id"], r.get("global_steps"))
        for d in r.get("decisions", []):
            decisions += 1
            fallbacks += bool(d.get("fallback"))
            per_server[d.get("server")] += 1
            cands = d.get("candidates", [])
            membership[len(cands)] += 1
            for c in cands:
                if c.get("stats_error"):
                    stats_err += 1
                else:
                    stats_ok += 1
                    if "num_waiting_reqs" in c:
                        waiting.append(c["num_waiting_reqs"]); running.append(c["num_running_reqs"]); kv.append(c["kv"])
    if not n:
        return {"note": "no gateway-logs.tar.gz in run folder"}
    if not n_sched:
        return {"active": False, "note": f"{n} gateway records, none routed by inference-scheduler"}
    steady = [sid for sid, mv in traj_mv.items() if mv is not None and mv >= 1]
    multi_all = sum(len(v) > 1 for v in traj_servers.values())
    multi_steady = sum(len(traj_servers[sid]) > 1 for sid in steady)
    return {
        "active": True,
        **_profile(Path(rf)),
        "requests": n_sched,
        "trajectories": len(traj_servers),
        "trajectories_touching_multiple_replicas": multi_all,
        "trajectories_touching_multiple_replicas_share_model_version_ge1":
            round(multi_steady / len(steady), 3) if steady else None,
        "decisions": decisions,
        "fallbacks_to_verl_lb": fallbacks,
        "decisions_per_server": dict(per_server.most_common()),
        "candidate_set_sizes": dict(sorted(membership.items())),
        "engine_stats_ok_fraction": round(stats_ok / (stats_ok + stats_err), 3) if (stats_ok + stats_err) else None,
        "mean_waiting_seen": round(statistics.mean(waiting), 2) if waiting else None,
        "mean_running_seen": round(statistics.mean(running), 2) if running else None,
        "mean_kv_usage_seen": round(statistics.mean(kv), 3) if kv else None,
    }


if __name__ == "__main__":
    print(json.dumps(evidence(sys.argv[1]), indent=2))
