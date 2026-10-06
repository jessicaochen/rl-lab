#!/usr/bin/env python3
"""Generation-side evidence of one run folder (importable: report(rf) -> dict).

    python3 tools/gen_latency.py runs/<run-id> [--json]

Three sources, all written by the setup's always-on instrumentation, so they
exist for baseline and feature runs alike:

- logs/gateway-logs.tar.gz — one JSON record per policy request at the uni-agent
  gateway (rlbench_verl_provider.rollout_adapter): per-turn latency, replica,
  tokens, resumes, model version (`global_steps`), router decision.
- logs/replica-metrics.tar.gz — vLLM /metrics snapshots per replica every 30 s:
  TTFT / queue / prefill / e2e histograms, prefix-cache counters, preemptions,
  token counters. Reported as deltas between the first and last snapshot.
- logs/verl-train-*.log — vLLM's 10-second stats line per replica
  (rollout.disable_log_stats=False; RAY_DEDUP_LOGS=0 keeps every replica's line):
  running / waiting / KV% / prefix hit — used for cross-replica balance per tick.
"""
import argparse, collections, glob, json, re, statistics, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path

# ---- gateway records --------------------------------------------------------

def gateway_records(rf: Path):
    t = rf / "logs" / "gateway-logs.tar.gz"
    if not t.exists():
        return []
    out = []
    with tarfile.open(t) as tar:
        for m in tar.getmembers():
            if m.name.endswith(".jsonl"):
                for line in tar.extractfile(m).read().decode(errors="replace").splitlines():
                    if line.strip():
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    return out

def pct(vals, q):
    if not vals: return None
    s = sorted(vals); k = (len(s) - 1) * q; f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)

def cv(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2 or not sum(vals): return None
    m = statistics.mean(vals)
    return statistics.pstdev(vals) / m if m else None

def _lat_block(recs):
    lat = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
    comp = [r["completion_tokens"] for r in recs if r.get("completion_tokens") is not None]
    per_server = collections.Counter(r.get("server_id") for r in recs)
    return {
        "requests": len(recs),
        "latency_p50_s": pct(lat, .5), "latency_p95_s": pct(lat, .95), "latency_p99_s": pct(lat, .99),
        "latency_mean_s": statistics.mean(lat) if lat else None,
        "completion_tokens": sum(comp) if comp else None,
        "prompt_tokens_mean": statistics.mean([r["prompt_tokens"] for r in recs if r.get("prompt_tokens") is not None]) if recs else None,
        "tok_per_s_per_request_mean": statistics.mean([r["completion_tokens"] / r["latency_s"] for r in recs
                                                       if r.get("completion_tokens") and r.get("latency_s")]) if recs else None,
        "resumed_requests": sum(1 for r in recs if (r.get("attempts") or 1) > 1),
        "errors": sum(1 for r in recs if r.get("error")),
        "requests_per_server": dict(per_server.most_common()),
        "server_request_cv": cv(list(per_server.values())),
    }

def _trajectories(recs):
    """Per session (= one agent trajectory): generation time = sum of its per-turn
    latencies, turns, generated tokens, replicas that served it; attributed to the
    model version of its first turn. This is the upstream A/B's unit of measure
    ("mean generation / trajectory", "slowest trajectory")."""
    by = {}
    for r in sorted(recs, key=lambda r: r.get("t_start") or 0):
        t = by.setdefault(r["session_id"], {"gen_s": 0.0, "turns": 0, "tokens": 0, "mv": r.get("global_steps"), "servers": set()})
        t["gen_s"] += r.get("latency_s") or 0.0
        t["turns"] += 1
        t["tokens"] += r.get("completion_tokens") or 0
        if r.get("server_id"):
            t["servers"].add(r["server_id"])
    return by

def _traj_block(trajs):
    if not trajs: return None
    g = [t["gen_s"] for t in trajs]
    return {
        "trajectories": len(trajs),
        "generation_s_per_trajectory_mean": statistics.mean(g),
        "generation_s_per_trajectory_p95": pct(g, .95),
        "generation_s_slowest_trajectory": max(g),
        "turns_per_trajectory_mean": statistics.mean(t["turns"] for t in trajs),
        "generated_tokens_per_trajectory_mean": statistics.mean(t["tokens"] for t in trajs),
        "trajectories_on_multiple_replicas_share": sum(len(t["servers"]) > 1 for t in trajs) / len(trajs),
    }

def gateway_report(rf: Path) -> dict:
    recs = gateway_records(rf)
    if not recs:
        return {"note": "no logs/gateway-logs.tar.gz (run predates the rollout adapter)"}
    by_step = collections.defaultdict(list)
    for r in recs:
        by_step[r.get("global_steps")].append(r)
    steps = sorted(k for k in by_step if k is not None)
    tail = [s for s in steps if s >= 1] or steps   # model version 0 = warmup/first batch
    trajs = _trajectories(recs)
    tr_by_mv = collections.defaultdict(list)
    for t in trajs.values():
        tr_by_mv[t["mv"]].append(t)
    return {
        "routers": dict(collections.Counter(r.get("router") for r in recs)),
        "all": _lat_block(recs),
        "steady": {"model_versions": tail, **_lat_block([r for s in tail for r in by_step[s]])} if tail else None,
        "per_model_version": {str(s): _lat_block(by_step[s]) for s in steps},
        "trajectories_all": _traj_block(list(trajs.values())),
        "trajectories_steady": _traj_block([t for s in tail for t in tr_by_mv.get(s, [])]) if tail else None,
        "trajectories_per_model_version": {str(s): _traj_block(tr_by_mv[s]) for s in steps if tr_by_mv.get(s)},
        "sessions": len(trajs),
    }

# ---- vLLM stats lines in the driver log ---------------------------------------

STATS = re.compile(
    r"^(?P<ts>\S+)?.*?\((?P<actor>vLLMHttpServer|AsyncvLLMServer|[A-Za-z_]*[Ss]erver[A-Za-z_]*) pid=(?P<pid>\d+)(?:, ip=(?P<ip>[\d.]+))?\).*?"
    r"(?:Avg prompt throughput: (?P<ptps>[\d.]+) tokens/s, Avg generation throughput: (?P<gtps>[\d.]+) tokens/s, )?"
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, GPU KV cache usage: (?P<kv>[\d.]+)%"
    r"(?:, Prefix cache hit rate: (?P<hit>[\d.]+)%)?")

def _ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None

def log_stats_report(rf: Path, tick_s: int = 10) -> dict:
    logs = sorted(glob.glob(str(rf / "logs" / "verl-train-*.log")))
    if not logs:
        return {"note": "no driver log"}
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    ticks = collections.defaultdict(dict)  # tick -> replica -> (running, waiting)
    for line in open(logs[-1], errors="replace"):
        m = STATS.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
        if not m: continue
        key = f"{m['ip'] or 'driver'}:{m['pid']}"
        d = per[key]
        d["running"].append(int(m["running"])); d["waiting"].append(int(m["waiting"])); d["kv"].append(float(m["kv"]))
        if m["hit"] is not None: d["hit"].append(float(m["hit"]))
        if m["gtps"] is not None: d["gtps"].append(float(m["gtps"]))
        t = _ts(m["ts"]) if m["ts"] else None
        if t is not None:
            ticks[int(t // tick_s)][key] = (int(m["running"]), int(m["waiting"]))
    if not per:
        return {"note": "no vLLM stats lines: verl launches the vLLM servers with VLLM_LOGGING_LEVEL=WARN, which hides "
                        "the periodic INFO stats line even with disable_log_stats=False; per-replica numbers come from the "
                        "/metrics snapshots (replica_metrics) instead"}
    cvs, idle_q, multi = [], 0, 0
    for _, reps in ticks.items():
        if len(reps) < 2: continue
        multi += 1
        c = cv([r for r, _ in reps.values()])
        if c is not None: cvs.append(c)
        if any(r == 0 for r, _ in reps.values()) and any(w > 0 for _, w in reps.values()): idle_q += 1
    return {
        "replicas": len(per),
        "per_replica": {k: {"samples": len(v["running"]), "running_mean": statistics.mean(v["running"]),
                            "waiting_mean": statistics.mean(v["waiting"]), "kv_usage_mean_pct": statistics.mean(v["kv"]),
                            "prefix_hit_mean_pct": statistics.mean(v["hit"]) if v["hit"] else None,
                            "gen_tok_per_s_mean": statistics.mean(v["gtps"]) if v["gtps"] else None} for k, v in per.items()},
        "ticks_with_multiple_replicas": multi,
        "running_cv_mean": statistics.mean(cvs) if cvs else None,
        "idle_while_queued_share": idle_q / multi if multi else None,
    }

# ---- /metrics snapshots per replica -------------------------------------------

LINE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+]?[\d.eE+naninf]+)')
HISTS = {"ttft": ["vllm:time_to_first_token_seconds"], "queue": ["vllm:request_queue_time_seconds"],
         "prefill": ["vllm:request_prefill_time_seconds"], "decode": ["vllm:request_decode_time_seconds"],
         "e2e": ["vllm:e2e_request_latency_seconds"]}
COUNTERS = {"prefix_queries": ["vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"],
            "prefix_hits": ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"],
            "preemptions": ["vllm:num_preemptions_total"], "generation_tokens": ["vllm:generation_tokens_total"],
            "prompt_tokens": ["vllm:prompt_tokens_total"], "requests_success": ["vllm:request_success_total"]}

def parse_prom(text):
    """-> {metric_name: {labels_str: value}} (labels kept so histogram buckets stay distinct)"""
    out = collections.defaultdict(dict)
    for line in text.splitlines():
        if not line or line[0] == "#": continue
        m = LINE.match(line)
        if m:
            try: out[m.group(1)][m.group(2) or ""] = float(m.group(3))
            except ValueError: pass
    return out

def _first(d, names):
    for n in names:
        if n in d: return d[n]
    return None

def _hist(snap, base):
    """(count, sum, {le: cumulative}) summed over label sets"""
    c = _first(snap, [base + "_count"]); t = _first(snap, [base + "_sum"])
    cnt = sum(c.values()) if c else None
    tot = sum(t.values()) if t else None
    buckets = collections.defaultdict(float)
    for labels, v in (_first(snap, [base + "_bucket"]) or {}).items():
        le = re.search(r'le="([^"]+)"', labels)
        if le: buckets[float(le.group(1)) if le.group(1) != "+Inf" else float("inf")] += v
    return cnt, tot, dict(buckets)

def _hist_delta(first, last, names):
    for b in names:
        c0, s0, b0 = _hist(first, b); c1, s1, b1 = _hist(last, b)
        if c1 is None: continue
        dc = c1 - (c0 or 0); ds = s1 - (s0 or 0)
        db = {le: b1[le] - b0.get(le, 0) for le in b1}
        def q(p):
            """Upper bound of the histogram bucket holding the p-quantile (Prometheus
            buckets are coarse near zero, so an interpolated value would be fiction)."""
            if not dc or not db: return None
            target = p * dc
            for le in sorted(db):
                if db[le] >= target:
                    return le if le != float("inf") else None
            return None
        return {"count": dc, "mean_s": ds / dc if dc else None, "p50_le_s": q(.5), "p95_le_s": q(.95)}
    return None

def _counter_delta(first, last, names):
    l = _first(last, names)
    if l is None: return None
    f = _first(first, names) or {}
    return sum(l.values()) - sum(f.values())

def replica_metrics_report(rf: Path) -> dict:
    t = rf / "logs" / "replica-metrics.tar.gz"
    if not t.exists():
        return {"note": "no logs/replica-metrics.tar.gz (run predates the replica poller)"}
    snaps = collections.defaultdict(dict)  # server -> ts -> parsed
    with tarfile.open(t) as tar:
        for m in tar.getmembers():
            parts = m.name.split("/")
            if m.name.endswith(".prom") and len(parts) >= 2:
                snaps[parts[-2]][parts[-1][:-5]] = parse_prom(tar.extractfile(m).read().decode(errors="replace"))
    per = {}
    for sid, by_ts in snaps.items():
        ts = sorted(by_ts)
        if len(ts) < 2: continue
        first, last = by_ts[ts[0]], by_ts[ts[-1]]
        row = {"snapshots": len(ts), "window": [ts[0], ts[-1]]}
        for k, names in HISTS.items(): row[k] = _hist_delta(first, last, names)
        for k, names in COUNTERS.items(): row[k] = _counter_delta(first, last, names)
        row["prefix_hit_rate"] = (row["prefix_hits"] / row["prefix_queries"]) if row.get("prefix_queries") else None
        per[sid] = row
    if not per:
        return {"note": "replica-metrics present but <2 snapshots per replica"}
    agg = {}
    for k in HISTS:
        rows = [r[k] for r in per.values() if r.get(k) and r[k]["count"]]
        n = sum(r["count"] for r in rows)
        agg[k] = {"count": n, "mean_s": sum(r["mean_s"] * r["count"] for r in rows) / n if n else None,
                  "p95_le_s_worst_replica": max((r["p95_le_s"] for r in rows if r["p95_le_s"] is not None), default=None)}
    for k in COUNTERS:
        vals = [r[k] for r in per.values() if r.get(k) is not None]
        agg[k] = sum(vals) if vals else None
    agg["prefix_hit_rate"] = (agg["prefix_hits"] / agg["prefix_queries"]) if agg.get("prefix_queries") else None
    # balance across the replicas that served the whole run (hybrid replicas on the
    # trainer GPUs leave after the first batch and would dominate a naive CV)
    full = max(r["snapshots"] for r in per.values())
    gen = [r["generation_tokens"] for r in per.values() if r.get("generation_tokens") and r["snapshots"] >= full // 2]
    agg["generation_tokens_cv_across_replicas"] = cv(gen)
    agg["replicas_in_cv"] = len(gen)
    return {"replicas": len(per), "per_replica": per, "aggregate": agg}

def report(rf) -> dict:
    rf = Path(rf)
    return {"gateway": gateway_report(rf), "vllm_log_stats": log_stats_report(rf), "replica_metrics": replica_metrics_report(rf)}

def _fmt(v):
    if isinstance(v, float): return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"
    return str(v)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_folder"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args(); r = report(a.run_folder)
    if a.json: print(json.dumps(r, indent=2, default=str)); return
    for sec, body in r.items():
        print(f"\n## {sec}")
        for k, v in body.items():
            if isinstance(v, dict) and k in ("per_replica", "per_model_version", "requests_per_server"):
                print(f"- {k}:"); [print(f"    {kk}: {json.dumps(vv, default=_fmt)[:400]}") for kk, vv in v.items()]
            elif isinstance(v, dict):
                print(f"- {k}:"); [print(f"    {kk}: {_fmt(vv) if not isinstance(vv, dict) else json.dumps(vv, default=_fmt)[:300]}") for kk, vv in v.items()]
            else: print(f"- {k}: {_fmt(v)}")

if __name__ == "__main__":
    main()
