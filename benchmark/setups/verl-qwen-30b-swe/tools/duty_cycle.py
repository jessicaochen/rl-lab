#!/usr/bin/env python3
"""GPU duty cycle per role (trainer / sampler) from a run folder.

Joins rlbench's DCGM scrapes (runs/<id>/metrics/dcgm-exporter-<ts>.prom — the
GKE-managed exporter, whose Hostname label is the node name) with the
placement snapshot written by hooks/post-run.sh (runs/<id>/events/placement.json:
node -> role) over the job's wall-clock window, and adds verl's own
trainer-busy fraction from the step metrics. A pod-name Hostname (our old
exporter) is resolved via placement's dcgm_pods map as a fallback. Importable (report() returns a dict) and a CLI.

    python3 tools/duty_cycle.py runs/<run-id>
"""
import glob, json, re, statistics, sys
from datetime import datetime, timezone
from pathlib import Path

LINE = re.compile(r'^(DCGM_FI_[A-Z_]+)\{([^}]*)\}\s+([-\d.eE+]+)')
FIELDS = ["DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_PROF_GR_ENGINE_ACTIVE", "DCGM_FI_PROF_SM_ACTIVE",
          "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "DCGM_FI_PROF_DRAM_ACTIVE", "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_FB_USED"]

def _labels(s):
    return dict(re.findall(r'(\w+)="([^"]*)"', s))

def _ts(name):
    m = re.search(r"(\d{8}T\d{6}Z)", name)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc) if m else None

def load_samples(rf: Path, window=None):
    """-> list of (ts, exporter_pod, gpu, field, value)"""
    out = []
    for f in sorted(glob.glob(str(rf / "metrics" / "dcgm-exporter-*.prom"))):
        ts = _ts(f)
        if window and ts and not (window[0] <= ts <= window[1]):
            continue
        for line in open(f, errors="replace"):
            m = LINE.match(line)
            if not m or m.group(1) not in FIELDS:
                continue
            lab = _labels(m.group(2))
            out.append((ts, lab.get("Hostname"), lab.get("gpu"), m.group(1), float(m.group(3))))
    return out

def job_window(rf: Path):
    r = json.loads((rf / "result.json").read_text())
    js = r.get("job_status", {})
    if js.get("startTime") and js.get("completionTime"):
        p = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return p(js["startTime"]), p(js["completionTime"])
    return None

def trainer_busy_fraction(rf: Path):
    logs = sorted(glob.glob(str(rf / "logs" / "verl-train-*.log")))
    if not logs: return None
    fr = []
    for line in open(logs[-1], errors="replace"):
        m = re.search(r"step:(\d+) - (.*)$", line)
        if not m: continue
        kv = dict(re.findall(r"(timing_s/(?:step|old_log_prob|update_actor|update_weights)):([\d.]+)", m.group(2)))
        if "timing_s/step" in kv:
            busy = sum(float(kv.get(k, 0)) for k in ("timing_s/old_log_prob", "timing_s/update_actor", "timing_s/update_weights"))
            fr.append(busy / float(kv["timing_s/step"]))
    return statistics.mean(fr) if fr else None

def report(rf: Path) -> dict:
    rf = Path(rf)
    pj = rf / "events" / "placement.json"
    placement = json.loads(pj.read_text()) if pj.exists() else {}
    pod_to_node = {p["pod"]: p["node"] for p in placement.get("dcgm_pods", [])}
    node_roles = placement.get("node_roles", {})
    window = job_window(rf)
    samples = load_samples(rf, window)
    per = {}  # role -> field -> list
    hosts = {}  # role -> set(node)
    for ts, host, gpu, field, val in samples:
        role = node_roles.get(host) or node_roles.get(pod_to_node.get(host), "unattributed")
        per.setdefault(role, {}).setdefault(field, []).append(val)
        hosts.setdefault(role, set()).add(host)
    out = {"window_utc": [w.isoformat() for w in window] if window else None,
           "n_scrapes": len({s[0] for s in samples}), "roles": {}}
    for role, fields in per.items():
        util = fields.get("DCGM_FI_DEV_GPU_UTIL", [])
        out["roles"][role] = {
            "gpu_util_mean_pct": round(statistics.mean(util), 1) if util else None,
            "gpu_util_frac_above_50": round(sum(u > 50 for u in util) / len(util), 2) if util else None,
            "gr_engine_active_mean": round(statistics.mean(fields["DCGM_FI_PROF_GR_ENGINE_ACTIVE"]), 3) if fields.get("DCGM_FI_PROF_GR_ENGINE_ACTIVE") else None,
            "sm_active_mean": round(statistics.mean(fields["DCGM_FI_PROF_SM_ACTIVE"]), 3) if fields.get("DCGM_FI_PROF_SM_ACTIVE") else None,
            "tensor_active_mean": round(statistics.mean(fields["DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"]), 3) if fields.get("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE") else None,
            "dram_active_mean": round(statistics.mean(fields["DCGM_FI_PROF_DRAM_ACTIVE"]), 3) if fields.get("DCGM_FI_PROF_DRAM_ACTIVE") else None,
            "power_mean_w": round(statistics.mean(fields["DCGM_FI_DEV_POWER_USAGE"]), 0) if fields.get("DCGM_FI_DEV_POWER_USAGE") else None,
            "gpu_samples": len(util),
            "nodes": sorted(hosts.get(role, ())),  # 'unattributed' = GPU nodes no Ray actor ran on (idle capacity)
        }
    out["trainer_busy_fraction_verl"] = (round(trainer_busy_fraction(rf), 3) if trainer_busy_fraction(rf) is not None else None)
    if not placement:
        out["note"] = "no events/placement.json (run predates the placement hook): GPUs unattributed"
    return out

if __name__ == "__main__":
    print(json.dumps(report(Path(sys.argv[1])), indent=2))
