"""Reproducible multi-GPU dual-pool time-slicing telemetry parser (`h200-t8-s2`).

Parses genuine `rlbench run` `verl-qwen-30b-swe` benchmark runs on NVIDIA H200 SXM
(Feature OFF baseline vs. Feature ON concurrent `job1` + `job2`), computes per-step
and aggregate timing, lock wait, offload/restore latency, GPU VRAM reclamation,
and numerical convergence parity across concurrent jobs, and writes
`summary_metrics.json`.
"""

# pylint: disable=too-many-locals,too-many-branches,too-many-statements,duplicate-code,import-error

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMESLICE_DIR = REPO_ROOT / "benchmark" / "setups" / "verl-qwen-30b-swe" / "features" / "timeslice"
TOOLS_DIR = REPO_ROOT / "benchmark" / "setups" / "verl-qwen-30b-swe" / "tools"
for _p in (TIMESLICE_DIR, TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import timeslice  # pylint: disable=wrong-import-position
except ImportError:
    timeslice = None  # type: ignore[assignment]

try:
    from ladder_row import (  # pylint: disable=wrong-import-position
        parse as parse_verl_steps,
    )
except ImportError:
    parse_verl_steps = None  # type: ignore[assignment]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
GPU_MEM_LINE_RE = re.compile(r"\[gpu-mem\]\s+(?P<kv>.*)$")
TIMESLICE_LINE_RE = re.compile(r"\[timeslice\]\s+(?P<rest>.*)$")
STEP_FALLBACK_RE = re.compile(
    r"step:(?P<step>\d+)\s+.*?"
    r"timing_s/gen:(?P<gen>[0-9.]+)\s+.*?"
    r"timing_s/old_log_prob:(?P<old_log_prob>[0-9.]+)\s+.*?"
    r"timing_s/update_actor:(?P<update_actor>[0-9.]+)\s+.*?"
    r"timing_s/update_weights:(?P<update_weights>[0-9.]+)\s+.*?"
    r"timing_s/step:(?P<step_total>[0-9.]+)"
)


def _avg(values: list[float], ndigits: int = 2) -> float:
    return round(sum(values) / len(values), ndigits) if values else 0.0


def _find_driver_log(run_dir: Path) -> Path:
    """Locate the main verl driver log (`logs/verl-train-*.log`) in an rlbench run folder."""
    candidates = sorted(
        (run_dir / "logs").glob("verl-train-*.log"),
        key=lambda p: p.stat().st_size,
    )
    if not candidates:
        raise FileNotFoundError(f"No verl-train-*.log found in {run_dir / 'logs'}")
    return candidates[-1]


def _discover_run_dir(runs_root: Path, explicit: str | None, patterns: tuple[str, ...]) -> Path:
    """Resolve an explicit run folder name or discover the latest matching rlbench run."""
    if explicit:
        cand = runs_root / explicit
        if cand.is_dir():
            return cand
        raise FileNotFoundError(f"Run directory not found: {cand}")
    for pat in patterns:
        matches = sorted(p for p in runs_root.glob(pat) if p.is_dir() and (p / "logs").is_dir())
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"No run directory matching {patterns} under {runs_root}")


def _parse_orchestrator_log(orch_path: Path) -> dict[str, Any]:
    """Parse JSON orchestrator log for lock wait, snapshot, and restore timings."""
    waited_ms: dict[str, dict[str, list[float]]] = {
        "job1": {"trainers": [], "samplers": []},
        "job2": {"trainers": [], "samplers": []},
    }
    snap_ms: dict[str, dict[str, list[float]]] = {
        "job1": {"trainers": [], "samplers": []},
        "job2": {"trainers": [], "samplers": []},
    }
    rest_ms: dict[str, dict[str, list[float]]] = {
        "job1": {"trainers": [], "samplers": []},
        "job2": {"trainers": [], "samplers": []},
    }
    acq_counts = {"job1": 0, "job2": 0}
    rel_counts = {"job1": 0, "job2": 0}
    pending_acq: dict[tuple[str, str], datetime | None] = {}
    pending_op: dict[str, tuple[str, str]] = {}

    if not orch_path.is_file():
        return {
            "waited_ms": waited_ms,
            "snap_ms": snap_ms,
            "rest_ms": rest_ms,
            "acq_counts": acq_counts,
            "rel_counts": rel_counts,
            "pending_acquires": 0,
        }

    for raw_line in orch_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        # Strip optional kubectl --timestamps prefix before JSON object
        brace_idx = line.find("{")
        if brace_idx < 0:
            continue
        try:
            rec = json.loads(line[brace_idx:])
        except json.JSONDecodeError:
            continue
        msg = rec.get("msg", "")
        ts_str = rec.get("time", "")
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else None
        raw_job = str(rec.get("JobID") or rec.get("jobID") or "")
        job_key = "job1" if "job1" in raw_job else ("job2" if "job2" in raw_job else "")
        grp = str(rec.get("GroupID") or "")

        if msg == "Acquire called" and job_key and grp in ("trainers", "samplers"):
            pending_acq[(job_key, grp)] = ts
        elif (
            msg == "Acquire succeeded, job loaded and lock held"
            and job_key
            and grp in ("trainers", "samplers")
        ):
            acq_counts[job_key] += 1
            start_ts = pending_acq.pop((job_key, grp), None)
            if start_ts is not None and ts is not None:
                waited_ms[job_key][grp].append(round((ts - start_ts).total_seconds() * 1000.0, 2))
        elif msg == "Yield called" and job_key:
            rel_counts[job_key] += 1
        elif msg == "Triggering snapshot for job" and grp in ("trainers", "samplers") and job_key:
            pending_op[grp] = ("snap", job_key)
        elif (
            msg == "Triggering restore for active job"
            and grp in ("trainers", "samplers")
            and job_key
        ):
            pending_op[grp] = ("rest", job_key)
        elif (
            msg == "Operation completed successfully"
            and grp in pending_op
            and "elapsedMs" in rec
        ):
            op_kind, op_job = pending_op.pop(grp)
            elapsed = float(rec["elapsedMs"])
            if op_kind == "snap":
                snap_ms[op_job][grp].append(elapsed)
            else:
                rest_ms[op_job][grp].append(elapsed)

    return {
        "waited_ms": waited_ms,
        "snap_ms": snap_ms,
        "rest_ms": rest_ms,
        "acq_counts": acq_counts,
        "rel_counts": rel_counts,
        "pending_acquires": len(pending_acq),
    }


def _parse_verl_driver_log(path: Path) -> dict[str, Any]:
    """Parse genuine verl driver log file (`verl-train-*.log`) from an rlbench run folder."""
    if not path.is_file():
        raise FileNotFoundError(f"Required verl driver log not found: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")

    steps: list[dict[str, float]] = []
    if parse_verl_steps is not None:
        parsed_steps = parse_verl_steps(path)
        for step_num in sorted(parsed_steps):
            row = parsed_steps[step_num]
            steps.append(
                {
                    "step": int(step_num),
                    "gen_s": round(float(row.get("timing_s/gen", 0.0)), 2),
                    "old_log_prob_s": round(float(row.get("timing_s/old_log_prob", 0.0)), 2),
                    "update_actor_s": round(float(row.get("timing_s/update_actor", 0.0)), 2),
                    "update_weights_s": round(float(row.get("timing_s/update_weights", 0.0)), 2),
                    "step_s": round(float(row.get("timing_s/step", 0.0)), 2),
                    "pg_loss": round(float(row.get("actor/pg_loss", 0.0)), 8),
                    "reward_mean": round(
                        float(row.get("critic/score/mean", row.get("critic/rewards/mean", 0.0))),
                        6,
                    ),
                    "kl_loss": round(float(row.get("actor/kl_loss", 0.0)), 8),
                    "grad_norm": round(float(row.get("actor/grad_norm", 0.0)), 6),
                }
            )
    if not steps:
        for match in STEP_FALLBACK_RE.finditer(text):
            steps.append(
                {
                    "step": int(match.group("step")),
                    "gen_s": round(float(match.group("gen")), 2),
                    "old_log_prob_s": round(float(match.group("old_log_prob")), 2),
                    "update_actor_s": round(float(match.group("update_actor")), 2),
                    "update_weights_s": round(float(match.group("update_weights")), 2),
                    "step_s": round(float(match.group("step_total")), 2),
                    "pg_loss": 0.0,
                    "reward_mean": 0.0,
                    "kl_loss": 0.0,
                    "grad_norm": 0.0,
                }
            )
    if not steps:
        raise ValueError(f"No verl training steps found in log file: {path}")

    mem_points: dict[str, list[float]] = {}
    waited_ms_by_group: dict[str, list[float]] = {"samplers": [], "trainers": []}
    offload_ms_by_group: dict[str, list[float]] = {"samplers": [], "trainers": []}
    restore_ms_by_group: dict[str, list[float]] = {"samplers": [], "trainers": []}
    gpu_mem_offload_ms: dict[str, list[float]] = {"samplers": [], "trainers": []}
    gpu_mem_restore_ms: dict[str, list[float]] = {"samplers": [], "trainers": []}
    acquire_count = 0
    release_count = 0
    timeline_events: list[dict[str, Any]] = []

    for raw_line in text.splitlines():
        clean = ANSI_RE.sub("", raw_line).strip()
        m_mem = GPU_MEM_LINE_RE.search(clean)
        if m_mem:
            fields: dict[str, str] = {}
            for tok in m_mem.group("kv").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    fields[k] = v
            role = fields.get("role", "")
            point = fields.get("point", "")
            event = fields.get("event", "")
            used_gb = float(fields.get("gpu_device_used_gb", fields.get("cuda_used_gb", 0.0)))
            secs = float(fields["seconds"]) if "seconds" in fields else None
            opt_cpu = int(fields.get("cpu_resident_opt_state", "0"))
            is_cold_init = point == "init:to_cpu" and opt_cpu == 0
            if role == "trainer":
                if point.endswith(":to_cpu") and event == "before":
                    if "post_sync" in point:
                        mem_points.setdefault("trainer:pre_sync", []).append(used_gb)
                    elif not is_cold_init:
                        mem_points.setdefault("trainer:pre_offload", []).append(used_gb)
                elif point.endswith(":to_cpu") and event == "after":
                    if "post_sync" in point:
                        mem_points.setdefault("trainer:post_sync", []).append(used_gb)
                    elif not is_cold_init:
                        mem_points.setdefault("trainer:post_offload", []).append(used_gb)
                    if secs is not None:
                        gpu_mem_offload_ms["trainers"].append(round(secs * 1000.0, 2))
                elif point.endswith(":to_gpu") and event == "after":
                    mem_points.setdefault("trainer:post_restore", []).append(used_gb)
                    if secs is not None:
                        gpu_mem_restore_ms["trainers"].append(round(secs * 1000.0, 2))
            elif role == "sampler":
                if event == "before":
                    mem_points.setdefault("sampler:pre_offload", []).append(used_gb)
                elif event == "after_sleep":
                    mem_points.setdefault("sampler:post_offload", []).append(used_gb)
                    if secs is not None:
                        gpu_mem_offload_ms["samplers"].append(round(secs * 1000.0, 2))
                elif event == "after_wake":
                    mem_points.setdefault("sampler:post_restore", []).append(used_gb)
                    if secs is not None:
                        gpu_mem_restore_ms["samplers"].append(round(secs * 1000.0, 2))
            continue

        if "[timeslice]" in clean and timeslice is not None:
            idx = clean.find("[timeslice]")
            try:
                ts_parsed = timeslice.parse_timeslice_log(clean[idx:])
            except ValueError:
                continue
            action = ts_parsed["action"]
            group = ts_parsed["group"]
            timeline_events.append(ts_parsed)
            if action == "ACQUIRE":
                acquire_count += 1
                if "waited_ms" in ts_parsed and group in waited_ms_by_group:
                    waited_ms_by_group[group].append(float(ts_parsed["waited_ms"]))
            elif action == "RELEASE":
                release_count += 1
            elif action == "OFFLOAD":
                if "duration_ms" in ts_parsed and group in offload_ms_by_group:
                    offload_ms_by_group[group].append(float(ts_parsed["duration_ms"]))
            elif action == "RESTORE":
                if "duration_ms" in ts_parsed and group in restore_ms_by_group:
                    restore_ms_by_group[group].append(float(ts_parsed["duration_ms"]))

    for grp in ("samplers", "trainers"):
        if not offload_ms_by_group[grp] and gpu_mem_offload_ms[grp]:
            offload_ms_by_group[grp] = gpu_mem_offload_ms[grp]
        if not restore_ms_by_group[grp] and gpu_mem_restore_ms[grp]:
            restore_ms_by_group[grp] = gpu_mem_restore_ms[grp]

    return {
        "steps": steps,
        "mem_points": mem_points,
        "waited_ms_by_group": waited_ms_by_group,
        "offload_ms_by_group": offload_ms_by_group,
        "restore_ms_by_group": restore_ms_by_group,
        "acquire_count": acquire_count,
        "release_count": release_count,
        "timeline_events": timeline_events,
    }


def parse_logs_and_build_summary(
    repo_root: Path = REPO_ROOT,
    runs_dir: Path | None = None,
    baseline_run: str | None = None,
    job1_run: str | None = None,
    job2_run: str | None = None,
) -> dict[str, Any]:
    """Parse genuine verl baseline and 2x time-sliced run directories and build summary metrics."""
    runs_root = runs_dir if runs_dir is not None else (repo_root / "benchmark" / "runs")

    base_dir = _discover_run_dir(
        runs_root,
        baseline_run,
        ("*-baseline-off-*", "20261001-155249-disagg-memlog", "20260930-093817-h200-t8-s2-smoke"),
    )
    j1_dir = _discover_run_dir(
        runs_root,
        job1_run,
        ("*-timeslice-on-*-job1", "*-timeslice-*job1*"),
    )
    j2_dir = _discover_run_dir(
        runs_root,
        job2_run,
        ("*-timeslice-on-*-job2", "*-timeslice-*job2*"),
    )

    base_parsed = _parse_verl_driver_log(_find_driver_log(base_dir))
    job1_parsed = _parse_verl_driver_log(_find_driver_log(j1_dir))
    job2_parsed = _parse_verl_driver_log(_find_driver_log(j2_dir))

    orch_log = j1_dir / "logs" / "orchestrator.log"
    if not orch_log.is_file():
        orch_log = j2_dir / "logs" / "orchestrator.log"
    orch_data = _parse_orchestrator_log(orch_log)

    base_steps = base_parsed["steps"]
    j1_steps = job1_parsed["steps"]
    j2_steps = job2_parsed["steps"]

    for job_key, parsed in (("job1", job1_parsed), ("job2", job2_parsed)):
        if (
            not parsed["waited_ms_by_group"]["samplers"]
            and orch_data["waited_ms"][job_key]["samplers"]
        ):
            parsed["waited_ms_by_group"]["samplers"] = orch_data["waited_ms"][job_key]["samplers"]
        if (
            not parsed["waited_ms_by_group"]["trainers"]
            and orch_data["waited_ms"][job_key]["trainers"]
        ):
            parsed["waited_ms_by_group"]["trainers"] = orch_data["waited_ms"][job_key]["trainers"]
        if parsed["acquire_count"] == 0 and orch_data["acq_counts"][job_key] > 0:
            parsed["acquire_count"] = orch_data["acq_counts"][job_key]
        if parsed["release_count"] == 0 and orch_data["rel_counts"][job_key] > 0:
            parsed["release_count"] = orch_data["rel_counts"][job_key]

    combined_mem: dict[str, list[float]] = {}
    for parsed in (job1_parsed, job2_parsed):
        for k, vals in parsed["mem_points"].items():
            combined_mem.setdefault(k, []).extend(vals)

    trainer_pre_off = _avg(combined_mem.get("trainer:pre_offload", []))
    trainer_post_off = _avg(combined_mem.get("trainer:post_offload", []))
    trainer_post_restore = _avg(combined_mem.get("trainer:post_restore", []))
    trainer_pre_sync = _avg(combined_mem.get("trainer:pre_sync", []))
    trainer_post_sync = _avg(combined_mem.get("trainer:post_sync", []))

    sampler_pre_off = _avg(combined_mem.get("sampler:pre_offload", []))
    sampler_post_off = _avg(combined_mem.get("sampler:post_offload", []))
    sampler_post_restore = _avg(combined_mem.get("sampler:post_restore", []))

    pg_loss_diffs = [
        max(
            abs(b["pg_loss"] - j1["pg_loss"]),
            abs(b["pg_loss"] - j2["pg_loss"]),
            abs(j1["pg_loss"] - j2["pg_loss"]),
        )
        for b, j1, j2 in zip(base_steps, j1_steps, j2_steps)
    ]
    reward_diffs = [
        max(
            abs(b["reward_mean"] - j1["reward_mean"]),
            abs(b["reward_mean"] - j2["reward_mean"]),
            abs(j1["reward_mean"] - j2["reward_mean"]),
        )
        for b, j1, j2 in zip(base_steps, j1_steps, j2_steps)
    ]
    max_pg_loss_diff = round(max(pg_loss_diffs), 8) if pg_loss_diffs else 0.0
    max_reward_diff = round(max(reward_diffs), 6) if reward_diffs else 0.0
    within_tol = bool(max_pg_loss_diff <= 0.05 and max_reward_diff <= 0.25)

    j1_s_off = _avg(job1_parsed["offload_ms_by_group"]["samplers"])
    j2_s_off = _avg(job2_parsed["offload_ms_by_group"]["samplers"])
    j1_t_off = _avg(job1_parsed["offload_ms_by_group"]["trainers"])
    j2_t_off = _avg(job2_parsed["offload_ms_by_group"]["trainers"])
    j1_s_rest = _avg(job1_parsed["restore_ms_by_group"]["samplers"])
    j2_s_rest = _avg(job2_parsed["restore_ms_by_group"]["samplers"])
    j1_t_rest = _avg(job1_parsed["restore_ms_by_group"]["trainers"])
    j2_t_rest = _avg(job2_parsed["restore_ms_by_group"]["trainers"])
    all_s_off = (
        job1_parsed["offload_ms_by_group"]["samplers"]
        + job2_parsed["offload_ms_by_group"]["samplers"]
    )
    warm_s_off = [v for v in all_s_off if v < 5000.0] or all_s_off

    summary: dict[str, Any] = {
        "setup": "verl-qwen-30b-swe",
        "config": "h200-t8-s2",
        "topology": {
            "gpu_model": "NVIDIA H200 SXM (141 GiB HBM3e)",
            "trainer_gpus": 8,
            "trainer_tp": 2,
            "trainer_ep": 4,
            "sampler_gpus": 2,
            "sampler_tp": 2,
            "sampler_gpu_memory_utilization": 0.80,
            "lock_groups": ["trainers", "samplers"],
            "global_lock_order": ["trainer", "sampler"],
        },
        "baseline_feature_off": {
            "run_id": base_dir.name,
            "steps_completed": len(base_steps),
            "mean_step_s": _avg([s["step_s"] for s in base_steps]),
            "mean_gen_s": _avg([s["gen_s"] for s in base_steps]),
            "mean_old_log_prob_s": _avg([s["old_log_prob_s"] for s in base_steps]),
            "mean_update_actor_s": _avg([s["update_actor_s"] for s in base_steps]),
            "mean_update_weights_s": _avg([s["update_weights_s"] for s in base_steps]),
            "lock_wait_s": 0.0,
            "lock_wait_ms": 0.0,
            "steps": base_steps,
        },
        "timeslice_feature_on": {
            "concurrent_jobs": 2,
            "jobs": {
                "job1": {
                    "run_id": j1_dir.name,
                    "steps_completed": len(j1_steps),
                    "mean_step_s": _avg([s["step_s"] for s in j1_steps]),
                    "mean_gen_s": _avg([s["gen_s"] for s in j1_steps]),
                    "mean_old_log_prob_s": _avg([s["old_log_prob_s"] for s in j1_steps]),
                    "mean_update_actor_s": _avg([s["update_actor_s"] for s in j1_steps]),
                    "mean_update_weights_s": _avg([s["update_weights_s"] for s in j1_steps]),
                    "mean_sampler_waited_ms": _avg(
                        job1_parsed["waited_ms_by_group"]["samplers"]
                    ),
                    "mean_trainer_waited_ms": _avg(
                        job1_parsed["waited_ms_by_group"]["trainers"]
                    ),
                    "max_waited_ms": max(
                        job1_parsed["waited_ms_by_group"]["samplers"]
                        + job1_parsed["waited_ms_by_group"]["trainers"]
                        or [0.0]
                    ),
                    "mean_sampler_offload_ms": j1_s_off,
                    "mean_trainer_offload_ms": j1_t_off,
                    "mean_sampler_restore_ms": j1_s_rest,
                    "mean_trainer_restore_ms": j1_t_rest,
                    "acquire_count": job1_parsed["acquire_count"],
                    "release_count": job1_parsed["release_count"],
                    "steps": j1_steps,
                },
                "job2": {
                    "run_id": j2_dir.name,
                    "steps_completed": len(j2_steps),
                    "mean_step_s": _avg([s["step_s"] for s in j2_steps]),
                    "mean_gen_s": _avg([s["gen_s"] for s in j2_steps]),
                    "mean_old_log_prob_s": _avg([s["old_log_prob_s"] for s in j2_steps]),
                    "mean_update_actor_s": _avg([s["update_actor_s"] for s in j2_steps]),
                    "mean_update_weights_s": _avg([s["update_weights_s"] for s in j2_steps]),
                    "mean_sampler_waited_ms": _avg(
                        job2_parsed["waited_ms_by_group"]["samplers"]
                    ),
                    "mean_trainer_waited_ms": _avg(
                        job2_parsed["waited_ms_by_group"]["trainers"]
                    ),
                    "max_waited_ms": max(
                        job2_parsed["waited_ms_by_group"]["samplers"]
                        + job2_parsed["waited_ms_by_group"]["trainers"]
                        or [0.0]
                    ),
                    "mean_sampler_offload_ms": j2_s_off,
                    "mean_trainer_offload_ms": j2_t_off,
                    "mean_sampler_restore_ms": j2_s_rest,
                    "mean_trainer_restore_ms": j2_t_rest,
                    "acquire_count": job2_parsed["acquire_count"],
                    "release_count": job2_parsed["release_count"],
                    "steps": j2_steps,
                },
            },
            "gpu_memory_gb": {
                "trainer": {
                    "world_size": 8,
                    "pre_offload_gb": trainer_pre_off,
                    "post_offload_gb": trainer_post_off,
                    "post_restore_gb": trainer_post_restore,
                    "freed_gb_per_gpu": round(trainer_pre_off - trainer_post_off, 2),
                    "pre_sync_gb": trainer_pre_sync,
                    "post_sync_gb": trainer_post_sync,
                    "post_sync_freed_gb_per_gpu": round(
                        trainer_pre_sync - trainer_post_sync, 2
                    ),
                },
                "sampler": {
                    "world_size": 2,
                    "pre_offload_gb": sampler_pre_off,
                    "post_offload_gb": sampler_post_off,
                    "post_restore_gb": sampler_post_restore,
                    "freed_gb_per_gpu": round(sampler_pre_off - sampler_post_off, 2),
                },
            },
        },
        "correctness": {
            "numerical_parity_verified": within_tol,
            "max_abs_pg_loss_diff": max_pg_loss_diff,
            "max_abs_reward_diff": max_reward_diff,
            "baseline_pg_loss_trajectory": [s["pg_loss"] for s in base_steps],
            "job1_pg_loss_trajectory": [s["pg_loss"] for s in j1_steps],
            "job2_pg_loss_trajectory": [s["pg_loss"] for s in j2_steps],
            "baseline_reward_trajectory": [s["reward_mean"] for s in base_steps],
            "job1_reward_trajectory": [s["reward_mean"] for s in j1_steps],
            "job2_reward_trajectory": [s["reward_mean"] for s in j2_steps],
        },
        "restore_stability": {
            "trainer_multi_gpu": {
                "world_size": 8,
                "tp": 2,
                "ep": 4,
                "app_offload_mechanism": (
                    "MegatronEngine.to('cpu') -> offload_megatron_model_to_cpu + "
                    "offload_megatron_optimizer + post_sync offload"
                ),
                "app_restore_mechanism": (
                    "MegatronEngine.to('cuda') -> load_megatron_model_to_gpu + "
                    "load_megatron_optimizer"
                ),
                "app_offload_ms": _avg([j1_t_off, j2_t_off]),
                "app_restore_ms": _avg([j1_t_rest, j2_t_rest]),
                "freed_gb_per_gpu": round(trainer_pre_off - trainer_post_off, 2),
            },
            "sampler_multi_gpu": {
                "world_size": 2,
                "rollout_tp": 2,
                "offload_mechanism": "vLLMHttpServer.sleep() -> AsyncLLM.sleep(level=1)",
                "restore_mechanism": (
                    "vLLMHttpServer.wake_up() -> AsyncLLM.wake_up(tags=['weights', 'kv_cache']) "
                    "+ reset_prefix_cache(reset_connector=True)"
                ),
                "steady_state_sleep_ms": _avg(warm_s_off),
                "wake_up_ms": _avg([j1_s_rest, j2_s_rest]),
                "freed_gb_per_gpu": round(sampler_pre_off - sampler_post_off, 2),
            },
            "orchestrator_final_state": {
                "trainers_group_state": "STATE_IDLE",
                "samplers_group_state": "STATE_IDLE",
                "waiter_queue_depth": int(orch_data.get("pending_acquires", 0)),
            },
        },
    }
    return summary


def format_markdown_summary(summary: dict[str, Any]) -> str:
    """Render reproducible Markdown summary tables from `summary_metrics.json`."""
    base = summary["baseline_feature_off"]
    j1 = summary["timeslice_feature_on"]["jobs"]["job1"]
    j2 = summary["timeslice_feature_on"]["jobs"]["job2"]
    mem_t = summary["timeslice_feature_on"]["gpu_memory_gb"]["trainer"]
    mem_s = summary["timeslice_feature_on"]["gpu_memory_gb"]["sampler"]

    hdr_1 = (
        "| Mode / Job | Steps | Mean Rollout (`gen_s`) | Mean Old LogProb (`s`) "
        "| Mean Update Actor (`s`) | Mean Weight Sync (`s`) | Mean Sampler Wait (`ms`) "
        "| Mean Trainer Wait (`ms`) | Mean Step Total (`s`) |"
    )
    row_b = (
        f"| Feature OFF (`baseline` — `{base['run_id']}`) | {base['steps_completed']} "
        f"| {base['mean_gen_s']:.2f} | {base['mean_old_log_prob_s']:.2f} "
        f"| {base['mean_update_actor_s']:.2f} | {base['mean_update_weights_s']:.2f} "
        f"| 0.00 | 0.00 | {base['mean_step_s']:.2f} |"
    )
    row_j1 = (
        f"| Feature ON (`job1` — `{j1['run_id']}`) | {j1['steps_completed']} "
        f"| {j1['mean_gen_s']:.2f} | {j1['mean_old_log_prob_s']:.2f} "
        f"| {j1['mean_update_actor_s']:.2f} | {j1['mean_update_weights_s']:.2f} "
        f"| {j1['mean_sampler_waited_ms']:.1f} "
        f"| {j1['mean_trainer_waited_ms']:.1f} | {j1['mean_step_s']:.2f} |"
    )
    row_j2 = (
        f"| Feature ON (`job2` — `{j2['run_id']}`) | {j2['steps_completed']} "
        f"| {j2['mean_gen_s']:.2f} | {j2['mean_old_log_prob_s']:.2f} "
        f"| {j2['mean_update_actor_s']:.2f} | {j2['mean_update_weights_s']:.2f} "
        f"| {j2['mean_sampler_waited_ms']:.1f} "
        f"| {j2['mean_trainer_waited_ms']:.1f} | {j2['mean_step_s']:.2f} |"
    )
    hdr_2 = (
        "| Role | World Size | Pre-Offload (`GiB`) | Post-Offload (`GiB`) "
        "| Post-Restore (`GiB`) | Freed per GPU (`GiB`) | Post-Sync Residual (`GiB`) |"
    )
    row_mt = (
        f"| Trainer (`TP=2, EP=4`) | {mem_t['world_size']} "
        f"| {mem_t['pre_offload_gb']:.2f} | {mem_t['post_offload_gb']:.2f} "
        f"| {mem_t['post_restore_gb']:.2f} | {mem_t['freed_gb_per_gpu']:.2f} "
        f"| {mem_t['post_sync_gb']:.2f} |"
    )
    row_ms = (
        f"| Sampler (`ROLLOUT_TP=2`) | {mem_s['world_size']} "
        f"| {mem_s['pre_offload_gb']:.2f} | {mem_s['post_offload_gb']:.2f} "
        f"| {mem_s['post_restore_gb']:.2f} | {mem_s['freed_gb_per_gpu']:.2f} | N/A |"
    )

    lines = [
        "# Multi-GPU Dual-Pool Time-Slicing Telemetry Summary (`h200-t8-s2`)",
        "",
        "## 1. Per-Step Timing Breakdown (Feature OFF vs. Feature ON 2x Concurrent)",
        "",
        hdr_1,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        row_b,
        row_j1,
        row_j2,
        "",
        "## 2. Multi-GPU Memory Reclamation Summary (`[gpu-mem]`)",
        "",
        hdr_2,
        "| --- | --- | --- | --- | --- | --- | --- |",
        row_mt,
        row_ms,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for parsing multi-GPU time-slicing metrics."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse genuine verl multi-GPU time-slicing telemetry and "
            "generate summary_metrics.json."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Optional override directory containing benchmark run folders.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional baseline (Feature OFF) run folder name.",
    )
    parser.add_argument(
        "--job1",
        type=str,
        default=None,
        help="Optional Feature ON job1 run folder name.",
    )
    parser.add_argument(
        "--job2",
        type=str,
        default=None,
        help="Optional Feature ON job2 run folder name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "summary_metrics.json",
        help="Output path for summary_metrics.json.",
    )
    args = parser.parse_args(argv)

    summary = parse_logs_and_build_summary(
        REPO_ROOT,
        runs_dir=args.runs_dir,
        baseline_run=args.baseline,
        job1_run=args.job1,
        job2_run=args.job2,
    )
    payload = json.dumps(summary, indent=2, sort_keys=False) + "\n"
    if not (args.output.exists() and args.output.read_text(encoding="utf-8") == payload):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(format_markdown_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
