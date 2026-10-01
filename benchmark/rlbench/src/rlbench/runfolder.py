"""Run folder layout and provenance recording.

Everything cluster-identifying (context name, project, node inventory, image
digests) is recorded HERE and only here — setup folders stay portable.
"""

from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from . import kube


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunFolder:
    def __init__(self, root: Path, run_id: str):
        self.run_id = run_id
        self.path = root / run_id
        self.config = self.path / "config"
        self.rendered = self.config / "rendered"
        self.logs = self.path / "logs"
        self.events = self.path / "events"
        self.metrics = self.path / "metrics"
        for d in (self.config, self.rendered, self.logs, self.events, self.metrics):
            d.mkdir(parents=True, exist_ok=True)
        # wall clock, not time.monotonic(): CLOCK_MONOTONIC pauses while the
        # host is suspended, which silently shrank a 3h run to "634s"
        self._timings: dict[str, float] = {}
        self._stamps: dict[str, str] = {"start": _utc_now()}
        self._t0 = time.time()

    @classmethod
    def create(cls, root: str | Path, name: str) -> "RunFolder":
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(Path(root), f"{ts}-{name}")

    # -- provenance -----------------------------------------------------

    def record_setup_ref(self, setup_root: Path) -> None:
        sha = subprocess.run(
            ["git", "-C", str(setup_root), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or "unknown"
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(setup_root), "status", "--porcelain"],
                capture_output=True, text=True,
            ).stdout.strip()
        )
        (self.config / "setup-ref.json").write_text(
            json.dumps({"path": str(setup_root), "git_sha": sha, "dirty": dirty}, indent=2)
        )

    def record_cluster_identity(self) -> None:
        identity = {
            "context": kube.current_context(),
            "nodes": [
                {
                    "name": n["metadata"]["name"],
                    "machine": n["metadata"]["labels"].get("node.kubernetes.io/instance-type"),
                    "nodepool": n["metadata"]["labels"].get("cloud.google.com/gke-nodepool"),
                    "spot": n["metadata"]["labels"].get("cloud.google.com/gke-spot"),
                    "gpu": n["status"]["allocatable"].get("nvidia.com/gpu"),
                }
                for n in kube.get_json("nodes").get("items", [])
            ],
        }
        (self.config / "cluster.json").write_text(json.dumps(identity, indent=2))

    def record_run_config(self, config_file: Path | None) -> None:
        if config_file is None:
            return
        if config_file.is_dir():
            shutil.copytree(config_file, self.config / config_file.name, dirs_exist_ok=True)
        else:
            (self.config / config_file.name).write_text(config_file.read_text())

    def write_rendered(self, name: str, docs: list[dict]) -> Path:
        p = self.rendered / name
        p.write_text(yaml.safe_dump_all(docs, sort_keys=False))
        return p

    # -- outcome ----------------------------------------------------------

    def mark(self, phase: str) -> None:
        self._timings[phase] = round(time.time() - self._t0, 1)
        self._stamps[phase] = _utc_now()

    def write_result(self, outcome: str, extra: dict | None = None) -> None:
        result = {
            "run_id": self.run_id,
            "outcome": outcome,
            "timings_s": self._timings,
            "timestamps_utc": self._stamps,
            "wall_clock_s": round(time.time() - self._t0, 1),
            **(extra or {}),
        }
        (self.path / "result.json").write_text(json.dumps(result, indent=2))
