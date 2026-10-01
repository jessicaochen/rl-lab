"""Thin kubectl wrappers. rlbench deliberately shells out to kubectl so it
inherits the user's context, auth, and plugins with zero client config."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

from .setup_folder import RUN_LABEL


class KubectlError(Exception):
    pass


def kubectl(*args: str, input_text: str | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        ["kubectl", *args], input=input_text, capture_output=True, text=True
    )
    if check and proc.returncode != 0:
        raise KubectlError(f"kubectl {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def apply(manifest_yaml: str) -> str:
    return kubectl("apply", "-f", "-", input_text=manifest_yaml)


def delete(manifest_yaml: str) -> str:
    return kubectl(
        "delete", "-f", "-", "--ignore-not-found", "--wait=false",
        input_text=manifest_yaml, check=False,
    )


def get_json(*args: str) -> dict:
    out = kubectl("get", *args, "-o", "json")
    return json.loads(out)


def current_context() -> str:
    return kubectl("config", "current-context").strip()


def wait_deployments_ready(namespace: str, timeout_s: int) -> None:
    deps = get_json("deployments", "-n", namespace).get("items", [])
    for d in deps:
        name = d["metadata"]["name"]
        kubectl(
            "rollout", "status", f"deployment/{name}", "-n", namespace,
            f"--timeout={timeout_s}s",
        )


def watch_job(namespace: str, name: str, timeout_s: int, poll_s: int = 15) -> dict:
    """Poll a (possibly Indexed) Job until Complete/Failed/timeout.

    Returns {"outcome": "Complete"|"Failed"|"Timeout", "job_status": {...}}.
    """
    deadline = time.monotonic() + timeout_s
    status: dict = {}
    failures = 0
    while time.monotonic() < deadline:
        try:
            status = get_json("job", name, "-n", namespace).get("status", {})
            failures = 0
        except KubectlError as e:
            # transient auth/DNS blips must not kill a multi-hour watch
            failures += 1
            if failures >= 40:  # ~10 min of consecutive failures
                raise
            print(f"    (job poll failed {failures}x, retrying: {str(e)[:120]})")
            time.sleep(poll_s)
            continue
        for cond in status.get("conditions", []):
            if cond.get("status") == "True" and cond.get("type") in ("Complete", "Failed"):
                return {"outcome": cond["type"], "job_status": status}
        time.sleep(poll_s)
    return {"outcome": "Timeout", "job_status": status}


def run_pods(namespace: str, run_id: str) -> list[dict]:
    sel = f"{RUN_LABEL}={run_id}"
    return get_json("pods", "-n", namespace, "-l", sel).get("items", [])


def dump_events(namespace: str, out_dir: Path) -> None:
    """Namespace events plus cluster-scoped Node events (spot preemptions
    surface on Node objects, not in the run namespace)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "namespace-events.json").write_text(
        kubectl("get", "events", "-n", namespace, "-o", "json", check=False) or "{}"
    )
    (out_dir / "node-events.json").write_text(
        kubectl(
            "get", "events", "-A", "--field-selector", "involvedObject.kind=Node",
            "-o", "json", check=False,
        ) or "{}"
    )


class LogStreamer:
    """Follows logs of every pod labeled with the run id into logs/<pod>.log.

    Streams (rather than fetching at the end) so logs survive pod deletion and
    spot preemption. A discovery thread picks up pods created mid-run.
    """

    def __init__(self, namespace: str, run_id: str, logs_dir: Path, poll_s: int = 20):
        self.namespace = namespace
        self.run_id = run_id
        self.logs_dir = logs_dir
        self.poll_s = poll_s
        self._streams: dict[str, subprocess.Popen] = {}
        self._files: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._discover_loop, daemon=True)

    def start(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def _discover_loop(self) -> None:
        while not self._stop.is_set():
            try:
                for pod in run_pods(self.namespace, self.run_id):
                    name = pod["metadata"]["name"]
                    phase = pod["status"].get("phase")
                    if phase == "Pending":
                        continue
                    proc = self._streams.get(name)
                    if proc is None:
                        self._follow(name)
                    elif proc.poll() is not None and phase == "Running":
                        # follow-stream died (API blip, host suspend) while the
                        # pod lives on: resume from now; the final snapshot at
                        # stop() backfills whatever the gap missed
                        self._follow(name, tail=0)
            except KubectlError:
                pass  # transient API errors: retry on next tick
            self._stop.wait(self.poll_s)

    def _follow(self, pod: str, tail: int | None = None) -> None:
        f = open(self.logs_dir / f"{pod}.log", "ab")
        self._files.append(f)
        cmd = ["kubectl", "logs", "-f", pod, "-n", self.namespace,
               "--all-containers", "--prefix", "--timestamps"]
        if tail is not None:
            cmd.append(f"--tail={tail}")
        self._streams[pod] = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.poll_s + 5)
        for proc in self._streams.values():
            proc.terminate()
        for f in self._files:
            f.close()
        self._final_snapshot()

    def _final_snapshot(self) -> None:
        """A follow-stream can lag or drop the last lines when a pod exits (or
        the stream was severed by an API blip). Re-fetch every run pod's full
        log once and keep whichever copy is longer, so the run folder always
        holds the complete log for pods that still exist."""
        try:
            pods = run_pods(self.namespace, self.run_id)
        except KubectlError:
            return
        for pod in pods:
            name = pod["metadata"]["name"]
            full = kubectl("logs", name, "-n", self.namespace, "--all-containers",
                           "--prefix", "--timestamps", check=False)
            target = self.logs_dir / f"{name}.log"
            if full and (not target.exists() or len(full) > target.stat().st_size):
                target.write_text(full)
