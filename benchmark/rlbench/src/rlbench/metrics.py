"""Periodic metrics sampling into the run folder.

Three sources, all timestamped:
- ``kubectl top`` nodes/pods (CPU/memory) appended as JSONL
- any Service in the run namespace labeled ``rlbench/scrape=true`` is scraped
  through the API-server proxy and each sample saved as a .prom file
- optional pod targets declared by the setup folder in ``scrape-targets.txt``
  (one per line: ``pods <namespace> <label-selector> <port> <path>``), e.g. a
  cluster-managed DCGM exporter in another namespace; pods are re-resolved
  every interval so node replacements don't break collection

The tool stays generic: it scrapes whatever the setup points it at.
"""

from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path

from .kube import KubectlError, get_json, kubectl

SCRAPE_LABEL = "rlbench/scrape=true"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class MetricsSampler:
    def __init__(self, namespace: str, metrics_dir: Path, interval_s: int = 30,
                 pod_targets: list[tuple[str, str, str, str]] | None = None):
        self.namespace = namespace
        self.metrics_dir = metrics_dir
        self.interval_s = interval_s
        self.pod_targets = pod_targets or []  # (namespace, selector, port, path)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def sample_once(self) -> None:
        ts = _now()
        self._top(ts, "nodes")
        self._top(ts, "pods", "-n", self.namespace)
        self._scrape_labeled_services(ts)
        self._scrape_pod_targets(ts)

    def _top(self, ts: str, kind: str, *extra: str) -> None:
        out = kubectl("top", kind, *extra, "--no-headers", check=False)
        if not out.strip():
            return
        line = json.dumps({"ts": ts, "kind": f"top-{kind}", "raw": out})
        with open(self.metrics_dir / f"top-{kind}.jsonl", "a") as f:
            f.write(line + "\n")

    def _scrape_labeled_services(self, ts: str) -> None:
        try:
            svcs = get_json("services", "-n", self.namespace, "-l", SCRAPE_LABEL)
        except KubectlError:
            return
        for svc in svcs.get("items", []):
            name = svc["metadata"]["name"]
            ports = svc["spec"].get("ports", [])
            if not ports:
                continue
            port = ports[0]["port"]
            path = (
                f"/api/v1/namespaces/{self.namespace}/services/"
                f"{name}:{port}/proxy/metrics"
            )
            body = kubectl("get", "--raw", path, check=False)
            if body:
                (self.metrics_dir / f"{name}-{ts}.prom").write_text(body)

    def _scrape_pod_targets(self, ts: str) -> None:
        for ns, selector, port, path in self.pod_targets:
            try:
                pods = get_json("pods", "-n", ns, "-l", selector).get("items", [])
            except KubectlError:
                continue
            for pod in pods:
                name = pod["metadata"]["name"]
                body = kubectl("get", "--raw", f"/api/v1/namespaces/{ns}/pods/{name}:{port}/proxy{path}",
                               check=False)
                if body:
                    (self.metrics_dir / f"{name}-{ts}.prom").write_text(body)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 5)
        self.sample_once()  # final sample at shutdown
