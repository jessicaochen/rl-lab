"""Pure helpers behind rollout_adapter.py: no ray / verl imports, unit-testable offline.

- RequestRecorder: buffered JSONL writer for per-request gateway records
- sync_endpoints: keep a (server_id -> handle) map in step with verl's
  GlobalRequestLoadBalancer membership, discovering handles for new ids by
  the drain-acquire trick the py-inference-scheduler hook uses
- router/env constants shared by the adapter, the feature fragment and the tools
"""

from __future__ import annotations

import atexit
import json
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

ROUTER_ENV = "RLBENCH_ROUTER"            # runtime-env variable selecting the routing mode
ROUTER_VERL = "verl"                     # verl's GlobalRequestLoadBalancer (default, baseline)
ROUTER_PYIS = "inference-scheduler"      # py-inference-scheduler decides per request
RUN_OUT_ENV = "RLBENCH_RUN_OUT"          # /data/outputs/<RUN_ID>, set by common.sh via the runtime env
GATEWAY_LOG_DIR = "gateway-logs"         # <RUN_OUT>/gateway-logs/<host>-<pid>.jsonl
REPLICA_METRICS_DIR = "replica-metrics"  # <RUN_OUT>/replica-metrics/<server>/<ts>.prom + replicas.json


class RequestRecorder:
    """Append JSON records to ``<dir>/<tag>.jsonl``; flushes every ``batch``
    records or ``max_age_s`` seconds (gateway actors write to Filestore, so
    one write per request would be needless NFS traffic). Thread-safe; the
    final flush runs at interpreter exit. Evidence never breaks generation:
    I/O errors drop the batch silently."""

    def __init__(self, directory: str | os.PathLike, tag: str | None = None,
                 batch: int = 100, max_age_s: float = 5.0):
        self.directory = Path(directory)
        self.tag = tag or f"{socket.gethostname()}-{os.getpid()}"
        self.batch = batch
        self.max_age_s = max_age_s
        self._buf: list[str] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()
        self.written = 0
        atexit.register(self.flush)

    @property
    def path(self) -> Path:
        return self.directory / f"{self.tag}.jsonl"

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, separators=(",", ":"))
        with self._lock:
            self._buf.append(line)
            due = len(self._buf) >= self.batch or (time.time() - self._last_flush) >= self.max_age_s
        if due:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buf:
                return
            lines, self._buf = self._buf, []
            self._last_flush = time.time()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write("\n".join(lines) + "\n")
            self.written += len(lines)
        except OSError:
            pass


async def sync_endpoints(
    known: dict[str, Any],
    current_ids: list[str],
    acquire: Callable[[str], Awaitable[tuple[str, Any]]],
    release: Callable[[str], Any],
    max_rounds: int = 3,
) -> tuple[list[str], list[str]]:
    """Bring ``known`` (server_id -> handle) in line with the balancer's
    ``current_ids``; returns (added, removed).

    Removed ids are dropped. Handles for new ids are discovered by acquiring
    with unique request ids and releasing right away: the balancer hands out
    its least-loaded server and a new server starts at zero inflight, so new
    servers surface within a few acquires. Bounded so a balancer that never
    returns a given id cannot spin forever."""
    current = set(current_ids)
    removed = sorted(sid for sid in known if sid not in current)
    for sid in removed:
        del known[sid]
    missing = current - set(known)
    added: list[str] = []
    acquired: list[str] = []
    budget = max_rounds * max(1, len(current))
    while missing and budget > 0:
        budget -= 1
        sid, handle = await acquire(f"rlbench-sync-{os.getpid()}-{time.time_ns()}-{budget}")
        acquired.append(sid)
        if sid in missing:
            known[sid] = handle
            missing.discard(sid)
            added.append(sid)
    for sid in acquired:
        release(sid)
    return added, removed


def sanitize(server_id: str) -> str:
    """server ids are "host:port" strings; make them path-safe."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in server_id)


_WAITING = re.compile(r"^(?:vllm:|vllm_)num_requests_waiting(?:\{.*?\})?\s+([\d.eE+-]+)", re.M)
_RUNNING = re.compile(r"^(?:vllm:|vllm_)num_requests_running(?:\{.*?\})?\s+([\d.eE+-]+)", re.M)
_KV = re.compile(r"^(?:vllm:|vllm_)(?:kv_cache_usage_perc|gpu_cache_usage_perc)(?:\{.*?\})?\s+([\d.eE+-]+)", re.M)


def parse_routing_stats(text: str) -> dict[str, Any]:
    """vLLM /metrics text -> the routing-stats dict py-inference-scheduler's scorers
    read (``num_waiting_reqs``, ``num_running_reqs``, ``kv`` in [0,1]); same metric
    names and semantics as its own verl/vLLM fetcher, summed over label sets for
    the request gauges and max over label sets for KV usage."""
    stats: dict[str, Any] = {"num_waiting_reqs": 0, "num_running_reqs": 0, "kv": 0.0, "error": None}
    w = _WAITING.findall(text)
    r = _RUNNING.findall(text)
    k = _KV.findall(text)
    if w:
        stats["num_waiting_reqs"] = int(sum(float(x) for x in w))
    if r:
        stats["num_running_reqs"] = int(sum(float(x) for x in r))
    if k:
        stats["kv"] = max(float(x) for x in k)
    if not (w or r or k):
        stats["error"] = "no vllm request/kv metrics in /metrics body"
    return stats
