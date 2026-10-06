"""rlbench's agent-loop-manager adapter for the verl + uni-agent setup.

Wired in by config/common.sh for EVERY run (feature or not):

    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=rlbench_verl_provider.rollout_adapter.RlbenchRolloutAdapter

It subclasses uni-agent's ``AgentFrameworkRolloutAdapter`` and changes one
thing: the LLM client handed to the gateway actors. The trainer passes a
``FullyAsyncLLMServerClient`` (routing = verl's ``GlobalRequestLoadBalancer``,
sticky per session; resumes requests aborted at weight-sync pauses). We wrap it
so that, in every gateway actor:

- ``InstrumentedClient`` (default, ``RLBENCH_ROUTER=verl``) records one JSON
  line per policy request (session, replica, tokens, latency, resumes, model
  version) under ``<RLBENCH_RUN_OUT>/gateway-logs/``; routing is unchanged.
- ``SchedulerClient`` (``RLBENCH_ROUTER=inference-scheduler``) additionally lets
  py-inference-scheduler pick the replica per request. It keeps the FullyAsync
  resume loop (the upstream hook subclasses the plain client and would lose it),
  follows verl's balancer membership (hybrid replicas come and go), scrapes each
  replica's vLLM ``/metrics`` itself for the scorers' inputs (no patch inside the
  server processes: a Ray ``worker_process_setup_hook`` that imported vLLM
  initialized CUDA before Ray assigned GPUs and put every trainer rank on GPU 0,
  measured 2026-10-05), and records the decision inputs in the same JSON line.

A ``ReplicaMetricsPoller`` actor snapshots every replica's vLLM ``/metrics`` into
``<RLBENCH_RUN_OUT>/replica-metrics/`` so TTFT / queue-time / prefix-cache
counters exist per replica in both arms. hooks/post-run.sh pulls both dirs into
the run folder.

The client is built lazily inside each gateway actor (``LazyClient``): the
scheduler holds an asyncio lock and plugin state that must not be pickled, and
uni-agent only ever calls ``backend.generate``.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
from uni_agent.framework.entry import AgentFrameworkRolloutAdapter
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

from .rollout_evidence import (
    GATEWAY_LOG_DIR,
    REPLICA_METRICS_DIR,
    ROUTER_ENV,
    ROUTER_PYIS,
    ROUTER_VERL,
    RUN_OUT_ENV,
    RequestRecorder,
    parse_routing_stats,
    sanitize,
    sync_endpoints,
)

logger = logging.getLogger(__name__)
LOG_TAG = "[rlbench-router]"


def _say(msg: str) -> None:
    """Evidence lines must reach the driver log: Ray forwards actor stdout, while
    logger.info is dropped by the default WARNING root level in these processes."""
    print(f"{LOG_TAG} {msg}", flush=True)

# per-generate() bookkeeping shared with _acquire_server/_release_server, which
# verl calls from inside the same asyncio task
_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("rlbench_gen_ctx", default=None)


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class ClientSpec:
    config: Any
    load_balancer_handle: Any
    only_hybrid: bool
    router: str
    out_dir: str | None


def build_client(spec: ClientSpec):
    cls = SchedulerClient if spec.router == ROUTER_PYIS else InstrumentedClient
    return cls(config=spec.config, load_balancer_handle=spec.load_balancer_handle,
               only_hybrid=spec.only_hybrid, out_dir=spec.out_dir)


class LazyClient:
    """Picklable stand-in for the gateway backend; builds the real client on
    first use in the receiving process."""

    def __init__(self, spec: ClientSpec):
        self._spec = spec
        self._client = None

    def __getstate__(self):
        return {"_spec": self._spec}

    def __setstate__(self, state):
        self._spec = state["_spec"]
        self._client = None

    def _get(self):
        if self._client is None:
            self._client = build_client(self._spec)
            _say(f"built {type(self._client).__name__} (router={self._spec.router}) pid={os.getpid()} out_dir={self._spec.out_dir}")
        return self._client

    async def generate(self, request_id, **kwargs):
        return await self._get().generate(request_id, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._get(), name)


class InstrumentedClient(FullyAsyncLLMServerClient):
    """verl routing, plus one JSON record per generate() call."""

    router = ROUTER_VERL

    def __init__(self, config, load_balancer_handle=None, only_hybrid: bool = False,
                 out_dir: str | None = None, **kwargs):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle,
                         only_hybrid=only_hybrid, **kwargs)
        self._recorder = RequestRecorder(Path(out_dir) / GATEWAY_LOG_DIR) if out_dir else None

    # -- routing hooks (verl calls these from LLMServerClient.generate) --------

    async def _acquire_server(self, request_id: str):
        server_id, handle = await super()._acquire_server(request_id)
        self._note_server(server_id)
        return server_id, handle

    @staticmethod
    def _note_server(server_id: str) -> None:
        ctx = _ctx.get()
        if ctx is not None:
            ctx["servers"].append(server_id)

    # -- instrumentation ----------------------------------------------------

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        ctx = {"servers": [], "prompt_ids": prompt_ids, "decisions": []}
        token = _ctx.set(ctx)
        t0 = time.time()
        output = None
        error = None
        try:
            output = await super().generate(request_id=request_id, prompt_ids=prompt_ids,
                                            sampling_params=sampling_params, **kwargs)
            return output
        except BaseException as e:  # noqa: BLE001  (re-raised; recorded first)
            error = type(e).__name__
            raise
        finally:
            _ctx.reset(token)
            if self._recorder is not None:
                self._recorder.write(self._record(request_id, prompt_ids, output, error, t0, time.time(), ctx))

    def _record(self, request_id, prompt_ids, output, error, t0, t1, ctx) -> dict:
        extra = getattr(output, "extra_fields", None) or {}
        rec = {
            "ts": _utc(),
            "t_start": round(t0, 3),
            "latency_s": round(t1 - t0, 4),
            "router": self.router,
            "session_id": request_id,
            "server_id": ctx["servers"][-1] if ctx["servers"] else None,
            "servers": ctx["servers"],
            "attempts": len(ctx["servers"]),
            "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None,
            "completion_tokens": len(output.token_ids) if output is not None and output.token_ids is not None else None,
            "stop_reason": getattr(output, "stop_reason", None),
            "num_preempted": getattr(output, "num_preempted", None),
            "global_steps": extra.get("global_steps"),
            "min_global_steps": extra.get("min_global_steps"),
            "error": error,
        }
        if ctx["decisions"]:
            rec["decisions"] = ctx["decisions"]
        return rec


class SchedulerClient(InstrumentedClient):
    """py-inference-scheduler picks the replica; everything else as above."""

    router = ROUTER_PYIS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # imported here: the scheduler package exists only in images >= v7 and
        # only matters when the feature is on
        import aiohttp
        from py_inference_scheduler import Scheduler
        from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
        from py_inference_scheduler.framework import Endpoint, LLMRequest

        self._Endpoint, self._LLMRequest, self._aiohttp = Endpoint, LLMRequest, aiohttp
        self._session = None            # aiohttp.ClientSession, created on the event loop at first use
        self._stats_max_age = 0.5       # seconds; coalesces bursts of decisions into one scrape per replica
        self._stats_timeout = 1.0       # seconds per replica; a slow replica must not stall routing
        self.scheduler = Scheduler()  # reads ROUTER_CONFIG_PATH, hot-reloads on mtime change
        self.inflight = InflightStore()
        self._handles: dict[str, Any] = {}
        self._endpoints: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._fallbacks = 0
        self._decisions = 0
        # the scheduler logs two INFO lines per decision; keep its warnings only
        logging.getLogger("py_inference_scheduler").setLevel(logging.WARNING)
        _say(f"SchedulerClient ready, config={os.environ.get('ROUTER_CONFIG_PATH')}")

    async def _sync(self) -> None:
        ids = await self._load_balancer.get_all_servers.remote()

        async def acquire(rid):
            return await self._load_balancer.acquire_server.remote(request_id=rid)

        def release(sid):
            self._load_balancer.release_server.remote(server_id=sid)

        added, removed = await sync_endpoints(self._handles, list(ids), acquire, release)
        for sid in removed:
            self._endpoints.pop(sid, None)
        for sid in added:
            self._endpoints[sid] = self._Endpoint(
                name=sid, attributes={"replica_obj": self._handles[sid], "routing_stats": {}, "stats_ts": 0.0})
        if added or removed:
            _say(f"membership: +{added} -{removed} -> {sorted(self._endpoints)}")

    async def _refresh(self, ep) -> None:
        """Populate ep.attributes["routing_stats"] from the replica's vLLM /metrics
        (what py-inference-scheduler's own verl fetcher does via a patched actor
        method; done client-side here, see module docstring)."""
        now = time.monotonic()
        if now - ep.attributes.get("stats_ts", 0.0) < self._stats_max_age:
            return
        url = ep.attributes.get("metrics_url")
        if url is None:
            try:
                host, port = await ep.attributes["replica_obj"].get_server_address.remote()
                url = ep.attributes["metrics_url"] = f"http://{host}:{port}/metrics"
            except Exception as e:  # noqa: BLE001
                ep.attributes["routing_stats"] = {"num_waiting_reqs": 0, "num_running_reqs": 0, "kv": 0.0,
                                                  "error": f"get_server_address: {type(e).__name__}: {e}"}
                ep.attributes["stats_ts"] = now
                return
        if self._session is None:
            self._session = self._aiohttp.ClientSession()
        try:
            async with self._session.get(url, timeout=self._aiohttp.ClientTimeout(total=self._stats_timeout)) as resp:
                text = await resp.text() if resp.status == 200 else ""
                status = resp.status
            stats = parse_routing_stats(text) if text else {"num_waiting_reqs": 0, "num_running_reqs": 0, "kv": 0.0,
                                                             "error": f"HTTP {status}"}
        except Exception as e:  # noqa: BLE001
            stats = {"num_waiting_reqs": 0, "num_running_reqs": 0, "kv": 0.0, "error": f"{type(e).__name__}: {e}"}
        ep.attributes["routing_stats"] = stats
        ep.attributes["stats_ts"] = now
        if stats.get("error") and not ep.attributes.get("stats_error_reported"):
            ep.attributes["stats_error_reported"] = True
            _say(f"WARNING {ep.name}: engine stats unavailable ({stats['error']}); scoring on inflight + prefix only")

    async def _schedule(self, request_id: str, prompt_ids):
        async with self._lock:
            await self._sync()
            eps = list(self._endpoints.values())
            if not eps:
                return None, []
            await asyncio.gather(*(self._refresh(ep) for ep in eps))
            for ep in eps:
                ep.attributes["queue_len"] = self.inflight.get(ep.name)
            request = self._LLMRequest(request_id=request_id, body=prompt_ids)
            selected = self.scheduler.run(request, candidates=eps)
            snapshot = [
                {"server": ep.name, "queue_len": ep.attributes.get("queue_len"),
                 **{k: v for k, v in (ep.attributes.get("routing_stats") or {}).items() if k != "error"},
                 **({"stats_error": True} if (ep.attributes.get("routing_stats") or {}).get("error") else {})}
                for ep in eps
            ]
            if not selected:
                return None, snapshot
            winner = selected[0].endpoint
            self.inflight.increment(winner.name)
            return winner, snapshot

    async def _acquire_server(self, request_id: str):
        ctx = _ctx.get()
        prompt_ids = ctx["prompt_ids"] if ctx else None
        winner, snapshot = await self._schedule(request_id, prompt_ids)
        self._decisions += 1
        decision = {"candidates": snapshot, "fallback": winner is None}
        if winner is None:
            self._fallbacks += 1
            if self._fallbacks in (1, 10, 100) or self._fallbacks % 1000 == 0:
                _say(f"WARNING scheduler returned no endpoint ({self._fallbacks}x), using verl's balancer")
            server_id, handle = await super()._acquire_server(request_id)  # InstrumentedClient notes the server
            self.inflight.increment(server_id)
            if ctx is not None:
                ctx.setdefault("lb_release", []).append(server_id)
        else:
            server_id, handle = winner.name, winner.attributes["replica_obj"]
            self._note_server(server_id)
        decision["server"] = server_id
        if ctx is not None:
            ctx["decisions"].append(decision)
        if self._decisions in (1, 100) or self._decisions % 5000 == 0:
            _say(f"decision #{self._decisions}: {json.dumps(decision, default=str)[:600]}")
        return server_id, handle

    def _release_server(self, server_id: str) -> None:
        self.inflight.decrement(server_id)
        ctx = _ctx.get()
        pending = ctx.get("lb_release") if ctx else None
        if pending and server_id in pending:
            pending.remove(server_id)
            super()._release_server(server_id)  # only fallback acquisitions touched verl's counters


@ray.remote(num_cpus=0.1)
class ReplicaMetricsPoller:
    """Every ``interval_s``: resolve the balancer's current servers, GET each
    replica's vLLM ``/metrics`` and store it as ``<out>/replica-metrics/<server>/<ts>.prom``;
    ``replicas.json`` maps server ids to host:port and first/last seen times."""

    def __init__(self, load_balancer_handle, out_dir: str, interval_s: float = 30.0):
        self._lb = load_balancer_handle
        self._out = Path(out_dir) / REPLICA_METRICS_DIR
        self._interval = interval_s
        self._handles: dict[str, Any] = {}
        self._addresses: dict[str, dict] = {}
        self._stop = False

    async def run(self) -> None:
        import aiohttp
        self._aiohttp = aiohttp  # imported lazily (head-only dependency), used by _tick
        _say(f"poller started: out={self._out} interval={self._interval}s")
        async with aiohttp.ClientSession() as session:
            while not self._stop:
                try:
                    await self._tick(session)
                except Exception as e:  # noqa: BLE001
                    _say(f"WARNING poller tick failed: {type(e).__name__}: {e}")
                await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._stop = True

    async def _tick(self, session) -> None:
        ids = await self._lb.get_all_servers.remote()

        async def acquire(rid):
            return await self._lb.acquire_server.remote(request_id=rid)

        def release(sid):
            self._lb.release_server.remote(server_id=sid)

        added, _ = await sync_endpoints(self._handles, list(ids), acquire, release)
        now = _utc()
        for sid in added:
            try:
                host, port = await self._handles[sid].get_server_address.remote()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{LOG_TAG} poller: get_server_address failed for {sid}: {e}")
                continue
            self._addresses[sid] = {"host": host, "port": port, "first_seen": now, "last_seen": now}
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for sid in ids:
            addr = self._addresses.get(sid)
            if not addr:
                continue
            try:
                async with session.get(f"http://{addr['host']}:{addr['port']}/metrics",
                                       timeout=self._aiohttp.ClientTimeout(total=5)) as resp:
                    body = await resp.text() if resp.status == 200 else ""
                    status = resp.status
            except Exception as e:  # noqa: BLE001
                body, status = "", f"error: {e}"
            addr["last_seen"] = now
            addr["last_status"] = status
            addr["snapshots"] = addr.get("snapshots", 0) + bool(body)
            if body:
                d = self._out / sanitize(sid)
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{ts}.prom").write_text(body)
        self._out.mkdir(parents=True, exist_ok=True)
        (self._out / "replicas.json").write_text(json.dumps(self._addresses, indent=2))


class RlbenchRolloutAdapter(AgentFrameworkRolloutAdapter):
    """uni-agent's adapter with the gateway backend swapped for our client."""

    @classmethod
    def create(cls, *, config, llm_client, **kwargs):
        router = os.environ.get(ROUTER_ENV, ROUTER_VERL)
        out_dir = os.environ.get(RUN_OUT_ENV) or None
        if router not in (ROUTER_VERL, ROUTER_PYIS):
            raise ValueError(f"{ROUTER_ENV}={router!r}: expected {ROUTER_VERL!r} or {ROUTER_PYIS!r}")
        spec = ClientSpec(
            config=llm_client.config,
            load_balancer_handle=llm_client._load_balancer,
            only_hybrid=bool(getattr(llm_client, "_only_hybrid", False)),
            router=router,
            out_dir=out_dir,
        )
        _say(f"router={router} out_dir={out_dir} wrapping {type(llm_client).__name__}")
        instance = super().create(config=config, llm_client=LazyClient(spec), **kwargs)
        instance.poller = None
        if out_dir:
            instance.poller = ReplicaMetricsPoller.remote(spec.load_balancer_handle, out_dir)
            instance.poller.run.remote()
        return instance
