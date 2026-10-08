#!/usr/bin/env python3
"""app-offload controller: an EXTERNAL process (a plain k8s pod on the same node as the Ray
workers, not a Ray node) that tells verl's TRAINER workers to offload GPU memory to the host in
the idle window after each weight sync, and to reload afterwards. (A sampler path exists but is
off by default: its first mechanism, scheduling engine coroutines through __ray_call__, does not
work — Ray runs __ray* methods outside the actor's event loop; the current mode-flip variant is
untested. See the feature README.)

It never changes verl's configuration. It discovers verl's actors through Ray's dashboard state
API, obtains handles over Ray Client, and calls methods verl already exposes:

  trainer (WorkerDict actors, one per rank)
      actor_rollout_to("cpu", model, optimizer, grad)   # MegatronEngine.to -- "irrespective of offload config"
      actor_rollout_to("device")
  sampler (vLLMHttpServer actors, node_rank 0 holds the engine)
      abort_all_requests -> wait_for_requests_to_drain -> engine.sleep(level=1) -> hold
      -> engine.wake_up(weights, kv_cache) -> engine.reset_prefix_cache -> resume_generation
      (verl's standalone sleep()/wake_up() are no-ops, so the engine coroutines are scheduled on
      the actor's event loop through Ray's generic __ray_call__.)

When it acts (both arms, identical):
  trainer  the post-weight-sync idle gap: no RUNNING task on any local WorkerDict for SETTLE_S and
           the two newest FINISHED verl tasks are update_weights then execute_checkpoint_engine
           (the finalize). offload -> hold TRAINER_HOLD_S -> reload. The gap ends with the params
           RESIDENT, exactly as verl itself leaves them after a sync.
  sampler  SAMPLER_CYCLES_PER_STEP controller-made pauses, started >= 20 s after a weight sync
           finished cluster-wide (next sync is a full step away): pause -> sleep -> hold
           SAMPLER_HOLD_S -> wake -> resume.

Everything it does and measures is one JSON record per line in
$APP_OFFLOAD_OUT/<node>.jsonl (collected into the run folder by hooks/post-run.sh) and mirrored
to stdout as "[app-offload] {...}" (streamed by rlbench). Memory numbers come from inside the
actor processes (pynvml device-used, torch allocator stats, RSS), so no verl patch is needed.
"""
from __future__ import annotations

import json
import os
import signal
import ssl
import sys
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

try:
    import ray
    from ray.util.state import list_actors, list_nodes, list_tasks
except ImportError:  # offline tests import the pure helpers only
    ray = None

# ----------------------------------------------------------------------------- configuration

ENV = os.environ.get
RUN_ID = ENV("RUN_ID", "run")
NODE_NAME = ENV("NODE_NAME", "") or os.uname().nodename
NODE_IP = ENV("NODE_IP", "")
POD_NAMESPACE = ENV("POD_NAMESPACE", "rlbench-verl-swe")
DASH = ENV("RAY_DASHBOARD_URL", "http://verl-head-svc:8265")
CLIENT = ENV("RAY_CLIENT_ADDRESS", "ray://verl-head-svc:10001")
OUT_DIR = ENV("APP_OFFLOAD_OUT", f"/data/outputs/{RUN_ID}/app-offload")
POLL_S = float(ENV("APP_OFFLOAD_POLL_S", "1"))
SETTLE_S = float(ENV("APP_OFFLOAD_SETTLE_S", "3"))
TRAINER_HOLD_S = float(ENV("APP_OFFLOAD_TRAINER_HOLD_S", "180"))
SAMPLER_HOLD_S = float(ENV("APP_OFFLOAD_SAMPLER_HOLD_S", "60"))
SAMPLER_CYCLES = int(ENV("APP_OFFLOAD_SAMPLER_CYCLES_PER_STEP", "1"))
SAMPLER_AFTER_SYNC_S = float(ENV("APP_OFFLOAD_SAMPLER_AFTER_SYNC_S", "20"))
DRY_RUN = ENV("APP_OFFLOAD_DRY_RUN", "0") == "1"            # probe + log only, never call offload
ALL_NODES = ENV("APP_OFFLOAD_ALL_NODES", "0") == "1"        # debug: treat every actor as local
CALL_TIMEOUT_S = float(ENV("APP_OFFLOAD_CALL_TIMEOUT_S", "300"))
PROBE_EVERY_S = 60.0

TRAINER_CLASS = "WorkerDict"
SERVER_CLASS = "vLLMHttpServer"
CKPT_CLASS = "CheckpointEngineWorker"
DONE_STATES = {"FINISHED", "FAILED"}
GIB = 1024**3


# ----------------------------------------------------------------------------- logging

class Log:
    def __init__(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        self.path = os.path.join(OUT_DIR, f"{NODE_NAME}.jsonl")
        self.f = open(self.path, "a", buffering=1)

    def __call__(self, action: str, **kw):
        rec = {"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "node": NODE_NAME, "node_ip": NODE_IP, "action": action, **kw}
        line = json.dumps(rec, default=str)
        self.f.write(line + "\n")
        print("[app-offload] " + line, flush=True)


log = Log()


# ----------------------------------------------------------------------------- k8s: which Ray pods are on my node

def ray_pod_ips_on_my_node() -> set[str]:
    """Pod IPs of the app=verl-ray pods scheduled on this k8s node (Ray's node_ip is the pod IP)."""
    sa = "/var/run/secrets/kubernetes.io/serviceaccount"
    with open(f"{sa}/token") as f:
        token = f.read().strip()
    host, port = ENV("KUBERNETES_SERVICE_HOST"), ENV("KUBERNETES_SERVICE_PORT", "443")
    q = urllib.parse.urlencode({"labelSelector": "app=verl-ray", "fieldSelector": f"spec.nodeName={NODE_NAME}"})
    req = urllib.request.Request(f"https://{host}:{port}/api/v1/namespaces/{POD_NAMESPACE}/pods?{q}",
                                 headers={"Authorization": f"Bearer {token}"})
    ctx = ssl.create_default_context(cafile=f"{sa}/ca.crt")
    with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
        items = json.load(r).get("items", [])
    ips = set()
    for it in items:
        st = it.get("status", {})
        if st.get("podIP"):
            ips.add(st["podIP"])
        for p in st.get("podIPs", []) or []:
            if p.get("ip"):
                ips.add(p["ip"])
    return ips


# ----------------------------------------------------------------------------- functions executed INSIDE the actors
# (shipped by cloudpickle through __ray_call__ / collective_rpc; must import everything they use)

def snap_in_actor(self=None):
    """GPU + host memory as seen from inside an actor process: pynvml device-used for the
    process's visible GPUs (no CUDA context created), torch allocator stats if torch already has
    a context here, RSS."""
    import os
    import sys
    import time
    out = {"pid": os.getpid(), "t": time.time(), "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import pynvml
        pynvml.nvmlInit()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        idx = [int(x) for x in vis.split(",") if x.strip().isdigit()] if vis else list(range(pynvml.nvmlDeviceGetCount()))
        gpus = {}
        for i in idx:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpus[str(i)] = round(m.used / 1024**3, 3)
        out["device_used_gib"] = gpus
    except Exception as e:  # noqa: BLE001
        out["nvml_error"] = repr(e)
    try:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            out["torch_allocated_gib"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
            out["torch_reserved_gib"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
    except Exception as e:  # noqa: BLE001
        out["torch_error"] = repr(e)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_gib"] = round(int(line.split()[1]) * 1024 / 1024**3, 3)
                    break
    except Exception:  # noqa: BLE001
        pass
    return out


def set_rollout_mode(self, mode_name):
    """vLLMHttpServer only. verl's standalone sleep()/wake_up() are no-ops because of
    `self.rollout_mode`; flipping it to COLOCATED makes the actor's own async methods run
    `engine.sleep(level=1)` / `engine.wake_up(tags)` + `reset_prefix_cache` on its event loop.
    (A sync __ray_call__ on this async actor executes in a helper thread with no running loop,
    so coroutines cannot be scheduled from it directly.) Returns the previous mode name."""
    from verl.workers.rollout.replica import RolloutMode
    prev = self.rollout_mode
    self.rollout_mode = getattr(RolloutMode, mode_name)
    return getattr(prev, "name", str(prev))


# ----------------------------------------------------------------------------- state-API helpers

@dataclass
class ActorInfo:
    actor_id: str
    name: str
    class_name: str
    node_id: str
    namespace: str
    pid: int | None = None
    node_ip: str | None = None


@dataclass
class TaskView:
    active: list = field(default_factory=list)      # verl tasks not finished (name, state, start)
    finished: list = field(default_factory=list)    # verl tasks finished, newest first (name, end_ms)
    ours_active: int = 0


def _task_name(t) -> str:
    return getattr(t, "name", None) or getattr(t, "func_or_class_name", "") or ""


def _is_ours(name: str) -> bool:
    return "__ray_call__" in name or name.endswith("actor_rollout_to") or name.endswith(".collective_rpc")


def gap_signature(views: dict) -> int | None:
    """The post-sync idle gap of the trainer, from the per-actor task views: every actor has no
    active verl task and its two newest finished verl tasks are execute_checkpoint_engine (the
    finalize) then update_weights. Returns the gap id (newest update_weights end_time_ms over the
    actors) or None. Pure function so it can be tested without Ray."""
    if not views:
        return None
    ids = []
    for v in views.values():
        if v.active:
            return None
        top = [x[0] for x in v.finished[:2]]
        if len(top) < 2 or not top[0].endswith("execute_checkpoint_engine") or not top[1].endswith("update_weights"):
            return None
        ids.append(v.finished[1][1])
    return max(ids)


def tasks_of(actor_id: str) -> TaskView:
    v = TaskView()
    ts = list_tasks(address=DASH, filters=[("actor_id", "=", actor_id)], detail=True, limit=5000, timeout=30,
                    raise_on_missing_output=False)
    for t in ts:
        name, state = _task_name(t), getattr(t, "state", "")
        if _is_ours(name):
            v.ours_active += state not in DONE_STATES
            continue
        if state in DONE_STATES:
            v.finished.append((name, getattr(t, "end_time_ms", None) or 0))
        else:
            v.active.append((name, state, getattr(t, "start_time_ms", None)))
    v.finished.sort(key=lambda x: x[1], reverse=True)
    return v


def alive_actors() -> list[ActorInfo]:
    nodes = {n.node_id: n for n in list_nodes(address=DASH, limit=1000, timeout=30, raise_on_missing_output=False)}
    out = []
    for a in list_actors(address=DASH, filters=[("state", "=", "ALIVE")], detail=True, limit=10000, timeout=30,
                         raise_on_missing_output=False):
        n = nodes.get(a.node_id)
        out.append(ActorInfo(actor_id=a.actor_id, name=a.name or "", class_name=a.class_name or "", node_id=a.node_id or "",
                             namespace=getattr(a, "ray_namespace", "") or "", pid=getattr(a, "pid", None),
                             node_ip=getattr(n, "node_ip", None) if n else None))
    return out


# ----------------------------------------------------------------------------- controller

class Controller:
    def __init__(self):
        self.my_ips: set[str] = set()
        self.my_ips_err: str | None = None
        self.connected = False
        self.ready = False
        self.namespace = ""
        self.handles: dict[str, ray.actor.ActorHandle] = {}
        self.trainers: list[ActorInfo] = []
        self.servers: list[ActorInfo] = []
        self.ckpt: list[ActorInfo] = []
        self.quiet_since: float | None = None
        self.last_gap_id: int | None = None
        self.pending_gap: dict | None = None        # waiting to observe the gap's end
        self.last_sync_end: int | None = None
        self.cycles_done_for_sync: int = 0
        self.last_probe = 0.0
        self.last_ip_refresh = 0.0
        self.stop = False
        self.in_flight: str | None = None            # "trainer_offloaded" | "sampler_asleep:<name>"

    # ---- discovery
    def refresh_local_ips(self):
        if ALL_NODES:
            return
        try:
            self.my_ips = ray_pod_ips_on_my_node()
            self.my_ips_err = None
        except Exception as e:  # noqa: BLE001
            self.my_ips_err = repr(e)

    def is_local(self, a: ActorInfo) -> bool:
        return ALL_NODES or (a.node_ip in self.my_ips)

    def discover(self):
        acts = alive_actors()
        # the state API reports the qualified class name, e.g. "create_colocated_worker_cls.<locals>.WorkerDict"
        is_cls = lambda a, c: a.class_name == c or a.class_name.endswith("." + c)  # noqa: E731
        self.trainers = [a for a in acts if is_cls(a, TRAINER_CLASS) and self.is_local(a)]
        self.servers = [a for a in acts if is_cls(a, SERVER_CLASS) and self.is_local(a) and a.name.endswith("_0")]
        self.ckpt = [a for a in acts if is_cls(a, CKPT_CLASS)]
        ns = next((a.namespace for a in acts if (is_cls(a, TRAINER_CLASS) or is_cls(a, SERVER_CLASS)) and a.namespace), "")
        if ns and ns != self.namespace:
            self.namespace = ns
            self.connected = False
        return acts

    def connect(self):
        if self.connected or not self.namespace:
            return
        if ray.is_initialized():
            ray.shutdown()
        ray.init(address=CLIENT, namespace=self.namespace, ignore_reinit_error=True, logging_level="WARNING")
        self.handles.clear()
        self.connected = True

    def handle(self, a: ActorInfo):
        h = self.handles.get(a.name)
        if h is None:
            h = ray.get_actor(a.name, namespace=self.namespace)
            self.handles[a.name] = h
        return h

    # ---- measurement
    def snap(self, actors: list[ActorInfo], tag: str) -> dict:
        refs = {a.name: self.handle(a).__ray_call__.remote(snap_in_actor) for a in actors}
        out = {}
        for name, ref in refs.items():
            try:
                out[name] = ray.get(ref, timeout=60)
            except Exception as e:  # noqa: BLE001
                out[name] = {"error": repr(e)}
        return out

    @staticmethod
    def device_used(snaps: dict) -> dict:
        """{actor: mean device-used GiB over its visible GPUs}"""
        res = {}
        for name, s in snaps.items():
            g = s.get("device_used_gib") if isinstance(s, dict) else None
            if g:
                res[name] = round(sum(g.values()) / len(g), 2)
        return res

    # ---- probe (assumption check, logged until it passes)
    def probe(self, acts: list[ActorInfo]):
        self.last_probe = time.time()
        rec = {"dashboard": DASH, "client": CLIENT, "namespace": self.namespace, "my_pod_ips": sorted(self.my_ips),
               "my_ips_error": self.my_ips_err, "dry_run": DRY_RUN,
               "actors_by_class": {}, "local_trainers": [a.name for a in self.trainers],
               "local_servers": [a.name for a in self.servers], "ckpt_workers": len(self.ckpt), "checks": {}}
        for a in acts:
            rec["actors_by_class"][a.class_name] = rec["actors_by_class"].get(a.class_name, 0) + 1
        ok = True
        try:
            if self.trainers:
                v = tasks_of(self.trainers[0].actor_id)
                rec["sample_trainer_tasks"] = {"active": v.active[:5], "finished_newest": v.finished[:6]}
                names = [n for n, _ in v.finished] + [n for n, _, _ in v.active]
                rec["checks"]["trainer_task_names_prefixed"] = any("actor_rollout_" in n for n in names)
            if self.ckpt:
                v = tasks_of(self.ckpt[0].actor_id)
                rec["sample_ckpt_tasks"] = {"active": v.active[:3], "finished_newest": v.finished[:4]}
            self.connect()
            rec["checks"]["ray_client_connected"] = self.connected
            if self.trainers:
                s = self.snap(self.trainers[:1], "probe")
                rec["probe_trainer_snapshot"] = s
                ok &= all("device_used_gib" in x for x in s.values())
                rec["checks"]["trainer_snapshot"] = ok
            if self.servers:
                s = self.snap(self.servers[:1], "probe")
                rec["probe_server_snapshot"] = s
                good = all("device_used_gib" in x for x in s.values())
                rec["checks"]["server_snapshot"] = good
                ok &= good
            ok &= self.connected and bool(self.trainers or self.servers) and not (self.my_ips_err and not ALL_NODES)
        except Exception as e:  # noqa: BLE001
            ok = False
            rec["error"] = repr(e)
            rec["trace"] = traceback.format_exc()[-2000:]
        rec["ready"] = bool(ok)
        self.ready = bool(ok)
        log("probe", **rec)

    # ---- trainer state machine
    def trainer_tick(self):
        if not self.trainers:
            return
        views = {a.name: tasks_of(a.actor_id) for a in self.trainers}
        active = {n: v.active for n, v in views.items() if v.active}
        now = time.time()
        if self.pending_gap is not None and active:
            g = self.pending_gap
            log("gap_end", gap_id=g["gap_id"], gap_s=round(now - g["t_start"], 1), first_activity=next(iter(active.values()))[0],
                offloaded_s=g.get("offloaded_s"))
            self.pending_gap = None
        if active:
            self.quiet_since = None
            return
        if self.quiet_since is None:
            self.quiet_since = now
            return
        if now - self.quiet_since < SETTLE_S:
            return
        gap_id = gap_signature(views)
        if gap_id is None or gap_id == self.last_gap_id:
            return
        self.last_gap_id = gap_id
        t_start = now
        log("gap_start", gap_id=gap_id, trainers=len(self.trainers), quiet_s=round(now - self.quiet_since, 1),
            sync_finished_ago_s=round(now - gap_id / 1000, 1) if gap_id else None)
        if DRY_RUN:
            self.pending_gap = {"gap_id": gap_id, "t_start": t_start}
            return
        # --- offload
        before = self.snap(self.trainers, "before_offload")
        t0 = time.time()
        refs = [self.handle(a).actor_rollout_to.remote("cpu", model=True, optimizer=True, grad=True) for a in self.trainers]
        err = None
        try:
            ray.get(refs, timeout=CALL_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        secs = time.time() - t0
        after = self.snap(self.trainers, "after_offload")
        self.in_flight = "trainer_offloaded"
        log("offload", gap_id=gap_id, seconds=round(secs, 2), error=err, device_used_before=self.device_used(before),
            device_used_after=self.device_used(after), snapshots_before=before, snapshots_after=after)
        # --- hold, watching for verl activity
        violated = None
        t_hold = time.time()
        while time.time() - t_hold < TRAINER_HOLD_S and not self.stop:
            time.sleep(POLL_S)
            act = {a.name: tasks_of(a.actor_id).active for a in self.trainers}
            act = {n: v for n, v in act.items() if v}
            if act:
                violated = next(iter(act.values()))[0]
                log("violation", gap_id=gap_id, held_s=round(time.time() - t_hold, 1), task=violated,
                    note="verl task started while params were offloaded by the controller; reloading now")
                break
        held = time.time() - t_hold
        # --- reload
        t0 = time.time()
        # grad=False: leave the rank exactly as verl does after a sync (params resident, grads not
        # allocated); grad=True re-allocated 8.2 GiB of zeroed grads in R1
        refs = [self.handle(a).actor_rollout_to.remote("device", model=True, optimizer=False, grad=False) for a in self.trainers]
        err = None
        try:
            ray.get(refs, timeout=CALL_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        secs = time.time() - t0
        self.in_flight = None
        after2 = self.snap(self.trainers, "after_reload")
        log("reload", gap_id=gap_id, seconds=round(secs, 2), held_s=round(held, 1), violated=violated, error=err,
            device_used_after=self.device_used(after2), snapshots_after=after2)
        self.pending_gap = {"gap_id": gap_id, "t_start": t_start, "offloaded_s": round(held, 1)}
        self.quiet_since = None

    # ---- sampler state machine
    def sync_state(self):
        """(sync running now?, end_ms of the newest finished CheckpointEngineWorker.update_weights)"""
        running, newest = False, None
        for a in self.ckpt:
            v = tasks_of(a.actor_id)
            running |= bool(v.active)
            for name, end in v.finished:
                if name.endswith("update_weights"):
                    newest = max(newest or 0, end)
                    break
        return running, newest

    def engine_call(self, a: ActorInfo, method: str, kwargs: dict, timeout: float) -> dict:
        """sleep / wake_up through the server's own async methods, with rollout_mode temporarily
        COLOCATED so they reach the engine (standalone mode makes them no-ops)."""
        h = self.handle(a)
        t0 = time.time()
        prev = ray.get(h.__ray_call__.remote(set_rollout_mode, "COLOCATED"), timeout=60)
        try:
            if method == "sleep":
                ray.get(h.sleep.remote(), timeout=timeout)                     # engine.sleep(level=1)
            elif method == "wake_up":
                ray.get(h.wake_up.remote(tags=kwargs.get("tags")), timeout=timeout)  # engine.wake_up + reset_prefix_cache
            else:
                raise ValueError(method)
            return {"state": "done", "method": method, "wall_s": round(time.time() - t0, 2), "prev_mode": prev}
        finally:
            ray.get(h.__ray_call__.remote(set_rollout_mode, prev), timeout=60)

    def sampler_snap(self, a: ActorInfo) -> dict:
        # device-level view from the server actor process (pynvml, CUDA_VISIBLE_DEVICES = its GPUs);
        # vLLM refuses to deserialize callables over collective_rpc, so no per-TP-worker torch stats
        return self.snap([a], "sampler")

    def sampler_cycle(self, a: ActorInfo, sync_id: int):
        h = self.handle(a)
        rec = {"server": a.name, "sync_id": sync_id}
        t_cycle = time.time()
        try:
            rec["before"] = self.sampler_snap(a)
            t0 = time.time()
            ray.get(h.abort_all_requests.remote(reset_prefix_cache=True), timeout=CALL_TIMEOUT_S)
            ray.get(h.wait_for_requests_to_drain.remote(), timeout=CALL_TIMEOUT_S)
            rec["pause_s"] = round(time.time() - t0, 2)
            log("sampler_pause", **rec)
            rec["sleep"] = self.engine_call(a, "sleep", {"level": 1}, timeout=CALL_TIMEOUT_S)  # {"state","wall_s"}
            self.in_flight = f"sampler_asleep:{a.name}"
            rec["asleep"] = self.sampler_snap(a)
            log("sampler_sleep", server=a.name, sync_id=sync_id, sleep=rec["sleep"], asleep=rec["asleep"],
                device_used_before=self.device_used(rec["before"]), device_used_asleep=self.device_used(rec["asleep"]))
            t_hold = time.time()
            while time.time() - t_hold < SAMPLER_HOLD_S and not self.stop:
                time.sleep(POLL_S)
            rec["held_s"] = round(time.time() - t_hold, 1)
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
            log("failed", stage="sampler_sleep", **rec)
        finally:
            # always bring the engine back: wake (if it slept) and resume are independent steps,
            # and resume_generation must run even when the wake path fails
            slept = rec.get("sleep", {}).get("state") == "done"
            try:
                if slept:
                    rec["wake"] = self.engine_call(a, "wake_up", {"tags": ["weights", "kv_cache"]}, timeout=CALL_TIMEOUT_S)
                    rec["awake"] = self.sampler_snap(a)
            except Exception as e:  # noqa: BLE001
                log("failed", stage="sampler_wake", server=a.name, sync_id=sync_id, error=repr(e), trace=traceback.format_exc()[-2000:])
            try:
                t0 = time.time()
                ray.get(h.resume_generation.remote(), timeout=CALL_TIMEOUT_S)
                rec["resume_s"] = round(time.time() - t0, 2)
                self.in_flight = None
                log("sampler_wake", server=a.name, sync_id=sync_id, wake=rec.get("wake"), held_s=rec.get("held_s"),
                    resume_s=rec["resume_s"], device_used_awake=self.device_used(rec.get("awake", {})), awake=rec.get("awake"),
                    generation_paused_s=round(time.time() - t_cycle, 1), slept=slept)
            except Exception as e:  # noqa: BLE001
                log("failed", stage="sampler_resume", server=a.name, sync_id=sync_id, error=repr(e), trace=traceback.format_exc()[-2000:])

    def sampler_tick(self):
        if not self.servers or not self.ckpt:
            return
        running, newest = self.sync_state()
        if newest is None or running:
            return
        if newest != self.last_sync_end:
            self.last_sync_end = newest
            self.cycles_done_for_sync = 0
        if self.cycles_done_for_sync >= SAMPLER_CYCLES:
            return
        if time.time() - newest / 1000 < SAMPLER_AFTER_SYNC_S:
            return
        self.cycles_done_for_sync += 1
        log("sampler_cycle_start", sync_id=newest, cycle=self.cycles_done_for_sync, servers=[a.name for a in self.servers],
            sync_finished_ago_s=round(time.time() - newest / 1000, 1))
        if DRY_RUN:
            return
        for a in self.servers:
            # never overlap verl's own sync pause
            if self.sync_state()[0]:
                log("sampler_cycle_skipped", server=a.name, sync_id=newest, reason="weight sync running")
                continue
            self.sampler_cycle(a, newest)

    # ---- main loop
    def run(self):
        log("start", config={"poll_s": POLL_S, "settle_s": SETTLE_S, "trainer_hold_s": TRAINER_HOLD_S, "sampler_hold_s": SAMPLER_HOLD_S,
                             "sampler_cycles_per_step": SAMPLER_CYCLES, "sampler_after_sync_s": SAMPLER_AFTER_SYNC_S,
                             "dry_run": DRY_RUN, "all_nodes": ALL_NODES}, out=log.path)
        while not self.stop:
            try:
                if (not self.my_ips or self.my_ips_err) and time.time() - self.last_ip_refresh >= 30:
                    self.last_ip_refresh = time.time()
                    self.refresh_local_ips()
                acts = self.discover()
                if not self.ready or time.time() - self.last_probe > PROBE_EVERY_S * 10:
                    if time.time() - self.last_probe >= PROBE_EVERY_S or not self.last_probe:
                        self.probe(acts)
                if self.ready:
                    self.connect()
                    self.trainer_tick()
                    self.sampler_tick()
            except Exception as e:  # noqa: BLE001
                log("error", error=repr(e), trace=traceback.format_exc()[-3000:])
                self.connected = False
                self.ready = False
                time.sleep(5)
            time.sleep(POLL_S)
        self.shutdown()

    def shutdown(self):
        """Never leave anything offloaded or asleep behind."""
        try:
            if self.in_flight == "trainer_offloaded" and self.trainers:
                ray.get([self.handle(a).actor_rollout_to.remote("device", model=True, optimizer=False, grad=False) for a in self.trainers], timeout=CALL_TIMEOUT_S)
                log("shutdown_reload", trainers=len(self.trainers))
            elif self.in_flight and self.in_flight.startswith("sampler_asleep:"):
                name = self.in_flight.split(":", 1)[1]
                a = next((s for s in self.servers if s.name == name), None)
                if a:
                    self.engine_call(a, "wake_up", {"tags": ["weights", "kv_cache"]}, timeout=CALL_TIMEOUT_S)
                    ray.get(self.handle(a).resume_generation.remote(), timeout=CALL_TIMEOUT_S)
                    log("shutdown_wake", server=name)
        except Exception as e:  # noqa: BLE001
            log("failed", stage="shutdown", error=repr(e))
        log("stop")


def main():
    c = Controller()

    def _sig(*_):
        c.stop = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    c.run()


if __name__ == "__main__":
    sys.exit(main())
