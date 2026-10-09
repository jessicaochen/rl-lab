"""Time-slicing orchestration, dual-pool lock lifecycle, and multi-GPU offload/restore for rl-lab.

Provides:
1. ``TimeSliceOrchestratorClient`` (alias ``OrchestratorClient``): gRPC / HTTP / socket client
   for ``TimeSliceOrchestratorService`` on port 50051 supporting ``acquire``, ``release`` /
   ``yield_lock``, ``get_status``, ``list_groups``, and ``on_accelerators`` context manager.
2. ``DualPoolRoleLocks`` (alias ``RoleLocks``): Deadlock-free dual-pool lock coordinator for
   ``trainers`` and ``samplers`` resource groups enforcing global lock order ``TRAINER`` before
   ``SAMPLER`` during ``init`` and ``weight_sync``, deferred snapshot reporting when
   ``pending_waiters == 0``, and waiting on ``sample_queue`` outside the ``trainers`` lock.
3. Multi-GPU Trainer (>= 2 GPUs, e.g. 8x H200 Megatron TP2*EP4) offload
   (``MegatronEngine.to("cpu", model=True, optimizer=True, grad=True)``, explicit
   ``point="post_sync"`` parameter offload, ``aggressive_empty_cache(force_sync=True)``) and
   restore (``MegatronEngine.to("cuda")``), plus driver-level ``cuda-checkpoint`` and
   ``universal_cr_shim_v2.c`` (``SIGRTMIN+1``=35 / ``SIGRTMIN+2``=36, ``NCCL_NVLS_ENABLE=0``).
4. Multi-GPU Sampler (>= 2 GPUs, e.g. 2x H200 vLLM TP2) in-flight request draining,
   ``vLLM.sleep(level=1)`` (``release_kv_cache``), and
   ``vLLM.wake_up(tags=["weights", "kv_cache"])`` + ``reset_prefix_cache`` across all TP ranks
   with safe ``gpu_memory_utilization <= 0.85`` (default ``0.80``).
5. Structured ``[timeslice]`` and ``[gpu-mem]`` telemetry emission and parsing helpers.
"""

from __future__ import annotations

# pylint: disable=too-many-lines,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-instance-attributes,too-many-branches,too-many-statements,broad-exception-caught
import atexit
import contextlib
import gc
import importlib
import os
import signal
import socket
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

GB = float(1024**3)
SIG_PRE_CHECKPOINT = 35  # SIGRTMIN+1 for universal_cr_shim_v2.c pre-freeze communicator teardown
SIG_POST_RESTORE = 36  # SIGRTMIN+2 for universal_cr_shim_v2.c post-restore lazy re-init


def _encode_varint(value: int) -> bytes:
    """Encode non-negative integer as protobuf base-128 varint."""
    out = bytearray()
    val = int(value)
    while val > 0x7F:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val & 0x7F)
    return bytes(out)


def _encode_job_group_request(job_id: str, group_id: str) -> bytes:
    """Serialize AcquireRequest / YieldRequest (field 1: job_id, field 2: group_id)."""
    jb = job_id.encode("utf-8")
    gb = group_id.encode("utf-8")
    out = bytearray()
    if jb:
        out.append(0x0A)
        out.extend(_encode_varint(len(jb)))
        out.extend(jb)
    if gb:
        out.append(0x12)
        out.extend(_encode_varint(len(gb)))
        out.extend(gb)
    return bytes(out)


def _decode_varint_fields(payload: bytes) -> dict[int, int]:
    """Decode varint fields (wire_type == 0) from a protobuf response message."""
    fields: dict[int, int] = {}
    idx = 0
    n = len(payload)
    while idx < n:
        tag = 0
        shift = 0
        while idx < n:
            b = payload[idx]
            idx += 1
            tag |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            val = 0
            shift = 0
            while idx < n:
                b = payload[idx]
                idx += 1
                val |= (b & 0x7F) << shift
                if not b & 0x80:
                    break
                shift += 7
            fields[field_num] = val
        elif wire_type == 2:
            length = 0
            shift = 0
            while idx < n:
                b = payload[idx]
                idx += 1
                length |= (b & 0x7F) << shift
                if not b & 0x80:
                    break
                shift += 7
            idx += length
        else:
            break
    return fields


class _DirectGrpcOrchestratorStub:
    """Version-independent gRPC client stub for TimeSliceOrchestratorService."""

    SERVICE_PREFIX = "/timeslice_orchestrator.v1alpha1.TimeSliceOrchestratorService"

    def __init__(self, target: str, channel_options: list[tuple[str, Any]] | None = None) -> None:
        host, _, port_str = target.rpartition(":")
        if not host or not port_str.isdigit():
            raise ValueError(f"Invalid gRPC target {target!r}")
        self._target = target
        self._host = host
        self._port = int(port_str)
        self._channel_options = channel_options or []
        self._channel: Any = None
        self._acquire_rpc: Any = None
        self._yield_rpc: Any = None
        self._connect()

    def _connect(self) -> None:
        """Open (or reopen) the gRPC channel and bind Acquire / Yield unary stubs."""
        with socket.create_connection((self._host, self._port), timeout=2.0):
            pass
        grpc_mod = importlib.import_module("grpc")
        self._channel = grpc_mod.insecure_channel(self._target, options=self._channel_options)
        self._acquire_rpc = self._channel.unary_unary(
            f"{self.SERVICE_PREFIX}/Acquire",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )
        self._yield_rpc = self._channel.unary_unary(
            f"{self.SERVICE_PREFIX}/Yield",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )

    def acquire(self, job_id: str, group_id: str, timeout_sec: float = 7200.0) -> AcquireResult:
        """Invoke TimeSliceOrchestratorService/Acquire over gRPC with reconnect retry."""
        req = _encode_job_group_request(job_id, group_id)
        t0 = time.perf_counter()
        deadline = t0 + timeout_sec
        while True:
            rem = max(5.0, deadline - time.perf_counter())
            try:
                raw = self._acquire_rpc(req, timeout=rem)
                fields = _decode_varint_fields(raw)
                waited_ms = float(fields.get(2, 0)) or (time.perf_counter() - t0) * 1000.0
                return AcquireResult(
                    success=bool(fields.get(1, 1)),
                    waited_ms=waited_ms,
                    context_restored=bool(fields.get(3, 0)),
                )
            except Exception:
                if time.perf_counter() + 2.0 >= deadline:
                    raise
                time.sleep(2.0)
                with contextlib.suppress(Exception):
                    self.close()
                    self._connect()

    def release(self, job_id: str, group_id: str, timeout_sec: float = 60.0) -> YieldResult:
        """Invoke TimeSliceOrchestratorService/Yield over gRPC with reconnect retry."""
        req = _encode_job_group_request(job_id, group_id)
        for attempt in range(3):
            try:
                raw = self._yield_rpc(req, timeout=timeout_sec)
                fields = _decode_varint_fields(raw)
                waiters = int(fields.get(2, 0))
                return YieldResult(
                    success=bool(fields.get(1, 1)),
                    pending_waiters=waiters,
                    snapshot_deferred=bool(fields.get(3, int(waiters == 0))),
                )
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.0)
                with contextlib.suppress(Exception):
                    self.close()
                    self._connect()
        raise RuntimeError("unreachable")

    def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class AcquireResult:
    """Response from TimeSliceOrchestratorService.Acquire."""

    success: bool
    waited_ms: float = 0.0
    context_restored: bool = False


@dataclass(frozen=True)
class YieldResult:
    """Response from TimeSliceOrchestratorService.Yield."""

    success: bool
    pending_waiters: int = 0
    snapshot_deferred: bool = True


@dataclass(frozen=True)
class OrchestratorGroupStatus:
    """Status snapshot for a time-slicing resource group (e.g. trainers or samplers)."""

    group_id: str
    state: str = "IDLE"
    locking_job: str = ""
    active_job: str = ""
    loaded_job: str = ""
    waiter_queue_depth: int = 0


def format_timeslice_log(
    *, job_id: str, role: str, action: str, group: str, ts: str | None = None, **fields: Any
) -> str:
    """Format a structured ``[timeslice]`` log line matching the PROJECT.md contract."""
    norm_role = role.strip().lower()
    if norm_role not in ("trainer", "sampler"):
        raise ValueError(f"Unsupported role {role!r}; expected 'trainer' or 'sampler'")
    norm_action = action.strip().upper()
    if norm_action not in ("ACQUIRE", "OFFLOAD", "RESTORE", "RELEASE"):
        raise ValueError(
            f"Unsupported action {action!r}; expected ACQUIRE, OFFLOAD, RESTORE, or RELEASE"
        )
    if not job_id or not job_id.strip():
        raise ValueError("job_id must be non-empty")
    if not group or not group.strip():
        raise ValueError("group must be non-empty")

    parts = [
        "[timeslice]",
        f"ts={ts or _utc_now_iso()}",
        f"job={job_id.strip()}",
        f"role={norm_role}",
        f"action={norm_action}",
        f"group={group.strip()}",
    ]
    if norm_action == "ACQUIRE":
        waited_ms = float(fields.get("waited_ms", 0.0))
        restored = bool(fields.get("context_restored", False))
        parts.extend([f"waited_ms={waited_ms:.1f}", f"context_restored={str(restored).lower()}"])
    elif norm_action in ("OFFLOAD", "RESTORE"):
        mode = str(fields.get("mode", "app"))
        duration_ms = float(fields.get("duration_ms", 0.0))
        status = str(fields.get("status", "ok"))
        gb_key = "freed_gb_per_gpu" if norm_action == "OFFLOAD" else "restored_gb_per_gpu"
        gb_val = float(fields.get(gb_key, 0.0))
        parts.extend(
            [
                f"mode={mode}",
                f"duration_ms={duration_ms:.1f}",
                f"{gb_key}={gb_val:.2f}",
                f"status={status}",
            ]
        )
    elif norm_action == "RELEASE":
        hold_ms = float(fields.get("hold_ms", 0.0))
        waiters = int(fields.get("pending_waiters", 0))
        deferred = bool(fields.get("snapshot_deferred", waiters == 0))
        parts.extend(
            [
                f"hold_ms={hold_ms:.1f}",
                f"pending_waiters={waiters}",
                f"snapshot_deferred={str(deferred).lower()}",
            ]
        )
    if fields.get("step") is not None:
        parts.append(f"step={int(fields['step'])}")
    return " ".join(parts)


def format_offload_event(
    *,
    role: str,
    rank: int,
    point: str,
    seconds: float,
    vram_before_gb: float,
    vram_after_gb: float,
    job_id: str = "job1",
    group: str | None = None,
) -> str:
    """Format a ``[timeslice] action=OFFLOAD`` log line for trainer/sampler offload transitions."""
    norm_role = role.strip().lower()
    resolved_group = group or ("trainers" if norm_role == "trainer" else "samplers")
    freed_gb = max(0.0, float(vram_before_gb) - float(vram_after_gb))
    base = format_timeslice_log(
        job_id=job_id,
        role=norm_role,
        action="OFFLOAD",
        group=resolved_group,
        mode="app",
        duration_ms=float(seconds) * 1000.0,
        freed_gb_per_gpu=freed_gb,
        status="ok",
    )
    return f"{base} rank={rank} point={point}"


def format_restore_event(
    *,
    role: str,
    rank: int,
    point: str,
    seconds: float,
    vram_before_gb: float,
    vram_after_gb: float,
    job_id: str = "job1",
    group: str | None = None,
) -> str:
    """Format a ``[timeslice] action=RESTORE`` log line for trainer/sampler restore transitions."""
    norm_role = role.strip().lower()
    resolved_group = group or ("trainers" if norm_role == "trainer" else "samplers")
    restored_gb = max(0.0, float(vram_after_gb) - float(vram_before_gb))
    base = format_timeslice_log(
        job_id=job_id,
        role=norm_role,
        action="RESTORE",
        group=resolved_group,
        mode="app",
        duration_ms=float(seconds) * 1000.0,
        restored_gb_per_gpu=restored_gb,
        status="ok",
    )
    return f"{base} rank={rank} point={point}"


def parse_timeslice_log(line: str) -> dict[str, str]:
    """Parse key=value pairs from a ``[timeslice]`` structured log line."""
    if "[timeslice]" not in line:
        raise ValueError(f"Missing [timeslice] prefix in line: {line!r}")
    kv_pairs: dict[str, str] = {}
    for token in line.strip().split():
        if "=" in token:
            key, val = token.split("=", 1)
            kv_pairs[key.strip()] = val.strip().rstrip(",")
        elif token in ("ACQUIRE", "RELEASE", "OFFLOAD", "RESTORE"):
            kv_pairs.setdefault("action", token)
    if "action" not in kv_pairs:
        raise ValueError(f"Missing action in [timeslice] log line: {line!r}")
    if kv_pairs["action"] not in ("ACQUIRE", "RELEASE", "OFFLOAD", "RESTORE"):
        raise ValueError(f"Invalid action {kv_pairs['action']!r} in line: {line!r}")
    return kv_pairs


def format_gpu_mem_log(
    *,
    point: str,
    role: str,
    rank: int,
    world_size: int = 8,
    torch_alloc_gb: float = 0.0,
    torch_reserved_gb: float = 0.0,
    cuda_used_gb: float = 0.0,
    cuda_total_gb: float = 141.0,
    **extra: Any,
) -> str:
    """Format a structured ``[gpu-mem]`` log line matching the PROJECT.md contract."""
    norm_role = role.strip().lower()
    if norm_role not in ("trainer", "sampler"):
        raise ValueError(f"Invalid role {role!r}; expected 'trainer' or 'sampler'")
    if rank < 0:
        raise ValueError(f"Negative rank {rank} is invalid")
    if world_size < 1:
        raise ValueError(f"Invalid world_size {world_size}")
    for name, val in (
        ("torch_alloc_gb", torch_alloc_gb),
        ("torch_reserved_gb", torch_reserved_gb),
        ("cuda_used_gb", cuda_used_gb),
        ("cuda_total_gb", cuda_total_gb),
    ):
        if val < 0.0:
            raise ValueError(f"Negative memory metric {name}={val}")

    base = (
        f"[gpu-mem] point={point.strip()} role={norm_role} rank={rank} world_size={world_size} "
        f"torch_alloc_gb={torch_alloc_gb:.2f} torch_reserved_gb={torch_reserved_gb:.2f} "
        f"cuda_used_gb={cuda_used_gb:.2f} cuda_total_gb={cuda_total_gb:.2f}"
    )
    if extra:
        extra_str = " ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
        if extra_str:
            base = f"{base} {extra_str}"
    return base


def parse_gpu_mem_log(line: str) -> dict[str, Any]:
    """Parse and validate a ``[gpu-mem]`` structured log line."""
    if "[gpu-mem]" not in line:
        raise ValueError(f"Missing [gpu-mem] prefix in line: {line!r}")
    kv_pairs: dict[str, Any] = {}
    for token in line.strip().split():
        if "=" in token:
            key, val = token.split("=", 1)
            kv_pairs[key.strip()] = val.strip().rstrip(",")
    for req in ("point", "role", "rank", "torch_alloc_gb", "cuda_used_gb"):
        if req not in kv_pairs:
            raise ValueError(f"Missing required field {req!r} in [gpu-mem] line: {line!r}")
    if kv_pairs["role"] not in ("trainer", "sampler"):
        raise ValueError(f"Invalid role {kv_pairs['role']!r} in [gpu-mem] line: {line!r}")
    rank_val = int(kv_pairs["rank"])
    if rank_val < 0:
        raise ValueError(f"Negative rank {rank_val} in [gpu-mem] line: {line!r}")
    kv_pairs["rank"] = rank_val
    if "world_size" in kv_pairs:
        kv_pairs["world_size"] = int(kv_pairs["world_size"])
    for float_key in ("torch_alloc_gb", "torch_reserved_gb", "cuda_used_gb", "cuda_total_gb"):
        if float_key in kv_pairs:
            fval = float(kv_pairs[float_key])
            if fval < 0.0:
                raise ValueError(f"Negative memory metric {float_key}={fval}")
            kv_pairs[float_key] = fval
    return kv_pairs


def capture_gpu_memory_snapshot() -> dict[str, float]:
    """Capture current GPU allocator and device-level memory usage in GiB."""
    with contextlib.suppress(ImportError, RuntimeError, AttributeError):
        torch_mod = importlib.import_module("torch")
        if torch_mod.cuda.is_available():
            torch_mod.cuda.synchronize()
            free_bytes, total_bytes = torch_mod.cuda.mem_get_info()
            return {
                "torch_alloc_gb": float(torch_mod.cuda.memory_allocated()) / GB,
                "torch_reserved_gb": float(torch_mod.cuda.memory_reserved()) / GB,
                "cuda_used_gb": float(total_bytes - free_bytes) / GB,
                "cuda_total_gb": float(total_bytes) / GB,
            }
    return {
        "torch_alloc_gb": 0.0,
        "torch_reserved_gb": 0.0,
        "cuda_used_gb": 0.0,
        "cuda_total_gb": 141.0,
    }


def emit_gpu_mem_log(
    point: str,
    role: str,
    rank: int = 0,
    world_size: int = 8,
    snapshot: dict[str, float] | None = None,
    **extra: Any,
) -> str:
    """Capture or format a ``[gpu-mem]`` line and print it with flush=True."""
    snap = snapshot if snapshot is not None else capture_gpu_memory_snapshot()
    line = format_gpu_mem_log(
        point=point,
        role=role,
        rank=rank,
        world_size=world_size,
        torch_alloc_gb=snap.get("torch_alloc_gb", 0.0),
        torch_reserved_gb=snap.get("torch_reserved_gb", 0.0),
        cuda_used_gb=snap.get("cuda_used_gb", 0.0),
        cuda_total_gb=snap.get("cuda_total_gb", 141.0),
        **extra,
    )
    print(line, flush=True)
    return line


def aggressive_empty_cache(force_sync: bool = True) -> None:
    """Synchronize CUDA streams, run GC, and release unoccupied PyTorch CUDA cache."""
    gc.collect()
    with contextlib.suppress(ImportError, RuntimeError, AttributeError):
        torch_mod = importlib.import_module("torch")
        if torch_mod.cuda.is_available():
            if force_sync:
                torch_mod.cuda.synchronize()
            torch_mod.cuda.empty_cache()
            if hasattr(torch_mod.cuda, "ipc_collect"):
                torch_mod.cuda.ipc_collect()


class TimeSliceOrchestratorClient:
    """Client for TimeSliceOrchestratorService (port 50051) with gRPC/CLI/in-memory fallback."""

    DEFAULT_ADDR = "timeslice-timesliceorchestrator.timeslice-system.svc.cluster.local:50051"

    def __init__(
        self,
        target: str | None = None,
        job_id: str | None = None,
        group_id: str | None = None,
        channel_options: list[tuple[str, Any]] | None = None,
        rlts_bin: str | None = None,
    ) -> None:
        self.target = (
            target
            or os.environ.get("TIMESLICE_ORCHESTRATOR_ADDR")
            or os.environ.get("TIMESLICE_ORCH_ADDR")
            or self.DEFAULT_ADDR
        )
        self.default_job_id = (
            job_id or os.environ.get("TIMESLICE_JOB_ID") or os.environ.get("JOB_ID")
        )
        self.default_group_id = group_id or os.environ.get("TIMESLICE_GROUP")
        self.rlts_bin = rlts_bin or os.environ.get("RLTS_BIN", "rlts")
        self.channel_options = channel_options or [
            ("grpc.keepalive_time_ms", 600000),
            ("grpc.keepalive_timeout_ms", 20000),
            ("grpc.keepalive_permit_without_calls", 0),
        ]
        self._sdk_client: Any = None
        self._local_group_holders: dict[str, str] = {}
        self._local_loaded_job: dict[str, str] = {}
        self._local_waiters: dict[str, int] = {"trainers": 0, "samplers": 0}
        self._init_sdk_client()

    def _init_sdk_client(self) -> None:
        """Attempt to bind to upstream timeslice SDK or direct gRPC wire stub."""
        with contextlib.suppress(ImportError, RuntimeError, ValueError, OSError):
            orch_mod = importlib.import_module("timeslice.orchestrator")
            upstream_cls = getattr(orch_mod, "TimeSliceOrchestratorClient", None)
            if upstream_cls is not None:
                self._sdk_client = upstream_cls(
                    self.target,
                    job_id=self.default_job_id,
                    group_id=self.default_group_id,
                    channel_options=self.channel_options,
                )
                return
        with contextlib.suppress(ImportError, RuntimeError, ValueError, OSError):
            self._sdk_client = _DirectGrpcOrchestratorStub(
                self.target, channel_options=self.channel_options
            )

    def set_simulated_waiters(self, group_id: str, count: int) -> None:
        """Configure waiter queue depth for local/offline testing."""
        if count < 0:
            raise ValueError(f"waiter count must be >= 0, got {count}")
        self._local_waiters[group_id] = count

    def _resolve_ids(self, job_id: str | None, group_id: str | None) -> tuple[str, str]:
        resolved_job = (job_id if job_id is not None else self.default_job_id) or ""
        resolved_group = (group_id if group_id is not None else self.default_group_id) or ""
        if not resolved_job.strip():
            raise ValueError("job_id must be non-empty")
        if not resolved_group.strip():
            raise ValueError("group_id must be non-empty")
        return resolved_job.strip(), resolved_group.strip()

    def acquire(
        self,
        job_id: str | None = None,
        group_id: str | None = None,
        timeout_sec: float = 7200.0,
    ) -> AcquireResult:
        """Acquire exclusive accelerator lock for ``job_id`` in ``group_id``."""
        r_job, r_group = self._resolve_ids(job_id, group_id)
        t0 = time.perf_counter()

        if self._sdk_client is not None:
            with contextlib.suppress(RuntimeError, TimeoutError, OSError, ValueError):
                resp = self._sdk_client.acquire(
                    job_id=r_job, group_id=r_group, timeout_sec=timeout_sec
                )
                return AcquireResult(
                    success=bool(getattr(resp, "success", True)),
                    waited_ms=float(
                        getattr(resp, "waited_ms", (time.perf_counter() - t0) * 1000.0)
                    ),
                    context_restored=bool(getattr(resp, "context_restored", False)),
                )

        prev_loaded = self._local_loaded_job.get(r_group)
        context_restored = prev_loaded is not None and prev_loaded != r_job
        self._local_group_holders[r_group] = r_job
        self._local_loaded_job[r_group] = r_job
        return AcquireResult(
            success=True,
            waited_ms=(time.perf_counter() - t0) * 1000.0,
            context_restored=context_restored,
        )

    def release(
        self,
        job_id: str | None = None,
        group_id: str | None = None,
        timeout_sec: float = 60.0,
    ) -> YieldResult:
        """Yield/release accelerator lock for ``job_id`` in ``group_id``."""
        r_job, r_group = self._resolve_ids(job_id, group_id)

        if self._sdk_client is not None:
            with contextlib.suppress(RuntimeError, TimeoutError, OSError, ValueError):
                resp = self._sdk_client.release(
                    job_id=r_job, group_id=r_group, timeout_sec=timeout_sec
                )
                waiters = int(getattr(resp, "pending_waiters", 0))
                return YieldResult(
                    success=bool(getattr(resp, "success", True)),
                    pending_waiters=waiters,
                    snapshot_deferred=bool(getattr(resp, "snapshot_deferred", waiters == 0)),
                )

        self._local_group_holders.pop(r_group, None)
        waiters = int(self._local_waiters.get(r_group, 0))
        return YieldResult(
            success=True,
            pending_waiters=waiters,
            snapshot_deferred=waiters == 0,
        )

    def yield_lock(
        self,
        job_id: str | None = None,
        group_id: str | None = None,
        timeout_sec: float = 60.0,
    ) -> YieldResult:
        """Alias for ``release`` matching ``TimeSliceOrchestratorService.Yield``."""
        return self.release(job_id=job_id, group_id=group_id, timeout_sec=timeout_sec)

    def get_status(
        self, group_id: str | None = None, timeout_sec: float = 30.0
    ) -> OrchestratorGroupStatus:
        """Return lock state and waiter queue depth for ``group_id``."""
        r_group = (group_id or self.default_group_id or "").strip()
        if not r_group:
            raise ValueError("group_id must be non-empty")
        if self._sdk_client is not None and hasattr(self._sdk_client, "get_status"):
            with contextlib.suppress(RuntimeError, TimeoutError, OSError, ValueError):
                st = self._sdk_client.get_status(r_group, timeout_sec=timeout_sec)
                return OrchestratorGroupStatus(
                    group_id=r_group,
                    state=str(getattr(st, "state", "IDLE")),
                    locking_job=str(getattr(st, "locking_job", "")),
                    active_job=str(getattr(st, "active_job", "")),
                    loaded_job=str(getattr(st, "loaded_job", "")),
                    waiter_queue_depth=int(getattr(st, "waiter_queue_depth", 0)),
                )
        holder = self._local_group_holders.get(r_group, "")
        loaded = self._local_loaded_job.get(r_group, "")
        waiters = int(self._local_waiters.get(r_group, 0))
        state = "LOCKED" if holder else ("IDLE_YIELDED" if loaded else "IDLE")
        return OrchestratorGroupStatus(
            group_id=r_group,
            state=state,
            locking_job=holder,
            active_job=holder,
            loaded_job=loaded,
            waiter_queue_depth=waiters,
        )

    def list_groups(self, timeout_sec: float = 30.0) -> list[str]:
        """List registered time-slicing resource groups."""
        if self._sdk_client is not None and hasattr(self._sdk_client, "list_groups"):
            with contextlib.suppress(RuntimeError, TimeoutError, OSError, ValueError):
                groups = self._sdk_client.list_groups(timeout_sec=timeout_sec)
                if groups:
                    return [str(g) for g in groups]
        return sorted(set(self._local_waiters.keys()) | {"trainers", "samplers"})

    @contextlib.contextmanager
    def on_accelerators(
        self,
        job_id: str | None = None,
        group_id: str | None = None,
        timeout_sec: float = 7200.0,
    ) -> Iterator[AcquireResult]:
        """Context manager that acquires the group lock on entry and yields it on exit."""
        r_job, r_group = self._resolve_ids(job_id, group_id)
        acq = self.acquire(job_id=r_job, group_id=r_group, timeout_sec=timeout_sec)
        try:
            yield acq
        finally:
            self.release(job_id=r_job, group_id=r_group)

    def close(self) -> None:
        """Close underlying gRPC channel if open."""
        if self._sdk_client is not None and hasattr(self._sdk_client, "close"):
            with contextlib.suppress(RuntimeError, OSError):
                self._sdk_client.close()


OrchestratorClient = TimeSliceOrchestratorClient


@dataclass
class DualPoolRoleLocks:
    """Deadlock-free dual-pool lock manager for ``trainers`` and ``samplers`` groups.

    Enforces strict Global Lock Order: ``TRAINER`` before ``SAMPLER``.
    - ``init`` and ``weight_sync`` acquire ``TRAINER`` then ``SAMPLER``, and release
      ``SAMPLER`` then ``TRAINER``.
    - Attempting to acquire ``TRAINER`` while already holding ``SAMPLER`` raises
      ``RuntimeError("lock-order violation: ...")``.
    """

    job_id: str
    trainer_group: str = "trainers"
    sampler_group: str = "samplers"
    orchestrator_addr: str = TimeSliceOrchestratorClient.DEFAULT_ADDR
    enabled: bool = True
    trainer_enabled: bool = True
    sampler_enabled: bool = True
    pending_waiters_trainer: int = 0
    pending_waiters_sampler: int = 0
    client: TimeSliceOrchestratorClient | None = None
    held_roles: set[str] = field(default_factory=set)
    acquire_timestamps: dict[str, float] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.job_id or not self.job_id.strip():
            raise ValueError("job_id must be non-empty")
        if not self.trainer_group or not self.trainer_group.strip():
            raise ValueError("trainer_group must be non-empty")
        if not self.sampler_group or not self.sampler_group.strip():
            raise ValueError("sampler_group must be non-empty")
        if self.trainer_group.strip() == self.sampler_group.strip():
            raise ValueError(
                f"trainer_group and sampler_group must be distinct, got {self.trainer_group!r}"
            )
        self.job_id = self.job_id.strip()
        self.trainer_group = self.trainer_group.strip()
        self.sampler_group = self.sampler_group.strip()
        if not self.trainer_enabled and not self.sampler_enabled:
            self.enabled = False
        if self.client is None:
            self.client = TimeSliceOrchestratorClient(
                target=self.orchestrator_addr, job_id=self.job_id
            )
        self.client.set_simulated_waiters(self.trainer_group, self.pending_waiters_trainer)
        self.client.set_simulated_waiters(self.sampler_group, self.pending_waiters_sampler)
        atexit.register(self.release_all)

    @staticmethod
    def _parse_bool_env(val: str | None, default: bool = False) -> bool:
        if val is None:
            return default
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    @classmethod
    def from_env(cls) -> DualPoolRoleLocks:
        """Instantiate DualPoolRoleLocks from TIMESLICE_* environment variables."""
        enabled = cls._parse_bool_env(os.environ.get("TIMESLICE_ENABLED", "0"), default=False)
        trainer_enabled = cls._parse_bool_env(
            os.environ.get("TIMESLICE_TRAINER_ENABLED"), default=enabled
        )
        sampler_enabled = cls._parse_bool_env(
            os.environ.get("TIMESLICE_SAMPLER_ENABLED"), default=enabled
        )
        job_id = (
            os.environ.get("TIMESLICE_JOB_ID")
            or os.environ.get("JOB_ID")
            or os.environ.get("RUN_ID")
            or "job1"
        )
        orch_addr = (
            os.environ.get("TIMESLICE_ORCHESTRATOR_ADDR")
            or os.environ.get("TIMESLICE_ORCH_ADDR")
            or TimeSliceOrchestratorClient.DEFAULT_ADDR
        )
        return cls(
            job_id=job_id,
            trainer_group=os.environ.get("TIMESLICE_TRAINER_GROUP", "trainers"),
            sampler_group=os.environ.get("TIMESLICE_SAMPLER_GROUP", "samplers"),
            orchestrator_addr=orch_addr,
            enabled=enabled and (trainer_enabled or sampler_enabled),
            trainer_enabled=trainer_enabled,
            sampler_enabled=sampler_enabled,
        )

    def _normalize_role(self, role: str) -> tuple[str, str]:
        norm = role.strip().lower()
        if norm not in ("trainer", "sampler"):
            raise ValueError(f"Unsupported role {role!r}; expected 'trainer' or 'sampler'")
        return norm, (self.trainer_group if norm == "trainer" else self.sampler_group)

    def is_role_enabled(self, role: str) -> bool:
        """Return whether time-slicing lock orchestration is active for ``role``."""
        norm_role, _ = self._normalize_role(role)
        if not self.enabled:
            return False
        return self.trainer_enabled if norm_role == "trainer" else self.sampler_enabled

    def acquire(
        self,
        role: str,
        timeout_sec: float = 7200.0,
        waited_ms: float | None = None,
        context_restored: bool | None = None,
        step: int | None = None,
    ) -> bool:
        """Acquire ``trainer`` or ``sampler`` lock enforcing global order TRAINER -> SAMPLER."""
        norm_role, group = self._normalize_role(role)
        if not self.is_role_enabled(norm_role) or norm_role in self.held_roles:
            return False
        if norm_role == "trainer" and "sampler" in self.held_roles:
            raise RuntimeError(
                "lock-order violation: acquiring TRAINER while holding SAMPLER "
                "(global order is trainer-first; release SAMPLER first)"
            )

        assert self.client is not None
        self.client.set_simulated_waiters(self.trainer_group, self.pending_waiters_trainer)
        self.client.set_simulated_waiters(self.sampler_group, self.pending_waiters_sampler)
        acq = self.client.acquire(job_id=self.job_id, group_id=group, timeout_sec=timeout_sec)
        self.held_roles.add(norm_role)
        self.acquire_timestamps[norm_role] = time.perf_counter()
        log_line = format_timeslice_log(
            job_id=self.job_id,
            role=norm_role,
            action="ACQUIRE",
            group=group,
            waited_ms=waited_ms if waited_ms is not None else acq.waited_ms,
            context_restored=(
                context_restored if context_restored is not None else acq.context_restored
            ),
            step=step,
        )
        self.events.append(log_line)
        print(log_line, flush=True)
        return True

    def release(
        self,
        role: str,
        hold_ms: float | None = None,
        step: int | None = None,
    ) -> bool:
        """Release ``trainer`` or ``sampler`` lock; idempotent if not currently held."""
        norm_role, group = self._normalize_role(role)
        if not self.is_role_enabled(norm_role) or norm_role not in self.held_roles:
            return False

        self.held_roles.remove(norm_role)
        t_acq = self.acquire_timestamps.pop(norm_role, None)
        computed_hold_ms = (
            hold_ms
            if hold_ms is not None
            else ((time.perf_counter() - t_acq) * 1000.0 if t_acq is not None else 0.0)
        )
        waiters = (
            self.pending_waiters_trainer
            if norm_role == "trainer"
            else self.pending_waiters_sampler
        )
        if self.client is not None:
            self.client.set_simulated_waiters(group, waiters)
            with contextlib.suppress(RuntimeError, TimeoutError, OSError, ValueError):
                res = self.client.release(job_id=self.job_id, group_id=group)
                waiters = res.pending_waiters

        log_line = format_timeslice_log(
            job_id=self.job_id,
            role=norm_role,
            action="RELEASE",
            group=group,
            hold_ms=computed_hold_ms,
            pending_waiters=waiters,
            snapshot_deferred=waiters == 0,
            step=step,
        )
        self.events.append(log_line)
        print(log_line, flush=True)
        return True

    def acquire_for_init_or_weight_sync(self, timeout_sec: float = 7200.0) -> None:
        """Acquire both pools in strict global order: TRAINER first, then SAMPLER."""
        if "sampler" in self.held_roles and "trainer" not in self.held_roles:
            self.release("sampler")
        self.acquire("trainer", timeout_sec=timeout_sec)
        self.acquire("sampler", timeout_sec=timeout_sec)

    def release_all(self) -> None:
        """Release all held locks in reverse global order: SAMPLER first, then TRAINER."""
        if "sampler" in self.held_roles:
            self.release("sampler")
        if "trainer" in self.held_roles:
            self.release("trainer")

    def wait_for_sample_batch_outside_trainer_lock(
        self,
        sample_queue: Any,
        offload_fn: Callable[[], None] | None = None,
        restore_fn: Callable[[], None] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Dequeue a rollout batch while ensuring ``trainers`` lock is NOT held during wait."""
        if "trainer" in self.held_roles:
            if offload_fn is not None:
                offload_fn()
            self.release("trainer")

        batch = sample_queue.get(timeout=timeout) if timeout is not None else sample_queue.get()
        self.acquire("trainer")
        if restore_fn is not None:
            restore_fn()
        return batch


RoleLocks = DualPoolRoleLocks
RoleLockConfig = DualPoolRoleLocks


def offload_megatron_trainer(
    engine: Any,
    *,
    job_id: str = "job1",
    group: str = "trainers",
    rank: int = 0,
    world_size: int = 8,
    point: str = "pre_offload",
    model: bool = True,
    optimizer: bool = True,
    grad: bool = True,
    mode: str = "app",
) -> dict[str, Any]:
    """Offload multi-GPU Megatron trainer state to host CPU and empty CUDA cache.

    Supports both full step-end offload (``model=True, optimizer=True, grad=True``) and
    explicit post-weight-sync offload (``point="post_sync", model=True, optimizer=False``).
    """
    if world_size < 2:
        raise ValueError(f"Multi-GPU trainer requires world_size >= 2, got {world_size}")

    snap_before = capture_gpu_memory_snapshot()
    pre_point = "pre_sync" if point == "post_sync" else "pre_offload"
    emit_gpu_mem_log(pre_point, "trainer", rank=rank, world_size=world_size, snapshot=snap_before)

    t0 = time.perf_counter()
    if mode == "cuda":
        trigger_cuda_checkpoint_transition(action="checkpoint")
    else:
        try:
            engine.to("cpu", model=model, optimizer=optimizer, grad=grad, point=point)
        except TypeError:
            engine.to("cpu", model=model, optimizer=optimizer, grad=grad)
        aggressive_empty_cache(force_sync=True)

    duration_ms = (time.perf_counter() - t0) * 1000.0
    snap_after = capture_gpu_memory_snapshot()
    post_point = "post_sync" if point == "post_sync" else "post_offload"
    emit_gpu_mem_log(post_point, "trainer", rank=rank, world_size=world_size, snapshot=snap_after)

    freed_gb = max(0.0, snap_before["cuda_used_gb"] - snap_after["cuda_used_gb"])
    ts_line = format_timeslice_log(
        job_id=job_id,
        role="trainer",
        action="OFFLOAD",
        group=group,
        mode=mode,
        duration_ms=duration_ms,
        freed_gb_per_gpu=freed_gb,
        status="ok",
    )
    if rank == 0:
        print(ts_line, flush=True)
    return {
        "duration_ms": duration_ms,
        "freed_gb_per_gpu": freed_gb,
        "timeslice_log": ts_line,
        "before": snap_before,
        "after": snap_after,
    }


def restore_megatron_trainer(
    engine: Any,
    *,
    job_id: str = "job1",
    group: str = "trainers",
    rank: int = 0,
    world_size: int = 8,
    model: bool = True,
    optimizer: bool = True,
    grad: bool = True,
    mode: str = "app",
) -> dict[str, Any]:
    """Restore multi-GPU Megatron trainer state back onto CUDA devices."""
    if world_size < 2:
        raise ValueError(f"Multi-GPU trainer requires world_size >= 2, got {world_size}")

    snap_before = capture_gpu_memory_snapshot()
    emit_gpu_mem_log(
        "pre_restore", "trainer", rank=rank, world_size=world_size, snapshot=snap_before
    )

    t0 = time.perf_counter()
    if mode == "cuda":
        trigger_cuda_checkpoint_transition(action="restore")
    else:
        try:
            engine.to("cuda", model=model, optimizer=optimizer, grad=grad, point="restore")
        except TypeError:
            engine.to("cuda", model=model, optimizer=optimizer, grad=grad)

    duration_ms = (time.perf_counter() - t0) * 1000.0
    snap_after = capture_gpu_memory_snapshot()
    emit_gpu_mem_log(
        "post_restore", "trainer", rank=rank, world_size=world_size, snapshot=snap_after
    )

    restored_gb = max(0.0, snap_after["cuda_used_gb"] - snap_before["cuda_used_gb"])
    ts_line = format_timeslice_log(
        job_id=job_id,
        role="trainer",
        action="RESTORE",
        group=group,
        mode=mode,
        duration_ms=duration_ms,
        restored_gb_per_gpu=restored_gb,
        status="ok",
    )
    if rank == 0:
        print(ts_line, flush=True)
    return {
        "duration_ms": duration_ms,
        "restored_gb_per_gpu": restored_gb,
        "timeslice_log": ts_line,
        "before": snap_before,
        "after": snap_after,
    }


def drain_sampler_inflight_requests(
    server_or_engine: Any,
    timeout_sec: float = 120.0,
    poll_interval_sec: float = 0.05,
) -> int:
    """Drain in-flight rollout requests before calling ``vLLM.sleep(level=1)``."""
    deadline = time.perf_counter() + timeout_sec
    polls = 0
    while time.perf_counter() < deadline:
        inflight = 0
        if hasattr(server_or_engine, "get_num_unfinished_requests"):
            inflight = int(server_or_engine.get_num_unfinished_requests())
        elif hasattr(server_or_engine, "inflight_requests"):
            inflight = int(getattr(server_or_engine, "inflight_requests", 0))
        if inflight <= 0:
            return polls
        polls += 1
        time.sleep(poll_interval_sec)
    raise TimeoutError(
        f"Timed out after {timeout_sec}s waiting to drain in-flight sampler requests"
    )


async def sleep_vllm_sampler_async(
    engine: Any,
    *,
    job_id: str = "job1",
    group: str = "samplers",
    tp_size: int = 2,
    gpu_memory_utilization: float = 0.80,
) -> dict[str, Any]:
    """Drain in-flight requests and put multi-GPU vLLM sampler to sleep (``sleep(level=1)``)."""
    if tp_size < 2:
        raise ValueError(f"Multi-GPU sampler requires tp_size >= 2, got {tp_size}")
    if not 0.10 <= gpu_memory_utilization <= 0.85:
        raise ValueError(
            f"gpu_memory_utilization={gpu_memory_utilization} outside safe "
            "time-slicing bound [0.10, 0.85]"
        )

    drain_sampler_inflight_requests(engine)
    snap_before = capture_gpu_memory_snapshot()
    for rank in range(tp_size):
        emit_gpu_mem_log(
            "pre_offload", "sampler", rank=rank, world_size=tp_size, snapshot=snap_before
        )

    t0 = time.perf_counter()
    await engine.sleep(level=1)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    snap_after = capture_gpu_memory_snapshot()
    for rank in range(tp_size):
        emit_gpu_mem_log(
            "post_offload", "sampler", rank=rank, world_size=tp_size, snapshot=snap_after
        )

    freed_gb = max(0.0, snap_before["cuda_used_gb"] - snap_after["cuda_used_gb"])
    ts_line = format_timeslice_log(
        job_id=job_id,
        role="sampler",
        action="OFFLOAD",
        group=group,
        mode="app",
        duration_ms=duration_ms,
        freed_gb_per_gpu=freed_gb,
        status="ok",
    )
    print(ts_line, flush=True)
    return {
        "duration_ms": duration_ms,
        "freed_gb_per_gpu": freed_gb,
        "timeslice_log": ts_line,
    }


async def wake_vllm_sampler_async(
    engine: Any,
    *,
    job_id: str = "job1",
    group: str = "samplers",
    tp_size: int = 2,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Wake multi-GPU vLLM sampler across TP ranks (wake_up(tags=['weights', 'kv_cache']))."""
    if tp_size < 2:
        raise ValueError(f"Multi-GPU sampler requires tp_size >= 2, got {tp_size}")

    wake_tags = tags if tags is not None else ["weights", "kv_cache"]
    snap_before = capture_gpu_memory_snapshot()
    for rank in range(tp_size):
        emit_gpu_mem_log(
            "pre_restore", "sampler", rank=rank, world_size=tp_size, snapshot=snap_before
        )

    t0 = time.perf_counter()
    await engine.wake_up(tags=wake_tags)
    if hasattr(engine, "reset_prefix_cache"):
        try:
            await engine.reset_prefix_cache(reset_connector=True)
        except TypeError:
            await engine.reset_prefix_cache()
    duration_ms = (time.perf_counter() - t0) * 1000.0

    snap_after = capture_gpu_memory_snapshot()
    for rank in range(tp_size):
        emit_gpu_mem_log(
            "post_restore", "sampler", rank=rank, world_size=tp_size, snapshot=snap_after
        )

    restored_gb = max(0.0, snap_after["cuda_used_gb"] - snap_before["cuda_used_gb"])
    ts_line = format_timeslice_log(
        job_id=job_id,
        role="sampler",
        action="RESTORE",
        group=group,
        mode="app",
        duration_ms=duration_ms,
        restored_gb_per_gpu=restored_gb,
        status="ok",
    )
    print(ts_line, flush=True)
    return {
        "duration_ms": duration_ms,
        "restored_gb_per_gpu": restored_gb,
        "timeslice_log": ts_line,
    }


def trigger_cuda_checkpoint_transition(
    action: str,
    pids: list[int] | None = None,
    use_cr_shim_signals: bool = True,
    cuda_checkpoint_bin: str = "cuda-checkpoint",
) -> dict[str, Any]:
    """Execute or prepare driver-level cuda-checkpoint with universal_cr_shim_v2.c signals."""
    norm_action = action.strip().lower()
    if norm_action not in ("checkpoint", "restore"):
        raise ValueError(f"Unsupported cuda-checkpoint action {action!r}")

    nvls_env = os.environ.get("NCCL_NVLS_ENABLE", "0")
    if nvls_env != "0":
        raise RuntimeError(
            "NCCL_NVLS_ENABLE must be set to '0' for multi-GPU cuda-checkpoint compatibility "
            "(otherwise NVLS multicast handles fail with CUDA error 801 / CUDA_ERROR_NOT_SUPPORTED)"
        )

    target_pids = pids if pids is not None else []
    commands_built: list[list[str]] = []

    if norm_action == "checkpoint":
        if use_cr_shim_signals:
            for pid in target_pids:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.Signals(SIG_PRE_CHECKPOINT))
        for step in ("lock", "checkpoint"):
            cmd = [cuda_checkpoint_bin, "--action", step]
            for pid in target_pids:
                cmd.extend(["--pid", str(pid)])
            commands_built.append(cmd)
    else:
        cmd = [cuda_checkpoint_bin, "--toggle"]
        for pid in target_pids:
            cmd.extend(["--pid", str(pid)])
        commands_built.append(cmd)
        if use_cr_shim_signals:
            for pid in target_pids:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.Signals(SIG_POST_RESTORE))

    return {
        "action": norm_action,
        "nccl_nvls_enable": nvls_env,
        "sig_pre_checkpoint": SIG_PRE_CHECKPOINT,
        "sig_post_restore": SIG_POST_RESTORE,
        "commands": commands_built,
    }


__all__ = [
    "SIG_POST_RESTORE",
    "SIG_PRE_CHECKPOINT",
    "AcquireResult",
    "DualPoolRoleLocks",
    "OrchestratorClient",
    "OrchestratorGroupStatus",
    "RoleLockConfig",
    "RoleLocks",
    "TimeSliceOrchestratorClient",
    "YieldResult",
    "aggressive_empty_cache",
    "capture_gpu_memory_snapshot",
    "drain_sampler_inflight_requests",
    "emit_gpu_mem_log",
    "format_gpu_mem_log",
    "format_offload_event",
    "format_restore_event",
    "format_timeslice_log",
    "offload_megatron_trainer",
    "parse_gpu_mem_log",
    "parse_timeslice_log",
    "restore_megatron_trainer",
    "sleep_vllm_sampler_async",
    "trigger_cuda_checkpoint_transition",
    "wake_vllm_sampler_async",
]
