"""Time-slicing orchestration, dual-pool lock lifecycle, and verl-app-channel-offload for rl-lab.

Provides:
1. ``TimeSliceOrchestratorClient`` (alias ``OrchestratorClient``): gRPC / HTTP / socket client
   for ``TimeSliceOrchestratorService`` on port 50051 supporting ``acquire``, ``release`` /
   ``yield_lock``, ``get_status``, ``list_groups``, and ``on_accelerators`` context manager.
2. ``DualPoolRoleLocks`` (alias ``RoleLocks``): Deadlock-free dual-pool lock coordinator for
   ``trainers`` and ``samplers`` resource groups enforcing global lock order ``TRAINER`` before
   ``SAMPLER`` during ``init`` and ``weight_sync``, deferred snapshot reporting when
   ``pending_waiters == 0``, and waiting on ``sample_queue`` outside the ``trainers`` lock.
3. ``WorkloadChannel`` gRPC client (``WorkloadHandle``, ``register_workload``) and Ray per-node
   offload relay (``OffloadSpec``, ``RankFanout``, ``OffloadProxy``, ``OffloadProxySet``,
   ``spawn_offload_proxies``, ``install_verl_offload``) for ``verl-app-channel-offload``
   (``origin/verl-app-channel-offload``, commit ``3628ef7c2939e0529b572e758efb7d712008fb86``).
4. Multi-GPU Trainer (>= 2 GPUs, e.g. 8x H200 Megatron TP2*EP4) application-level offload
   (``MegatronEngine.to("cpu", model=True, optimizer=True, grad=True)`` +
   ``aggressive_empty_cache(force_sync=True)``) and restore
   (``MegatronEngine.to("cuda", model=True, optimizer=True, grad=False)``).
5. Structured ``[timeslice]`` and ``[gpu-mem]`` telemetry emission and parsing helpers.
"""

from __future__ import annotations

# pylint: disable=too-many-lines,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-instance-attributes,too-many-branches,too-many-statements,broad-exception-caught,import-outside-toplevel,global-statement
import asyncio
import atexit
import contextlib
import gc
import importlib
import inspect
import logging
import os
import queue
import random
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class _FallbackGrpcModule:  # pylint: disable=too-few-public-methods
    """Fallback namespace when grpcio is not installed in the host lint environment."""

    @staticmethod
    def insecure_channel(target: str, options: Any = None) -> Any:
        """Raise RuntimeError if invoked without grpcio or mock injection."""
        del options
        raise RuntimeError(f"grpcio not installed; cannot open channel to {target}")


try:
    import grpc
except ImportError:
    grpc = _FallbackGrpcModule()  # type: ignore[assignment]

logger = logging.getLogger(__name__)

GB = float(1024**3)

SUSPEND_MODE_UNSPECIFIED = 0
SUSPEND_MODE_OFFLOAD = 1
SUSPEND_MODE_DISCARD = 2

ENV_APP_OFFLOAD = "TIMESLICE_APP_OFFLOAD"
ENV_AGENT_ADDR = "TIMESLICE_AGENT_ADDR"
DEFAULT_READY_TIMEOUT_SEC = 60.0
DEFAULT_CALL_TIMEOUT_SEC = 110.0
DEFAULT_NAME_PREFIX = "timeslice-offload"

SNAPSHOT_KWARGS: dict[str, Any] = {
    "device": "cpu",
    "model": True,
    "optimizer": True,
    "grad": True,
}
RESTORE_KWARGS: dict[str, Any] = {
    "device": "device",
    "model": True,
    "optimizer": True,
    "grad": False,
}

_MODE_NAMES: dict[str, int] = {
    "offload": SUSPEND_MODE_OFFLOAD,
    "discard": SUSPEND_MODE_DISCARD,
}

_CLOSE = object()
_INITIAL_BACKOFF_SEC = 0.5
_MAX_BACKOFF_SEC = 30.0
_STABLE_CONNECTION_SEC = 5.0
_CONNECT_TIMEOUT_SEC = 10.0


def _encode_varint(value: int) -> bytes:
    """Encode non-negative integer as protobuf base-128 varint."""
    out = bytearray()
    val = int(value)
    while val > 0x7F:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val & 0x7F)
    return bytes(out)


def _encode_varint_field(field_num: int, value: int) -> bytes:
    """Encode a protobuf varint field (wire_type == 0)."""
    if not value:
        return b""
    return _encode_varint((field_num << 3) | 0) + _encode_varint(value)


def _encode_length_delimited(field_num: int, payload: bytes) -> bytes:
    """Encode a protobuf length-delimited field (wire_type == 2)."""
    tag = _encode_varint((field_num << 3) | 2)
    return tag + _encode_varint(len(payload)) + payload


def _encode_string_field(field_num: int, value: str) -> bytes:
    """Encode a non-empty UTF-8 string field (wire_type == 2)."""
    if not value:
        return b""
    return _encode_length_delimited(field_num, value.encode("utf-8"))


def _encode_job_group_request(job_id: str, group_id: str) -> bytes:
    """Serialize AcquireRequest / YieldRequest (field 1: job_id, field 2: group_id)."""
    return _encode_string_field(1, job_id) + _encode_string_field(2, group_id)


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


def _decode_proto_fields(payload: bytes) -> tuple[dict[int, list[int]], dict[int, list[bytes]]]:
    """Decode both varint (wire_type 0) and length-delimited (wire_type 2) fields."""
    varints: dict[int, list[int]] = {}
    blobs: dict[int, list[bytes]] = {}
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
            varints.setdefault(field_num, []).append(val)
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
            blobs.setdefault(field_num, []).append(payload[idx : idx + length])
            idx += length
        else:
            break
    return varints, blobs


def _encode_workload_register_message(
    job_id: str,
    group: str = "",
    supported_modes: Sequence[int] = (SUSPEND_MODE_OFFLOAD,),
    default_mode: int = SUSPEND_MODE_OFFLOAD,
    tags: Sequence[str] = (),
) -> bytes:
    """Serialize WorkloadMessage(register=RegisterWorkload(...)) for WorkloadChannel."""
    caps_buf = bytearray()
    for mode in supported_modes:
        if int(mode) != 0:
            caps_buf.extend(_encode_varint_field(1, int(mode)))
    if int(default_mode) != 0:
        caps_buf.extend(_encode_varint_field(2, int(default_mode)))
    for tag in tags:
        caps_buf.extend(_encode_string_field(3, str(tag)))

    reg_buf = bytearray()
    reg_buf.extend(_encode_string_field(1, job_id))
    reg_buf.extend(_encode_string_field(2, group))
    if caps_buf:
        reg_buf.extend(_encode_length_delimited(3, bytes(caps_buf)))
    return _encode_length_delimited(1, bytes(reg_buf))


def _encode_workload_result_message(command_id: str, ok: bool, error: str = "") -> bytes:
    """Serialize WorkloadMessage(result=CommandResult(...)) for WorkloadChannel."""
    res_buf = bytearray()
    res_buf.extend(_encode_string_field(1, command_id))
    if ok:
        res_buf.extend(_encode_varint_field(2, 1))
    if error:
        res_buf.extend(_encode_string_field(3, error))
    return _encode_length_delimited(2, bytes(res_buf))


def _decode_agent_command(payload: bytes) -> dict[str, Any]:
    """Deserialize an AgentCommand protobuf payload from WorkloadChannel."""
    _, top_blobs = _decode_proto_fields(payload)
    cmd_id = top_blobs.get(1, [b""])[0].decode("utf-8", errors="replace")
    if 2 in top_blobs:
        snap_varints, snap_blobs = _decode_proto_fields(top_blobs[2][-1])
        mode = snap_varints.get(1, [SUSPEND_MODE_OFFLOAD])[-1]
        tags = [b.decode("utf-8", errors="replace") for b in snap_blobs.get(2, [])]
        return {"command_id": cmd_id, "kind": "snapshot", "mode": mode, "tags": tags}
    if 3 in top_blobs:
        _, rest_blobs = _decode_proto_fields(top_blobs[3][-1])
        tags = [b.decode("utf-8", errors="replace") for b in rest_blobs.get(1, [])]
        return {"command_id": cmd_id, "kind": "restore", "mode": 0, "tags": tags}
    return {"command_id": cmd_id, "kind": "unknown", "mode": 0, "tags": []}


def mode_from_name(name: str | int) -> int:
    """Map a suspend mode name ('offload'/'discard') or int to SuspendMode enum value."""
    if isinstance(name, int):
        return name
    try:
        return _MODE_NAMES[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown suspend mode {name!r}; expected one of {sorted(_MODE_NAMES)}"
        ) from None


def mode_name(mode: int | str) -> str:
    """Map a SuspendMode enum value to its public string name."""
    if isinstance(mode, str):
        return mode.strip().lower()
    for name, value in _MODE_NAMES.items():
        if value == mode:
            return name
    return str(mode)

COMMAND_OP_UNSPECIFIED = 0
COMMAND_OP_SNAPSHOT = 1
COMMAND_OP_RESTORE = 2


@dataclass
class RegisterWorkload:
    """Protobuf-compatible RegisterWorkload message for SnapshotAgentService/WorkloadChannel."""

    job_id: str = ""
    group: str = ""
    pid: int = 0
    supported_modes: Sequence[str | int] = ("offload",)
    default_mode: str | int = "offload"
    tags: Sequence[str] = ()

    def SerializeToString(self) -> bytes:  # pylint: disable=invalid-name
        """Serialize RegisterWorkload to protobuf wire bytes."""
        modes_int = [mode_from_name(m) for m in self.supported_modes]
        def_int = mode_from_name(self.default_mode)
        raw_msg = _encode_workload_register_message(
            job_id=self.job_id,
            group=self.group,
            supported_modes=modes_int,
            default_mode=def_int,
            tags=self.tags,
        )
        _, top_blobs = _decode_proto_fields(raw_msg)
        return top_blobs.get(1, [b""])[0]

    @classmethod
    def FromString(cls, payload: bytes) -> RegisterWorkload:  # pylint: disable=invalid-name
        """Deserialize RegisterWorkload from protobuf wire bytes."""
        _, blobs = _decode_proto_fields(payload)
        job_id = blobs.get(1, [b""])[0].decode("utf-8", errors="replace")
        group = blobs.get(2, [b""])[0].decode("utf-8", errors="replace")
        modes: list[str] = ["offload"]
        def_mode = "offload"
        tags: list[str] = []
        if 3 in blobs:
            c_vars, c_blobs = _decode_proto_fields(blobs[3][-1])
            if 1 in c_vars:
                modes = [mode_name(m) for m in c_vars[1]]
            if 2 in c_vars:
                def_mode = mode_name(c_vars[2][-1])
            if 3 in c_blobs:
                tags = [b.decode("utf-8", errors="replace") for b in c_blobs[3]]
        return cls(
            job_id=job_id,
            group=group,
            supported_modes=modes,
            default_mode=def_mode,
            tags=tags,
        )


@dataclass
class CommandResult:
    """Protobuf-compatible CommandResult message for SnapshotAgentService/WorkloadChannel."""

    command_id: str = ""
    ok: bool = True
    success: bool | None = None
    error: str = ""
    error_message: str | None = None
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if self.success is not None:
            self.ok = bool(self.success)
        else:
            self.success = bool(self.ok)
        if self.error_message is not None and not self.error:
            self.error = str(self.error_message)
        else:
            self.error_message = str(self.error)

    def SerializeToString(self) -> bytes:  # pylint: disable=invalid-name
        """Serialize CommandResult to protobuf wire bytes."""
        raw_msg = _encode_workload_result_message(
            command_id=self.command_id, ok=bool(self.ok), error=self.error
        )
        _, top_blobs = _decode_proto_fields(raw_msg)
        return top_blobs.get(2, [b""])[0]

    @classmethod
    def FromString(cls, payload: bytes) -> CommandResult:  # pylint: disable=invalid-name
        """Deserialize CommandResult from protobuf wire bytes."""
        varints, blobs = _decode_proto_fields(payload)
        cmd_id = blobs.get(1, [b""])[0].decode("utf-8", errors="replace")
        ok = bool(varints.get(2, [0])[-1])
        err = blobs.get(3, [b""])[0].decode("utf-8", errors="replace") if 3 in blobs else ""
        return cls(command_id=cmd_id, ok=ok, success=ok, error=err, error_message=err)


@dataclass
class WorkloadMessage:
    """Protobuf-compatible WorkloadMessage wrapping RegisterWorkload or CommandResult."""

    register: RegisterWorkload | None = None
    result: CommandResult | None = None

    def SerializeToString(self) -> bytes:  # pylint: disable=invalid-name
        """Serialize WorkloadMessage to protobuf wire bytes."""
        if self.register is not None:
            modes_int = [mode_from_name(m) for m in self.register.supported_modes]
            def_int = mode_from_name(self.register.default_mode)
            return _encode_workload_register_message(
                job_id=self.register.job_id,
                group=self.register.group,
                supported_modes=modes_int,
                default_mode=def_int,
                tags=self.register.tags,
            )
        if self.result is not None:
            return _encode_workload_result_message(
                command_id=self.result.command_id,
                ok=bool(self.result.ok),
                error=self.result.error,
            )
        return b""

    @classmethod
    def FromString(cls, payload: bytes) -> WorkloadMessage:  # pylint: disable=invalid-name
        """Deserialize WorkloadMessage from protobuf wire bytes."""
        _, blobs = _decode_proto_fields(payload)
        if 1 in blobs:
            return cls(register=RegisterWorkload.FromString(blobs[1][-1]))
        if 2 in blobs:
            return cls(result=CommandResult.FromString(blobs[2][-1]))
        return cls()


@dataclass
class AgentCommand:
    """Protobuf-compatible AgentCommand sent by snapshot-agent over WorkloadChannel."""

    command_id: str = ""
    op: int = COMMAND_OP_UNSPECIFIED
    mode: str = "offload"
    tags: Sequence[str] = ()

    @property
    def kind(self) -> str:
        """Return 'snapshot', 'restore', or 'unknown'."""
        if self.op == COMMAND_OP_SNAPSHOT:
            return "snapshot"
        if self.op == COMMAND_OP_RESTORE:
            return "restore"
        return "unknown"

    def SerializeToString(self) -> bytes:  # pylint: disable=invalid-name
        """Serialize AgentCommand to protobuf wire bytes."""
        buf = bytearray()
        buf.extend(_encode_string_field(1, self.command_id))
        if self.op == COMMAND_OP_SNAPSHOT:
            snap_buf = bytearray()
            snap_buf.extend(_encode_varint_field(1, mode_from_name(self.mode)))
            for tag in self.tags:
                snap_buf.extend(_encode_string_field(2, str(tag)))
            buf.extend(_encode_length_delimited(2, bytes(snap_buf)))
        elif self.op == COMMAND_OP_RESTORE:
            rest_buf = bytearray()
            for tag in self.tags:
                rest_buf.extend(_encode_string_field(1, str(tag)))
            buf.extend(_encode_length_delimited(3, bytes(rest_buf)))
        return bytes(buf)

    @classmethod
    def FromString(cls, payload: bytes) -> AgentCommand:  # pylint: disable=invalid-name
        """Deserialize AgentCommand from protobuf wire bytes."""
        decoded = _decode_agent_command(payload)
        kind = decoded["kind"]
        op = (
            COMMAND_OP_SNAPSHOT
            if kind == "snapshot"
            else (COMMAND_OP_RESTORE if kind == "restore" else COMMAND_OP_UNSPECIFIED)
        )
        return cls(
            command_id=str(decoded["command_id"]),
            op=op,
            mode=mode_name(int(decoded["mode"])) if kind == "snapshot" else "offload",
            tags=tuple(decoded["tags"]),
        )


class SnapshotAgentServiceStub:  # pylint: disable=too-few-public-methods
    """gRPC client stub for snapshot_agent.v1alpha1.SnapshotAgentService."""

    SERVICE_PATH = "/snapshot_agent.v1alpha1.SnapshotAgentService/WorkloadChannel"

    def __init__(self, channel: Any) -> None:
        self._channel = channel

    def WorkloadChannel(  # pylint: disable=invalid-name
        self, outbound_iter: Iterator[WorkloadMessage | bytes]
    ) -> Iterator[AgentCommand]:
        """Open bidirectional WorkloadChannel stream on the underlying gRPC channel."""
        stream_rpc = self._channel.stream_stream(
            self.SERVICE_PATH,
            request_serializer=lambda m: (
                m.SerializeToString() if hasattr(m, "SerializeToString") else bytes(m)
            ),
            response_deserializer=lambda b: (
                AgentCommand.FromString(b) if isinstance(b, (bytes, bytearray)) else b
            ),
        )
        return stream_rpc(outbound_iter)


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


class WorkloadAdapter:
    """Base adapter translating SnapshotAgentService WorkloadChannel commands."""

    name = "base"
    supported_modes: Sequence[int] = (SUSPEND_MODE_OFFLOAD,)
    default_mode: int = SUSPEND_MODE_OFFLOAD
    tags: Sequence[str] = ()

    def snapshot(self, mode: int, tags: list[str]) -> Any:
        """Suspend/offload the workload state."""
        raise NotImplementedError

    def restore(self, tags: list[str]) -> Any:
        """Resume/restore the workload state."""
        raise NotImplementedError


class CallbackAdapter(WorkloadAdapter):
    """Adapter backed by explicit on_snapshot / on_restore callables."""

    name = "callbacks"

    def __init__(self, on_snapshot: Callable[..., Any], on_restore: Callable[..., Any]) -> None:
        self._on_snapshot = on_snapshot
        self._on_restore = on_restore

    def snapshot(self, mode: int, tags: list[str]) -> Any:
        return self._on_snapshot(mode_name(mode), tags)

    def restore(self, tags: list[str]) -> Any:
        return self._on_restore(tags)


class SnapshottableAdapter(WorkloadAdapter):
    """Adapter wrapping any object implementing snapshot(mode, tags) and restore(tags)."""

    name = "snapshottable"

    def __init__(self, obj: Any) -> None:
        self._obj = obj
        declared = getattr(obj, "supported_modes", None)
        if declared:
            self.supported_modes = [mode_from_name(m) for m in declared]
        default = getattr(obj, "default_mode", None)
        if default:
            self.default_mode = mode_from_name(default)

    def snapshot(self, mode: int, tags: list[str]) -> Any:
        return self._obj.snapshot(mode_name(mode), tags)

    def restore(self, tags: list[str]) -> Any:
        return self._obj.restore(tags)


def resolve_adapter(
    workload: Any | None,
    on_snapshot: Callable[..., Any] | None,
    on_restore: Callable[..., Any] | None,
) -> WorkloadAdapter:
    """Resolve a WorkloadAdapter from a workload object or callback pair."""
    if on_snapshot is not None or on_restore is not None:
        if on_snapshot is None or on_restore is None:
            raise ValueError("on_snapshot and on_restore must be provided together")
        if workload is not None:
            raise ValueError("pass either workload= or callbacks, not both")
        return CallbackAdapter(on_snapshot, on_restore)
    if workload is None:
        raise ValueError("a workload object or on_snapshot/on_restore callbacks are required")
    if callable(getattr(workload, "snapshot", None)) and callable(
        getattr(workload, "restore", None)
    ):
        return SnapshottableAdapter(workload)
    raise TypeError(
        f"don't know how to suspend {type(workload).__name__}: "
        "not Snapshottable (snapshot/restore methods); pass on_snapshot=/on_restore= callbacks"
    )


async def _wrap_awaitable(awaitable: Any) -> Any:
    return await awaitable


class WorkloadHandle:
    """Handle for a registered workload over SnapshotAgentService/WorkloadChannel."""

    SERVICE_PATH = "/snapshot_agent.v1alpha1.SnapshotAgentService/WorkloadChannel"

    def __init__(
        self,
        agent: str = "",
        job_id: str = "",
        group: str = "",
        register_payload: bytes | None = None,
        adapter: WorkloadAdapter | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        *,
        agent_addr: str | None = None,
        on_offload: Callable[..., Any] | None = None,
        on_reload: Callable[..., Any] | None = None,
        supported_modes: Sequence[str] | None = None,
        default_mode: str = "offload",
        tags: Sequence[str] = (),
        auto_start: bool | None = None,
    ) -> None:
        resolved_agent = agent_addr or agent or os.environ.get(ENV_AGENT_ADDR, "127.0.0.1:9001")
        if adapter is None:
            if on_offload is not None or on_reload is not None:
                off_fn = on_offload or (lambda _m: None)
                rel_fn = on_reload or (lambda _m: None)
                adapter = CallbackAdapter(
                    on_snapshot=lambda m, _tags=None: off_fn(m),
                    on_restore=lambda _tags=None: rel_fn(default_mode or "offload"),
                )
            else:
                raise ValueError("WorkloadHandle requires adapter or on_offload/on_reload")
        modes_list = list(supported_modes) if supported_modes else ["offload"]
        self._register_msg = WorkloadMessage(
            register=RegisterWorkload(
                job_id=job_id,
                group=group,
                pid=os.getpid(),
                supported_modes=modes_list,
                default_mode=default_mode,
                tags=tuple(tags),
            )
        )
        self._agent = resolved_agent
        self._job_id = job_id
        self._group = group
        self._register_payload = (
            register_payload
            if register_payload is not None
            else self._register_msg.SerializeToString()
        )
        self._adapter = adapter
        self._loop = loop
        self._stub_factory: Callable[[Any], Any] | None = None
        self._auto_reconnect = register_payload is not None
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._started = False
        self._call: Any = None
        self._call_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=f"workload-channel-{job_id}",
            daemon=True,
        )
        should_start = auto_start if auto_start is not None else (register_payload is not None)
        if should_start:
            self.start()

    def start(self) -> None:
        """Start the background WorkloadChannel stream thread if not already started."""
        if not self._started:
            self._started = True
            self._thread.start()

    @property
    def connected(self) -> bool:
        """True while a WorkloadChannel gRPC stream to the agent is open."""
        return self._connected.is_set()

    def wait_connected(self, timeout: float | None = None) -> bool:
        """Block until the workload channel is open, or timeout expires."""
        return self._connected.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Stop servicing commands and close the WorkloadChannel stream."""
        self._stop.set()
        with self._call_lock:
            if self._call is not None:
                with contextlib.suppress(Exception):
                    self._call.cancel()
        if self._started:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        grpc_mod = grpc if grpc is not None else importlib.import_module("grpc")
        backoff = _INITIAL_BACKOFF_SEC
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._connect_and_serve(grpc_mod)
                if not self._auto_reconnect:
                    return
            except (RuntimeError, OSError, ValueError, TimeoutError, LookupError, TypeError) as err:
                if not self._stop.is_set():
                    logger.warning(
                        "workload channel for job %s (agent %s) disconnected: %s",
                        self._job_id,
                        self._agent,
                        err,
                    )
                if not self._auto_reconnect:
                    return
            if self._stop.is_set():
                return
            if time.monotonic() - started > _STABLE_CONNECTION_SEC:
                backoff = _INITIAL_BACKOFF_SEC
            sleep_for = backoff * (0.5 + random.random())
            backoff = min(backoff * 2, _MAX_BACKOFF_SEC)
            self._stop.wait(sleep_for)

    def _connect_and_serve(self, grpc_mod: Any) -> None:
        sendq: queue.SimpleQueue[Any] = queue.SimpleQueue()

        def requests() -> Iterator[WorkloadMessage]:
            yield self._register_msg
            while True:
                item = sendq.get()
                if item is _CLOSE:
                    return
                yield item

        raw_ch = grpc_mod.insecure_channel(self._agent)
        ctx_mgr = raw_ch if hasattr(raw_ch, "__enter__") else contextlib.nullcontext(raw_ch)
        with ctx_mgr as channel:
            if hasattr(channel, "stream_stream") and hasattr(grpc_mod, "channel_ready_future"):
                with contextlib.suppress(Exception):
                    grpc_mod.channel_ready_future(channel).result(timeout=_CONNECT_TIMEOUT_SEC)
            stub_fn = self._stub_factory or SnapshotAgentServiceStub
            stub = stub_fn(channel)
            call = stub.WorkloadChannel(requests())
            with self._call_lock:
                self._call = call
            self._connected.set()
            print(
                f"[timeslice] registering workload channel job={self._job_id} "
                f"group={self._group} agent={self._agent}",
                flush=True,
            )
            try:
                for raw_cmd in call:
                    result_msg = self._execute(raw_cmd)
                    sendq.put(result_msg)
            finally:
                self._connected.clear()
                with self._call_lock:
                    self._call = None
                sendq.put(_CLOSE)

    def _execute(self, raw_cmd: AgentCommand | bytes) -> WorkloadMessage:
        cmd_obj = (
            raw_cmd
            if isinstance(raw_cmd, AgentCommand)
            else AgentCommand.FromString(bytes(raw_cmd))
        )
        kind = cmd_obj.kind
        cmd_id = cmd_obj.command_id
        t0 = time.monotonic()
        try:
            if kind == "snapshot":
                outcome = self._adapter.snapshot(
                    mode_from_name(cmd_obj.mode), list(cmd_obj.tags)
                )
            elif kind == "restore":
                outcome = self._adapter.restore(list(cmd_obj.tags))
            else:
                raise ValueError(f"unknown command type {kind!r}")
            self._await_if_needed(outcome)
            dur_ms = int((time.monotonic() - t0) * 1000)
            return WorkloadMessage(
                result=CommandResult(
                    command_id=cmd_id,
                    ok=True,
                    success=True,
                    duration_ms=dur_ms,
                )
            )
        except Exception as err:
            logger.exception("%s command %s failed", kind, cmd_id)
            err_str = f"{type(err).__name__}: {err}"
            return WorkloadMessage(
                result=CommandResult(
                    command_id=cmd_id,
                    ok=False,
                    success=False,
                    error=err_str,
                    error_message=err_str,
                )
            )

    def _await_if_needed(self, outcome: Any) -> None:
        if not inspect.isawaitable(outcome):
            return
        coro = outcome if inspect.iscoroutine(outcome) else _wrap_awaitable(outcome)
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        else:
            asyncio.run(coro)


def register_workload(
    agent: str,
    job_id: str,
    group: str = "",
    workload: Any | None = None,
    on_snapshot: Callable[..., Any] | None = None,
    on_restore: Callable[..., Any] | None = None,
    supported_modes: list[str] | None = None,
    default_mode: str | None = None,
    tags: list[str] | None = None,
) -> WorkloadHandle:
    """Register this process's workload with the node's snapshot-agent via WorkloadChannel."""
    if not job_id:
        raise ValueError("job_id is required")
    adapter = resolve_adapter(workload, on_snapshot, on_restore)
    modes = (
        [mode_from_name(m) for m in supported_modes]
        if supported_modes
        else list(adapter.supported_modes)
    )
    default = mode_from_name(default_mode) if default_mode else int(adapter.default_mode)
    resolved_tags = list(tags) if tags is not None else list(adapter.tags)
    register_payload = _encode_workload_register_message(
        job_id=job_id,
        group=group,
        supported_modes=modes,
        default_mode=default,
        tags=resolved_tags,
    )
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    return WorkloadHandle(
        agent=agent,
        job_id=job_id,
        group=group,
        register_payload=register_payload,
        adapter=adapter,
        loop=loop,
    )


def _default_log(msg: str) -> None:
    print(f"[timeslice] {msg}", flush=True)


@dataclass(frozen=True)
class OffloadSpec:
    """Specification of the rank actor RPC used to offload/restore state."""

    method: str
    snapshot_kwargs: Mapping[str, Any] = field(default_factory=lambda: dict(SNAPSHOT_KWARGS))
    restore_kwargs: Mapping[str, Any] = field(default_factory=lambda: dict(RESTORE_KWARGS))
    restore_method: str | None = None
    call_timeout: float = DEFAULT_CALL_TIMEOUT_SEC
    call_timeout_sec: float = DEFAULT_CALL_TIMEOUT_SEC
    supported_modes: tuple[str, ...] = ("offload",)
    default_mode: str = "offload"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("OffloadSpec.method is required")
        for mode in self.supported_modes:
            mode_from_name(mode)
        mode_from_name(self.default_mode)


class RankFanout:
    """Snapshottable implementation that fans out actor RPCs to node-local rank handles."""

    def __init__(
        self,
        handles: Sequence[Any],
        spec: OffloadSpec,
        *,
        job_id: str = "",
        node: str = "",
        node_id: str = "",
        log: Callable[[str], None] | None = None,
    ) -> None:
        if not handles:
            raise ValueError("RankFanout requires at least one rank handle")
        self._handles = list(handles)
        self._spec = spec
        self._job_id = job_id
        self._node = node or (node_id[:8] if node_id else "")
        self._log = log or _default_log
        self.supported_modes = list(spec.supported_modes)
        self.default_mode = spec.default_mode

    @property
    def ranks(self) -> int:
        """Number of local rank handles managed by this fanout."""
        return len(self._handles)

    def on_offload(self, mode: str = "offload") -> None:
        """Alias for snapshot(mode, []) used by direct callback consumers."""
        self.snapshot(mode, [])

    def on_reload(self, mode: str = "offload") -> None:
        """Alias for restore([]) used by direct callback consumers."""
        del mode
        self.restore([])

    def snapshot(self, mode: str, tags: list[str] | None = None) -> None:
        """Fan out snapshot RPC across all local rank actors."""
        del tags
        if mode != "offload":
            raise ValueError(
                f"RankFanout only supports mode='offload' (got {mode!r}); "
                "trainer state cannot be reconstructed after discard"
            )
        self._fanout(self._spec.method, self._spec.snapshot_kwargs, "OFFLOAD")

    def restore(self, tags: list[str] | None = None) -> None:
        """Fan out restore RPC across all local rank actors."""
        del tags
        self._fanout(
            self._spec.restore_method or self._spec.method,
            self._spec.restore_kwargs,
            "RELOAD",
        )

    def _fanout(self, method: str, kwargs: Mapping[str, Any], verb: str) -> None:
        ray_mod = importlib.import_module("ray")
        prefix = (
            f"job={self._job_id} {verb} node={self._node} "
            f"ranks={len(self._handles)} method={method}"
        )
        started = time.monotonic()
        try:
            refs = [getattr(h, method).remote(**dict(kwargs)) for h in self._handles]
            ray_mod.get(refs, timeout=self._spec.call_timeout_sec)
        except Exception as err:
            self._log(
                f"{prefix} FAILED after {time.monotonic() - started:.1f}s "
                f"err={type(err).__name__}: {err}"
            )
            raise
        self._log(f"{prefix} took={time.monotonic() - started:.1f}s")


class OffloadProxy:
    """Per-node zero-CPU Ray actor holding the WorkloadChannel and relaying to local ranks."""

    def __init__(
        self,
        job_id: str,
        group: str,
        handles: Sequence[Any],
        spec: OffloadSpec,
        agent: str | None = None,
        node: str = "",
        log: Callable[[str], None] | None = None,
    ) -> None:
        resolved_agent = agent or os.environ.get(ENV_AGENT_ADDR, "")
        if not resolved_agent:
            with contextlib.suppress(Exception):
                ray_mod = importlib.import_module("ray")
                node_ip = ray_mod.util.get_node_ip_address()
                if node_ip:
                    resolved_agent = f"{node_ip}:9001"
        if not resolved_agent:
            raise ValueError(
                f"no snapshot-agent address: pass agent= or set {ENV_AGENT_ADDR} on the node's pods"
            )
        self._job_id = job_id
        self._agent = resolved_agent
        self._node = node
        self._spec = spec
        self._fanout = RankFanout(handles, spec, job_id=job_id, node=node, log=log)
        self._handle = register_workload(
            resolved_agent,
            job_id=job_id,
            group=group,
            workload=self._fanout,
            supported_modes=list(spec.supported_modes),
            default_mode=spec.default_mode,
            tags=list(spec.tags),
        )

    def ready(self, timeout: float = DEFAULT_READY_TIMEOUT_SEC) -> bool:
        """Block until the WorkloadChannel to snapshot-agent is open."""
        return self._handle.wait_connected(timeout)

    def info(self) -> dict[str, Any]:
        """Return proxy registration diagnostics."""
        return {
            "job_id": self._job_id,
            "node": self._node,
            "agent": self._agent,
            "ranks": self._fanout.ranks,
            "method": self._spec.method,
            "connected": self._handle.connected,
        }

    def close(self, timeout: float = 5.0) -> None:
        """Close the WorkloadChannel stream."""
        self._handle.close(timeout=timeout)


_PROXY_ACTOR_CLS: Any = None


def _proxy_cls() -> Any:
    """Return OffloadProxy wrapped as a zero-CPU Ray actor class."""
    global _PROXY_ACTOR_CLS
    if _PROXY_ACTOR_CLS is None:
        ray_mod = importlib.import_module("ray")
        _PROXY_ACTOR_CLS = ray_mod.remote(num_cpus=0)(OffloadProxy)
    return _PROXY_ACTOR_CLS


def _node_id(actor_self: Any) -> str:
    """Execute inside a rank actor via __ray_call__ to retrieve its Ray NodeID."""
    del actor_self
    ray_mod = importlib.import_module("ray")
    return str(ray_mod.get_runtime_context().get_node_id())


def group_handles_by_node(
    handles: Sequence[Any], *, timeout: float = 60.0
) -> dict[str, list[Any]]:
    """Map Ray NodeID -> rank actor handles living on that node."""
    ray_mod = importlib.import_module("ray")
    node_ids = ray_mod.get(
        [h.__ray_call__.remote(_node_id) for h in handles], timeout=timeout
    )
    groups: dict[str, list[Any]] = {}
    for handle, node_id in zip(handles, node_ids):
        groups.setdefault(str(node_id), []).append(handle)
    return groups


@dataclass
class OffloadProxySet:
    """Collection of per-node OffloadProxy actors spawned for one job."""

    job_id: str
    actors: list[Any]
    nodes: list[str]
    log: Callable[[str], None] = _default_log

    def info(self, timeout: float = 10.0) -> list[dict[str, Any]]:
        """Query info() from every proxy actor."""
        ray_mod = importlib.import_module("ray")
        return list(ray_mod.get([a.info.remote() for a in self.actors], timeout=timeout))

    def close(self, timeout: float = 5.0) -> None:
        """Deregister every proxy and terminate the proxy actors."""
        ray_mod = importlib.import_module("ray")
        with contextlib.suppress(Exception):
            ray_mod.get([a.close.remote(timeout) for a in self.actors], timeout=timeout + 5)
        for actor in self.actors:
            with contextlib.suppress(Exception):
                ray_mod.kill(actor, no_restart=True)


def spawn_offload_proxies(
    handles: Sequence[Any],
    spec: OffloadSpec,
    *,
    job_id: str,
    group: str = "trainers",
    agent: str | None = None,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SEC,
    name_prefix: str = DEFAULT_NAME_PREFIX,
    log: Callable[[str], None] | None = None,
) -> OffloadProxySet:
    """Start one pinned OffloadProxy actor per node hosting handles and wait until ready."""
    if not job_id:
        raise ValueError("job_id is required")
    if not handles:
        raise ValueError("no handles to proxy")
    ray_mod = importlib.import_module("ray")
    sched_mod = importlib.import_module("ray.util.scheduling_strategies")
    node_affinity_cls = sched_mod.NodeAffinitySchedulingStrategy
    log_fn = log or _default_log

    actors: list[Any] = []
    nodes: list[str] = []
    for node_id, local in group_handles_by_node(handles).items():
        short = node_id[:8]
        actor = (
            _proxy_cls()
            .options(
                name=f"{name_prefix}-{job_id}-{short}",
                get_if_exists=True,
                num_cpus=0,
                scheduling_strategy=node_affinity_cls(node_id=node_id, soft=False),
            )
            .remote(
                job_id=job_id,
                group=group,
                handles=local,
                spec=spec,
                agent=agent,
                node=short,
                log=log_fn,
            )
        )
        actors.append(actor)
        nodes.append(node_id)
    proxies = OffloadProxySet(job_id=job_id, actors=actors, nodes=nodes, log=log_fn)

    try:
        ready = ray_mod.get(
            [a.ready.remote(ready_timeout) for a in actors], timeout=ready_timeout + 15
        )
    except Exception as err:
        proxies.close()
        raise RuntimeError(
            f"offload proxies for job {job_id} failed to start: {type(err).__name__}: {err}"
        ) from err
    not_ready = [node[:8] for node, ok in zip(nodes, ready) if not ok]
    if not_ready:
        proxies.close()
        raise TimeoutError(
            f"offload proxies for job {job_id} on nodes {not_ready} did not connect to "
            f"snapshot-agent within {ready_timeout:.0f}s (agent={agent or '$' + ENV_AGENT_ADDR})"
        )
    return proxies


def offload_method_for_role(train_role: Any = "actor_rollout") -> str:
    """Name of the manual offload RPC on a raw rank handle for a verl role."""
    role = str(train_role).strip()
    if not role:
        raise ValueError("train_role is required")
    return f"{role}_to"


def offload_spec_for_role(train_role: Any = "actor_rollout") -> OffloadSpec:
    """Return the OffloadSpec for a verl trainer worker group of the given role."""
    return OffloadSpec(
        method=offload_method_for_role(train_role),
        snapshot_kwargs=dict(SNAPSHOT_KWARGS),
        restore_kwargs=dict(RESTORE_KWARGS),
    )


def install_verl_offload(
    handles: Sequence[Any],
    *,
    job_id: str | None = None,
    group: str | None = None,
    train_role: Any = "actor_rollout",
    agent: str | None = None,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SEC,
) -> OffloadProxySet:
    """Spawn and register per-node OffloadProxy actors for a verl trainer worker group."""
    resolved_job = (
        job_id
        or os.environ.get("TIMESLICE_JOB_ID")
        or os.environ.get("JOB_ID")
        or "job1"
    )
    resolved_group = group or os.environ.get("TIMESLICE_TRAINER_GROUP", "trainers")
    spec = offload_spec_for_role(train_role)
    started = time.monotonic()
    proxies = spawn_offload_proxies(
        handles,
        spec,
        job_id=resolved_job,
        group=resolved_group,
        agent=agent,
        ready_timeout=ready_timeout,
        log=_default_log,
    )
    _default_log(
        f"job={resolved_job} OFFLOAD-PROXIES ready nodes={len(proxies.nodes)} "
        f"ranks={len(handles)} method={spec.method} took={time.monotonic() - started:.1f}s"
    )
    return proxies


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
    sampler_enabled: bool = False
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
            os.environ.get("TIMESLICE_SAMPLER_ENABLED"), default=False
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
    point: str = "app_channel",
    model: bool = True,
    optimizer: bool = True,
    grad: bool = True,
    mode: str = "app",
) -> dict[str, Any]:
    """Offload multi-GPU Megatron trainer state to host CPU and empty CUDA cache."""
    if world_size < 2:
        raise ValueError(f"Multi-GPU trainer requires world_size >= 2, got {world_size}")

    snap_before = capture_gpu_memory_snapshot()
    emit_gpu_mem_log(
        "pre_offload", "trainer", rank=rank, world_size=world_size, snapshot=snap_before
    )

    t0 = time.perf_counter()
    try:
        engine.to("cpu", model=model, optimizer=optimizer, grad=grad, point=point)
    except TypeError:
        engine.to("cpu", model=model, optimizer=optimizer, grad=grad)
    aggressive_empty_cache(force_sync=True)

    duration_ms = (time.perf_counter() - t0) * 1000.0
    snap_after = capture_gpu_memory_snapshot()
    emit_gpu_mem_log(
        "post_offload", "trainer", rank=rank, world_size=world_size, snapshot=snap_after
    )

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
    grad: bool = False,
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
    try:
        engine.to("cuda", model=model, optimizer=optimizer, grad=grad, point="app_channel")
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


snapshot_pb2 = sys.modules[__name__]
snapshot_pb2_grpc = sys.modules[__name__]

__all__ = [
    "COMMAND_OP_RESTORE",
    "COMMAND_OP_SNAPSHOT",
    "COMMAND_OP_UNSPECIFIED",
    "RESTORE_KWARGS",
    "SNAPSHOT_KWARGS",
    "SUSPEND_MODE_DISCARD",
    "SUSPEND_MODE_OFFLOAD",
    "SUSPEND_MODE_UNSPECIFIED",
    "AcquireResult",
    "AgentCommand",
    "CallbackAdapter",
    "CommandResult",
    "DualPoolRoleLocks",
    "OffloadProxy",
    "OffloadProxySet",
    "OffloadSpec",
    "OrchestratorClient",
    "OrchestratorGroupStatus",
    "RankFanout",
    "RegisterWorkload",
    "RoleLockConfig",
    "RoleLocks",
    "SnapshotAgentServiceStub",
    "SnapshottableAdapter",
    "TimeSliceOrchestratorClient",
    "WorkloadAdapter",
    "WorkloadHandle",
    "WorkloadMessage",
    "YieldResult",
    "aggressive_empty_cache",
    "capture_gpu_memory_snapshot",
    "drain_sampler_inflight_requests",
    "emit_gpu_mem_log",
    "format_gpu_mem_log",
    "format_offload_event",
    "format_restore_event",
    "format_timeslice_log",
    "group_handles_by_node",
    "grpc",
    "install_verl_offload",
    "mode_from_name",
    "mode_name",
    "offload_megatron_trainer",
    "offload_method_for_role",
    "offload_spec_for_role",
    "parse_gpu_mem_log",
    "parse_timeslice_log",
    "register_workload",
    "resolve_adapter",
    "restore_megatron_trainer",
    "sleep_vllm_sampler_async",
    "spawn_offload_proxies",
    "wake_vllm_sampler_async",
]
