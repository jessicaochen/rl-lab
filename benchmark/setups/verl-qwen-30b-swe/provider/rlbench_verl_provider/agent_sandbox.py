"""uni-agent sandbox provider backed by kubernetes-sigs/agent-sandbox.

Creates one ``Sandbox`` CR (agents.x-k8s.io/v1beta1, OSS controller) per
episode on the cluster's gVisor node pool and drives it via the Kubernetes
pod-exec API. Ported from the battle-tested verifiers runtime of the prime-rl
setup: root user (SWE task images assume root-owned /testbed), caps dropped,
no SA token, per-image registry rewriting handled upstream by uni-agent's
``image_map`` config.

The recipe's "tool image mount" (mini-swe-agent sidecar at
/opt/mini-swe-agent) is an openyuanrong feature; here each mount is
translated to an initContainer that copies the tool image's payload into an
emptyDir shared with the task container — same result, plain Kubernetes.

Registered as provider ``agent_sandbox`` (the driver image appends this
module to uni-agent's ``SANDBOX_MODULES`` for lazy loading).
"""

from __future__ import annotations

import asyncio
import datetime
import shlex
import time
import uuid
from typing import Any, ClassVar

from uni_agent.sandbox.base import ExecResult, Sandbox, SandboxConfig
from uni_agent.sandbox.registry import register_sandbox

GROUP = "agents.x-k8s.io"
VERSION = "v1beta1"
PLURAL = "sandboxes"
GVISOR_LABEL = "sandbox.gke.io/runtime"


@register_sandbox("agent_sandbox")
class AgentSandbox(Sandbox):
    supports_shell: ClassVar[bool] = False

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        runtime_timeout: float = 7200.0,
        namespace: str = "default",
        cpu: float = 2.0,
        memory_gb: float = 4.0,
        ready_timeout: float = 1800.0,
        mounts: list[dict[str, str]] | None = None,  # [{target, image_url}]
        **_extra: Any,
    ) -> None:
        self.image = image
        self.runtime_timeout = runtime_timeout
        self.namespace = namespace
        self.cpu = cpu
        self.memory_gb = memory_gb
        self.ready_timeout = ready_timeout
        self.mounts = mounts or []
        self._name: str | None = None
        self._core = None
        self._custom = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> "AgentSandbox":
        return cls(
            image=config.image,
            runtime_timeout=config.runtime_timeout,
            **config.sandbox_kwargs,
        )

    # ----- k8s plumbing ---------------------------------------------------

    def _connect(self) -> None:
        if self._core is not None:
            return
        from kubernetes import client, config as kconfig

        try:
            kconfig.load_incluster_config()
        except Exception:
            kconfig.load_kube_config()
        self._core = client.CoreV1Api()
        self._custom = client.CustomObjectsApi()

    def _manifest(self, name: str) -> dict:
        volumes: list[dict] = []
        init_containers: list[dict] = []
        mounts: list[dict] = []
        for i, m in enumerate(self.mounts):
            vol = f"tool-{i}"
            volumes.append({"name": vol, "emptyDir": {}})
            # the tool image carries its payload at exactly `target`
            init_containers.append({
                "name": f"tool-copy-{i}",
                "image": m["image_url"],
                "command": ["/bin/sh", "-c", f"cp -a {m['target']}/. /__tool_dst/"],
                "volumeMounts": [{"name": vol, "mountPath": "/__tool_dst"}],
                "securityContext": {"runAsUser": 0},
            })
            mounts.append({"name": vol, "mountPath": m["target"]})

        shutdown_at = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=self.runtime_timeout + self.ready_timeout)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": "Sandbox",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "labels": {"rlbench/component": "rollout-sandbox"},
            },
            "spec": {
                # safety net: controller deletes leaked sandboxes
                "shutdownTime": shutdown_at,
                "shutdownPolicy": "Delete",
                "podTemplate": {
                    "spec": {
                        "runtimeClassName": "gvisor",
                        "automountServiceAccountToken": False,
                        "nodeSelector": {GVISOR_LABEL: "gvisor"},
                        "tolerations": [{
                            "key": GVISOR_LABEL, "operator": "Equal",
                            "value": "gvisor", "effect": "NoSchedule",
                        }],
                        "initContainers": init_containers,
                        "containers": [{
                            "name": "sandbox",
                            "image": self.image,
                            "command": ["/bin/sh", "-c", "sleep infinity"],
                            "securityContext": {
                                "capabilities": {"drop": ["ALL"]},
                                "allowPrivilegeEscalation": False,
                                "runAsUser": 0,
                            },
                            "resources": {
                                "requests": {
                                    "cpu": str(self.cpu),
                                    "memory": f"{int(self.memory_gb * 1024)}Mi",
                                },
                                "limits": {
                                    "cpu": str(self.cpu),
                                    "memory": f"{int(self.memory_gb * 1024)}Mi",
                                },
                            },
                            "volumeMounts": mounts,
                        }],
                        "volumes": volumes,
                    }
                },
            },
        }

    # ----- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        if self._name is not None:
            return
        name = f"ua-{uuid.uuid4().hex[:10]}"
        await asyncio.to_thread(self._create_and_wait, name)

    def _create_and_wait(self, name: str) -> None:
        self._connect()
        self._custom.create_namespaced_custom_object(
            GROUP, VERSION, self.namespace, PLURAL, self._manifest(name)
        )
        self._name = name  # recorded immediately so stop() can't orphan it
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            obj = self._custom.get_namespaced_custom_object(
                GROUP, VERSION, self.namespace, PLURAL, name
            )
            for cond in obj.get("status", {}).get("conditions", []):
                if cond.get("type") == "Ready" and cond.get("status") == "True":
                    return
            time.sleep(3)
        raise TimeoutError(
            f"agent-sandbox {name} not Ready after {self.ready_timeout}s (image {self.image})"
        )

    async def stop(self) -> None:
        name, self._name = self._name, None
        if name is None:
            return
        await asyncio.to_thread(self._delete, name)

    def _delete(self, name: str) -> None:
        try:
            self._connect()
            self._custom.delete_namespaced_custom_object(
                GROUP, VERSION, self.namespace, PLURAL, name
            )
        except Exception:
            pass  # idempotent: 404 / transient API errors must not fail teardown

    async def is_alive(self) -> bool:
        if self._name is None:
            return False

        def _check() -> bool:
            try:
                self._connect()
                obj = self._custom.get_namespaced_custom_object(
                    GROUP, VERSION, self.namespace, PLURAL, self._name
                )
                return any(
                    c.get("type") == "Ready" and c.get("status") == "True"
                    for c in obj.get("status", {}).get("conditions", [])
                )
            except Exception:
                return False

        return await asyncio.to_thread(_check)

    # ----- exec ----------------------------------------------------------

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if self._name is None:
            raise RuntimeError("AgentSandbox not started; call start() first")
        prefix = ""
        if workdir:
            prefix += f"cd {shlex.quote(workdir)} && "
        for k, v in (env or {}).items():
            prefix += f"export {k}={shlex.quote(v)}; "
        command = ["/bin/sh", "-c", f"{prefix}exec {shlex.join(argv)}"]
        return await asyncio.to_thread(self._exec_sync, command, timeout)

    def _exec_sync(self, command: list[str], timeout: float | None) -> ExecResult:
        from kubernetes.stream import stream

        self._connect()
        resp = stream(
            self._core.connect_get_namespaced_pod_exec,
            self._name, self.namespace,
            command=command, stdout=True, stderr=True, stdin=False, tty=False,
            _preload_content=False,
        )
        out: list[str] = []
        err: list[str] = []
        deadline = time.monotonic() + (timeout or self.runtime_timeout)
        while resp.is_open():
            if time.monotonic() > deadline:
                resp.close()
                raise TimeoutError(f"exec timed out after {timeout}s")
            resp.update(timeout=5)
            if resp.peek_stdout():
                out.append(resp.read_stdout())
            if resp.peek_stderr():
                err.append(resp.read_stderr())
        rc = resp.returncode
        resp.close()
        return ExecResult(
            exit_code=rc if rc is not None else -1,
            stdout="".join(out), stderr="".join(err),
        )

    def _is_timeout_error(self, exc: BaseException) -> bool:
        return isinstance(exc, TimeoutError)
