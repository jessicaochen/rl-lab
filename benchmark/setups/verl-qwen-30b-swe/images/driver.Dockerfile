# Driver image for the verl + uni-agent setup: runs on every Ray node (head,
# trainer workers, rollout workers) and in the submitter Job.
#
# Base: verl's app image (torch + vLLM + Megatron deps). verl itself is
# installed from source at release/v0.9.0 — the tag uni-agent's mini-swe-agent
# recipe is developed and validated against (separate_async trainer + the
# black-box agent framework).
ARG VERL_BASE=verlai/verl:vllm024.dev2
FROM ${VERL_BASE}

# pinned upstreams (build args so bumps don't edit the Dockerfile)
ARG VERL_REF=adc7eefa16dad75c5f7b878823d5a76eac90c7b3
# release/v0.9.0
ARG UNI_AGENT_REF=9f7b024
# last commit before Continuous-Token integration (#164), which requires verl>v0.9.0
ARG PYIS_REF=291b31be078c88e28aa0dd00dc67f03d82d81312
# py-inference-scheduler main @ 2026-09-23 (rlbench feature "inference-scheduler")

RUN git clone --filter=blob:none https://github.com/verl-project/verl /opt/verl \
    && git -C /opt/verl checkout -q ${VERL_REF} \
    && pip install --no-deps -e /opt/verl

RUN git clone --filter=blob:none https://github.com/verl-project/uni-agent /opt/uni-agent \
    && git -C /opt/uni-agent checkout -q ${UNI_AGENT_REF} \
    && pip install -e /opt/uni-agent

# k8s client for the agent-sandbox provider; megatron-bridge + modelopt for
# use_mbridge HF->Megatron loading. Constraints pin the base image's critical
# wheels (an unconstrained resolve uninstalled flashinfer and tripped over the
# distro-owned blinker), while still letting pip pull the small leaf deps
# modelopt needs (pulp, ...).
RUN pip install "kubernetes>=31.0.0" "cupy-cuda12x" \
    && pip install --no-deps "megatron-bridge" "nvidia-modelopt"
# a full-dep resolve breaks the base image (uninstalls flashinfer, trips on
# distro blinker) and a constrained resolve is unsatisfiable — so install the
# heavy packages dep-less and pull in only the small leaf modules they
# actually import, discovered by attempting the import
RUN python - <<'EOF'
import importlib, subprocess, sys
PIP_NAME = {"yaml": "pyyaml", "PIL": "pillow", "cv2": "opencv-python-headless"}
# leaf deps resolve normally but pinned to the base image's numpy/torch so a
# leaf (e.g. scipy) can't pull an incompatible version of either
subprocess.check_call(
    "pip freeze 2>/dev/null | grep -E '^(numpy|torch)==' > /tmp/base-pins.txt", shell=True
)
for _ in range(15):
    try:
        importlib.import_module("megatron.bridge.models.conversion.auto_bridge")
        print("mbridge OK")
        break
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        name = getattr(e, "name", None)
        if not name:
            raise
        pkg = PIP_NAME.get(name, name)
        print("installing missing leaf dep:", pkg, flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-c", "/tmp/base-pins.txt",
             "--upgrade", pkg]
        )
else:
    raise SystemExit("mbridge import still failing after 15 leaf installs")
EOF

# swebench harness: uni-agent's swe_bench task grades F2P/P2P with it (used
# for SWE-bench Verified validation); not a declared uni-agent dependency.
# Pinned to 4.1.0 = the version uni-agent's docs install (5.x moved the
# harness modules the grader imports). No torch/numpy deps of its own, so a
# normal resolve pinned to the base image's numpy/torch is safe.
RUN pip freeze 2>/dev/null | grep -E '^(numpy|torch)==' > /tmp/base-pins.txt \
    && pip install -c /tmp/base-pins.txt "swebench==4.1.0" \
    && python - <<'EOF2'
import importlib, subprocess, sys
seen = set()
for _ in range(20):
    try:
        importlib.import_module("swebench.harness.grading"); importlib.import_module("swebench.harness.test_spec.python")
        print("swebench OK"); break
    except (ModuleNotFoundError, ImportError) as e:
        mod = (getattr(e, "name", None) or str(e).split("'")[1]).split(".")[0]
        pkg = {"yaml": "pyyaml", "PIL": "pillow", "dotenv": "python-dotenv", "git": "GitPython", "bs4": "beautifulsoup4"}.get(mod, mod)
        if pkg in seen or pkg == "swebench":
            raise SystemExit(f"swebench import failing inside an installed package: {type(e).__name__}: {e}")
        seen.add(pkg)
        print("installing missing leaf dep:", pkg, "(because:", str(e)[:80], ")", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-c", "/tmp/base-pins.txt", "--upgrade", pkg])
else:
    raise SystemExit("swebench import still failing")
EOF2

# py-inference-scheduler (rlbench feature "inference-scheduler"): the package
# proper is a src layout (pip -e), but its verl integration (integration/) and
# backend patches (backends/) are plain top-level dirs outside the wheel, so the
# repo root goes on PYTHONPATH. Present in every image so baseline and feature
# runs share one image; only imported when the feature is on. Leaf deps only
# (ray/fastapi/uvicorn come with the base), pinned to the base numpy/torch.
RUN git clone --filter=blob:none https://github.com/llm-d-incubation/py-inference-scheduler /opt/py-inference-scheduler \
    && git -C /opt/py-inference-scheduler checkout -q ${PYIS_REF} \
    && pip install --no-deps -e /opt/py-inference-scheduler \
    && pip freeze 2>/dev/null | grep -E '^(numpy|torch)==' > /tmp/base-pins.txt \
    && pip install -c /tmp/base-pins.txt "prometheus-client>=0.20" "setproctitle>=1.3" "aiohttp>=3.9"
ENV PYTHONPATH=/opt/py-inference-scheduler${PYTHONPATH:+:$PYTHONPATH}

# our provider (agent-sandbox Sandbox CRs + rollout adapter) + r2e-gym task module
COPY provider /opt/rlbench-verl/provider
COPY tasks /opt/rlbench-verl/tasks
RUN pip install -e /opt/rlbench-verl/provider -e /opt/rlbench-verl/tasks

# register both in uni-agent's lazy module maps so pure-config selection works
RUN python - <<'EOF'
import pathlib, uni_agent.sandbox.registry as sr, uni_agent.tasks.registry as tr
for mod, key, val in (
    (sr, "agent_sandbox", "rlbench_verl_provider.agent_sandbox"),
    (tr, "r2e_gym", "rlbench_verl_tasks.r2e_gym.task"),
):
    p = pathlib.Path(mod.__file__)
    text = p.read_text()
    marker = f'"{key}": "{val}",'
    if marker not in text:
        anchor = "_MODULES: dict[str, str] = {"
        i = text.index(anchor) + len(anchor)
        p.write_text(text[:i] + f'\n    {marker}' + text[i:])
print("registered agent_sandbox provider + r2e_gym task")
EOF

# build-time verification: everything importable, both registrations resolve
RUN python - <<'EOF'
import importlib
import verl, megatron.core  # noqa
importlib.import_module("verl.trainer.main_ppo")
from uni_agent.sandbox.registry import get_sandbox_cls
assert get_sandbox_cls("agent_sandbox").provider == "agent_sandbox"
from uni_agent.tasks.registry import get_task_cls
assert get_task_cls("r2e_gym").name == "r2e_gym"
# rollout adapter (always on) + py-inference-scheduler (feature): importable, modern verl layout detected
import rlbench_verl_provider.rollout_adapter  # noqa
from py_inference_scheduler import Scheduler  # noqa
import integration.verl.verl_hook as hook
assert hook._VERL_LAYOUT == "modern", hook._VERL_LAYOUT
print("verl + uni-agent + agent_sandbox provider + r2e_gym task + rollout adapter + py-inference-scheduler: OK")
EOF
