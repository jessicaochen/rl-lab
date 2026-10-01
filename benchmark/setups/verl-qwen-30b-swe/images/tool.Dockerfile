# Mini-swe-agent sidecar tool image — adapted from uni-agent's
# Dockerfile.mini-swe-agent-tool with one change: the final stage is busybox
# (not scratch) with the payload at its real path, because our agent_sandbox
# provider materializes "mounts" as an initContainer that runs
# `cp -a /opt/mini-swe-agent/. <emptyDir>` (k8s has no image-overlay mounts).
FROM debian:bookworm-slim AS builder

ARG PBS_RELEASE="20260602"
ARG PBS_PYTHON="3.12.13"

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget \
    && rm -rf /var/lib/apt/lists/* \
    && wget -q \
        "https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PBS_PYTHON}%2B${PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz" \
        -O /tmp/python.tar.gz \
    && mkdir -p /opt/mini-swe-agent \
    && tar -xzf /tmp/python.tar.gz -C /opt/mini-swe-agent --strip-components=1 \
    && rm /tmp/python.tar.gz

RUN /opt/mini-swe-agent/bin/pip install --no-cache-dir \
    "mini-swe-agent==2.2.8" \
    "litellm==1.81.7"

COPY images/run_agent.py /opt/mini-swe-agent/bin/run_agent.py

FROM busybox:1.36
COPY --from=builder /opt/mini-swe-agent /opt/mini-swe-agent
