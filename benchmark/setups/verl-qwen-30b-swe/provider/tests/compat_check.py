"""GPU-free compatibility check for rollout_adapter against the INSTALLED verl +
py-inference-scheduler (run inside the driver image, e.g. on the Ray head pod):

    ROUTER_CONFIG_PATH=/opt/py-inference-scheduler/integration/verl/examples/scheduler.yaml \
    RLBENCH_RUN_OUT=/tmp/compat RLBENCH_ROUTER=inference-scheduler \
    python3 -m rlbench_verl_provider_tests.compat_check   # or: python3 provider/tests/compat_check.py

Fake vLLM server actors + the real GlobalRequestLoadBalancer, exercising the
real FullyAsyncLLMServerClient subclasses: bootstrap, membership add/remove,
inflight accounting back to zero, resume on an aborted first attempt, JSONL
records written, and that LazyClient survives pickling.
"""
import asyncio
import json
import os
import pickle
import sys
from pathlib import Path

import ray
from omegaconf import OmegaConf
from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer
from verl.workers.rollout.replica import TokenOutput

from rlbench_verl_provider.rollout_adapter import ClientSpec, LazyClient, build_client


@ray.remote
class FakeServer:
    def __init__(self, name, abort_first=False):
        self.name, self.calls, self.abort_first = name, 0, abort_first

    async def generate(self, request_id, prompt_ids, sampling_params, **kw):
        self.calls += 1
        if self.abort_first and self.calls == 1:
            return TokenOutput(token_ids=[7], log_probs=[0.0], stop_reason="aborted", extra_fields={"global_steps": 1})
        return TokenOutput(token_ids=[1, 2, 3], log_probs=[0.0] * 3, stop_reason="stop", extra_fields={"global_steps": 1})

    def get_routing_stats(self):
        return {"num_waiting_reqs": 0, "num_running_reqs": self.calls, "kv": 0.1, "error": None}

    def get_server_address(self):
        return ("127.0.0.1", 1)

    def get_calls(self):
        return self.calls


async def main():
    router = os.environ.get("RLBENCH_ROUTER", "verl")
    out = os.environ.get("RLBENCH_RUN_OUT", "/tmp/rlbench-compat")
    servers = {f"srv-{i}": FakeServer.remote(f"srv-{i}", abort_first=(i == 0)) for i in range(3)}
    lb = ray.remote(GlobalRequestLoadBalancer).remote(servers)
    cfg = OmegaConf.create({"actor_rollout_ref": {"rollout": {"response_length": 64, "ignore_eos": False}}})
    spec = ClientSpec(config=cfg, load_balancer_handle=lb, only_hybrid=False, router=router, out_dir=out)
    lazy = pickle.loads(pickle.dumps(LazyClient(spec)))
    client = lazy._get()
    print("client:", type(client).__name__)

    prefix = list(range(200))
    for i in range(4):  # same session, growing context: multi-turn shape
        o = await lazy.generate(request_id="sess-A", prompt_ids=prefix + list(range(10 * i)),
                                sampling_params={"max_tokens": 32})
        assert o.stop_reason == "stop", o
    status = await lb.get_status.remote() if hasattr(lb, "get_status") else None
    total = await lb.get_total_inflight.remote()
    assert total == 0, f"verl balancer inflight not back to zero: {total} ({status})"
    if router == "inference-scheduler":
        assert sorted(client._endpoints) == sorted(servers), client._endpoints
        assert all(v == 0 for v in client.inflight._counts.values()) if hasattr(client.inflight, "_counts") else True
        # membership follows the balancer
        await lb.remove_servers.remote(["srv-2"])
        await lazy.generate(request_id="sess-B", prompt_ids=prefix, sampling_params={"max_tokens": 8})
        assert "srv-2" not in client._endpoints, client._endpoints
        await lb.add_servers.remote({"srv-2": servers["srv-2"]})
        await lazy.generate(request_id="sess-C", prompt_ids=prefix, sampling_params={"max_tokens": 8})
        assert "srv-2" in client._endpoints, client._endpoints
        print("fallbacks:", client._fallbacks, "decisions:", client._decisions)
    client._recorder.flush()
    recs = [json.loads(l) for p in Path(out, "gateway-logs").glob("*.jsonl") for l in p.read_text().splitlines()]
    assert recs and all(r["router"] == router for r in recs), recs[:1]
    resumed = [r for r in recs if r["attempts"] > 1]
    print(f"records={len(recs)} resumed={len(resumed)} servers={sorted({r['server_id'] for r in recs})}")
    assert resumed, "expected the aborted first attempt on srv-0 to be resumed (attempts > 1)"
    print("compat check OK")


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    asyncio.run(main())
    ray.shutdown()
