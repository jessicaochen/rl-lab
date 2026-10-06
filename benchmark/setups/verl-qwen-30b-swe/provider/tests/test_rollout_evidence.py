"""Offline tests for the pure parts of the rollout adapter (stdlib only).

    python3 -m unittest discover -s benchmark/setups/verl-qwen-30b-swe/provider/tests -v
"""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlbench_verl_provider.rollout_evidence import RequestRecorder, parse_routing_stats, sanitize, sync_endpoints  # noqa: E402


class FakeBalancer:
    """Mimics GlobalRequestLoadBalancer.acquire_server: least inflight wins."""

    def __init__(self, servers):
        self.servers = dict(servers)
        self.inflight = {s: 0 for s in servers}
        self.acquires = 0

    async def acquire(self, request_id):
        self.acquires += 1
        sid = min(self.inflight, key=lambda s: (self.inflight[s], s))
        self.inflight[sid] += 1
        return sid, self.servers[sid]

    def release(self, sid):
        self.inflight[sid] -= 1


class SyncEndpointsTests(unittest.TestCase):
    def test_bootstrap_discovers_every_server_and_releases(self):
        lb = FakeBalancer({"a": "ha", "b": "hb", "c": "hc"})
        known = {}
        added, removed = asyncio.run(sync_endpoints(known, ["a", "b", "c"], lb.acquire, lb.release))
        self.assertEqual(sorted(added), ["a", "b", "c"])
        self.assertEqual(removed, [])
        self.assertEqual(known, {"a": "ha", "b": "hb", "c": "hc"})
        self.assertEqual(set(lb.inflight.values()), {0})  # every bootstrap acquire released

    def test_follows_add_and_remove(self):
        lb = FakeBalancer({"a": "ha", "b": "hb"})
        known = {}
        asyncio.run(sync_endpoints(known, ["a", "b"], lb.acquire, lb.release))
        lb.inflight["a"] = 5  # a is busy; a new server d appears
        lb.servers["d"] = "hd"
        lb.inflight["d"] = 0
        added, removed = asyncio.run(sync_endpoints(known, ["a", "b", "d"], lb.acquire, lb.release))
        self.assertEqual(added, ["d"])
        self.assertLessEqual(lb.acquires, 4)  # d (zero inflight) surfaces within the first two acquires
        added, removed = asyncio.run(sync_endpoints(known, ["a", "d"], lb.acquire, lb.release))
        self.assertEqual((added, removed), ([], ["b"]))
        self.assertEqual(sorted(known), ["a", "d"])

    def test_bounded_when_server_never_surfaces(self):
        lb = FakeBalancer({"a": "ha"})
        known = {}
        added, removed = asyncio.run(sync_endpoints(known, ["a", "ghost"], lb.acquire, lb.release, max_rounds=2))
        self.assertEqual(added, ["a"])
        self.assertNotIn("ghost", known)
        self.assertLessEqual(lb.acquires, 4)


class RequestRecorderTests(unittest.TestCase):
    def test_batches_then_flushes(self):
        with tempfile.TemporaryDirectory() as d:
            rec = RequestRecorder(Path(d) / "gateway-logs", tag="t", batch=3, max_age_s=1e9)
            rec.write({"n": 1}); rec.write({"n": 2})
            self.assertFalse(rec.path.exists())
            rec.write({"n": 3})
            self.assertEqual([json.loads(l)["n"] for l in rec.path.read_text().splitlines()], [1, 2, 3])
            rec.write({"n": 4}); rec.flush()
            self.assertEqual(rec.written, 4)

    def test_parse_routing_stats(self):
        body = ("# HELP x\nvllm:num_requests_running{model_name=\"m\",engine=\"0\"} 3.0\n"
                "vllm:num_requests_running{model_name=\"m\",engine=\"1\"} 2.0\n"
                "vllm:num_requests_waiting{model_name=\"m\"} 4.0\n"
                "vllm:kv_cache_usage_perc{model_name=\"m\",engine=\"0\"} 0.25\nvllm:kv_cache_usage_perc{engine=\"1\"} 0.75\n")
        self.assertEqual(parse_routing_stats(body), {"num_waiting_reqs": 4, "num_running_reqs": 5, "kv": 0.75, "error": None})
        self.assertIsNotNone(parse_routing_stats("nothing here")["error"])

    def test_sanitize(self):
        self.assertEqual(sanitize("10.0.0.5:41234"), "10.0.0.5_41234")


if __name__ == "__main__":
    unittest.main()
