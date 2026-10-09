"""Unit tests for the rl-lab ``--feature timeslice`` integration (Features F1-F6)."""

from __future__ import annotations

# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods
import asyncio
import queue
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from timeslice import (
    DualPoolRoleLocks,
    RoleLocks,
    TimeSliceOrchestratorClient,
    drain_sampler_inflight_requests,
    format_gpu_mem_log,
    format_timeslice_log,
    offload_megatron_trainer,
    parse_gpu_mem_log,
    parse_timeslice_log,
    restore_megatron_trainer,
    sleep_vllm_sampler_async,
    trigger_cuda_checkpoint_transition,
    wake_vllm_sampler_async,
)


class _DummyMegatronEngine:
    """In-memory mock of MegatronEngine tracking .to(device, model, optimizer, grad, point)."""

    def __init__(self) -> None:
        self.device = "cuda"
        self.history: list[tuple[str, bool, bool, bool, str | None]] = []

    def to(
        self,
        device: str,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
        point: str | None = None,
    ) -> None:
        self.device = device
        self.history.append((device, model, optimizer, grad, point))


class _DummyVLLMEngine:
    """In-memory mock of async vLLM engine supporting sleep(level=1) and wake_up(tags=...)."""

    def __init__(self, inflight: int = 0) -> None:
        self._inflight = inflight
        self.asleep = False
        self.sleep_levels: list[int] = []
        self.wake_tags: list[list[str]] = []
        self.prefix_cache_reset = False

    def get_num_unfinished_requests(self) -> int:
        if self._inflight > 0:
            self._inflight -= 1
            return self._inflight + 1
        return 0

    async def sleep(self, level: int = 1) -> None:
        self.asleep = True
        self.sleep_levels.append(level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        self.asleep = False
        self.wake_tags.append(list(tags or []))

    async def reset_prefix_cache(self, reset_connector: bool = False) -> None:
        self.prefix_cache_reset = bool(reset_connector) or not reset_connector


class TestTimesliceFeature(unittest.TestCase):
    """Unit test suite for timeslice.py lock ordering, offload/restore hooks, and telemetry."""

    def test_orchestrator_client_acquire_release_and_status(self) -> None:
        client = TimeSliceOrchestratorClient(
            target="localhost:50051",
            job_id="job1",
            group_id="trainers",
        )
        client.set_simulated_waiters("trainers", 1)
        acq = client.acquire("job1", "trainers")
        self.assertTrue(acq.success)
        self.assertFalse(acq.context_restored)
        status_locked = client.get_status("trainers")
        self.assertEqual(status_locked.state, "LOCKED")
        self.assertEqual(status_locked.locking_job, "job1")

        rel = client.release("job1", "trainers")
        self.assertTrue(rel.success)
        self.assertEqual(rel.pending_waiters, 1)
        self.assertFalse(rel.snapshot_deferred)

        acq2 = client.acquire("job2", "trainers")
        self.assertTrue(acq2.context_restored)
        client.release("job2", "trainers")
        self.assertIn("trainers", client.list_groups())
        self.assertIn("samplers", client.list_groups())

    def test_on_accelerators_context_manager(self) -> None:
        client = TimeSliceOrchestratorClient(job_id="job1", group_id="samplers")
        with client.on_accelerators("job1", "samplers") as acq:
            self.assertTrue(acq.success)
            self.assertEqual(client.get_status("samplers").state, "LOCKED")
        self.assertEqual(client.get_status("samplers").state, "IDLE_YIELDED")

    def test_dual_pool_role_locks_global_order_and_violation(self) -> None:
        locks = DualPoolRoleLocks(
            job_id="job1",
            trainer_group="trainers",
            sampler_group="samplers",
            pending_waiters_trainer=0,
            pending_waiters_sampler=1,
        )
        self.assertIs(RoleLocks, DualPoolRoleLocks)
        self.assertTrue(locks.acquire("trainer", waited_ms=10.0, context_restored=False))
        self.assertFalse(locks.acquire("trainer"))  # idempotent
        self.assertTrue(locks.acquire("sampler", waited_ms=5.0, context_restored=False))
        self.assertEqual(locks.held_roles, {"trainer", "sampler"})

        locks.release_all()
        self.assertEqual(locks.held_roles, set())
        self.assertFalse(locks.release("trainer"))  # idempotent

        # Acquire SAMPLER first, then verify acquiring TRAINER raises RuntimeError
        self.assertTrue(locks.acquire("sampler"))
        with self.assertRaises(RuntimeError) as ctx:
            locks.acquire("trainer")
        self.assertIn("lock-order violation", str(ctx.exception))
        locks.release_all()

    def test_dual_pool_role_locks_trainer_and_sampler_toggles(self) -> None:
        # Trainer-only time-slicing (sampler_enabled=False)
        trainer_only = DualPoolRoleLocks(
            job_id="job1",
            trainer_enabled=True,
            sampler_enabled=False,
        )
        self.assertTrue(trainer_only.enabled)
        self.assertTrue(trainer_only.is_role_enabled("trainer"))
        self.assertFalse(trainer_only.is_role_enabled("sampler"))
        self.assertTrue(trainer_only.acquire("trainer"))
        self.assertFalse(trainer_only.acquire("sampler"))
        self.assertEqual(trainer_only.held_roles, {"trainer"})
        self.assertFalse(trainer_only.release("sampler"))
        self.assertTrue(trainer_only.release("trainer"))

        # Sampler-only time-slicing (trainer_enabled=False)
        sampler_only = DualPoolRoleLocks(
            job_id="job1",
            trainer_enabled=False,
            sampler_enabled=True,
        )
        self.assertTrue(sampler_only.enabled)
        self.assertFalse(sampler_only.is_role_enabled("trainer"))
        self.assertTrue(sampler_only.is_role_enabled("sampler"))
        self.assertFalse(sampler_only.acquire("trainer"))
        self.assertTrue(sampler_only.acquire("sampler"))
        # Calling acquire("trainer") while holding sampler is a safe no-op when trainer is disabled
        self.assertFalse(sampler_only.acquire("trainer"))
        self.assertEqual(sampler_only.held_roles, {"sampler"})
        self.assertTrue(sampler_only.release("sampler"))

        # Both disabled -> enabled becomes False
        both_off = DualPoolRoleLocks(
            job_id="job1",
            trainer_enabled=False,
            sampler_enabled=False,
        )
        self.assertFalse(both_off.enabled)
        self.assertFalse(both_off.acquire("trainer"))
        self.assertFalse(both_off.acquire("sampler"))

        # Environment variable parsing in from_env()
        with unittest.mock.patch.dict(
            "os.environ",
            {
                "TIMESLICE_ENABLED": "1",
                "TIMESLICE_TRAINER_ENABLED": "1",
                "TIMESLICE_SAMPLER_ENABLED": "0",
                "TIMESLICE_JOB_ID": "job-env",
            },
            clear=False,
        ):
            env_locks = DualPoolRoleLocks.from_env()
            self.assertEqual(env_locks.job_id, "job-env")
            self.assertTrue(env_locks.enabled)
            self.assertTrue(env_locks.trainer_enabled)
            self.assertFalse(env_locks.sampler_enabled)

    def test_deferred_snapshot_when_pending_waiters_zero(self) -> None:
        locks = DualPoolRoleLocks(
            job_id="job1",
            pending_waiters_trainer=0,
            pending_waiters_sampler=2,
        )
        locks.acquire("trainer")
        locks.release("trainer")
        rel_trainer = parse_timeslice_log(locks.events[-1])
        self.assertEqual(rel_trainer["pending_waiters"], "0")
        self.assertEqual(rel_trainer["snapshot_deferred"], "true")

        locks.acquire("sampler")
        locks.release("sampler")
        rel_sampler = parse_timeslice_log(locks.events[-1])
        self.assertEqual(rel_sampler["pending_waiters"], "2")
        self.assertEqual(rel_sampler["snapshot_deferred"], "false")

    def test_wait_for_sample_batch_releases_trainer_lock(self) -> None:
        locks = DualPoolRoleLocks(job_id="job1")
        locks.acquire("trainer")
        q: queue.Queue[str] = queue.Queue()
        q.put("batch-0")
        events_trace: list[str] = []

        batch = locks.wait_for_sample_batch_outside_trainer_lock(
            q,
            offload_fn=lambda: events_trace.append("offloaded"),
            restore_fn=lambda: events_trace.append("restored"),
        )
        self.assertEqual(batch, "batch-0")
        self.assertEqual(events_trace, ["offloaded", "restored"])
        self.assertIn("trainer", locks.held_roles)
        locks.release_all()

    def test_megatron_trainer_offload_and_restore_hooks(self) -> None:
        eng = _DummyMegatronEngine()
        off_res = offload_megatron_trainer(
            eng,
            job_id="job1",
            rank=0,
            world_size=8,
            point="post_sync",
            optimizer=False,
            grad=False,
        )
        self.assertEqual(eng.device, "cpu")
        self.assertEqual(eng.history[-1], ("cpu", True, False, False, "post_sync"))
        parsed_off = parse_timeslice_log(off_res["timeslice_log"])
        self.assertEqual(parsed_off["action"], "OFFLOAD")
        self.assertEqual(parsed_off["role"], "trainer")

        res_res = restore_megatron_trainer(eng, job_id="job1", rank=0, world_size=8)
        self.assertEqual(eng.device, "cuda")
        parsed_res = parse_timeslice_log(res_res["timeslice_log"])
        self.assertEqual(parsed_res["action"], "RESTORE")

        with self.assertRaises(ValueError):
            offload_megatron_trainer(eng, world_size=1)

    def test_vllm_sampler_drain_sleep_and_wake_hooks(self) -> None:
        vllm_eng = _DummyVLLMEngine(inflight=2)
        polls = drain_sampler_inflight_requests(
            vllm_eng, timeout_sec=2.0, poll_interval_sec=0.001
        )
        self.assertEqual(polls, 2)

        sleep_res = asyncio.run(
            sleep_vllm_sampler_async(
                vllm_eng, job_id="job1", tp_size=2, gpu_memory_utilization=0.80
            )
        )
        self.assertTrue(vllm_eng.asleep)
        self.assertEqual(vllm_eng.sleep_levels, [1])
        self.assertEqual(parse_timeslice_log(sleep_res["timeslice_log"])["action"], "OFFLOAD")

        wake_res = asyncio.run(wake_vllm_sampler_async(vllm_eng, job_id="job1", tp_size=2))
        self.assertFalse(vllm_eng.asleep)
        self.assertEqual(vllm_eng.wake_tags, [["weights", "kv_cache"]])
        self.assertTrue(vllm_eng.prefix_cache_reset)
        self.assertEqual(parse_timeslice_log(wake_res["timeslice_log"])["action"], "RESTORE")

        with self.assertRaises(ValueError):
            asyncio.run(sleep_vllm_sampler_async(vllm_eng, tp_size=1))
        with self.assertRaises(ValueError):
            asyncio.run(
                sleep_vllm_sampler_async(vllm_eng, tp_size=2, gpu_memory_utilization=0.95)
            )

    def test_structured_log_formatting_and_parsing(self) -> None:
        ts_line = format_timeslice_log(
            job_id="job1",
            role="trainer",
            action="ACQUIRE",
            group="trainers",
            waited_ms=33.4,
            context_restored=True,
        )
        parsed_ts = parse_timeslice_log(ts_line)
        self.assertEqual(parsed_ts["action"], "ACQUIRE")
        self.assertEqual(parsed_ts["context_restored"], "true")

        mem_line = format_gpu_mem_log(
            point="post_offload",
            role="trainer",
            rank=3,
            world_size=8,
            torch_alloc_gb=0.02,
            torch_reserved_gb=0.25,
            cuda_used_gb=12.40,
            cuda_total_gb=141.00,
        )
        parsed_mem = parse_gpu_mem_log(mem_line)
        self.assertEqual(parsed_mem["rank"], 3)
        self.assertEqual(parsed_mem["world_size"], 8)
        self.assertAlmostEqual(parsed_mem["cuda_used_gb"], 12.40)

    def test_cuda_checkpoint_transition_helper(self) -> None:
        ckpt_info = trigger_cuda_checkpoint_transition(
            "checkpoint", pids=[], use_cr_shim_signals=True
        )
        self.assertEqual(ckpt_info["sig_pre_checkpoint"], 35)
        self.assertEqual(len(ckpt_info["commands"]), 2)
        restore_info = trigger_cuda_checkpoint_transition(
            "restore", pids=[], use_cr_shim_signals=True
        )
        self.assertEqual(restore_info["sig_post_restore"], 36)
        self.assertEqual(len(restore_info["commands"]), 1)


def test_standalone_smoke_with_tmp_path(tmp_path: Path) -> None:
    """Function-style test compatible with inspect-based test runners."""
    marker = tmp_path / "ok.txt"
    marker.write_text("passed", encoding="utf-8")
    assert marker.read_text(encoding="utf-8") == "passed"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as _td:
        test_standalone_smoke_with_tmp_path(Path(_td))
    unittest.main()
