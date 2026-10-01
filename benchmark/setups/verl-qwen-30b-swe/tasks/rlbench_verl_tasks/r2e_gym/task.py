"""R2E-Gym task for uni-agent: mini-swe-agent (or any configured agent) works
on /testbed inside a per-task sandbox; reward = hidden tests re-run afterwards.

Lifecycle (ported from the prime-envs r2e_gym taskset, which this replaces):
1. link the testbed venv + drop stale bytecode
2. HIDE the grading tests: archive /r2e_tests, download the archive to the
   driver, delete it in the sandbox (agent must not see the tests)
3. run the agent against /testbed
4. restore the tests into /testbed/r2e_tests and run `run_tests.sh`
5. reward 1.0 iff the pytest pass/fail map matches `expected_output_json`
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from pydantic import Field
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

from .reward import calculate_reward, extract_gold_patch

logger = logging.getLogger(__name__)

REPO_PATH = "/testbed"
REMOTE_TEST_ARCHIVE = "/tmp/r2e_tests.tar.gz"
REMOTE_ROUNDTRIP_ARCHIVE = "/tmp/r2e_tests_roundtrip.tar.gz"

# The testbed venv (project + pytest) plus quiet, non-interactive tooling.
ENV = {
    "PATH": (
        "/opt/miniconda3/bin:/testbed/.venv/bin:/root/.local/bin:"
        "/root/.cargo/bin:/go/bin:/usr/local/go/bin:/usr/local/cargo:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "CI": "1",
}

LINK = r"""
ln -sfn /testbed/.venv /root/.venv 2>/dev/null || true
mkdir -p /root/.local/bin
ln -sfn /testbed/.venv/bin/python /root/.local/bin/python 2>/dev/null || true
ln -sfn /testbed/.venv/bin/python /root/.local/bin/python3 2>/dev/null || true
find /testbed/.venv/bin -type f -executable -exec ln -sfn {} /root/.local/bin/ \; 2>/dev/null || true
# also surface the venv on the default PATH: the in-sandbox agent (uni-agent
# mini-swe-agent launch) builds its own env and won't include /root/.local/bin
ln -sfn /testbed/.venv/bin/python /usr/local/bin/python 2>/dev/null || true
ln -sfn /testbed/.venv/bin/python /usr/local/bin/python3 2>/dev/null || true
find /testbed/.venv/bin -type f -executable -exec ln -sfn {} /usr/local/bin/ \; 2>/dev/null || true
"""

CLEAN_PYCACHE = (
    "timeout 30 bash -c 'shopt -s globstar; rm -rf **/*.pyc **/__pycache__' "
    "2>/dev/null || timeout 30 find . -name '*.pyc' -delete 2>/dev/null || true"
)

RESTORE_TESTS = f"rm -rf {REPO_PATH}/r2e_tests && tar -C {REPO_PATH} -xzf {REMOTE_ROUNDTRIP_ARCHIVE}"


class R2EGymTaskConfig(TaskConfig):
    name: str = "r2e_gym"
    eval_timeout: float = Field(
        default=600.0,
        description="Per-sample reward-eval timeout (s) inside the sandbox.",
    )
    run_oracle_solution: bool = Field(
        default=False,
        description="Oracle mode: skip the agent, apply the gold patch — must score 1.0.",
    )


@register_task("r2e_gym")
class R2EGymTask(Task):
    name = "r2e_gym"
    config_model = R2EGymTaskConfig

    async def _sh(self, sandbox, cmd: str, *, what: str, timeout: float | None = None,
                  workdir: str | None = None) -> str:
        result = await sandbox.exec(["sh", "-c", cmd], env=ENV, timeout=timeout, workdir=workdir)
        if result.exit_code != 0:
            raise RuntimeError(f"r2e {what} failed (exit {result.exit_code}): {result.stderr.strip()[-500:]}")
        return result.stdout

    async def run(self) -> TaskResult:
        cfg: R2EGymTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata
        name = sample.get("commit_hash", "?")
        logger.info("starting r2e_gym task (%s)", name)

        async with self.build_sandbox() as sandbox:
            # 1. setup
            await self._sh(sandbox, LINK, what="venv link")
            await sandbox.exec(["sh", "-c", CLEAN_PYCACHE], env=ENV, workdir=REPO_PATH)

            # 2. hide the grading tests (archive lives on the driver during the episode)
            await self._sh(
                sandbox, f"tar -C / -czf {REMOTE_TEST_ARCHIVE} r2e_tests", what="archive tests"
            )
            with tempfile.NamedTemporaryFile(prefix="r2e_tests_", suffix=".tar.gz", delete=False) as fh:
                local_archive = Path(fh.name)
            try:
                await sandbox.download_file(REMOTE_TEST_ARCHIVE, local_archive)
                await self._sh(
                    sandbox, f"rm -rf /r2e_tests {REMOTE_TEST_ARCHIVE}", what="hide tests"
                )

                # 3. the agent works on /testbed (oracle: gold patch instead)
                if cfg.run_oracle_solution:
                    patch = extract_gold_patch(sample.get("parsed_commit_content", ""))
                    if not patch.strip():
                        raise RuntimeError(f"empty gold patch for {name}")
                    await sandbox.write_file("/tmp/gold.patch", patch)
                    await self._sh(
                        sandbox, "git apply --whitespace=fix /tmp/gold.patch",
                        what="gold apply", workdir=REPO_PATH,
                    )
                    finished = True
                else:
                    agent = self.build_agent()
                    agent_result = await agent.run(
                        sandbox=sandbox,
                        messages=cfg.prompt,
                        workdir=REPO_PATH,
                    )
                    finished = agent_result.finished

                # 4. restore tests + grade
                await sandbox.upload_file(local_archive, REMOTE_ROUNDTRIP_ARCHIVE)
                await self._sh(sandbox, RESTORE_TESTS, what="restore tests")
                test = await sandbox.exec(
                    ["sh", "-c", "/bin/bash run_tests.sh 2>&1"],
                    env=ENV, workdir=REPO_PATH, timeout=cfg.eval_timeout,
                )
            finally:
                local_archive.unlink(missing_ok=True)

            reward = calculate_reward(test.stdout or "", sample["expected_output_json"])
            logger.info("r2e_gym task done (%s): reward=%s finished=%s", name, reward, finished)
            return TaskResult(
                reward=reward,
                accuracy=reward,
                finished=finished,
                extra_info={
                    "resolved": bool(reward),
                    "test_exit_code": test.exit_code,
                    "test_output_tail": (test.stdout or "")[-2000:],
                },
            )
