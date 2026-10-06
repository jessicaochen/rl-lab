"""rlbench: setup, run, and collect RL training benchmarks on the cluster
kubectl currently points at.

    rlbench run <setup-folder> [--config f] [--feature NAME]... [--var K=V]... [--keep] [--out runs/] [--timeout 24h]
    rlbench cleanup <run-folder>
    rlbench collect <run-folder>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

from . import kube
from .metrics import MetricsSampler
from .runfolder import RunFolder
from .setup_folder import (
    SetupFolder,
    feature_builtins,
    feature_variables,
    namespace_of,
    render_manifest,
    render_text_lenient,
)

DEFAULT_TIMEOUT_S = 24 * 3600
READY_TIMEOUT_S = 60 * 60   # inference readiness includes model download + load


def _parse_timeout(value: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600}
    if value[-1] in units:
        return int(float(value[:-1]) * units[value[-1]])
    return int(value)


def _run_hook(hook: Path | None, env: dict[str, str]) -> None:
    if hook:
        print(f"--> hook: {hook.name}")
        subprocess.run(["bash", str(hook)], env={**os.environ, **env}, check=True)


def cmd_run(args: argparse.Namespace) -> int:
    setup = SetupFolder.load(args.setup_folder)
    features = [setup.feature(n) for n in args.feature]  # unknown names fail here, before any run folder exists
    run = RunFolder.create(args.out, args.name or setup.root.name)
    print(f"run id: {run.run_id}\nrun folder: {run.path}")
    if features:
        print("features: " + ", ".join(f.name for f in features))

    extra: dict[str, str] = {}
    for item in args.var:
        if "=" not in item:
            print(f"error: --var expects KEY=VALUE, got {item!r}")
            return 1
        k, v = item.split("=", 1)
        extra[k] = v
    # precedence: environment < feature vars.env < --var < built-ins
    variables = {
        **os.environ,
        **feature_variables(features),
        **extra,
        "RUN_ID": run.run_id,
        "RUN_NAME": run.run_id,
        **feature_builtins(features),
    }
    config_file = Path(args.config).resolve() if args.config else None
    feature_config_files = [(f, c) for f in features for c in f.config_files]
    if feature_config_files and config_file is not None and not config_file.is_dir():
        print("error: feature config files need --config to be a directory (or no --config)")
        return 1

    # Render everything up front: a template error should fail before the
    # cluster is touched, and the rendered manifests are the cleanup manifest.
    setup_docs: list[dict] = []
    for m in setup.setup_manifests:
        docs = render_manifest(m, variables)
        run.write_rendered(m.name, docs)
        setup_docs.extend(docs)
    # feature manifests render after the setup's: "feature-<name>-<file>" sorts
    # after the numbered base files, so they apply last and are deleted first
    for f in features:
        for m in f.setup_manifests:
            docs = render_manifest(m, variables)
            run.write_rendered(f"feature-{f.name}-{m.name}", docs)
            setup_docs.extend(docs)
    job_docs = render_manifest(setup.job_manifest, variables)
    job_path = run.write_rendered("job.yaml", job_docs)
    namespace = namespace_of(setup_docs + job_docs)
    jobs = [d for d in job_docs if d.get("kind") == "Job"]
    if len(jobs) != 1:
        print(f"error: job.yaml must contain exactly one Job (found {len(jobs)})")
        return 1
    job_name = jobs[0]["metadata"]["name"]

    run.record_setup_ref(setup.root)
    run.record_features(features)
    run.record_cluster_identity()

    # Run configs are templated like manifests (e.g. per-run output dirs);
    # the rendered copy in the run folder is exactly what ran.
    try:
        rendered_config = _render_run_config(run.config, config_file, feature_config_files, variables)
    except ValueError as e:
        print(f"error: {e}")
        return 1

    pod_targets = setup.scrape_targets()
    for f in features:
        pod_targets += f.scrape_targets()
    streamer = kube.LogStreamer(namespace, run.run_id, run.logs)
    sampler = MetricsSampler(namespace, run.metrics, pod_targets=pod_targets)
    hook_env = {"RUN_ID": run.run_id, "NAMESPACE": namespace, "RLBENCH_FEATURES": variables["RLBENCH_FEATURES"]}
    outcome = "SetupFailed"
    job_status: dict = {}
    try:
        for owner in (setup, *features):  # setup hook first, then features in --feature order
            _run_hook(owner.hook("pre-setup.sh"), hook_env)

        print(f"--> applying setup to namespace {namespace}")
        for p in sorted(run.rendered.iterdir()):
            if p.name != "job.yaml":
                kube.apply(p.read_text())
        if rendered_config:
            source = (
                f"--from-file={rendered_config}" if rendered_config.is_dir()
                else f"--from-file={rendered_config.name}={rendered_config}"
            )
            manifest = kube.kubectl(
                "create", "configmap", "rlbench-run-config", "-n", namespace,
                source, "--dry-run=client", "-o", "yaml",
            )
            kube.apply(manifest)
        kube.wait_deployments_ready(namespace, READY_TIMEOUT_S)
        run.mark("setup_ready")

        streamer.start()
        sampler.start()

        outcome = "RunError"  # setup succeeded; later exceptions are run errors
        print(f"--> submitting job {job_name}")
        # Job templates are immutable: a leftover job from a previous attempt
        # (e.g. after --keep) must go before re-submission
        kube.kubectl("delete", "job", job_name, "-n", namespace,
                     "--ignore-not-found", check=False)
        kube.apply(job_path.read_text())
        result = kube.watch_job(namespace, job_name, _parse_timeout(args.timeout))
        outcome, job_status = result["outcome"], result["job_status"]
        run.mark("job_finished")
        print(f"--> job outcome: {outcome}")
    except Exception:
        traceback.print_exc()
    finally:
        print("--> collecting")
        sampler.stop()
        streamer.stop()
        kube.dump_events(namespace, run.events)
        run.mark("collected")
        run.write_result(outcome, {"namespace": namespace, "features": [f.name for f in features],
                                   "job_status": job_status})
        for owner in (setup, *features):
            _run_hook(owner.hook("post-run.sh"), {**hook_env, "RUN_FOLDER": str(run.path)})
        if args.keep:
            print(f"--> --keep: leaving resources in namespace {namespace}")
        else:
            _cleanup_rendered(run.rendered)
    return 0 if outcome == "Complete" else 1


def _render_run_config(config_dir: Path, config_file: Path | None,
                       feature_config_files: list[tuple], variables: dict[str, str]) -> Path | None:
    """Render the ``--config`` file/dir plus every feature's ``config/*`` into
    the run folder; returns the path that becomes ConfigMap ``rlbench-run-config``.

    Feature files land flat next to the run config (ConfigMap keys are flat);
    a feature may not shadow a run-config file or another feature's file."""
    rendered_config: Path | None = None
    if config_file:
        rendered_config = config_dir / config_file.name
        if config_file.is_dir():
            rendered_config.mkdir(exist_ok=True)
            for f in sorted(config_file.iterdir()):
                if f.is_file():
                    (rendered_config / f.name).write_text(render_text_lenient(f, variables))
        else:
            rendered_config.write_text(render_text_lenient(config_file, variables))
    elif feature_config_files:
        rendered_config = config_dir / "run-config"
        rendered_config.mkdir(exist_ok=True)
    for feat, c in feature_config_files:
        target = rendered_config / c.name
        if target.exists():
            raise ValueError(f"feature {feat.name!r} config file {c.name!r} collides with an existing run config file")
        target.write_text(render_text_lenient(c, variables))
    return rendered_config


def _cleanup_rendered(rendered_dir: Path) -> None:
    print("--> cleanup: deleting everything this run created")
    for p in sorted(rendered_dir.iterdir(), reverse=True):
        kube.delete(p.read_text())


def _load_run_folder(path: str) -> tuple[Path, str]:
    root = Path(path).resolve()
    rendered = root / "config" / "rendered"
    if not rendered.is_dir():
        print(f"error: {root} is not a run folder (no config/rendered/)")
        raise SystemExit(1)
    docs: list[dict] = []
    for p in rendered.iterdir():
        docs.extend(d for d in yaml.safe_load_all(p.read_text()) if d)
    return rendered, namespace_of(docs)


def cmd_cleanup(args: argparse.Namespace) -> int:
    rendered, _ = _load_run_folder(args.run_folder)
    _cleanup_rendered(rendered)
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    root = Path(args.run_folder).resolve()
    rendered, namespace = _load_run_folder(args.run_folder)
    run_id = root.name
    print(f"--> collecting from namespace {namespace} for run {run_id}")
    # refresh the recorded outcome from the live job: a watcher that died
    # mid-run (auth blip, session loss) leaves a stale/wrong result.json
    result_file = root / "result.json"
    try:
        jobs = [d for d in yaml.safe_load_all((rendered / "job.yaml").read_text()) if d]
        status = kube.get_json("job", jobs[0]["metadata"]["name"], "-n", namespace).get("status", {})
        for cond in status.get("conditions", []):
            if cond.get("status") == "True" and cond.get("type") in ("Complete", "Failed"):
                result = json.loads(result_file.read_text()) if result_file.exists() else {"run_id": run_id}
                result.update({"outcome": cond["type"], "job_status": status})
                result_file.write_text(json.dumps(result, indent=2))
                print(f"--> outcome refreshed from live job: {cond['type']}")
    except kube.KubectlError:
        pass  # job already gone; keep the recorded outcome
    kube.dump_events(namespace, root / "events")
    MetricsSampler(namespace, root / "metrics").sample_once()  # pod targets need the setup folder; service targets still work
    for pod in kube.run_pods(namespace, run_id):
        name = pod["metadata"]["name"]
        out = kube.kubectl("logs", name, "-n", namespace, "--all-containers",
                           "--prefix", "--timestamps", check=False)
        if out:
            (root / "logs" / f"{name}.log").write_text(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rlbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="set up, run the job, collect, clean up")
    p_run.add_argument("setup_folder")
    p_run.add_argument("--config", help="run config file, exposed in-cluster as ConfigMap 'rlbench-run-config'")
    p_run.add_argument("--feature", action="append", default=[], metavar="NAME",
                       help="enable <setup>/features/NAME (repeatable); recorded in config/features.json")
    p_run.add_argument("--keep", action="store_true", help="skip cleanup after the run")
    p_run.add_argument("--var", action="append", default=[], metavar="KEY=VALUE",
                       help="extra render-time variable for ${KEY} in manifests/configs (repeatable)")
    p_run.add_argument("--out", default="runs", help="parent folder for run folders")
    p_run.add_argument("--name", help="run name (default: setup folder name)")
    p_run.add_argument("--timeout", default=str(DEFAULT_TIMEOUT_S), help="job timeout, e.g. 90m, 12h")
    p_run.set_defaults(func=cmd_run)

    p_clean = sub.add_parser("cleanup", help="delete everything a previous run created")
    p_clean.add_argument("run_folder")
    p_clean.set_defaults(func=cmd_cleanup)

    p_collect = sub.add_parser("collect", help="(re-)collect artifacts from a live/finished run")
    p_collect.add_argument("run_folder")
    p_collect.set_defaults(func=cmd_collect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
