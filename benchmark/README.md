# RL Bench

A simple Python CLI to set up, run, and benchmark reinforcement learning training jobs on a Kubernetes cluster.

The tool is generic: it knows nothing about any particular RL framework. Everything specific to a setup (veRL, RLlib, prime-rl, custom, ...) lives in a *setup folder* that follows a small convention. The tool orchestrates: apply the setup, run the job, watch it, collect artifacts, and (optionally) clean up.

Any analysis done must always be rooted in the metrics and logs collected for a particular run.

## Usage

```
rlbench run <setup-folder> [--config <run-config>] [--feature <name>]... [--keep] [--out runs/] [--timeout 12h]
rlbench cleanup <run-folder>            # remove everything a previous run created
rlbench collect <run-folder>            # (re-)collect artifacts from a live/finished run
```

- Operates on whatever cluster `kubectl` currently points at.
- `--keep` skips cleanup after the run (default: clean up everything the tool created).
- `--feature <name>` (repeatable) opts the run into `<setup>/features/<name>/` — see *Features* below. Everything else about the run stays identical, which is what makes baseline-vs-feature runs comparable.
- Every run gets a run folder `runs/<timestamp>-<name>/` — the single source of truth for analysis. See `runs/README.md`; run folders are never committed (PII) and are backed up externally.

## Setup folder contract

A setup folder defines one type of RL infrastructure. Convention:

```
my-setup/
  setup/          # k8s manifests applied (in filename order) before the job
  job.yaml        # exactly one k8s Job (Indexed Jobs fine) — its Complete/Failed is the run's outcome
  config/         # optional run configs to pass via --config
  hooks/          # optional: pre-setup.sh, post-run.sh escape hatches
  provision.sh    # optional one-time cluster prep (never run by rlbench)
  scrape-targets.txt  # optional extra metrics targets (pods by label selector)
  features/<name>/    # optional opt-in features, enabled per run with --feature <name>
  README.md       # what this setup is and which variables it needs
```

Rules the tool enforces or relies on:

- **Labeling**: every applied object (and its pod template) is labeled `rlbench/run=<run-id>`, so collection and cleanup never depend on the setup folder being tidy.
- **Templating**: `${NAME}` and `${NAME:-default}` are render-time variables (environment, `--var KEY=VALUE`, built-ins `RUN_ID`/`RUN_NAME`); bare `$name` is never touched, so shell scripts work as run configs. Manifests render strictly (unresolved `${NAME}` fails before the cluster is touched); `--config` files render leniently and the rendered copy in the run folder is exactly what ran.
- **Portability**: nothing in a setup folder may name a specific cluster, project, zone, registry, or user. Cluster identity is resolved at run time (env + current kubectl context) and recorded only in the run folder.
- **Run config**: the `--config` file is copied into the run folder and exposed in-cluster as ConfigMap `rlbench-run-config` in the run's namespace.
- **Metrics scraping**: any Service labeled `rlbench/scrape=true` is scraped through the API-server proxy every sampling interval; a setup may also list pod targets in `scrape-targets.txt` (`pods <namespace> <label-selector> <port> <path>`), e.g. a cluster-managed DCGM exporter, re-resolved each interval.

## Features (A/B toggles)

A *feature* is a named, self-contained unit a run opts into so that its impact can be measured against the same setup and run config without it. Each lives in `<setup>/features/<name>/`:

```
features/<name>/
  README.md            # what it is, what it needs, how to verify from the run folder that it was active
  vars.env             # KEY=VALUE render-time variables; precedence: environment < vars.env < --var
  setup/*.yaml         # optional extra manifests, rendered + labeled like the setup's, applied after them
  config/*             # optional files copied flat into the run config dir (ConfigMap keys are flat);
                       #   convention: a feature-<name>.sh fragment the job's entrypoint sources
  hooks/pre-setup.sh, hooks/post-run.sh   # optional, run after the setup's hooks
  scrape-targets.txt   # optional, merged into the setup's targets
  report.py            # optional: evidence(run_folder) -> dict for the setup's comparison tools
```

Rules:

- rlbench sets the built-ins `RLBENCH_FEATURES` (comma-joined names, `""` when none) and `FEATURE_<NAME>=1` (upper-case, `-`→`_`) for every enabled feature, so base manifests and run configs can use `${FEATURE_X:-0}`.
- Base manifests stay a superset with `${VAR:-default}` knobs; a feature flips knobs through `vars.env` and may **add** objects, but never patches base objects.
- Feature manifests render to `config/rendered/feature-<name>-<file>`, which sorts after the numbered base files: applied last, deleted first.
- A feature's `config/*` may not shadow a run-config file or another feature's file; with several features, later `vars.env` values win.
- Provenance: `config/features.json` (name, path, resolved vars) and a `features` list in `result.json`. Name runs explicitly (`--name <rung>-<feature>`).
- **Observability needed to measure a feature is never part of the feature** — it belongs to the setup, so both arms of a comparison carry identical instrumentation and differ only in the feature itself.

## What gets collected

Into `runs/<run-id>/` (see `runs/README.md` for the full layout):

- `config/` — run config, rendered manifests, setup-folder git SHA, resolved cluster identity, enabled features (`features.json`)
- `logs/` — logs from every pod belonging to the run, *streamed* so they survive pod deletion and spot preemption
- `events/` — namespace events plus Node events (spot preemptions surface there)
- `metrics/` — timestamped `kubectl top` samples and scrapes of `rlbench/scrape=true` services
- `result.json` — outcome (Complete/Failed/Timeout), phase timings, job status

Training-quality metrics (reward curves, step counts) are whatever the job itself writes to stdout/files — captured via logs, or copied out by a `post-run.sh` hook.

## Lifecycle

1. render — all manifests (setup, then features) rendered and written to the run folder first; template errors and unknown features fail before the cluster is touched
2. setup — apply `setup/` manifests, then feature manifests, wait for Deployments to become ready
3. run — submit `job.yaml`, poll until Complete/Failed/timeout; logs and metrics stream throughout
4. collect — finalized at the end (`rlbench collect` can re-run it anytime). If the watcher process dies mid-run (lost credentials, host suspend) the in-cluster Job keeps going; afterwards `rlbench collect <run-folder>` refreshes the outcome and pod logs, and the setup's `post-run.sh` can be run by hand with `RUN_ID`, `NAMESPACE`, `RUN_FOLDER` set to pull the shared-volume artifacts (done for `20261005-161446`; only the periodic metrics sampled before the watcher died are available)
5. cleanup — `kubectl delete` of the rendered manifests, reverse order (skipped with `--keep`)

The cluster itself is never touched beyond what the tool created; one-time infrastructure (node pools, addons, registries) belongs in a setup's `provision.sh`, run manually.
