# Recorded runs

Every `rlbench run` creates one folder here — the single source of truth for
analyzing that run. Any analysis must be rooted in the artifacts of a
particular run folder, never in ad-hoc cluster inspection after the fact.

## Never commit run folders

Recorded runs are **not checked in** (enforced by the `.gitignore` in this
directory): logs and metrics can contain PII and cluster identifiers
(project ids, node names, usernames). Runs are backed up to external storage
instead — the repo only carries this README and the ignore rules.

## Expected structure

```
runs/<timestamp>-<name>/
├── config/         # exactly what ran: resolved run config, rendered manifests,
│                   # setup-folder git SHA, resolved cluster identity (project,
│                   # cluster, node pools, image digests), features.json
│                   # (which --feature toggles were on, with their resolved vars)
├── logs/           # <pod>.log — streamed from every rlbench-labeled pod, so
│                   # they survive pod deletion and spot preemption
├── events/         # kubernetes events for the run's objects, incl. node
│                   # preemptions
├── metrics/        # timestamped samples: DCGM GPU metrics, kubectl top
├── logs/agent-logs.tar.gz      # (verl setup) per-episode session logs + token-level trajectories
├── logs/rollouts.tar.gz        # (verl setup) verl per-step rollout dumps (readable JSONL)
├── logs/val-rollouts.tar.gz    # (verl setup) validation rollout dumps
├── logs/gateway-logs.tar.gz    # (verl setup) one JSON record per policy request at the gateway:
│                               # session, replica, prompt/completion tokens, latency, resumes, router decision
├── logs/replica-metrics.tar.gz # (verl setup) periodic vLLM /metrics snapshots per replica (TTFT, queue
│                               # time, prefix-cache hits, KV usage, tokens)
├── events/placement.json       # (verl setup) Ray pod->node->role + DCGM pod->node (duty-cycle join key)
├── events/gke-*.json           # (verl setup) GKE operations + cluster/node-pool state at run end
└── result.json     # outcome (Complete/Failed/Timeout), wall clock, UTC phase
                    # timestamps, enabled features, k8s job status
```

Cluster-identifying values live **only** here (under `config/`) — setup
folders and scripts stay portable and carry no reference to any particular
cluster.
