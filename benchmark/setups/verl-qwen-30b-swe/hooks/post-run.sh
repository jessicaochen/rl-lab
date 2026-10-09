#!/usr/bin/env bash
# rlbench post-run hook: capture cloud-side evidence that k8s events lose when
# nodes are replaced (auto-upgrades, node-pool ops, preemptions). Cluster
# identity is derived from the active kubectl context — never hardcoded — and
# lands only in the run folder, per the portability rule.
#   env from rlbench: RUN_ID, NAMESPACE, RUN_FOLDER
set -uo pipefail
out="${RUN_FOLDER:?}/events"
mkdir -p "$out"
ctx="$(kubectl config current-context 2>/dev/null || true)"
# GKE context names look like gke_<project>_<location>_<cluster>
if [[ "$ctx" =~ ^gke_([^_]+)_([^_]+)_(.+)$ ]]; then
  project="${BASH_REMATCH[1]}" location="${BASH_REMATCH[2]}" cluster="${BASH_REMATCH[3]}"
  gcloud container operations list --project "$project" --location "$location" \
    --filter="targetLink~${cluster} AND startTime>-P1D" --sort-by=startTime \
    --format=json > "$out/gke-operations.json" 2>/dev/null || true
  gcloud container clusters describe "$cluster" --project "$project" --location "$location" \
    --format="json(currentMasterVersion,releaseChannel,maintenancePolicy,nodePools[].name,nodePools[].version,nodePools[].management)" \
    > "$out/gke-cluster.json" 2>/dev/null || true
fi
# node inventory at run end (names + creation times reveal replacements)
kubectl get nodes -o custom-columns='NODE:.metadata.name,CREATED:.metadata.creationTimestamp,VERSION:.status.nodeInfo.kubeletVersion,SPOT:.metadata.labels.cloud\.google\.com/gke-spot' \
  > "$out/nodes-at-end.txt" 2>/dev/null || true
# per-episode uni-agent session logs live only on the shared volume
# (/data/outputs/<RUN_ID>/agent-logs); pull them into the run folder so they
# survive teardown of the PVC and analysis stays rooted in the run folder
head="$(kubectl get pods -n "${NAMESPACE:?}" -l "ray.io/node-type=head,rlbench.timeslice.io/job-id=${JOB_ID:-job1}" -o name 2>/dev/null | head -1)"
if [ -z "$head" ]; then
  head="$(kubectl get pods -n "${NAMESPACE:?}" -l ray.io/node-type=head -o name 2>/dev/null | head -1)"
fi
if [ -n "$head" ] && kubectl exec -n "$NAMESPACE" "${head#pod/}" -c ray-head -- test -d "/data/outputs/${RUN_ID:?}/agent-logs" 2>/dev/null; then
  mkdir -p "${RUN_FOLDER}/logs"
  kubectl exec -n "$NAMESPACE" "${head#pod/}" -c ray-head -- tar czf - -C "/data/outputs/${RUN_ID}" agent-logs \
    > "${RUN_FOLDER}/logs/agent-logs.tar.gz" 2>/dev/null && echo "post-run: saved agent-logs.tar.gz ($(du -h "${RUN_FOLDER}/logs/agent-logs.tar.gz" | cut -f1))"
fi
# placement snapshot: Ray places verl's trainer/rollout pools by GPU count,
# so roles must be read per run. DCGM samples join on Hostname = node name
# (GKE's managed exporter); dcgm_pods is kept for pod-name joins.
python3 - "$NAMESPACE" "$RUN_FOLDER" "${head#pod/}" "$RUN_ID" <<'PY'
import json, re, subprocess, sys, glob, collections
ns, rf = sys.argv[1], sys.argv[2]
def pods(sel, in_ns=ns):
    out = subprocess.run(["kubectl","get","pods","-n",in_ns,"-l",sel,"-o","json"], capture_output=True, text=True).stdout
    items = json.loads(out).get("items", []) if out else []
    return [{"pod": i["metadata"]["name"], "node": i["spec"].get("nodeName"), "ip": i["status"].get("podIP"),
             "group": i["metadata"]["labels"].get("ray.io/group") or i["metadata"]["labels"].get("ray.io/node-type")} for i in items]
roles = collections.defaultdict(collections.Counter)
for log in glob.glob(f"{rf}/logs/verl-train-*.log"):
    for m in re.finditer(r"\((WorkerDict|vLLMHttpServer|_GatewayActor|TaskRunnerV1) pid=\d+(?:, ip=([0-9.]+))?\)", open(log, errors="replace").read()):
        roles[m.group(2) or "driver-node"][m.group(1)] += 1
snap = {"ray_pods": pods("app=verl-ray"), "dcgm_pods": pods("app.kubernetes.io/name=gke-managed-dcgm-exporter", "gke-managed-system"), "actor_counts_by_ip": {k: dict(v) for k, v in roles.items()}}
# vLLM replicas (server id = host:port as verl's balancer names them) -> node, from the
# poller's replicas.json on the shared volume (fetched before this snapshot runs)
replicas = {}
try:
    out = subprocess.run(["kubectl", "exec", "-n", ns, sys.argv[3], "-c", "ray-head", "--", "cat", f"/data/outputs/{sys.argv[4]}/replica-metrics/replicas.json"],
                         capture_output=True, text=True).stdout
    replicas = json.loads(out) if out.strip() else {}
except Exception:
    replicas = {}
# role per node: 'trainer' if it hosts WorkerDict actors, else 'sampler' if it hosts vLLM servers
ip_to_node = {p["ip"]: p["node"] for p in snap["ray_pods"] if p["ip"]}
node_roles = {}
for ip, c in roles.items():
    node = ip_to_node.get(ip)
    if not node: continue
    node_roles[node] = "trainer" if c.get("WorkerDict") else ("sampler" if c.get("vLLMHttpServer") else "other")
snap["node_roles"] = node_roles
snap["replicas"] = [{"server_id": sid, "host": a.get("host"), "port": a.get("port"), "node": ip_to_node.get(a.get("host")),
                     "first_seen": a.get("first_seen"), "last_seen": a.get("last_seen")} for sid, a in replicas.items()]
json.dump(snap, open(f"{rf}/events/placement.json", "w"), indent=2)
print("post-run: placement.json", {n: r for n, r in node_roles.items()})
PY
# verl rollout / validation dumps (readable per-step JSONL), the gateway's
# per-request records and the per-replica vLLM /metrics snapshots (both written
# by rlbench_verl_provider.rollout_adapter) live on the shared volume too
if [ -n "$head" ]; then
  # app-offload: JSONL of the external offload controller (features/app-offload)
  for d in rollouts val-rollouts gateway-logs replica-metrics app-offload; do
    if kubectl exec -n "$NAMESPACE" "${head#pod/}" -c ray-head -- test -d "/data/outputs/${RUN_ID}/$d" 2>/dev/null; then
      kubectl exec -n "$NAMESPACE" "${head#pod/}" -c ray-head -- tar czf - -C "/data/outputs/${RUN_ID}" "$d" \
        > "${RUN_FOLDER}/logs/$d.tar.gz" 2>/dev/null && echo "post-run: saved $d.tar.gz ($(du -h "${RUN_FOLDER}/logs/$d.tar.gz" | cut -f1))"
    fi
  done
fi
echo "post-run: wrote gke-operations.json, gke-cluster.json, nodes-at-end.txt"
