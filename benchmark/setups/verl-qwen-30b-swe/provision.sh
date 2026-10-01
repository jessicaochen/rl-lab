#!/usr/bin/env bash
# One-time cluster preparation for the verl-qwen-30b-swe rlbench setup.
#
# Idempotent: safe to re-run; each step checks state before mutating.
# Portable: all cluster identity comes from env vars / active gcloud config —
# never hardcode a project, cluster, zone, or registry path in this repo.
#
# Required env:
#   CLUSTER    target GKE cluster name
#   LOCATION   cluster zone or region
# Optional env (defaults shown):
#   PROJECT                 active gcloud project
#   GPU_POOLS               "trainer-gpu-pool sampler-gpu-pool"  (all pools that may host Ray GPU workers)
#   SANDBOX_POOL            sandbox-pool
#   SANDBOX_MACHINE_TYPE    n4-standard-32
#   SANDBOX_NODES           10
#   SANDBOX_DISK_TYPE       hyperdisk-balanced
#   SANDBOX_DISK_GB         1000
#   SANDBOX_SPOT            true
#   SANDBOX_NODE_LOCATIONS  (unset) sibling zone(s) when the cluster zone lacks capacity
#   AGENT_SANDBOX_VERSION   v1.0.1   (OSS kubernetes-sigs/agent-sandbox release)
#   AR_REPO                 rlbench  (Artifact Registry docker repo for our images)
#   CKPT_BUCKET             ${PROJECT}-rlbench-ckpt
#   RUN_NAMESPACE           rlbench-verl-swe   (namespace whose KSA gets bucket access)
#   RUN_KSA                 verl-driver
set -euo pipefail

: "${CLUSTER:?set CLUSTER to the target GKE cluster name}"
: "${LOCATION:?set LOCATION to the cluster zone or region}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-$(sed -E 's/-[a-z]$//' <<<"$LOCATION")}"
GPU_POOLS="${GPU_POOLS:-trainer-gpu-pool sampler-gpu-pool}"
SANDBOX_POOL="${SANDBOX_POOL:-sandbox-pool}"
SANDBOX_MACHINE_TYPE="${SANDBOX_MACHINE_TYPE:-n4-standard-32}"
SANDBOX_NODES="${SANDBOX_NODES:-10}"
SANDBOX_DISK_TYPE="${SANDBOX_DISK_TYPE:-hyperdisk-balanced}"
SANDBOX_DISK_GB="${SANDBOX_DISK_GB:-1000}"
SANDBOX_SPOT="${SANDBOX_SPOT:-true}"
AGENT_SANDBOX_VERSION="${AGENT_SANDBOX_VERSION:-v1.0.1}"
AR_REPO="${AR_REPO:-rlbench}"
CKPT_BUCKET="${CKPT_BUCKET:-${PROJECT}-rlbench-ckpt}"
RUN_NAMESPACE="${RUN_NAMESPACE:-rlbench-verl-swe}"
RUN_KSA="${RUN_KSA:-verl-driver}"

gc() { gcloud --project "$PROJECT" --quiet "$@"; }
describe_cluster() { gc container clusters describe "$CLUSTER" --location "$LOCATION" --format="value($1)"; }
step() { printf '\n==> %s\n' "$*"; }

step "Target: cluster=$CLUSTER location=$LOCATION project=$PROJECT"
gc container clusters get-credentials "$CLUSTER" --location "$LOCATION"
project_number="$(gc projects describe "$PROJECT" --format='value(projectNumber)')"

# --- OSS agent-sandbox (NOT the GKE managed addon) -------------------------
# The managed addon's admission policy forces runAsNonRoot on Sandbox CRs,
# which breaks SWE task images that require root. The OSS install ships the
# same CRDs (v1beta1) + an in-cluster controller, with no such policy.
step "OSS agent-sandbox $AGENT_SANDBOX_VERSION"
if [[ "$(describe_cluster addonsConfig.agentSandboxConfig.enabled)" == "True" ]]; then
  echo "disabling the GKE managed addon (conflicts with the OSS install)"
  gc beta container clusters update "$CLUSTER" --location "$LOCATION" --no-enable-agent-sandbox
  # the addon leaves orphaned v1alpha1 CRDs + its admission policy behind
  kubectl delete validatingadmissionpolicybinding secure-sandbox-binding --ignore-not-found
  kubectl delete validatingadmissionpolicy secure-sandbox-policy --ignore-not-found
  kubectl delete crd sandboxes.agents.x-k8s.io sandboxclaims.extensions.agents.x-k8s.io \
    sandboxtemplates.extensions.agents.x-k8s.io sandboxwarmpools.extensions.agents.x-k8s.io --ignore-not-found
fi
kubectl apply --server-side -f \
  "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/sandbox-with-extensions.yaml"
kubectl -n agent-sandbox-system rollout status deployment/agent-sandbox-controller --timeout=300s

step "Filestore CSI driver (RWX shared volume)"
if [[ "$(describe_cluster addonsConfig.gcpFilestoreCsiDriverConfig.enabled)" != "True" ]]; then
  gc container clusters update "$CLUSTER" --location "$LOCATION" --update-addons=GcpFilestoreCsiDriver=ENABLED
fi

step "Workload Identity + GCS FUSE CSI (checkpoints to GCS)"
if [[ -z "$(describe_cluster workloadIdentityConfig.workloadPool)" ]]; then
  gc container clusters update "$CLUSTER" --location "$LOCATION" --workload-pool="${PROJECT}.svc.id.goog"
fi
if [[ "$(describe_cluster addonsConfig.gcsFuseCsiDriverConfig.enabled)" != "True" ]]; then
  gc container clusters update "$CLUSTER" --location "$LOCATION" --update-addons=GcsFuseCsiDriver=ENABLED
fi
# EVERY pool that may host a Ray GPU worker needs the GKE metadata server:
# Ray places verl's trainer/rollout pools by GPU count, not by node pool, and
# the GCS FUSE mount fails on nodes without it. Flipping the mode rolls nodes.
for pool in $GPU_POOLS; do
  mode="$(gc container node-pools describe "$pool" --cluster "$CLUSTER" --location "$LOCATION" \
          --format='value(config.workloadMetadataConfig.mode)')"
  if [[ "$mode" != "GKE_METADATA" ]]; then
    gc container node-pools update "$pool" --cluster "$CLUSTER" --location "$LOCATION" --workload-metadata=GKE_METADATA
  else
    echo "$pool: already GKE_METADATA"
  fi
done

step "gVisor sandbox node pool: $SANDBOX_POOL ($SANDBOX_MACHINE_TYPE x$SANDBOX_NODES, spot=$SANDBOX_SPOT)"
if gc container node-pools describe "$SANDBOX_POOL" --cluster "$CLUSTER" --location "$LOCATION" >/dev/null 2>&1; then
  echo "already exists"
else
  flags=()
  [[ "$SANDBOX_SPOT" == "true" ]] && flags+=(--spot)
  [[ -n "${SANDBOX_NODE_LOCATIONS:-}" ]] && flags+=(--node-locations "$SANDBOX_NODE_LOCATIONS")
  gc container node-pools create "$SANDBOX_POOL" \
    --cluster "$CLUSTER" --location "$LOCATION" \
    --machine-type "$SANDBOX_MACHINE_TYPE" "${flags[@]}" \
    --num-nodes "$SANDBOX_NODES" \
    --sandbox type=gvisor \
    --image-type cos_containerd \
    --disk-type "$SANDBOX_DISK_TYPE" \
    --disk-size "$SANDBOX_DISK_GB"
  # note: 10 x 1TB boot disks may exceed the regional DISKS_TOTAL_GB quota;
  # request a bump via `gcloud beta quotas preferences create` if creation fails
fi

step "Checkpoint bucket: gs://$CKPT_BUCKET (hierarchical namespace)"
if ! gc storage buckets describe "gs://$CKPT_BUCKET" >/dev/null 2>&1; then
  gc storage buckets create "gs://$CKPT_BUCKET" --location "$REGION" \
    --uniform-bucket-level-access --enable-hierarchical-namespace
fi
gc storage buckets add-iam-policy-binding "gs://$CKPT_BUCKET" \
  --member="principal://iam.googleapis.com/projects/${project_number}/locations/global/workloadIdentityPools/${PROJECT}.svc.id.goog/subject/ns/${RUN_NAMESPACE}/sa/${RUN_KSA}" \
  --role=roles/storage.objectAdmin >/dev/null
echo "objectAdmin granted to ${RUN_NAMESPACE}/${RUN_KSA}"

step "Artifact Registry: $AR_REPO (our images) + dockerhub-cache (pull-through) in $REGION"
for repo in "$AR_REPO" dockerhub-cache; do
  if ! gc artifacts repositories describe "$repo" --location "$REGION" >/dev/null 2>&1; then
    if [[ "$repo" == dockerhub-cache ]]; then
      gc artifacts repositories create dockerhub-cache --location "$REGION" \
        --repository-format=docker --mode=remote-repository --remote-docker-repo=DOCKER-HUB \
        --description="pull-through cache for Docker Hub task images"
    else
      gc artifacts repositories create "$repo" --location "$REGION" --repository-format=docker --description="rlbench images"
    fi
  fi
  gc artifacts repositories add-iam-policy-binding "$repo" --location "$REGION" \
    --member="serviceAccount:${project_number}-compute@developer.gserviceaccount.com" \
    --role=roles/artifactregistry.reader >/dev/null
done

step "Acceptance checks"
kubectl get crds | grep 'agents.x-k8s.io' || { echo "ERROR: agent-sandbox CRDs missing"; exit 1; }
kubectl wait --for=condition=Ready node -l cloud.google.com/gke-nodepool="$SANDBOX_POOL" --timeout=300s
kubectl get storageclass | grep -q rwx || echo "WARN: no *-rwx storage class yet (Filestore CSI may still be rolling out)"
kubectl get crd rayclusters.ray.io >/dev/null 2>&1 || echo "WARN: KubeRay CRDs not found — install the kuberay-operator (verl runs on Ray)"

step "Provisioning complete"
