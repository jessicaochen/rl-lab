#!/usr/bin/env bash
# Apply / check / revert the temporary verl instrumentation in the LIVE Ray pods of the
# verl setup (no image rebuild). verl is an editable install at /opt/verl
# (release/v0.9.0 @ adc7eefa) inside the driver image; Ray starts fresh actor processes
# per job, so patched source is picked up by the next `rlbench run` without restarting
# pods. The driver runs on the head pod, actors on the workers -> all pods get the patch.
#
#   patch-pods.sh apply  [namespace]   # kubectl cp + git apply (refuses if it does not apply cleanly)
#   patch-pods.sh check  [namespace]   # HEAD sha + git status of /opt/verl in every pod
#   patch-pods.sh revert [namespace]   # git checkout -- . + remove the added file
#   PATCH_FILE=<other.patch> PATCH_NEW_FILES="" patch-pods.sh apply   # reuse with another patch
set -euo pipefail
ACTION=${1:?usage: patch-pods.sh apply|check|revert [namespace]}
NS=${2:-rlbench-verl-swe}
HERE="$(cd "$(dirname "$0")" && pwd)"
PATCH="${PATCH_FILE:-$HERE/verl-disagg-memlog.patch}"   # override: PATCH_FILE=<path> (experiments/trainer-app-offload reuses this script)
EXPECTED_SHA=adc7eefa16dad75c5f7b878823d5a76eac90c7b3
NEW_FILES="${PATCH_NEW_FILES-verl/utils/gpu_mem_log.py}"   # files the patch ADDS (removed on revert); PATCH_NEW_FILES="" if none

pods=$(kubectl get pods -n "$NS" -l app=verl-ray -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
[ -n "$pods" ] || { echo "no app=verl-ray pods in namespace $NS" >&2; exit 1; }

for pod in $pods; do
  case "$pod" in verl-head-*) c=ray-head ;; *) c=ray-worker ;; esac
  ex() { kubectl exec -n "$NS" "$pod" -c "$c" -- "$@"; }
  ex git config --global --add safe.directory /opt/verl >/dev/null 2>&1 || true
  case "$ACTION" in
    apply)
      kubectl cp "$PATCH" "$NS/$pod:/tmp/verl-exp.patch" -c "$c"
      ex git -C /opt/verl apply --check /tmp/verl-exp.patch
      ex git -C /opt/verl apply /tmp/verl-exp.patch
      ;;
    revert)
      ex git -C /opt/verl checkout -- .
      ex sh -c "cd /opt/verl && rm -f $NEW_FILES"
      ;;
    check) ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
  esac
  sha=$(ex git -C /opt/verl rev-parse HEAD | tr -d '\r')
  status=$(ex git -C /opt/verl status --short | tr -d '\r')
  if [ "$sha" = "$EXPECTED_SHA" ]; then ok=ok; else ok="UNEXPECTED (want $EXPECTED_SHA)"; fi
  echo "== $pod ($c): HEAD=${sha:0:8} $ok"
  echo "${status:-   clean}"
done
