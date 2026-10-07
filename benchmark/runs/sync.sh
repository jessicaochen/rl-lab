#!/usr/bin/env bash
# Two-way backup of recorded run folders against a GCS bucket.
#
#   RUNS_BUCKET=<bucket> [PROJECT=<gcp-project>] ./sync.sh [--dry-run]
#
# - pushes every local run that is not in the bucket
# - pulls every bucket run that is not here
# - never deletes anything on either side; runs present on both are left alone
#
# A run counts as present only if its result.json exists: rlbench writes that
# file last, so an in-progress run is never pushed, and this script copies it
# last too, so an interrupted transfer leaves no marker and is simply resumed
# (rsync is idempotent) on the next invocation.
#
# Required env:
#   RUNS_BUCKET  bucket holding the backups ("gs://" prefix optional)
# Optional env (defaults shown):
#   PROJECT      active gcloud project
#
# Bucket and project names are deliberately NOT in the repo: run folders carry
# PII and cluster identifiers, and the repo stays portable across clusters.
set -euo pipefail

: "${RUNS_BUCKET:?set RUNS_BUCKET to the bucket that holds run backups}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
BUCKET="gs://${RUNS_BUCKET#gs://}"
BUCKET="${BUCKET%/}"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "error: unknown argument $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")"
gc() { gcloud --project "$PROJECT" "$@"; }
RUN_ID_RE='^[0-9]{8}-[0-9]{6}-'

# complete local runs: a run dir with a result.json
local_ids=$(for f in */result.json; do
  [ -f "$f" ] || continue
  d="${f%/result.json}"
  [[ "$d" =~ $RUN_ID_RE ]] && echo "$d"
done | sort)

# complete remote runs: "matched no objects" means an empty bucket, not an error
remote_err=$(mktemp); trap 'rm -f "$remote_err"' EXIT
if remote_ls=$(gc storage ls "$BUCKET/*/result.json" 2>"$remote_err"); then
  :
elif grep -q "matched no objects" "$remote_err"; then
  remote_ls=""
else
  cat "$remote_err" >&2
  echo "error: could not list $BUCKET" >&2
  exit 1
fi
remote_ids=$(printf '%s\n' "$remote_ls" | sed -nE "s#^$BUCKET/([^/]+)/result\.json\$#\1#p" | sort)

push=$(comm -23 <(printf '%s\n' "$local_ids") <(printf '%s\n' "$remote_ids") | sed '/^$/d')
pull=$(comm -13 <(printf '%s\n' "$local_ids") <(printf '%s\n' "$remote_ids") | sed '/^$/d')
both=$(comm -12 <(printf '%s\n' "$local_ids") <(printf '%s\n' "$remote_ids") | sed '/^$/d')
count() { [ -z "$1" ] && echo 0 || printf '%s\n' "$1" | wc -l; }

echo "bucket: $BUCKET (project $PROJECT)"
echo "local $(count "$local_ids"), remote $(count "$remote_ids"), push $(count "$push"), pull $(count "$pull"), both $(count "$both")"
[ -n "$push" ] && printf '  push %s\n' $push
[ -n "$pull" ] && printf '  pull %s\n' $pull
if [ "$DRY_RUN" = 1 ]; then
  echo "--dry-run: nothing transferred"
  exit 0
fi

# transfer: everything but the marker, then the marker
transfer() {  # src dst
  gc storage rsync -r --exclude='^result\.json$' "$1" "$2"
  gc storage cp "$1/result.json" "$2/result.json"
}
n=0; total=$(count "$push")
for id in $push; do
  n=$((n + 1)); echo "--> push $id ($n of $total)"
  transfer "$id" "$BUCKET/$id"
done
n=0; total=$(count "$pull")
for id in $pull; do
  n=$((n + 1)); echo "--> pull $id ($n of $total)"
  mkdir -p "$id"
  transfer "$BUCKET/$id" "$id"
done
echo "--> done"
