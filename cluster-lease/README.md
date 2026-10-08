# cluster-lease

One agent on the hardware at a time. The cluster this repo targets is a shared
GPU cluster, and several agents (people, Claude sessions, CI jobs) may want it.
This directory holds a tiny CLI, `lease`, that lets them take turns. A single
`Lease` object (`coordination.k8s.io/v1`) in the well-known namespace
`cluster-lease` says who holds the hardware right now.

**If you are an agent about to touch the cluster: read the next section and run
`./lease acquire` before anything else.** Nothing enforces this. It works only
because every agent does it.

## The contract

1. **Acquire before you touch anything.** Before `kubectl apply`, `rlbench run`,
   `provision.sh`, hot patches, probe pods, anything that uses nodes:

   ```sh
   cd cluster-lease
   LEASE_HOLDER="<you>/<session>" ./lease acquire --purpose "what you are about to do"
   ```

   Exit `0`: you hold it, go ahead. Exit `2`: someone else holds it. The output
   tells you who, since when, and why. **Do not proceed. Do not delete their
   resources.** Either wait (`--wait 3600` polls for up to an hour) or contact
   them. Exit `3`: you hold it, but the cluster is dirty (see step 4).

2. **Keep it alive.** `acquire` starts a background renew loop for you, tied to
   the shell that ran it. If your process dies, renewals stop and the lease
   expires within `LEASE_TTL` (default 15 minutes), so a crashed agent cannot
   block the cluster. If you cannot run background processes, pass
   `--no-renew` and call `./lease renew` at least every `TTL/3` seconds yourself.

3. **Clean up, then release.**

   ```sh
   # tear down everything you created: rlbench cleanup, probe pods, patches, PVCs
   ./lease release
   ```

   `release` scans the cluster for hardware-using resources (GPU pods outside
   system namespaces, RayClusters, RayJobs, Sandboxes). If any remain it refuses
   with exit `3` and lists them. Remove them and run it again. `--force`
   releases anyway and records the leftovers on the lease under your name.

4. **Dirty cluster on acquire.** If `acquire` finds leftovers it still gives you
   the lease (you now have the exclusive right to touch the hardware) but exits
   `3` and names the previous holder. You may delete those leftovers, since
   nobody else can legitimately be using them. Then start your work. If the
   leftovers are deliberately yours from an earlier session, pass
   `--adopt-dirty`.

5. **Never edit the Lease object by hand**, and never `kubectl delete` it to
   "fix" a blocked state. If you believe a holder is dead, wait for expiry (at
   most 15 minutes after their last renewal). `./lease status` shows the
   countdown. Taking over an expired lease is automatic.

## First use on a new cluster

Nothing to install. Any `lease` command checks for the `cluster-lease`
namespace and creates it, plus a `Role` named `cluster-lease-user`, if missing.
The Lease object itself is created on the first `acquire`. Provisioning is
idempotent and race-safe, so two agents starting at once is fine. `./lease
init` does only the provisioning if you want it explicit.

Identities that are not cluster admins need the Role bound to them:

```sh
kubectl create rolebinding <name> -n cluster-lease --role=cluster-lease-user --user=<identity>
```

They also need cluster-wide read on pods and on the CRDs listed in
`LEASE_CHECK_KINDS` for the cleanup scan. The built-in `view` ClusterRole covers
pods; CRDs may need their own read rule.

## Commands

| Command | Does | Exit codes |
|---|---|---|
| `lease status` | holder, purpose, expiry, cleanliness; provisions if needed | `0` free or yours, `2` held by another |
| `lease acquire --purpose "..."` | take the lease, start renew loop, scan for leftovers | `0` ok, `2` held by another, `3` yours but dirty |
| `lease renew [--loop]` | heartbeat once, or forever | `0` ok |
| `lease release [--force]` | verify cleanup, free the lease, stop renew loop | `0` ok, `2` not yours, `3` dirty |
| `lease check` | list hardware-using leftovers only | `0` clean, `3` dirty |
| `lease init` | provision namespace and Role only | `0` |

Options on `acquire`: `--wait SECONDS` to queue, `--adopt-dirty`, `--no-renew`,
`--watch-pid PID` to tie the renew loop to a process other than your shell
(`0` disables the tie; the loop then relies on `LEASE_MAX_HOLD`).

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `LEASE_HOLDER` | `$USER@hostname` | Who you are. Make it findable: `jane/claude-job-1a2b`, `ci/run-4412`. Shown to anyone you block. |
| `LEASE_CONTACT` | empty | Free text: chat handle, email. Shown in `status`. |
| `LEASE_TTL` | `900` | Seconds without renewal before a holder is considered dead. |
| `LEASE_MAX_HOLD` | `43200` | Renew loop stops after this many seconds, so a forgotten session expires within `TTL` after 12 h. |
| `LEASE_NAMESPACE` / `LEASE_NAME` | `cluster-lease` / `hardware` | Where the Lease lives. One Lease per hardware pool if you ever split. |
| `LEASE_SYSTEM_NS` | `kube-,gke-,gmp-,cluster-lease` | Namespace prefixes ignored by the cleanup scan. |
| `LEASE_CHECK_KINDS` | `rayclusters,rayjobs,sandboxes` | Extra kinds the cleanup scan lists. Kinds whose CRD is absent are skipped. |
| `KUBECTL` | `kubectl` | Binary to use. The active kubeconfig context is the cluster. |

Renew loop state (pid file, log) lives in `$XDG_RUNTIME_DIR/cluster-lease/`,
falling back to `~/.cache/cluster-lease/`.

## Using it from rlbench hooks

The setup's `hooks/pre-setup.sh` and `hooks/post-run.sh` are the natural
places. The pre-setup hook's shell exits as soon as the hook finishes, so tie
the loop to the rlbench process instead:

```sh
# pre-setup.sh, first lines
"$REPO/cluster-lease/lease" acquire --purpose "rlbench ${RUN_ID}" --watch-pid "$PPID" || {
  rc=$?; [ "$rc" = 3 ] && echo "pre-setup: cluster dirty, cleaning up leftovers" || exit "$rc"; }

# post-run.sh, last line (after the setup's own teardown)
"$REPO/cluster-lease/lease" release || "$REPO/cluster-lease/lease" release --force
```

## How it works

- **Atomic acquire.** A missing Lease is created with `kubectl create`, which
  the API server rejects with `AlreadyExists` for everyone but the first caller.
- **Compare-and-swap takeover.** A free or expired Lease is taken with
  `kubectl replace` on a manifest carrying the `resourceVersion` just read. If
  another agent replaced it first the API server returns `Conflict`, the loser
  re-reads and sees a live holder. Two agents cannot both win.
- **Expiry.** A holder is dead when `renewTime + leaseDurationSeconds` is in
  the past. The renew loop writes `renewTime` every `TTL/3` seconds, also via
  compare-and-swap, and exits if the holder identity changed under it.
- **Release keeps the object.** Releasing sets `holderIdentity` to empty and
  writes `previous-holder`, `released-at`, `cleanup-status` and `leftovers`
  annotations (prefix `cluster-lease/`). So `status` on a free lease still shows
  who was last on the hardware and whether they left it clean.
- **Cleanup predicate.** "Dirty" means any non-system-namespace pod requesting
  `nvidia.com/gpu` that is not Succeeded/Failed, or any instance of the kinds in
  `LEASE_CHECK_KINDS`. Tune the predicate for your cluster rather than reaching
  for `--force`.

## Inspecting by hand

```sh
kubectl get lease -n cluster-lease hardware -o yaml
```

Reading is always fine. Writing by hand is not; the CLI's compare-and-swap
assumes it is the only writer.

## What it does not do

It does not stop an agent that never runs `lease acquire`. If that becomes a
problem, add a `ValidatingAdmissionPolicy` whose CEL rule rejects pods
requesting `nvidia.com/gpu` unless their namespace carries a label equal to the
Lease's current `holderIdentity`, with the Lease as the policy's parameter
resource. Until then this is a cooperative protocol, and the README you are
reading is the enforcement.
