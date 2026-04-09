# RDMA Progress - April 9, 2026

This note captures the debugging session from April 9, 2026, what changed,
what we measured, and what the likely architectural implication is.

## Goal

Get `runsc-rdma` within 10% of `runc` on multi-node NCCL bus bandwidth.

For the current 2-node, 8-GPU-per-node setup, that means getting close to the
`runc` baseline of about `484 GB/s` bus bandwidth, with a target floor of about
`436 GB/s`.

## What Worked Today

### Environment and baseline checks

- Both Modal H200 nodes were reachable and configured earlier in the day.
- Bare-metal 2-node `torchrun` remained healthy at:
  - `algbw 257.238 GB/s`
  - `busbw 482.321 GB/s`
- The prior `runc` 2-node baseline from the same setup remained:
  - `algbw 258.129 GB/s`
  - `busbw 483.992 GB/s`

### gVisor correctness improvements

The earlier "overlap" failure mode was fixed:

- `rdmaproxy` now keeps shared NVIDIA-backed GPU VMAs instead of dropping back
  to raw sandbox addresses when `MAP_FIXED_NOREPLACE` collides.
- When the target GPU VA was already occupied in the sentry, `rdmaproxy`
  successfully created a relocated NVIDIA-backed VMA at another sentry VA.
- The old warning pattern:
  - `GPU device memory mirror failed (... file exists), raw VA passthrough`
  stopped appearing in the failing 8-GPU run.

Related local code areas:

- `pkg/sentry/fsimpl/sys/rdma.go`
- `pkg/sentry/devices/rdmaproxy/rdmaproxy.go`
- `pkg/sentry/devices/rdmaproxy/rdmaproxy_ioctl_unsafe.go`

### Partial GDR success under gVisor

`runsc-rdma` with GPUDirect RDMA was no longer "completely broken":

- 1 GPU per node succeeded:
  - `algbw 47.879 GB/s`
  - `busbw 47.879 GB/s`
- 2 GPUs per node succeeded:
  - `algbw 88.948 GB/s`
  - `busbw 133.422 GB/s`

This matters because it shows:

- IB transport works
- GDRDMA is active
- `nvidia-peermem` is not generically broken under gVisor
- the remaining problem is not basic setup

## What Still Failed

### 8 GPUs per node in one `runsc-rdma` sandbox

The 8-GPU-per-node single-container run still failed with:

- `ibv_reg_mr_iova2 failed with error Bad address`

The failure correlated with GPU memory regions whose sentry VMA had to move
away from the original app GPU VA.

In the failing logs, the pattern looked like:

- app GPU VA in the `0xa...` range
- sentry mirror created at a different `0x7ef...` or `0x7f1c...` VA
- MR registration rewritten from app VA to sentry VA
- `hca_va` or app-facing addressing remained on the original VA

Examples seen in logs:

- `start 0xa1a000000 -> sentry 0x7ef784f80000`
- `start 0xa19400000 -> sentry 0x7eff9d000000`
- `start 0xa17600000 -> sentry 0x7f14401db000`

### Multi-sandbox workaround was not viable

A practical workaround was tested: one `runsc-rdma` sandbox per local rank.

Setup:

- 4 total ranks
- 1 GPU per container
- 2 containers per node
- `--ipc=host`
- matching hostnames per physical node

Result:

- `algbw 32.029 GB/s`
- `busbw 48.043 GB/s`

That is far below:

- the 2-GPU single-sandbox result (`133.422 GB/s`)
- the `runc` target (`483.992 GB/s`)

Why it failed:

- NCCL logs showed same-host peers going over `NET/IB/.../GDRDMA`
- same-host peers did not stay on local `P2P/direct pointer` paths

So splitting ranks across sandboxes destroys the local topology behavior we
need for near-`runc` performance.

## Interpretation

The single-sandbox design is still the only realistic path to the 10% target.

The data points suggest:

1. GDR works while the sentry mapping is effectively the same address as the
   app-visible GPU VA.
2. Once the GPU region has to be relocated in the sentry address space, MR
   registration starts failing.
3. The current proxy logic rewrites only the registration fields. It does not
   have a later fast-path translation stage for work requests or other app VA
   usage.

That makes the remaining problem look like an address-space model mismatch, not
just a missing env var or a generic performance bug.

## Local Code Change Prepared After The Benchmarks

A follow-up diagnostic patch was prepared locally but could not be benchmarked
because the Modal worker endpoints timed out later in the evening.

The patch does two things in
`pkg/sentry/devices/rdmaproxy/rdmaproxy_ioctl_unsafe.go`:

1. Adds explicit handling for the modern REG_MR `IOVA` attribute.
2. For relocated GPU-backed MRs only, rewrites:
   - legacy `hca_va`
   - modern `IOVA`
   by the same delta as the `start` or `addr` rewrite.

This is a narrow diagnostic experiment. It tests whether the remaining failure
is caused by the RDMA-visible address still pointing at the app VA while the
host-side peer-memory registration is happening against a different sentry VMA.

## What "Per-Rank / Per-MM Architecture" Means

This phrase refers to changing the host-side ownership model for RDMA
registration.

### Current model

Today, one sentry process owns one host address space for the whole local
sandbox.

That means:

- all local ranks share one host mm
- all GPU VA mirrors have to coexist in that single host address space
- when two ranks want conflicting GPU VA ranges, some mappings must be
  relocated
- once relocation happens, the proxy has to fake address identity using field
  rewrites

### Per-rank / per-mm model

Instead, each local rank, task, or GPU-owning execution context would get its
own host address space for RDMA-visible memory registration.

In practice, that could mean:

- one helper process per rank, or
- one dedicated registration context per task/rank, or
- a broader multi-process sandboxing model where each rank has its own host mm

The key idea is:

- register GPU memory from a host mm where the GPU VA can stay at the same
  address the app expects
- avoid relocations between app VA and host registration VA
- avoid having to repair that mismatch later with `hca_va`, `IOVA`, or work
  request address rewriting

### Why this may be better than more field rewriting

Field rewriting is attractive when only registration is involved, but it gets
fragile once the app and kernel continue to use the memory region later.

Potential problems with "just rewrite more fields":

- legacy REG_MR uses `start` plus `hca_va`
- modern REG_MR uses `addr` plus `IOVA`
- later work requests may still carry app VAs
- CQ/QP or driver-owned fast paths may assume the registered IOVA is stable
- peer-memory integration may be sensitive to the VA handed down by the RDMA
  core, not just to the pages that were pinned once

In other words: if the architecture fundamentally wants "this GPU VA exists in
the caller's mm at this address", then increasingly complex field rewriting may
keep fixing symptoms without fixing the address-space mismatch itself.

## Why We Believe This Is The Likely Direction

Three pieces of evidence point there:

1. Low-fanout GDR runs work.
   - 1 GPU/node and 2 GPUs/node both succeeded.
2. High-fanout single-sandbox runs fail exactly when GPU VAs must be relocated.
3. Multi-sandbox runs preserve correctness but destroy local NCCL topology and
   bandwidth.

That combination strongly suggests:

- one shared host mm is the root of the GPU VA collision problem
- multiple independent sandboxes are too expensive from NCCL's point of view
- the winning design is likely "single job semantics, but with per-rank host
  address-space ownership for RDMA-sensitive memory"

## Open Questions

The next live benchmark should answer these:

1. Does rewriting relocated GPU `hca_va` and modern `IOVA` make 8-GPU MR
   registration succeed?
2. If registration succeeds, does NCCL still behave correctly at runtime?
3. If it succeeds but bandwidth remains far below `runc`, where does the next
   gap come from:
   - address translation overhead
   - missing local NCCL optimization
   - systrap overhead
   - another GPU memory registration edge case

## Recommended Next Step

When the worker nodes are back:

1. Deploy the local diagnostic patch.
2. Rebuild and install `runsc-rdma` on both nodes.
3. Rerun the benchmark ladder:
   - 1 GPU/node
   - 2 GPUs/node
   - 8 GPUs/node
4. If 8-GPU still fails or is badly under target, move on from field rewriting
   and prototype the per-rank/per-mm design.

## Session Blocker

Late in the session, both worker SSH endpoints stopped responding:

- `wo-a1sh1kw61wt69jvetucf3dnum`
- `wo-q2doub248ut22pma3364yn5dk`

The attempted `-1` fallback hostnames did not resolve, so the final diagnostic
patch could not be deployed or benchmarked during this session.
