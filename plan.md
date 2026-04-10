# RDMA Plan - 2026-04-10

## Blocking Error

Current blocking failure:

- `ibv_reg_mr_iova2` returns `Bad address`
- the host `REG_MR` ioctl returns `EFAULT`
- this first appears once the run exercises true GPU-backed registration
- `3+ GPU/node` fails before meaningful RDMA traffic reaches the HCAs

What that means right now:

- this is a control-path / registration bug
- it is happening before the RDMA data plane is established
- the immediate goal is not scaling or performance tuning
- the immediate goal is to make one true GPU-backed `MR_REG` succeed

## Immediate Debug Goal

The next debugger should treat this as the primary question:

- why does a sentry-created NVIDIA-backed VMA still fail GPU-backed
  `ibv_reg_mr`?

The first success criterion is:

- one confirmed successful true GPU-backed `MR_REG` on a `0xa1...` or `0x50...`
  range
- with a real MR handle returned
- and no `EFAULT`

## Project Objective

The broader project objective is still:

- get `runsc-rdma` within `10%` of `runc` on multi-node NCCL `busbw`
- `runc` reference: about `484 GB/s`
- target floor: about `436 GB/s`

That objective remains important, but it is downstream of the current blocking
GPU-backed `MR_REG` bug.

Do not start with:

- per-rank/per-`mm` architecture work
- broad performance tuning
- helper-process routing changes

until that one path succeeds.

## Immediate Debug Checklist

For the first debugging pass, answer only these questions:

1. Is the failing `MR_REG` actually GPU-backed?
2. Did it go through `mirrorGPUDeviceMemory()`?
3. Was the resulting VMA:
   - direct
   - prepared
   - reused from `sharedGPUVMAs`
   - relocated after `EEXIST`
4. Which `hostFD` / device / mmap provenance produced that VMA?
5. Was the VMA identity-mapped or relocated?
6. Did the host `REG_MR` return an MR handle or `EFAULT`?

Until those are answered for a failing case and then for one successful
GPU-backed case, later architecture work should be considered out of scope.

## Context

The current `runsc-rdma` design uses one sentry process and one host address
space for the whole local sandbox. That works for low-fanout GPUDirect RDMA
cases, but it appears to break down once enough local ranks and GPU mappings
are present that GPU virtual address ranges collide inside that shared host
`mm`.

That context matters because the successful `2 GPUs/node` result of about
`133.4 GB/s` is strong evidence that the architecture works in principle, while
still being far from the actual end goal.

The working hypothesis has now changed:

- `1 GPU/node` and `2 GPUs/node` still show that the IB and NCCL GDR plumbing
  can work under gVisor.
- The successful `2 GPU/node` case does not appear to exercise true GPU-backed
  `MR_REG` at all; it likely uses CPU proxy buffers for RDMA.
- The smallest known failing reproducer is now `4 GPUs/node`, which is also the
  first case that clearly exercises real GPU-backed registration.
- Collisions and relocation still matter, but they no longer appear to be the
  primary bug by themselves.
- The stronger current theory is that each `ibv_reg_mr` needs a VMA backed by
  the correct NVIDIA allocation provenance, not just a VMA at the correct VA.
- A sentry-created NVIDIA VMA may exist and still fail `ibv_reg_mr_iova2` if
  `nvidia_peermem` cannot resolve the specific GPU pages for that registration.

## Prior Data Points That Matter

The strongest positive evidence so far is that low-fanout runs can show
apparent GDR behavior under gVisor. That is important, but it is not the same
thing as proving that true GPU-backed GPUDirect RDMA is working end to end.

### NCCL and bandwidth data

- NCCL reported GPUDirect paths active under gVisor:
  - `GDR 1`
  - `PXN 1`
  - transfers via `NET/IB/GDRDMA`
- `busbw` reached about `134.4 GB/s` with `2 GPUs/node`
- the comparable `GDR=0` case was about `22 GB/s`
- this is about a `6x` improvement from enabling the GDR path

### NIC utilization data

For the successful 2-GPU-per-node case, only two HCAs were active, which is the
expected topology outcome when each GPU routes through its nearest NIC.

Node 0:

- `mlx5_0`
  - `rcv delta`: `12,822,572,343`
  - `xmit delta`: `12,823,783,080`
  - `rcv`: `51.3 GB`
  - `xmit`: `51.3 GB`
- `mlx5_3`
  - `rcv delta`: `12,820,380,111`
  - `xmit delta`: `12,820,382,901`
  - `rcv`: `51.3 GB`
  - `xmit`: `51.3 GB`
- all other HCAs:
  - `0`
- total:
  - `102.6 GB` receive
  - `102.6 GB` transmit

Node 1:

- `mlx5_0`
  - `rcv delta`: `12,823,783,080`
  - `xmit delta`: `12,822,572,343`
  - `rcv`: `51.3 GB`
  - `xmit`: `51.3 GB`
- `mlx5_3`
  - `rcv delta`: `12,820,382,901`
  - `xmit delta`: `12,820,380,111`
  - `rcv`: `51.3 GB`
  - `xmit`: `51.3 GB`
- all other HCAs:
  - `0`
- total:
  - `102.6 GB` receive
  - `102.6 GB` transmit

### GPU-count staircase and port-counter behavior

The GPU-count staircase makes a very important distinction between the working
low-fanout case and the failing higher-fanout cases:

| GPUs/node | Total ranks | GDR=3 result | Port counter delta |
| --- | --- | --- | --- |
| 2 | 4 | works (`~134 GB/s`) | `~101 GB` across 2 HCAs |
| 3 | 6 | `EFAULT` | zero |
| 4 | 8 | `EFAULT` | zero |
| 8 | 16 | `EFAULT` | zero |

This means:

- `2 GPU/node` is the last configuration that reaches real network data
  transfer
- `3+ GPU/node` fails before measurable RDMA traffic appears on the HCAs
- the current blocking bug is therefore pre-data-plane
- the failure is happening during bring-up or registration, not during a
  steady-state transport path that already works and then regresses

### Additional April 9 datapoints carried forward

These points were preserved from `RDMA_PROGRESS_2026-04-09.md` so the dated
note can be retired without losing the most useful quantitative context.

- bare-metal 2-node `torchrun` baseline:
  - `algbw 257.238 GB/s`
  - `busbw 482.321 GB/s`
- prior `runc` baseline on the same setup:
  - `algbw 258.129 GB/s`
  - `busbw 483.992 GB/s`
- low-fanout `1 GPU/node` result:
  - `algbw 47.879 GB/s`
  - `busbw 47.879 GB/s`
- multi-sandbox workaround result:
  - `algbw 32.029 GB/s`
  - `busbw 48.043 GB/s`
- why the multi-sandbox workaround failed:
  - same-host peers went over `NET/IB/.../GDRDMA`
  - same-host peers did not stay on local `P2P/direct pointer` paths
- related local code areas during the April 9 debug session:
  - `pkg/sentry/fsimpl/sys/rdma.go`
  - `pkg/sentry/devices/rdmaproxy/rdmaproxy.go`
  - `pkg/sentry/devices/rdmaproxy/rdmaproxy_ioctl_unsafe.go`
- follow-up local diagnostic patch prepared after the benchmarks:
  - explicit handling for modern `REG_MR` `IOVA`
  - relocated-GPU rewrite of legacy `hca_va` and modern `IOVA`

### Interpretation of those data points

- NCCL can report `GDR 1` and take the intended `NET/IB/GDRDMA` path under
  gVisor in the `2 GPU/node` case
- local topology behavior looks sensible in that low-fanout case
- this shows the transport and topology story can look healthy
- but it does not prove that true GPU-backed `ibv_reg_mr` is working
- `3+ GPU/node` failing with zero port-counter delta shows the current blocking
  bug happens before the RDMA data plane is meaningfully established
- the remaining problem is whether the GPU-backed registration path preserves
  the correct NVIDIA allocation provenance as scale increases

## Updated Interpretation From The New 4-GPU Reproducer

The newer data changes the primary thesis in an important way.

### What the newer runs show

- `2 GPU/node` succeeds, but appears to issue no true GPU-backed `MR_REG`
  operations.
- The concrete `2 GPU/node` control-path facts were:
  - `GPU MR_REGs: 0`
  - `Relocated MRs: 0`
  - `EFAULT: 0`
  - `GPU VMAs created: 80`
- `4 GPU/node` is the smallest known failing reproducer.
- The `4 GPU/node` run creates real GPU VMAs at `0xa1...` addresses.
- Some failing registrations are for identity-mapped GPU VMAs where app VA and
  sentry VA are the same.
- Only a small number of relocated GPU MRs were observed, and relocation alone
  does not explain all failures.
- At `3+ GPU`, the failure mode is `ibv_reg_mr_iova2` returning `EFAULT` even
  for some identity-mapped NVIDIA VMAs.

### What that means

- the `2 GPU/node` success case does not validate the full GPU-backed
  `ibv_reg_mr` path
- the sentry can create NVIDIA-backed GPU VMAs in the `2 GPU/node` case, but
  that still does not mean those VMAs are ever used in true GPU-backed
  registration there
- the first real exercise of that path appears at `4 GPU/node`
- identity mapping by itself is not sufficient
- therefore the primary missing invariant is probably not just VA identity
- the current failure is not just cross-process sharing or relocation pressure
- even identity-mapped NVIDIA VMAs can still be unusable for GPU-backed
  registration

The stronger current theory is:

- each GPU-backed `ibv_reg_mr` needs a VMA backed by the correct NVIDIA mmap
  context or allocation provenance
- `nvidia_peermem` must be able to resolve the exact GPU pages for that
  registration
- a generic or reused NVIDIA VMA at the same virtual address may still be wrong
  for the current allocation
- the sentry-side NVIDIA VMA may exist and still fail `pin_user_pages` /
  peermem resolution for the actual GPU physical pages needed by `ibv_reg_mr`

### Immediate architecture consequence

This means a per-rank/per-`mm` redesign may still help later, but it is no
longer obviously the first fix. The first thing to prove is whether the sentry
is creating a peermem-valid VMA for each specific GPU allocation.

## What Was Fixed Before The Current Architecture Question

Before the current per-rank/per-`mm` discussion, one important class of failure
had already been improved in `rdmaproxy`.

The meaningful correctness changes were:

- shared NVIDIA-backed GPU VMAs are now kept instead of immediately falling back
  to raw sandbox addresses
- when the target GPU VA is already occupied, `rdmaproxy` can create a
  relocated NVIDIA-backed VMA at another sentry VA
- the old raw-VA passthrough warning pattern stopped appearing in the failing
  8-GPU run

The specific old warning that stopped appearing was effectively:

- `GPU device memory mirror failed (... file exists), raw VA passthrough`

Why this matters:

- it narrows the remaining problem from basic GPU mirror failure
- to what happens after the mirror succeeds, especially whether the resulting
  VMA has the correct provenance for GPU-backed registration

In other words, the architecture question is no longer "can `rdmaproxy` mirror
GPU memory at all?" It is now closer to "does each `MR_REG` see a sentry VMA
that `nvidia_peermem` can use to resolve the exact GPU allocation for that
registration?"

## Instrumentation Added Today

Today we added lightweight counters to
`pkg/sentry/devices/rdmaproxy/rdmaproxy_ioctl_unsafe.go` so that steady-state
RDMA control-plane activity can be measured without `strace` overhead.

### Existing counters retained

- `ioctl_rate`
- `write_rate`
- `read_rate`

### New counters added

- `mr_reg`
- `mr_dereg`
- `cq_create`
- `cq_destroy`
- `qp_create`
- `qp_destroy`

### New log lines added

- explicit `GPU VA collision ... retrying at any free sentry VA`
- compact `MR_REG handle=...` summary lines

The collision logs include:

- requested GPU VA range
- `sandboxFD`
- `hostFD`
- device name
- `mmapLen`
- task ownership metadata:
  - `tid`
  - `tgid_root`

The compact MR summary logs include:

- app VA range
- sentry VA range
- `relocated=true/false`
- `hca_va` old and new values
- `iova` old and new values
- final MR handle
- task ownership metadata

## Prior Ioctl Frequency Data

One especially useful prior data point is the earlier low-overhead ioctl
frequency analysis from a node-0 `2-GPU`, `GDR=3`, `BW_ONLY` run. This gives a
baseline for what the control plane actually looked like in a working low-fanout
configuration.

### rdmaproxy ioctl summary

- total `rdmaproxy` log lines:
  - `74,209`
- total ioctls:
  - `8,832`
- successful ioctls:
  - `4,752`
- failed ioctls:
  - `4,040`
- `mmap`s:
  - `88`
- device opens:
  - `24`
  - `12` `uverbs` devices x `2` opens each
- async event FDs:
  - `64`
- pinned MR releases:
  - `20`
- pinned CQ releases:
  - `7`
- pinned QP releases:
  - `13`

### Error breakdown

- `errno=61` (`ENODATA`):
  - `4,016` calls
  - this was the hot error
  - all from `obj=0x0000 method=6`
- `errno=28` (`ENOSPC`):
  - `24` calls
  - expected
  - one per device open capability check

### Ioctl hot path

- `obj=0x0000 method=6`
  - `8,144` calls
  - likely GID query
  - by far the hottest path, about `92%` of all ioctls
- `obj=0x0000 method=0`
  - `258` calls
  - likely device query
- `obj=0x1000 method=4096`
  - `96` calls
  - likely PD or context operations
- `obj=0x0007 method=1`
  - `54` calls
  - likely QP modify
- `obj=0x0000 method=2`
  - `48` calls
  - likely port query
- `obj=0x1008 method=4096`
  - `40` calls
  - likely completion channel operations
- `obj=0x0007 method=4`
  - `40` calls
  - likely QP create
- `obj=0x0001 method=0`
  - `40` calls
  - likely MR reg

### Why this prior frequency profile matters

- the hottest ioctl path in the working `2-GPU` case was not MR churn
- it was overwhelmingly one device-object query path
- MR registration existed, but it was not dominating the control plane

That is useful context for the helper-process question:

- if the future `8-GPU` runs show collisions and relocation pressure while MR
  lifecycle rates remain relatively modest, then the architecture problem is
  more likely address-space ownership than raw MR volume
- if helper routing is eventually added, it should avoid inserting IPC into the
  dominant query-heavy or steady-state fast paths unless absolutely necessary

### Where the counters are hooked

- Modern verbs path:
  - `handleRDMAVerbsIoctl()`
  - increment per-op counters immediately after `classifyIoctl()`
- Legacy verbs path:
  - `uverbsFD.Write()`
  - map legacy write command IDs to the same action buckets
- Read path:
  - `uverbsFD.Read()`
  - `asyncEventFD.Read()`

### Why this matters

This gives a low-overhead answer to a key architecture question:

- Are `MR_REG` / `DEREG_MR` and `QP` / `CQ` lifecycle operations mostly setup
  activity?
- Or do they continue at a high rate during the hot NCCL loop?

If they are mostly setup-path operations, a per-rank helper-process design is
much more likely to be viable.

## Immediate Usage

Run the usual NCCL workload and inspect the `rdmaproxy: PERF ...` log lines.

The main signals to look for are:

- low `mr_reg` / `mr_dereg` rates after startup
- low `cq_create` / `cq_destroy` and `qp_create` / `qp_destroy` rates after
  startup
- whether overall `ioctl_rate`, `write_rate`, and `read_rate` stay modest in
  the steady-state allreduce loop

If the hot loop mostly uses already-created queue state and mapped buffers,
that argues in favor of a helper-based separation of address spaces.

## How To Analyze The Results

The original goal of the low-overhead instrumentation is to replace
high-overhead `strace`-style syscall counting with proxy-local measurement that
is specific to the verbs paths we care about.

The right way to read the results is to correlate four streams:

- periodic `rdmaproxy: PERF ...` rate lines
- explicit `GPU VA collision ...` lines
- compact `MR_REG handle=...` summary lines
- per-run notes about whether a given `MR_REG` is truly GPU-backed or only a
  CPU/userspace range

It is also useful to compare the new runs against the earlier `2-GPU GDR=3`
frequency profile above, especially the fact that the hottest prior ioctl path
was `obj=0x0000 method=6` and not MR registration itself.

### Step 1: Split startup from steady state

For each run:

1. capture the first burst of logs during communicator and queue bring-up
2. separately capture a later steady-state window during the allreduce loop

Do this for:

- 1 GPU/node
- 2 GPUs/node
- 8 GPUs/node

The main mistake to avoid is averaging startup-heavy object creation together
with the hot loop.

### Step 2: Read the `PERF` lines as a control-plane profile

Interpret the `rdmaproxy: PERF ...` line this way:

- `ioctl_rate`
  - overall modern verbs ioctl pressure
- `write_rate`
  - legacy verbs command pressure
- `read_rate`
  - async-event and related read-side activity
- `mr_reg` / `mr_dereg`
  - MR lifecycle churn
- `cq_create` / `cq_destroy`
  - CQ object churn
- `qp_create` / `qp_destroy`
  - QP object churn

The most important question is:

- do these rates collapse after startup?

If yes, then the workload is mostly using pre-created RDMA objects during the
hot loop, which means a later architecture split is less likely to hurt
steady-state performance.

If no, and control-plane churn remains high in steady state, then helper IPC
has a higher chance of becoming expensive.

In particular, compare the new runs against the earlier baseline:

- if `MR_REG`, `DEREG_MR`, `QP create`, and `CQ create` stay small relative to
  the hottest query paths, that favors helper-based ownership for setup-path
  objects
- if those object-lifecycle counters explode in the `8-GPU` case, that raises
  the cost risk of routing them through helpers
- if the `8-GPU` case still looks query-dominated but now also shows repeated
  collision and relocation logs, that strongly points to the shared-`mm`
  address-space model as the scaling bottleneck

### Step 3: Separate true GPU-backed `MR_REG` from broad address matches

Before drawing architecture conclusions, classify the registrations correctly.

Questions to answer first:

- is the `MR_REG` app address in a true GPU VA range such as `0x50...` or
  `0xa1...`?
- or is it just a userspace heap or mmap range such as `0x7f...`?
- does the run actually issue GPU-backed `MR_REG`, or does it stay on CPU proxy
  buffers?

This matters because the earlier `2 GPU/node` success case appears not to have
exercised real GPU-backed `MR_REG` at all.

### Step 4: Read collision logs as evidence of address-space pressure

Each `GPU VA collision ...` line means:

- the code attempted to map the NVIDIA-backed VMA at the app-aligned GPU VA
- `MAP_FIXED_NOREPLACE` found that range already occupied in the sentry
- the code had to retry at some free sentry VA instead

That is the direct evidence for address-space pressure in the shared sentry
`mm`.

Questions to answer from these logs:

- how often do collisions happen?
- do they appear in the first failing `4 GPU/node` reproducer?
- are they concentrated in one task or spread across many `tgid_root` values?
- do repeated collisions target the same VA ranges?

Important interpretation rule:

- collisions are now best treated as a secondary pressure signal
- they are no longer sufficient by themselves to explain failure, because some
  identity-mapped GPU VMAs also fail with `EFAULT`

### Step 5: Read compact MR summaries as the bridge between provenance and failure

Each `MR_REG handle=...` summary line should be used to answer:

- did the MR preserve VA identity?
- if not, how often was it relocated?
- was `hca_va` rewritten?
- was `iova` rewritten?
- which task owned the MR?
- did the failing registrations involve identity-mapped GPU VMAs anyway?

This is the best low-overhead bridge between:

- the frequency view from `PERF`
- the address-space view from collision logs
- the provenance question of which VMA was actually used for registration
- the failure symptom from `ibv_reg_mr_iova2 failed with error Bad address`

### Step 6: Build a simple interpretation table

For each run configuration, summarize:

- startup `mr_reg`
- steady-state `mr_reg`
- startup `qp_create` / `cq_create`
- steady-state `qp_create` / `cq_create`
- collision count
- relocated MR count
- count of rewritten `hca_va`
- count of rewritten `iova`
- count of true GPU-backed `MR_REG`
- count of identity-mapped GPU-backed failures
- count of relocated GPU-backed failures
- dominant ioctl object and method pairs
- dominant error codes
- whether the run succeeded
- final `busbw`

The architecture implications are then much easier to read:

- low steady-state syscall churn + high collision rate at 8 GPUs
  - suggests host-`mm` separation may still be a useful later step after
    correctness is restored
- high steady-state churn + high collision rate
  - helper design may still be right, but cost risk is higher
- query-dominated control plane + high collision and relocation rate
  - supports keeping helpers focused on MR ownership and setup-path state, not
    every ioctl class
- identity-mapped GPU VMA failures
  - strongly suggests the primary bug is not just shared-`mm` relocation
  - points instead to missing NVIDIA allocation provenance or an otherwise
    non-pinnable sentry VMA
- low collision rate + persistent failures
  - suggests another registration or post-registration edge case
- collisions only for one task/rank
  - may indicate uneven VA behavior or one rank-specific reuse pattern
- collisions across many tasks/ranks
  - more strongly supports the shared-`mm` root-cause theory

### Step 7: What result would most strongly justify a provenance-first fix?

The strongest justification would look like this:

- `2 GPU/node` succeeds without true GPU-backed `MR_REG`
- `4 GPU/node` is the first failing reproducer and does exercise true
  GPU-backed `MR_REG`
- some failing registrations are identity-mapped GPU VMAs
- collisions and relocation appear, but do not explain all failures
- CQ/QP creation continues to succeed while `MR_REG` fails

That combination would say:

- the RDMA architecture works
- the hot loop is not dominated by verbs setup traffic
- the failure is centered on GPU-backed MR registration
- the primary missing invariant is correct VMA provenance for
  `nvidia_peermem`, not just VA identity

At that point, the first implementation question is how to preserve the correct
NVIDIA mmap or allocation context per `MR_REG`, not whether to split into
multiple host `mm`s immediately.

## Additional Diagnoses To Refine The Architecture

The next round of diagnosis should focus first on VMA provenance, and only
secondarily on VA collisions.

### 1. Prove which registrations are truly GPU-backed

Do not rely on broad address-pattern matching alone.

For every failing and successful `MR_REG`, classify:

- app VA range
- whether the range is a true GPU VA or a CPU/userspace range
- whether the registration took the GPU-specific action path
- whether a NVIDIA-backed VMA was created or reused

This prevents false conclusions from broad `MR_REG` greps.

### 2. Record the NVIDIA mmap provenance used for each GPU VMA

The most important missing datum is which NVIDIA allocation context produced the
VMA used for registration.

For each created or reused GPU VMA, record:

- `sandboxFD`
- `hostFD`
- device name
- whether it came from the direct mmap path or a prepared candidate
- whether it was created fresh, reused from `sharedGPUVMAs`, or relocated after
  `EEXIST`

If possible, extend this with any allocation-specific identity available from
the nvproxy side, not just FD and VA range.

### 3. Audit the `sharedGPUVMAs` reuse model

The current reuse model is keyed primarily by VA range. The newer evidence
suggests that may be too weak.

The questions to answer are:

- can two different GPU allocations appear at the same VA range over time?
- can those allocations require different NVIDIA mmap provenance?
- is `sharedGPUVMAs` returning a VMA that was created for one allocation but is
  later reused for another?

If the answer is yes, then the cache key is fundamentally wrong for peermem.

### 4. Compare failing identity-mapped GPU VMAs with successful ones

This is now one of the most important experiments.

For each identity-mapped GPU VMA:

- app VA
- sentry VA
- length
- `hostFD`
- whether MR registration succeeded
- whether it came from direct, prepared, or reused mapping

If identity-mapped VMAs still fail, that is strong evidence that VA equality is
not the missing invariant.

### 5. Keep collision logs, but demote them to secondary evidence

`EEXIST` collision logs are still useful because they show address-space
pressure in the shared sentry `mm`.

But they should now answer secondary questions:

- how much pressure does one shared `mm` create?
- how often does the fallback path run?
- how often does one VMA get reused across overlapping registrations?

They should no longer be treated as sufficient proof of the root cause by
themselves.

### 6. Keep one narrow field-rewrite experiment in reserve

The `hca_va` / `IOVA` rewrite patch is still a useful control experiment.

It can still help separate:

- stale app-visible addressing after relocation
from
- non-pinnable or wrong-provenance sentry VMAs

But it is now a secondary diagnostic, not the primary plan.

## Code Anchors For Debugging

The next debugger should start from the exact code paths that define the current
GPU-backed MR registration surface.

### Primary files

- `pkg/sentry/devices/rdmaproxy/rdmaproxy_ioctl_unsafe.go`
- `pkg/sentry/devices/rdmaproxy/rdmaproxy.go`
- `pkg/sentry/fsimpl/sys/rdma.go`

### Most important functions

In `pkg/sentry/devices/rdmaproxy/rdmaproxy_ioctl_unsafe.go`:

- `handleRDMAVerbsIoctl()`
- `classifyIoctl()`
- `prepareMRRegModern()`
- `prepareMRRegInvokeWrite()`
- `mirrorSandboxPages()`
- `mirrorGPUDeviceMemory()`
- `translatedRelocatedGPUVA()`
- `extractMRHandle()`
- `extractDeregMRHandle()`

In `pkg/sentry/devices/rdmaproxy/rdmaproxy.go`:

- `mirroredPages`
- `sharedGPUVMAKey`
- `sharedGPUVMA`
- `acquireSharedGPUVMA()`
- `mapOrAcquireSharedGPUVMA()`
- `uverbsFD.pinnedMRs`

### Most important branch points

The current bug is likely to be explained by one of these branch points:

- in `mirrorSandboxPages()`:
  - `mm.Pin()` success
  - fallback to `mirrorProxyDevicePages()`
  - fallback to `mirrorGPUDeviceMemory()`
  - raw passthrough last resort
- in `mirrorGPUDeviceMemory()`:
  - direct candidate mmap at the app GPU VA
  - prepared candidate mmap path
  - `sharedGPUVMAs` reuse path
  - `MAP_FIXED_NOREPLACE` failure with `EEXIST`
  - retry at a free sentry VA

### State that is currently suspicious

The `sharedGPUVMAs` reuse model is especially important to audit.

What is clearly represented today:

- GPU VA start
- GPU VA length

What may not be represented strongly enough:

- NVIDIA `hostFD`
- direct vs prepared candidate origin
- allocation-specific NVIDIA mmap provenance
- any per-allocation identity beyond VA range

This is one reason the current cache key is a prime suspect.

### Questions to answer in code for each failing GPU-backed `MR_REG`

- did the request go through `prepareMRRegModern()` or
  `prepareMRRegInvokeWrite()`?
- did `mirrorSandboxPages()` fall into `mirrorGPUDeviceMemory()`?
- did `mirrorGPUDeviceMemory()` create a fresh VMA, reuse one from
  `sharedGPUVMAs`, or relocate after `EEXIST`?
- which `candidate.hostFD` and `candidate.devName` were used?
- was the resulting mapping identity-mapped or relocated?
- did the host `REG_MR` ioctl return a real MR handle or `EFAULT`?

### First success criterion to trace in code

The first meaningful milestone is not performance. It is one confirmed
successful true GPU-backed `MR_REG` on a `0xa1...` or `0x50...` range.

That success should be traceable through:

- `prepareMRReg*()`
- `mirrorSandboxPages()`
- `mirrorGPUDeviceMemory()`
- host `REG_MR` ioctl return
- `extractMRHandle()`

Until that path succeeds, later architecture work should be treated as
secondary.

## Updated Architecture Direction

The likely near-term direction is now:

- preserve single-job NCCL semantics
- preserve the current useful low-overhead instrumentation
- first fix the blocking bug:
  - make GPU-backed `MR_REG` use sentry VMAs with the correct NVIDIA allocation
    provenance
- only then decide whether additional host-`mm` separation is still needed
- the architecture work below should therefore be read as a possible
  after-the-bug-is-fixed plan, not as the first implementation step

Per-rank/per-`mm` remains a plausible later step, but it is now a contingent
optimization or secondary fix, not the first assumed answer.

## Proposed Implementation Procedure

### Phase 0: Lock down the smallest true reproducer

Use:

- `2 GPU/node` as the low-fanout control case
- `4 GPU/node` as the first true failing reproducer
- `8 GPU/node` as the worst-case confirmation

The most important change here is that `4 GPU/node` is now the primary debug
target because it appears to be the smallest case that actually exercises
GPU-backed `MR_REG`.

This phase is still part of fixing the current blocking bug, not the later
architecture overhaul.

### Phase 1: Instrument provenance, not just addresses

Before any architecture overhaul:

1. log whether each `MR_REG` is truly GPU-backed
2. log which NVIDIA candidate or mmap context created the corresponding VMA
3. log whether the VMA was direct, prepared, reused, or relocated
4. correlate that with success or `EFAULT`

Exit criteria:

- you can explain each failing GPU-backed registration in terms of the VMA that
  was actually handed to `nvidia_peermem`

This is the minimum gate before any broader architecture work should begin.

### Phase 2: Validate or falsify the cache-key hypothesis

Explicitly test whether:

- the same VA range can correspond to different GPU allocations
- `sharedGPUVMAs` reuses a VMA across registrations that need different NVIDIA
  backing
- overlap plus reuse is causing a wrong-context VMA to be handed to peermem

If that hypothesis is correct, then the next fix is likely to be:

- change or disable broad VMA reuse
- key reuse by stronger provenance than just VA range
- or recreate VMAs per registration or per allocation context

### Phase 3: Make GPU-backed `MR_REG` correct first

The primary implementation target should be:

- for each GPU-backed `MR_REG`, produce a sentry VMA that maps the correct GPU
  allocation for that exact registration

Possible directions include:

- tighter reuse rules
- provenance-aware cache keys
- per-registration VMA creation
- preserving allocation-specific mmap context from nvproxy into rdmaproxy

The success criterion is correctness, not yet scalability.

This is the bug-fix milestone that should happen before the rest of the
architecture plan is treated as active work.

### Phase 4: Re-evaluate whether host-`mm` separation is still needed

Only after Phase 3 is working should the plan re-open the per-rank/per-`mm`
question.

At that point, the remaining reasons to split host address spaces would be:

- reducing `MAP_FIXED_NOREPLACE` collisions
- reducing cross-rank VMA contention
- avoiding incorrect reuse pressure in one shared sentry `mm`
- simplifying ownership and lifetime if provenance remains rank-local

### Phase 5: If needed, introduce per-rank/per-`mm` with provenance preserved

If host-`mm` separation is still required after the provenance problem is
solved, then the design should become:

- one brokered sandbox-visible `uverbsFD`
- one helper or registration context per rank
- helper affinity for MR-sensitive state
- no loss of allocation provenance across helper routing

The crucial lesson from the updated diagnosis is:

- splitting `mm`s without preserving correct NVIDIA allocation provenance is
  unlikely to be sufficient

## Expected Benefits

- targets the newly identified likely root cause directly
- avoids overcommitting to a large architecture overhaul too early
- preserves the option of per-rank/per-`mm` later if it is still needed
- keeps single-job NCCL semantics intact while the real failure is isolated

## Main Risks

- the required NVIDIA allocation identity may be hard to recover cleanly
- nvproxy or NVIDIA mmap state may not expose enough provenance directly
- disabling broad reuse may increase mapping churn or overhead
- a later per-rank/per-`mm` split may still be needed after provenance is fixed
- the failure may still involve more than one issue: provenance plus collision
  pressure

## Validation Plan

After implementing the provenance-first changes:

1. Verify that `2 GPU/node` still works.
2. Verify that `4 GPU/node` no longer fails on GPU-backed `MR_REG`.
3. Retry `8 GPU/node`.
4. Compare:
   - GPU-backed `MR_REG` success and failure counts
   - identity-mapped versus relocated success rates
   - `PERF` counter patterns
   - NCCL `busbw`
5. Only if correctness is restored but scale still regresses, evaluate
   per-rank/per-`mm` as the next step.

## Recommendation

Do not start with the per-rank/per-`mm` overhaul.

Start by proving and fixing the provenance invariant:

- each GPU-backed `ibv_reg_mr` must see a sentry VMA created from the correct
  NVIDIA allocation context for that exact registration

That is the blocking bug and should be fixed first.

Once that is working, re-measure `4 GPU/node` and `8 GPU/node`. If collisions
and cross-rank pressure still dominate after correctness is restored, then a
per-rank/per-`mm` design becomes one possible next logical step.
