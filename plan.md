# RDMA Plan - 2026-04-10

## Context

The current `runsc-rdma` design uses one sentry process and one host address
space for the whole local sandbox. That works for low-fanout GPUDirect RDMA
cases, but it appears to break down once enough local ranks and GPU mappings
are present that GPU virtual address ranges collide inside that shared host
`mm`.

The performance target remains:

- about `484 GB/s` `busbw` as the `runc` / bare-metal reference point
- about `436 GB/s` `busbw` as the 10%-from-`runc` target floor

That context matters because the successful `2 GPUs/node` result of about
`133.4 GB/s` is strong evidence that the architecture works in principle, while
still being far from the actual end goal.

The working hypothesis remains:

- 1 GPU/node and 2 GPUs/node succeed because the RDMA and GDR plumbing is
  fundamentally working.
- 8 GPUs/node fails because some GPU-backed mappings must be relocated in the
  sentry address space.
- Once relocation happens, field rewriting (`start`, `addr`, `hca_va`, `IOVA`)
  becomes fragile and may not be sufficient for the lifetime of the MR and the
  later fast path.

## Prior Data Points That Matter

The strongest positive evidence so far is that GPUDirect RDMA already works
under gVisor in the low-fanout case. The architecture is not fundamentally
wrong; it appears to need a scalable answer for more local GPUs.

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

### Interpretation of those data points

- GPUDirect RDMA works under gVisor
- NCCL is taking the intended `NET/IB/GDRDMA` path
- local topology behavior looks sensible in the 2-GPU case
- the current architecture is viable in principle
- the remaining problem is scaling the address-space model to more GPUs, not
  discovering whether GDR works at all

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
- to relocation and address-identity mismatch after the mirror succeeds

In other words, the architecture question is no longer "can `rdmaproxy` mirror
GPU memory at all?" It is now much closer to "what happens when the mirrored
GPU VMA cannot stay at the app-visible VA in one shared sentry `mm`?"

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

The right way to read the results is to correlate three streams:

- periodic `rdmaproxy: PERF ...` rate lines
- explicit `GPU VA collision ...` lines
- compact `MR_REG handle=...` summary lines

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
hot loop, which supports a helper-based design.

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

### Step 3: Read collision logs as proof of shared-`mm` pressure

Each `GPU VA collision ...` line means:

- the code attempted to map the NVIDIA-backed VMA at the app-aligned GPU VA
- `MAP_FIXED_NOREPLACE` found that range already occupied in the sentry
- the code had to retry at some free sentry VA instead

That is the direct evidence for address-space pressure in the shared sentry
`mm`.

Questions to answer from these logs:

- how often do collisions happen?
- do they appear only at 8 GPUs or already at 2 GPUs?
- are they concentrated in one task or spread across many `tgid_root` values?
- do repeated collisions target the same VA ranges?

### Step 4: Read compact MR summaries as the bridge between frequency and failure

Each `MR_REG handle=...` summary line should be used to answer:

- did the MR preserve VA identity?
- if not, how often was it relocated?
- was `hca_va` rewritten?
- was `iova` rewritten?
- which task owned the MR?

This is the best low-overhead bridge between:

- the frequency view from `PERF`
- the address-space view from collision logs
- the failure symptom from `ibv_reg_mr_iova2 failed with error Bad address`

### Step 5: Build a simple interpretation table

For each run configuration, summarize:

- startup `mr_reg`
- steady-state `mr_reg`
- startup `qp_create` / `cq_create`
- steady-state `qp_create` / `cq_create`
- collision count
- relocated MR count
- count of rewritten `hca_va`
- count of rewritten `iova`
- dominant ioctl object and method pairs
- dominant error codes
- whether the run succeeded
- final `busbw`

The architecture implications are then much easier to read:

- low steady-state syscall churn + high collision rate at 8 GPUs
  - strongly supports per-rank/per-`mm`
- high steady-state churn + high collision rate
  - helper design may still be right, but cost risk is higher
- query-dominated control plane + high collision and relocation rate
  - supports keeping helpers focused on MR ownership and setup-path state, not
    every ioctl class
- low collision rate + persistent failures
  - suggests another registration or post-registration edge case
- collisions only for one task/rank
  - may indicate uneven VA behavior or one rank-specific reuse pattern
- collisions across many tasks/ranks
  - more strongly supports the shared-`mm` root-cause theory

### Step 6: What result would most strongly justify the helper architecture?

The strongest justification would look like this:

- 1 GPU and 2 GPU runs show low steady-state control-plane churn
- 8 GPU run shows repeated `GPU VA collision ...` lines
- the corresponding MR summaries show frequent relocation
- failures cluster around relocated MRs
- NCCL topology still looks correct in the low-fanout case

That combination would say:

- the RDMA architecture works
- the hot loop is not dominated by verbs setup traffic
- the shared sentry `mm` is the scaling bottleneck

At that point, the main remaining question is implementation cost, not
architectural direction.

## Additional Diagnoses To Refine The Architecture

In addition to the new low-overhead op counters, there are a few diagnoses that
would materially improve confidence about where the architecture must change.

### 1. Pinpoint exactly where VA collisions occur

Today the most relevant collision and relocation evidence already comes from the
GPU VMA mapping path and the MR rewrite path.

Current useful log sites in
`pkg/sentry/devices/rdmaproxy/rdmaproxy_ioctl_unsafe.go` include:

- `MR_REG (modern) sandbox_va=%#x length=%d`
- `MR REG (INVOKE_WRITE) sandbox_va=%#x length=%d`
- `relocated GPU MR rewrote hca_va ... (start ... -> ...)`
- `relocated GPU MR rewrote iova ... (addr ... -> ...)`
- `GPU VMA mmap failed ... at %#x len %d: %v`
- `GPU device memory VMA created at %#x len %d -> sentry %#x`
- `created nvidia-backed VMA for GPU VA ... -> sentry %#x`
- `reused existing nvidia-backed VMA for GPU VA ...`

Those tell us:

- which sandbox GPU VA is being registered
- whether it stayed identity-mapped or had to move
- whether `MAP_FIXED_NOREPLACE` hit an existing mapping and forced a fallback
- which NVIDIA frontend candidate was used

### 2. Use the explicit collision log at the `EEXIST` branch

The code now emits one explicit "collision happened here" line at the point of
`MAP_FIXED_NOREPLACE` returning `EEXIST`.

That is one of the most useful new diagnoses.

Current log content:

- requested GPU VA range
- requested aligned length
- candidate `sandboxFD`
- candidate `hostFD`
- device name
- whether the fallback reused an existing shared mapping or created a new free
  VMA elsewhere

This makes it much easier to answer:

- how often collisions occur
- which GPU VA ranges collide
- whether collisions are mostly exact-range reuse or true incompatible overlap

### 3. Use the compact summary for every GPU-backed MR

For each successful GPU-backed MR registration, the compact summary log now
captures:

- app VA start and end
- sentry VA start and end
- whether identity mapping held
- whether relocation happened
- whether `hca_va` or `IOVA` was rewritten
- MR handle on success

This lets us correlate:

- registration failures
- relocation events
- later `ibv_reg_mr_iova2` bad-address failures

### 4. Identify which rank/task owns each conflicting range

The next most important missing datum is ownership:

- which sandbox task or thread group attempted the mapping
- whether the collision came from the same rank reusing a prior range or a
  different rank conflicting in the shared sentry `mm`

Recommended extra metadata in logs:

- task ID
- thread-group ID
- sandbox PID if available
- host `uverbsFD`

This is the cleanest way to validate the "one shared host `mm` across local
ranks causes the collision" hypothesis.

### 5. Distinguish exact reuse from partial overlap

Today the shared-GPU-VMA path is keyed by exact `[gpuStart, gpuLen]`. That
means it helps for exact-range reuse, but not necessarily for partial overlap
or same-base-different-length cases unless the fallback path succeeds.

An additional diagnosis worth adding is overlap classification:

- exact existing range reuse
- same start, different length
- partial overlap
- disjoint but same candidate device

If most collisions are exact or same-start-different-length reuse, the current
sharing model may be salvageable further. If many are true cross-rank partial
overlaps, that more strongly supports per-rank/per-`mm`.

### 6. Correlate MR churn with collision and relocation rates

With the new counters in place, the next refinement is to correlate:

- `mr_reg`
- `mr_dereg`
- relocation logs
- `EEXIST` collision logs

This answers whether:

- collisions happen only during startup object creation
- or the workload continues to create VA pressure during steady state

That distinction matters a lot for how expensive a helper-based solution can be.

### 7. Keep one narrow experiment in reserve

The current `hca_va` / `IOVA` rewrite patch is still a useful diagnostic.
Even if the long-term answer is per-rank/per-`mm`, it can still help separate:

- "registration fails because the visible RDMA address is stale"
from
- "registration fails because the shared-`mm` model is fundamentally wrong"

If that patch makes 8-GPU registration succeed but runtime still regresses, the
architecture problem is more likely in post-registration address identity or
topology behavior, not the initial pinning step alone.

## Collision-Avoidance Direction

The likely long-term direction is to separate GPU VA-sensitive RDMA
registration into multiple host address spaces while preserving single-job
semantics for NCCL.

The key idea is:

- keep one container / one NCCL job from the application's point of view
- stop forcing all local ranks to share one host `mm` for GPU VA mirroring
- instead, give each local rank or GPU-owning execution context its own host
  registration context

This should let each rank preserve GPU VA identity locally and avoid the
relocations that currently trigger MR registration failures.

## Proposed Implementation Procedure

### Phase 0: Confirm the control-plane cost model

Before major architecture work:

1. Run the new counters on:
   - 1 GPU/node
   - 2 GPUs/node
   - 8 GPUs/node
2. Compare startup behavior versus steady state.
3. Confirm whether object creation and MR registration are mostly front-loaded.

Exit criteria:

- helper routing is plausible if steady-state lifecycle activity is low

### Phase 1: Introduce broker/helper vocabulary in `rdmaproxy`

Refactor the current single-host-FD mental model into:

- front-end broker:
  - the existing sandbox-visible `uverbsFD`
- helper contexts:
  - one helper process or dedicated registration context per rank

Each helper context should own:

- its own host `uverbs` FD
- its own host `mm`
- its own mirrored GPU VA mappings
- its own pinned object bookkeeping

At this stage the code can still route everything to one helper by default; the
goal is to establish the abstraction boundaries first.

### Phase 2: Define the routing key

Route requests by local rank identity, not by virtual address.

Practical options:

- sandbox thread group / process identity
- the process that opened the verbs FD
- an explicit rank tag if one becomes available

For NCCL and `torchrun`, the most natural first approximation is:

- one sandbox process == one local rank

This means the broker can maintain:

- `task group -> helper`

### Phase 3: Move MR registration ownership to helpers

For `MR_REG`:

1. broker identifies the calling rank
2. broker selects that rank's helper
3. broker forwards the registration request to the helper
4. helper mirrors the GPU VA into its own host `mm`
5. helper performs the host-side MR registration
6. helper returns the resulting host MR handle

The crucial property is:

- the helper performs registration from an `mm` where the GPU VA can remain at
  the address the application expects

### Phase 4: Add stable ownership tables

The sentry cannot rediscover helper ownership later by re-examining the caller.
It must remember ownership at creation time.

At minimum the broker needs:

- `mr handle -> helper`

In practice, a safer design is:

- sandbox-visible synthetic MR ID -> `{ helper, hostMRHandle }`

That avoids assuming raw host handles are globally unique across helpers.

### Phase 5: Route MR teardown to the owning helper

For `DEREG_MR`:

1. extract the sandbox-visible handle
2. look up ownership in the broker table
3. forward the request to the owning helper
4. helper deregisters the host MR and releases its mirrored mapping
5. broker drops the ownership entry

This is the same lifetime pattern the current code already uses with
`pinnedMRs`, but the value becomes ownership metadata instead of only mirrored
pages.

### Phase 6: Determine object-affinity requirements beyond MR

The next question is how much more than MR must be helper-affine.

Likely candidates:

- PD
- CQ
- QP
- async event FD handling

The conservative rule is:

- any object whose handles or backing memory are only meaningful in the helper
  that created them must remain bound to that helper

This likely means the broker eventually needs ownership maps for more than MRs.
The implementation should start narrow and expand only when traces or failures
show that it is required.

### Phase 7: Keep the hot path out of helper IPC if possible

The helper architecture is only attractive if it mostly affects setup and
ownership, not the packet-rate data path.

The design goal should be:

- helper IPC for creation / destruction / registration / teardown
- no helper IPC for every queue operation in the steady-state allreduce loop

If the architecture forces cross-process mediation for every work request,
completion poll, or doorbell interaction, the overhead may become too high.

### Phase 8: Prototype order

Recommended implementation order:

1. preserve the current single-broker front-end
2. create one helper per local rank
3. route only `MR_REG` / `DEREG_MR` first
4. validate that 8-GPU MR registration now succeeds more reliably
5. only then expand helper affinity to PD / CQ / QP if required

This minimizes the amount of new complexity introduced before the first
architectural answer is available.

## Expected Benefits

- avoids GPU VA collisions caused by one shared host `mm`
- reduces dependence on increasingly fragile field rewriting
- keeps single-job NCCL semantics intact
- may preserve local topology behavior better than the multi-sandbox workaround

## Main Risks

- helper lifecycle complexity
- handle namespace translation complexity
- more IPC and context switching
- possible need to make more RDMA objects helper-affine than initially expected
- risk that some fast-path operations still assume stronger address identity or
  object locality than the initial helper design provides

## Validation Plan

After implementing a minimal helper prototype:

1. Verify 1 GPU/node still works.
2. Verify 2 GPUs/node still works.
3. Retry 8 GPUs/node.
4. Compare:
   - MR registration correctness
   - steady-state `PERF` counter patterns
   - NCCL `busbw`
5. If 8-GPU registration succeeds but performance remains poor, investigate:
   - helper IPC overhead
   - loss of local NCCL optimizations
   - remaining GPU VA-sensitive objects outside MR

## Recommendation

Use the newly added counters first to confirm whether MR and object lifecycle
traffic is mostly startup-path. If it is, move to a minimal per-rank/per-`mm`
prototype centered on MR ownership and helper routing, rather than continuing
to add more ad hoc address-rewrite logic on top of one shared host address
space.
