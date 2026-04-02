---

# Sentry CPU & Mutex Profiling Results

Collected during a 16-rank (2-node × 8 GPU) NCCL `all_reduce` benchmark
over InfiniBand with GPUDirect RDMA, running under `runsc-rdma` with the
sched_yield stub-side fast-path enabled and `--strace`/`--debug` disabled.

- **Wall clock:** 11.04s
- **Total CPU samples:** 12.98s (117.6% — multiple cores)
- **Benchmark result:** 212.39 GB/s bus bandwidth at 128 MB (runc baseline: ~306 GB/s)

## CPU Profile

### Flat time (where CPU cycles are actually burned)

| Time | % | Function | What it does |
|------|------|----------|--------------|
| 4.00s | 30.8% | `internal/runtime/syscall.Syscall6` | Go runtime syscall wrapper (futex wake/wait, context switches) |
| 1.60s | 12.3% | `hostsyscall.RawSyscall` | Host ioctl forwarding (nvproxy/rdmaproxy → host kernel) |
| 0.39s | 3.0% | `contextQueue.add` | Systrap context queue management |
| 0.34s | 2.6% | `SeqCount.BeginWrite` | Sequence lock contention |
| 0.26s | 2.0% | `subprocess.switchToApp` | Systrap app context switch |
| 0.25s | 1.9% | `runtime.futex` | Go runtime futex (scheduler, channels) |
| 0.24s | 1.8% | `atomic.CompareAndSwap` | CAS contention (lock-free queues) |
| 0.24s | 1.8% | `sync.Mutex.Lock` | Mutex acquisition |

43% of sentry CPU is spent making host syscalls (`Syscall6` + `RawSyscall`).

### Cumulative time (call tree)

| Time | % | Path | Notes |
|------|------|------|-------|
| 5.33s | 41.1% | `doSyscall` (total) | All sentry syscall processing |
| 4.03s | 31.0% | `platformContext.Switch` | Systrap round-trip to/from stub |
| 3.09s | 23.8% | `subprocess.waitOnState` | Sentry waiting for stub to trap back |
| 2.56s | 19.7% | `Ioctl` (total) | nvproxy + rdmaproxy ioctl forwarding |
| 1.84s | 14.2% | `kickSysmsgThread` | Waking stub threads |
| 1.40s | 10.8% | `nvproxy.frontendFD.Ioctl` | NVIDIA frontend ioctls |
| 1.30s | 10.0% | `RecvFrom` | NCCL bootstrap TCP socket ops |
| 0.87s | 6.7% | `fastPathDispatcher.waitFor` | Dispatcher spin-waiting for contexts |
| 0.80s | 6.2% | `nvproxy.uvmFD.Ioctl` | NVIDIA UVM ioctls |
| 0.69s | 5.3% | `accept` | Socket accept (NCCL bootstrap) |
| 0.61s | 4.7% | `fastPathDispatcher.loop` | Dispatcher main loop |

## Mutex Contention Profile

Total mutex delay: 1.88s. Dominated by memory management locks, not
systrap or futex.

| Delay | % | Lock / Path | What it does |
|-------|------|-------------|--------------|
| 1.09s | 58.1% | `mm.activeRWMutex.DowngradeLock` → `HandleUserFault` | Page fault handling RW lock contention |
| 0.58s | 30.7% | `mm.mappingRWMutex.Unlock` → `MProtect` | mprotect syscall write-lock contention |
| 0.07s | 3.6% | `mm.CopyOutFrom` → `RecvMsg` | Socket recv buffer copy |
| 0.06s | 3.3% | `mm.mappingRWMutex` → `MUnmap` | munmap syscall |
| 0.05s | 2.7% | `mm.mappingRWMutex` → `MMap` | mmap syscall |

89% of mutex contention is in the memory manager — `HandleUserFault` and
`MProtect` fighting over the same RW mutex. CUDA/NCCL calls mprotect
frequently for GPU buffer management; each mprotect takes the write lock
and blocks concurrent page fault handling.

## Optimization Targets

### 1. Reduce systrap round-trip overhead (31% of CPU)

`switchToApp` + `waitOnState` + `kickSysmsgThread` together consume 31% of
CPU, mostly on host futex syscalls to wake/wait on stub threads. Each
application syscall that reaches the sentry pays this cost. Reducing the
number of sentry-visible syscalls (as we did with sched_yield) is the
primary lever. Candidates:

- **clock_gettime**: if called frequently, handle via stub-side vDSO read.
- **Futex fast-path for FUTEX_WAKE with no waiters**: return immediately
  in the stub when the futex word indicates no waiters.

### 2. MM lock contention (89% of mutex delay)

HandleUserFault and MProtect contend on the same RW mutex. If CUDA/NCCL
calls mprotect frequently (for GPU buffer registration), each call holds
the write lock and blocks all concurrent page fault resolution. Options:

- **Finer-grained MM locking**: split the address space into regions with
  independent locks.
- **Batch mprotect operations**: coalesce adjacent mprotect calls.
- **Reduce page faults**: pre-populate mappings for known GPU buffer
  regions (already partially done via `PlatformEffectPopulate`).

### 3. Nvproxy ioctl forwarding (17% of CPU)

Each nvproxy ioctl calls `RawSyscall` which blocks the OS thread without
releasing the Go P. At 10.8% frontend + 6.2% UVM = 17% of CPU in ioctl
forwarding. Options:

- **Switch `RawSyscall` to `Syscall`** for long-running ioctls so the Go
  scheduler can reuse the P while the thread blocks in the kernel.
- **Dedicated forwarding goroutine/thread**: offload ioctls to a pool of
  OS threads that don't hold Go P's.

### 4. Dispatcher overhead (7% of CPU)

`fastPathDispatcher.waitFor` + `.loop` consume ~7%. The dispatcher uses a
global mutex to manage the context list. For workloads with many concurrent
syscalls, this serializes context scheduling. Possible improvements:

- Per-CPU or sharded dispatcher queues.
- Lock-free context handoff.

### 5. Network overhead (15% of CPU)

`RecvFrom` (10%) + `accept` (5%) are NCCL bootstrap operations (TCP
signaling for rank coordination). These are transient (setup/teardown
phase) and not in the steady-state data path, so optimizing them has
limited impact on large-message bandwidth.
