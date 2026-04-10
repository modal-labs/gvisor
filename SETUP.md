# Multi-Node RDMA Test Setup

Two-node distributed PyTorch training over InfiniBand with GPUDirect RDMA.
Covers bare metal, Docker (runc), and gVisor (runsc-rdma).

## Current status (2026-04-10)

| Configuration | GPUs/node | GDR | busbw | Status |
|--------------|-----------|-----|-------|--------|
| Bare metal | 8 | 3 | ~480 GB/s | works |
| Docker runc | 8 | 3 | ~480 GB/s | works |
| gVisor runsc-rdma | 2 | 3 | ~134 GB/s | works (but no true GPU MR_REG — see note) |
| gVisor runsc-rdma | 3+ | 3 | — | EFAULT in ibv_reg_mr |
| gVisor runsc-rdma | 8 | 0 | ~22 GB/s | works (CPU-staged, slow) |

**GDR=3 at 2 GPUs is misleading:** NCCL reports GDR=1 but never calls
`ibv_reg_mr` on GPU memory (`0xa1.../0x50...` VA ranges). It only registers
CPU proxy buffers (`0x7f...`). At 3+ GPUs, NCCL proxy threads attempt
cross-process GPU memory registration, triggering the sentry's
`mirrorGPUDeviceMemory` path which EFAULTs — `nvidia_peermem` can't resolve
the sentry-side VMA to GPU physical pages.

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ibv_reg_mr_iova2 failed with error Bad address` | GPU memory VMA not pinnable by nvidia_peermem | Reduce to 2 GPUs/node, or use GDR=0 |
| `Could not find NET with id N` | Topo/graph XML from wrong hardware | Regenerate via `/baseline` |
| `socketProgress: Connection closed by remote peer` | Master never started | Launch both nodes in one coordinated command |
| `waitForInput: socket timed out after 300000ms` | Master didn't start within 5 min | Same — check launch coordination |
| GDR 0, 2 channels, ~4 GB/s | Container missing libibverbs | Rebuild torch-slim image |
| `modprobe nvidia_peermem: Invalid argument` | Already loaded | Ignore |
| Zero port counter deltas | Crash before data flowed | Check test output for NCCL errors |

## Variables

All skills and commands assume these are set (discovered during `/node-setup`):

```bash
export MASTER_ADDR=<Node 0 IP>       # eth0 or ens7
export NCCL_IB_HCA=<active HCA list> # comma-separated, only "4: ACTIVE" ports
export DEVS=$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
export PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export IFNAME=$(ip -4 addr show ens7 &>/dev/null && echo ens7 || echo eth0)
```

## Workflow

### 1. Provision nodes → `/node-setup`

```
/node-setup wo-abc123 wo-def456
```

Checks InfiniBand exists (rejects A100/non-IB nodes), clones repo, installs
PyTorch via get-pip.py bootstrap, loads nvidia_peermem, discovers IPs and
active HCAs, verifies cross-node connectivity.

### 2. Establish baselines → `/baseline`

```
/baseline wo-abc123 wo-def456
```

Runs bare-metal NCCL allreduce (generates topo XML that gVisor needs), then
Docker runc allreduce. Reports busbw, GDR, PXN, channels for both.

### 3. Test gVisor → `/runsc-test`

```
/runsc-test wo-abc123 wo-def456 --gpus 2 --gdr 3 --build
```

Builds runsc from source, deploys as runsc-rdma Docker runtime, runs NCCL
allreduce under gVisor with port counter measurement. Flags:

- `--gpus N` — GPUs per node (default 2; 3+ EFAULTs with GDR=3)
- `--gdr 0|3` — GPUDirect level (default 3)
- `--build` — rebuild runsc before testing
- `--debug` — enable sentry debug logging, analyze rdmaproxy stats after run

Port counters are always captured before/after. Reports per-HCA deltas,
total bytes, rail balance.

With `--debug`, also reports: MR_REG count, GPU MR_REGs, relocated MRs,
EFAULT count, ioctl breakdown, and EFAULT context from sentry logs.

## Hardware notes

**HCA naming:** Not all HCAs carry inter-node traffic. `mlx5_1` and `mlx5_7`
are typically NVSwitch management ports (always zero port counters). `mlx5_2`
and `mlx5_8` are often DOWN. Auto-discover with the active HCA loop.

**Network interface:** `eth0` on H200 Oracle nodes, `ens7` on some B200 nodes.
The skills auto-detect this.

**Python version:** 3.9 or 3.10 depending on the base image. Affects the
volume mount path for PyTorch site-packages.

**Topo XML:** gVisor hides PCI topology from the container. NCCL needs
`NCCL_TOPO_FILE` pointing to a topo XML generated on bare metal (via
`NCCL_TOPO_DUMP_FILE`). The `/baseline` skill generates this automatically.
Topo XML is hardware-specific — regenerate when switching node types.

## Test script

`torch_mnist_train.py` runs:
1. Allreduce bandwidth measurement (5 warmup + 20 trials, 4 GB payload)
2. Optional MNIST training (1 epoch, synthetic data)

Set `BW_ONLY=1` to skip training (~10s instead of ~60s).
