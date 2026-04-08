# NCCL RDMA Debug Checklist

Binary pass/fail checks for diagnosing multi-node NCCL bandwidth issues.
Run each check in order — stop at the first failure.

## 1. Is libibverbs visible inside the container?

```bash
docker exec <cid> ls /usr/lib/x86_64-linux-gnu/libibverbs.so.1
```

- **PASS:** file exists
- **FAIL:** `No such file` → NCCL falls back to sockets. Rebuild image with `libibverbs1` and `ibverbs-providers`, or mount from host.

**NCCL log signal:** `Failed to open libibverbs.so[.1]` followed by `transport/net_ib.cc:852 -> 3`

## 2. Does NCCL detect IB devices?

Look for `NET/IB : Using [0]mlx5_...` in NCCL_DEBUG=INFO output.

- **PASS:** `NET/IB : Using [0]mlx5_X:1/RoCE ...`
- **FAIL:** no NET/IB line → libibverbs missing (check 1) or uverbs devices not passed

**Fix:** pass `--device /dev/infiniband/uverbs*` to docker run.

## 3. Is GDR (GPUDirect RDMA) active?

```
Connected all rings, use ring PXN ? GDR ?
```

- **PASS:** `GDR 1`
- **FAIL:** `GDR 0` → NCCL can't register GPU memory for RDMA DMA

**Common causes of GDR 0:**
- `nvidia_peermem` not loaded: `sudo modprobe nvidia_peermem`
- libibverbs missing (check 1)
- NCCL can't open `/dev/infiniband/uverbs*` devices

## 4. How many coll channels?

```
N coll channels, N collnet channels, N nvls channels, N p2p channels
```

- **PASS:** 20 coll channels (matches bare metal)
- **FAIL:** 2 coll channels → CPU affinity mismatch or topology issue

**Common causes of 2 channels:**
- Container cgroup restricts CPUs (check 5)
- Topo file mismatch with actual hardware

## 5. Does the container have all CPUs?

```bash
docker exec <cid> cat /sys/fs/cgroup/cpuset.cpus.effective
docker exec <cid> nproc
```

- **PASS:** `0-111` (or all host CPUs), `nproc` >= 100
- **FAIL:** `0-11` or `12-111` → cgroup cpuset restricting container

**Fix:**
```bash
sudo systemctl set-property modal-containers.slice AllowedCPUs=0-111
# Also set cgroup-parent=modal-containers.slice in daemon.json
```

## 6. Does nvidia_peermem work?

```bash
lsmod | grep nvidia_peermem
cat /sys/module/nvidia_peermem/version
```

- **PASS:** module loaded, version printed
- **FAIL:** module not found → `sudo modprobe nvidia_peermem`

## 7. Is IB traffic actually flowing?

Compare port counters before/after test:

```bash
cat /sys/class/infiniband/mlx5_5/ports/1/counters/port_rcv_data
cat /sys/class/infiniband/mlx5_5/ports/1/counters/port_xmit_data
```

- **PASS:** counters increase by ~5 GB per NIC per direction
- **FAIL:** counters unchanged → NCCL not using IB (check 1-2)

## 8. Quick bandwidth reference

| Scenario | busbw (GB/s) | What's wrong |
|----------|-------------|--------------|
| ~386 | Nothing — full speed, GDR 1, 20 channels |
| ~4-5 | Missing libibverbs → GDR 0, 2 channels |
| ~3-4 | gVisor without libibverbs → rdmaproxy only path |
| ~0 | No IB at all — check device passthrough |

## 9. ib_write_bw sanity check

Tests raw RDMA independently of NCCL:

```bash
# Server: ib_write_bw -d mlx5_5 --report_gbits
# Client: ib_write_bw -d mlx5_5 <SERVER_IP> --report_gbits
```

- **PASS:** ~386 Gb/s
- **FAIL:** check memlock limits, IB device state, nvidia_peermem
