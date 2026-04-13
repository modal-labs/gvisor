---
name: runsc-test
description: Build runsc, deploy, run gVisor RDMA test with port counters and optional debug logging
user-invocable: true
allowed-tools: Bash
argument-hint: [node0-host node1-host] [--gpus N] [--gdr 0|3] [--debug] [--build]
---

# gVisor runsc-rdma Test

Build and deploy runsc, run NCCL allreduce under gVisor with port counter
measurement and optional sentry debug logging.

Assumes `/node-setup` has been run and `/baseline` has generated a topo XML.

## Arguments

$ARGUMENTS should start with two SSH hostnames, then optional flags:
- `wo-abc123 wo-def456` — the two node hostnames
- `--gpus N` — GPUs per node (default 2). **Known limitation:** GDR=3 EFAULTs at 3+.
- `--gdr 0|3` — GDR level (default 3). Use 0 for CPU-staged RDMA fallback.
- `--debug` — enable sentry debug logging for this run
- `--build` — rebuild runsc from source before testing

## SSH convention

All SSH commands MUST use `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`.
Pipe stderr through `2>&1 | grep -v "^Warning"` to suppress known-hosts noise.
See `/node-setup` for details.
- SSH user is `modal`

If no arguments given, ask the user for the node hostnames.

## Environment discovery

Same as `/baseline` — run on node 0 to get MASTER_IP, IFNAME, NCCL_IB_HCA, PYVER.

## Step 1: Build and deploy (if `--build` or first run)

### Pull latest code
```
ssh modal@<node> 'cd ~/gvisor && git remote set-url origin https://github.com/modal-labs/gvisor.git && git pull'
```
Run on both nodes in parallel.

### Build runsc
```
ssh modal@<node> 'cd ~/gvisor && sudo make copy TARGETS=runsc DESTINATION=/usr/local/bin/runsc'
```
Run on both nodes in parallel. First build: 5-15 min. Incremental: ~1 min.
Monitor: `ssh modal@<node> 'sudo docker ps | grep bazel'`

### Deploy
```
ssh modal@<node> 'sudo cp /usr/local/bin/runsc /usr/local/bin/runsc-rdma && sudo chmod +x /usr/local/bin/runsc-rdma'
```

### Register Docker runtime
```
ssh modal@<node> 'sudo python3 -c "
import json, pathlib
p = pathlib.Path(\"/etc/docker/daemon.json\")
cfg = json.loads(p.read_text().strip()) if p.exists() else {}
cfg.setdefault(\"runtimes\", {})[\"runsc-rdma\"] = {
    \"path\": \"/usr/local/bin/runsc-rdma\",
    \"runtimeArgs\": [
        \"--rdmaproxy\", \"--nvproxy\",
        \"--nvproxy-allowed-driver-capabilities=compute,utility,video\",
        \"--network=host\", \"--rdma-expected-ipoib=-1\",
    ],
}
p.write_text(json.dumps(cfg, indent=2) + chr(10))
" && sudo systemctl restart docker'
```

### Build torch-slim image
```
ssh modal@<node> 'cd ~/gvisor && sudo docker build -f Dockerfile.torch-slim -t torch-slim .'
```

### Verify
```
ssh modal@<node> '/usr/local/bin/runsc-rdma --version && sudo docker info 2>&1 | grep -A3 Runtimes'
```

## Step 2: Enable debug logging (if `--debug`)

```
ssh modal@<node> 'sudo python3 -c "
import json, pathlib
p = pathlib.Path(\"/etc/docker/daemon.json\")
cfg = json.loads(p.read_text())
args = cfg[\"runtimes\"][\"runsc-rdma\"][\"runtimeArgs\"]
if \"--debug\" not in args:
    args.extend([\"--debug-log=/tmp/runsc-logs/\", \"--debug\"])
    p.write_text(json.dumps(cfg, indent=2) + chr(10))
" && sudo mkdir -p /tmp/runsc-logs && sudo rm -f /tmp/runsc-logs/*.txt && sudo systemctl restart docker'
```

Run on both nodes. **Warning:** Debug logging can cause timing failures at 4+ GPUs.

## Steps 3-5: Run the test (SINGLE Bash tool call)

**CRITICAL:** Steps 3-5 (BEFORE counters → launch node 1 → launch node 0 →
AFTER counters → node 1 log) MUST be executed as ONE Bash tool call. Do NOT
split these into separate tool calls — node 1 will time out waiting for node 0
if there is any gap between launches.

Build this as one bash script with shell variables at the top:

```bash
export MASTER_IP="<from env discovery>"
export PORT=<29500 + random offset>
export NODE0="<node0-host>"
export NODE1="<node1-host>"
export GPUS=<gpu count>
export GDR=<0 or 3>
export DMABUF=$( [ "$GDR" = "3" ] && echo 1 || echo 0 )

# Cleanup any stale containers
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 'sudo docker rm -f nccl-runsc 2>/dev/null; echo ok' 2>&1
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE0 'sudo docker kill $(sudo docker ps -q) 2>/dev/null; echo ok' 2>&1

# BEFORE port counters (both nodes)
echo "=== BEFORE ==="
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE0 'for dev in $(ls -d /sys/class/infiniband/mlx5_* | xargs -I{} basename {}); do rcv=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data); xmit=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_xmit_data); echo "N0 $dev: rcv=$rcv xmit=$xmit"; done' 2>&1
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 'for dev in $(ls -d /sys/class/infiniband/mlx5_* | xargs -I{} basename {}); do rcv=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data); xmit=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_xmit_data); echo "N1 $dev: rcv=$rcv xmit=$xmit"; done' 2>&1

# Launch node 1 (background via nohup on remote)
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 "bash -c '
DEVS=\$(ls /dev/infiniband/uverbs* | sed \"s/^/--device=/\" | tr \"\n\" \" \")
NCCL_IB_HCA=\$(for dev in /sys/class/infiniband/mlx5_*; do d=\$(basename \$dev); grep -q \"4: ACTIVE\" \$dev/ports/1/state 2>/dev/null && echo -n \"\$d,\"; done | sed \"s/,\$//\")
nohup sudo docker run --runtime=runsc-rdma --name nccl-runsc --gpus all \$DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v \$HOME/.local/lib/python<PYVER>/site-packages:/usr/local/lib/python<PYVER>/dist-packages \
  -v \$HOME/.local/bin:/usr/local/bin \
  -v \$HOME/gvisor:/workspace \
  -e NCCL_NET_GDR_LEVEL=$GDR -e NCCL_DMABUF_ENABLE=$DMABUF \
  -e NCCL_TOPO_FILE=/workspace/nccl_topo.xml \
  -e NCCL_SOCKET_IFNAME=<IFNAME> -e GLOO_SOCKET_IFNAME=<IFNAME> \
  -e NCCL_IB_HCA=\$NCCL_IB_HCA -e BW_ONLY=1 -e OMP_NUM_THREADS=1 \
  -w /workspace torch-slim \
  torchrun --nproc_per_node=$GPUS --nnodes=2 --master_addr=$MASTER_IP --master_port=$PORT --node_rank=1 ./torch_mnist_train.py \
  > /tmp/nccl-runsc.log 2>&1 &
echo launched
'" 2>&1

# Wait for container to start
sleep 3
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 'sudo docker ps --filter name=nccl-runsc --format "{{.Names}} {{.Status}}"' 2>&1

# Launch node 0 (foreground — blocks until test completes)
echo "=== NODE 0 OUTPUT ==="
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE0 "
DEVS=\$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
NCCL_IB_HCA=\$(for dev in /sys/class/infiniband/mlx5_*; do d=\$(basename \$dev); grep -q '4: ACTIVE' \$dev/ports/1/state 2>/dev/null && echo -n \"\$d,\"; done | sed 's/,$//')
sudo docker run --runtime=runsc-rdma --rm --gpus all \$DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v \$HOME/.local/lib/python<PYVER>/site-packages:/usr/local/lib/python<PYVER>/dist-packages \
  -v \$HOME/.local/bin:/usr/local/bin \
  -v \$HOME/gvisor:/workspace \
  -e NCCL_NET_GDR_LEVEL=$GDR -e NCCL_DMABUF_ENABLE=$DMABUF \
  -e NCCL_TOPO_FILE=/workspace/nccl_topo.xml \
  -e NCCL_SOCKET_IFNAME=<IFNAME> -e GLOO_SOCKET_IFNAME=<IFNAME> \
  -e NCCL_IB_HCA=\$NCCL_IB_HCA -e BW_ONLY=1 -e OMP_NUM_THREADS=1 \
  -w /workspace torch-slim \
  torchrun --nproc_per_node=$GPUS --nnodes=2 --master_addr=$MASTER_IP --master_port=$PORT --node_rank=0 ./torch_mnist_train.py
" 2>&1

# AFTER port counters (both nodes)
echo "=== AFTER ==="
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE0 'for dev in $(ls -d /sys/class/infiniband/mlx5_* | xargs -I{} basename {}); do rcv=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data); xmit=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_xmit_data); echo "N0 $dev: rcv=$rcv xmit=$xmit"; done' 2>&1
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 'for dev in $(ls -d /sys/class/infiniband/mlx5_* | xargs -I{} basename {}); do rcv=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data); xmit=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_xmit_data); echo "N1 $dev: rcv=$rcv xmit=$xmit"; done' 2>&1

# Node 1 log and cleanup
echo "=== NODE 1 LOG ==="
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 'cat /tmp/nccl-runsc.log | grep -E "busbw|algbw|FATAL|ibv_reg_mr|BW_ONLY" | head -10; sudo docker rm -f nccl-runsc 2>/dev/null' 2>&1
```

After running, compute port counter deltas:
- Per-HCA: `(after - before) * 4 / 1e9` = GB
- Total across all HCAs
- Flag zero-delta HCAs (mlx5_1, mlx5_7 = NVSwitch mgmt, expected)
- Verify traffic is balanced across rails
- Verify rcv ≈ xmit (symmetric for allreduce)
- Zero total delta = crash before data flowed

## Step 6: Analyze sentry logs (if `--debug`)

```
ssh modal@<node0> 'LOG=$(sudo ls -t /tmp/runsc-logs/runsc.log.*boot.txt 2>/dev/null | head -1)
echo "Log: $LOG ($(sudo wc -c < $LOG) bytes)"
echo "MR_REGs: $(sudo grep -c "MR_REG handle" $LOG)"
echo "GPU MR_REGs: $(sudo grep "MR_REG handle" $LOG | sudo grep -cE "app=0x50|app=0xa1")"
echo "Relocated MRs: $(sudo grep -c "relocated GPU MR" $LOG)"
echo "GPU VMAs created: $(sudo grep -c "GPU device memory VMA created" $LOG)"
echo "EFAULT errors: $(sudo grep -c "errno=14" $LOG)"
echo "=== Error breakdown ==="
sudo grep "host ioctl returned.*errno" $LOG | sed "s/.*errno=\([0-9]*\).*/errno=\1/" | sort | uniq -c | sort -rn
echo "=== GPU MR_REGs ==="
sudo grep "MR_REG handle" $LOG | grep -E "app=0x50|app=0xa1"
echo "=== Relocated GPU MRs ==="
sudo grep "relocated GPU MR" $LOG
echo "=== EFAULT context ==="
sudo grep -B10 "errno=14" $LOG | grep -E "MR_REG|MR REG rewrote|GPU device memory VMA|forwarding ioctl|errno=14"'
```

Then disable debug logging:
```
ssh modal@<node> 'sudo python3 -c "
import json, pathlib
p = pathlib.Path(\"/etc/docker/daemon.json\")
cfg = json.loads(p.read_text())
cfg[\"runtimes\"][\"runsc-rdma\"][\"runtimeArgs\"] = [a for a in cfg[\"runtimes\"][\"runsc-rdma\"][\"runtimeArgs\"] if \"debug\" not in a]
p.write_text(json.dumps(cfg, indent=2) + chr(10))
" && sudo systemctl restart docker'
```

## Step 7: Report

Cleanup node 1:
```
ssh modal@<node1> 'cat /tmp/nccl-runsc.log | tail -10; sudo docker rm -f nccl-runsc 2>/dev/null'
```

Present results:
- busbw and algbw (GB/s and Gb/s)
- GDR level, PXN, channel count
- Port counter delta table (per-HCA and total)
- If `--debug`: sentry log summary (MR_REGs, GPU MR_REGs, EFAULTs)
- Any FATAL errors with root cause analysis

### Common failures

| Error | Cause | Fix |
|-------|-------|-----|
| `Could not find NET with id N` | Wrong topo/graph XML | Regenerate via `/baseline` on this hardware |
| `Connection closed by remote peer` | Node 1 started alone, node 0 never launched | Check coordinated launch timing |
| Zero port counter deltas | Crashed before data flowed | Check NCCL errors in output |
