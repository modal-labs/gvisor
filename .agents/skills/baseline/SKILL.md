---
name: baseline
description: Run bare-metal and Docker runc NCCL baselines, generate topo XML
user-invocable: true
allowed-tools: Bash
argument-hint: [node0-host node1-host]
---

# Baseline NCCL Tests

Run bare-metal and Docker runc NCCL allreduce benchmarks on two nodes.
Establishes the reference busbw and generates topo XML that gVisor needs.

Assumes `/node-setup` has already been run on both nodes.

## Arguments

$ARGUMENTS should be two SSH hostnames, e.g.: `wo-abc123 wo-def456`
- SSH user is `modal`

## SSH convention

All SSH commands MUST use `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`.
Pipe stderr through `2>&1 | grep -v "^Warning"` to suppress known-hosts noise.
See `/node-setup` for details.

## Environment discovery

Run on node 0 and parse the output. All subsequent steps use these values.

```
ssh modal@<node0> 'echo MASTER_IP=$(ip -4 addr show eth0 2>/dev/null | grep inet | awk "{print \$2}" | cut -d/ -f1 || ip -4 addr show ens7 | grep inet | awk "{print \$2}" | cut -d/ -f1); \
  echo IFNAME=$(ip -4 addr show ens7 &>/dev/null && echo ens7 || echo eth0); \
  echo NCCL_IB_HCA=$(for dev in /sys/class/infiniband/mlx5_*; do d=$(basename $dev); grep -q "4: ACTIVE" $dev/ports/1/state 2>/dev/null && echo -n "$d,"; done | sed "s/,$//"); \
  echo PYVER=$(python3 -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")'
```

## Coordinated launch

**CRITICAL:** Each test MUST run as ONE Bash tool call — nohup node 1 → sleep 3
→ node 0 foreground → collect results. Do NOT split into separate tool calls;
node 1 times out (~300 s) if node 0 doesn't start promptly.

Pick a unique port for each test (e.g. 29500 + random offset).

## Part 1: Bare-metal test

Generates topo XML that gVisor needs. Run as a single Bash call:

```bash
export MASTER_IP="<from env discovery>"
export PORT=<unique port>
export NODE0="<node0-host>"
export NODE1="<node1-host>"
export IFNAME="<eth0 or ens7>"
export NCCL_IB_HCA="<active HCA list>"

# Node 1 (background)
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 "bash -c '
sudo prlimit --pid=\$\$ --memlock=unlimited:unlimited && export PATH=\$HOME/.local/bin:\$PATH && cd ~/gvisor && \
  NCCL_DEBUG=INFO NCCL_NET_GDR_LEVEL=3 NCCL_DMABUF_ENABLE=1 \
  NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME \
  NCCL_IB_HCA=$NCCL_IB_HCA BW_ONLY=1 OMP_NUM_THREADS=1 \
  torchrun --nproc_per_node=8 --nnodes=2 --master_addr=$MASTER_IP --master_port=$PORT --node_rank=1 ./torch_mnist_train.py \
  > /tmp/bare-metal-n1.log 2>&1 &
echo launched
'" 2>&1

sleep 3

# Node 0 (foreground, with topo dump)
echo "=== BARE METAL ==="
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE0 "
sudo prlimit --pid=\$\$ --memlock=unlimited:unlimited && export PATH=\$HOME/.local/bin:\$PATH && cd ~/gvisor && \
  NCCL_DEBUG=INFO NCCL_NET_GDR_LEVEL=3 NCCL_DMABUF_ENABLE=1 \
  NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME \
  NCCL_IB_HCA=$NCCL_IB_HCA BW_ONLY=1 OMP_NUM_THREADS=1 \
  NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml \
  NCCL_GRAPH_DUMP_FILE=/tmp/nccl_graph.xml NCCL_GRAPH_DUMP_FILE_RANK=0 \
  torchrun --nproc_per_node=8 --nnodes=2 --master_addr=$MASTER_IP --master_port=$PORT --node_rank=0 ./torch_mnist_train.py
" 2>&1

echo "=== NODE 1 LOG ==="
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 'cat /tmp/bare-metal-n1.log | grep -E "busbw|algbw|FATAL" | head -5' 2>&1
```

After the run, copy topo XML to the repo and to node 1:
```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE0 'cp /tmp/nccl_topo.xml ~/gvisor/nccl_topo.xml'
scp -o StrictHostKeyChecking=no modal@$NODE0:~/gvisor/nccl_topo.xml /tmp/nccl_topo.xml
scp -o StrictHostKeyChecking=no /tmp/nccl_topo.xml modal@$NODE1:~/gvisor/nccl_topo.xml
```

**Expected:** ~480 GB/s busbw (H200), ~386 GB/s (B200). GDR 1, PXN 1, 16+ channels.

## Part 2: Docker runc test

Same pattern, single Bash call. Assumes torch-slim image is built (from `/runsc-test --build`
or manually via `sudo docker build -f Dockerfile.torch-slim -t torch-slim .`).

```bash
export PORT=<different unique port>
export PYVER="<from env discovery>"

# Cleanup
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 'sudo docker rm -f runc-test 2>/dev/null' 2>&1

# Node 1 (background)
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 "bash -c '
DEVS=\$(ls /dev/infiniband/uverbs* | sed \"s/^/--device=/\" | tr \"\n\" \" \")
nohup sudo docker run --runtime=runc --name runc-test --gpus all \$DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v \$HOME/.local/lib/python$PYVER/site-packages:/usr/local/lib/python$PYVER/dist-packages \
  -v \$HOME/.local/bin:/usr/local/bin \
  -v \$HOME/gvisor:/workspace \
  -e NCCL_DEBUG=INFO -e NCCL_NET_GDR_LEVEL=3 -e NCCL_DMABUF_ENABLE=1 \
  -e NCCL_SOCKET_IFNAME=$IFNAME -e GLOO_SOCKET_IFNAME=$IFNAME \
  -e NCCL_IB_HCA=$NCCL_IB_HCA -e BW_ONLY=1 -e OMP_NUM_THREADS=1 \
  -w /workspace torch-slim \
  torchrun --nproc_per_node=8 --nnodes=2 --master_addr=$MASTER_IP --master_port=$PORT --node_rank=1 ./torch_mnist_train.py \
  > /tmp/runc-n1.log 2>&1 &
echo launched
'" 2>&1

sleep 3
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 'sudo docker ps --filter name=runc-test --format "{{.Names}} {{.Status}}"' 2>&1

# Node 0 (foreground)
echo "=== RUNC ==="
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE0 "
DEVS=\$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
sudo docker run --runtime=runc --rm --gpus all \$DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v \$HOME/.local/lib/python$PYVER/site-packages:/usr/local/lib/python$PYVER/dist-packages \
  -v \$HOME/.local/bin:/usr/local/bin \
  -v \$HOME/gvisor:/workspace \
  -e NCCL_DEBUG=INFO -e NCCL_NET_GDR_LEVEL=3 -e NCCL_DMABUF_ENABLE=1 \
  -e NCCL_SOCKET_IFNAME=$IFNAME -e GLOO_SOCKET_IFNAME=$IFNAME \
  -e NCCL_IB_HCA=$NCCL_IB_HCA -e BW_ONLY=1 -e OMP_NUM_THREADS=1 \
  -w /workspace torch-slim \
  torchrun --nproc_per_node=8 --nnodes=2 --master_addr=$MASTER_IP --master_port=$PORT --node_rank=0 ./torch_mnist_train.py
" 2>&1

echo "=== NODE 1 LOG ==="
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null modal@$NODE1 'cat /tmp/runc-n1.log | grep -E "busbw|algbw|FATAL" | head -5; sudo docker rm -f runc-test 2>/dev/null' 2>&1
```

**Expected:** Matches bare-metal. If busbw is ~4 GB/s with 2 channels, the
torch-slim image is missing libibverbs — rebuild it.

## Report

For each test, grep node 0 output for `busbw`, `algbw`, `GDR`, `PXN`, `channels`.
Present a comparison table:

| Test | busbw (GB/s) | GDR | PXN | Channels |
|------|-------------|-----|-----|----------|
| Bare metal | | | | |
| Docker runc | | | | |
