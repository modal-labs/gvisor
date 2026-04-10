# Multi-Node RDMA Test Setup

Two-node distributed PyTorch training over InfiniBand with GPUDirect RDMA.
Covers bare metal, Docker (runc), and gVisor (runsc-rdma).

## 1. Host prerequisites (both nodes)

```bash
# Verify InfiniBand exists before proceeding
ls /dev/infiniband/uverbs* || { echo "No InfiniBand — wrong node type"; exit 1; }

# Clone and build gVisor
git clone https://github.com/modal-labs/gvisor.git && cd gvisor
git checkout alessio/development
sudo make copy TARGETS=runsc DESTINATION=/usr/local/bin/runsc
sudo chmod +x /usr/local/bin/runsc

# Load nvidia_peermem for GPUDirect RDMA
sudo modprobe nvidia_peermem

# Install PyTorch (pip3 may not be preinstalled on Modal workers)
export PATH="$HOME/.local/bin:$PATH"
pip3 --version 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | python3 -
pip3 install torch torchvision
```

## 2. Discover node IPs and IB HCAs

```bash
# Node A IP (use eth0 or ens7, whichever exists)
export MASTER_ADDR=$(ip -4 addr show eth0 2>/dev/null | grep inet | awk '{print $2}' | awk -F/ '{print $1}')
[ -z "$MASTER_ADDR" ] && export MASTER_ADDR=$(ip -4 addr show ens7 | grep inet | awk '{print $2}' | awk -F/ '{print $1}')
echo "export MASTER_ADDR=$MASTER_ADDR"
# Copy and paste this export on Node B.

# List active IB devices (exclude DOWN ports)
for dev in /sys/class/infiniband/mlx5_*; do
  d=$(basename $dev)
  state=$(cat $dev/ports/1/state)
  echo "$d: $state"
done

# Set HCA list — only include ACTIVE ports (state "4: ACTIVE")
# Examples seen in practice:
#   B200 nodes:  mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12
#   H200 nodes:  mlx5_0,mlx5_1,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_9,mlx5_10,mlx5_11
export NCCL_IB_HCA=$(for dev in /sys/class/infiniband/mlx5_*; do
  d=$(basename $dev)
  grep -q "4: ACTIVE" $dev/ports/1/state 2>/dev/null && echo -n "$d,"
done | sed 's/,$//')
echo "NCCL_IB_HCA=$NCCL_IB_HCA"
```

## 3. Bare-metal test

```bash
sudo prlimit --pid=$$ --memlock=unlimited:unlimited

NCCL_DEBUG=INFO \
NCCL_NET_GDR_LEVEL=3 \
NCCL_DMABUF_ENABLE=1 \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}" \
GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}" \
NCCL_IB_HCA="$NCCL_IB_HCA" \
OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=8 --nnodes=2 \
  --master_addr="$MASTER_ADDR" --master_port=29541 \
  --node_rank=0 ./torch_mnist_train.py
```

Set `--node_rank=1` on Node B. Use `NCCL_SOCKET_IFNAME=ens7` on nodes
that have `ens7` instead of `eth0`.

Add `BW_ONLY=1` to skip training and only run the allreduce bandwidth
measurement (5 warmup + 20 trials at 4 GB). Much faster feedback loop.

**Expected:** ~386 GB/s busbw, GDR 1, PXN 1, 20 coll channels.

## 4. Docker setup (both nodes)

### Fix cgroup CPU allocation (Modal workers)

Modal workers restrict `system.slice` to CPUs 0-11. Docker containers inherit
this, starving NCCL proxy threads. Fix:

```bash
# Move Docker service to the container slice
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/50-modal-slice.conf <<'EOF'
[Service]
Slice=modal-containers.slice
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker

# Give containers all CPUs
sudo systemctl set-property modal-containers.slice AllowedCPUs=0-111

# Place containers in the right cgroup
sudo python3 -c "
import json, pathlib
p = pathlib.Path('/etc/docker/daemon.json')
cfg = json.loads(p.read_text().strip()) if p.exists() else {}
cfg['cgroup-parent'] = 'modal-containers.slice'
p.write_text(json.dumps(cfg, indent=2) + chr(10))
"
sudo systemctl restart docker

# Verify
sudo docker run --rm ubuntu:22.04 bash -c "cat /sys/fs/cgroup/cpuset.cpus.effective; nproc"
# Should show 0-111 / 112
```

### Build the test image

The image must include `libibverbs` — without it NCCL cannot use IB and falls
back to sockets (GDR 0, 2 channels, ~4 GB/s instead of ~386 GB/s).

```bash
sudo docker build -f Dockerfile.torch-slim -t torch-slim .
```

### Register runsc-rdma runtime (for gVisor tests)

```bash
sudo cp /usr/local/bin/runsc /usr/local/bin/runsc-rdma
sudo chmod +x /usr/local/bin/runsc-rdma

sudo python3 -c "
import json, pathlib
p = pathlib.Path('/etc/docker/daemon.json')
cfg = json.loads(p.read_text().strip()) if p.exists() else {}
cfg.setdefault('runtimes', {})['runsc-rdma'] = {
    'path': '/usr/local/bin/runsc-rdma',
    'runtimeArgs': [
        '--rdmaproxy', '--nvproxy',
        '--nvproxy-allowed-driver-capabilities=compute,utility,video',
        '--network=host', '--rdma-expected-ipoib=-1',
    ],
}
p.write_text(json.dumps(cfg, indent=2) + chr(10))
"
sudo systemctl restart docker
```

## 5. Docker (runc) test

```bash
DEVS=$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
IFNAME=$(ip -4 addr show ens7 &>/dev/null && echo ens7 || echo eth0)

sudo docker run --runtime=runc --rm --gpus all $DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v $HOME/.local/lib/python${PYVER}/site-packages:/usr/local/lib/python${PYVER}/dist-packages \
  -v $HOME/.local/bin:/usr/local/bin \
  -v $HOME/gvisor:/workspace \
  -e NCCL_DEBUG=INFO \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=1 \
  -e NCCL_SOCKET_IFNAME=$IFNAME \
  -e GLOO_SOCKET_IFNAME=$IFNAME \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e OMP_NUM_THREADS=1 \
  -w /workspace \
  torch-slim \
  torchrun --nproc_per_node=8 --nnodes=2 \
    --master_addr="$MASTER_ADDR" --master_port=29541 \
    --node_rank=0 ./torch_mnist_train.py
```

**Expected:** ~386 GB/s busbw, GDR 1, 20 coll channels (matches bare metal).

## 6. Docker (gVisor / runsc-rdma) test

Same command, replace `--runtime=runc` with `--runtime=runsc-rdma`.

```bash
# Generate topo XML once on bare metal (single-node is enough):
#   NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml torchrun --nproc_per_node=8 --nnodes=1 ...

sudo docker run --runtime=runsc-rdma --rm --gpus all $DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v $HOME/.local/lib/python${PYVER}/site-packages:/usr/local/lib/python${PYVER}/dist-packages \
  -v $HOME/.local/bin:/usr/local/bin \
  -v $HOME/gvisor:/workspace \
  -e NCCL_DEBUG=INFO \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=1 \
  -e NCCL_TOPO_FILE=/workspace/nccl_topo.xml \
  -e NCCL_SOCKET_IFNAME=$IFNAME \
  -e GLOO_SOCKET_IFNAME=$IFNAME \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e BW_ONLY=1 \
  -e OMP_NUM_THREADS=1 \
  -w /workspace \
  torch-slim \
  torchrun --nproc_per_node=8 --nnodes=2 \
    --master_addr="$MASTER_ADDR" --master_port=29541 \
    --node_rank=0 ./torch_mnist_train.py
```

Add `BW_ONLY=1` (`-e BW_ONLY=1`) to skip training and only measure bandwidth.

Requires `NCCL_TOPO_FILE` pointing to a topo XML generated on bare metal
(gVisor hides PCI topology from the container). Generate one with a single-node
bare-metal run using `NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml`.

**Expected (GDR=3, 8 GPUs):** busbw comparable to bare metal.
**Expected (GDR=0, 8 GPUs):** ~22 GB/s busbw (CPU-staged, ~20x slower).

## 7. Verify RDMA with port counters

Read IB port counters before and after a test to confirm bytes actually flow
over the wire. Counters are in 4-byte words — multiply deltas by 4 to get bytes.

```bash
# Build the list of active HCAs (same as NCCL_IB_HCA)
ACTIVE_HCAS=$(for dev in /sys/class/infiniband/mlx5_*; do
  d=$(basename $dev)
  grep -q "4: ACTIVE" $dev/ports/1/state 2>/dev/null && echo "$d"
done)

# Snapshot counters (run before AND after the test)
for dev in $ACTIVE_HCAS; do
  rcv=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data)
  xmit=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_xmit_data)
  echo "$dev: rcv=$rcv xmit=$xmit"
done
```

Run the snapshot, execute the test, run the snapshot again, then compute deltas:

```bash
# Compute deltas (paste before/after values into a script, or eyeball them).
# Convert words to GB:  delta_words * 4 / 1e9
#
# Example from an 8xH200 gVisor run (GDR=0, 8 active rails):
#   mlx5_0:  rcv=12.87 GB  xmit=12.89 GB
#   mlx5_3:  rcv=12.99 GB  xmit=13.33 GB
#   ...
#   Total:   rcv≈103 GB    xmit≈105 GB   (across 8 HCAs)
#
# HCAs with zero delta (e.g. mlx5_1, mlx5_7) are management/NVSwitch ports,
# not inter-node NICs — this is expected.
```

**What to look for:**
- All inter-node HCAs should show non-zero deltas (confirms multi-rail).
- Traffic should be roughly balanced across rails (~equal per-HCA).
- rcv ≈ xmit (symmetric for allreduce).
- Total bytes should be consistent with the reported busbw × test duration.

## 8. Verify RDMA bandwidth with ib_write_bw

```bash
# Node A (server):
ib_write_bw -d mlx5_5 --report_gbits

# Node B (client):
ib_write_bw -d mlx5_5 <NODE_A_IP> --report_gbits
```

**Expected:** ~386 Gb/s (400G line rate). Works identically through gVisor rdmaproxy.
