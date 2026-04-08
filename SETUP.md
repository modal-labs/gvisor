# Multi-Node RDMA Test Setup

Two-node distributed PyTorch training over InfiniBand with GPUDirect RDMA.
Covers bare metal, Docker (runc), and gVisor (runsc-rdma).

## 1. Host prerequisites (both nodes)

```bash
# Clone and build gVisor
git clone https://github.com/modal-labs/gvisor.git && cd gvisor
git checkout alessio/development
sudo make copy TARGETS=runsc DESTINATION=/usr/local/bin/runsc
sudo chmod +x /usr/local/bin/runsc

# Load nvidia_peermem for GPUDirect RDMA
sudo modprobe nvidia_peermem

# Install PyTorch
sudo apt install -y python3-pip
pip3 install torch torchvision
export PATH="$HOME/.local/bin:$PATH"
```

## 2. Discover node IPs and IB HCAs

```bash
# Node A IP (use eth0 or ens7, whichever exists)
export MASTER_ADDR=$(ip -4 addr show eth0 | grep inet | awk '{print $2}' | awk -F/ '{print $1}')
echo "export MASTER_ADDR=$MASTER_ADDR"
# Copy and paste this export on Node B.

# List active IB devices
ibstat | grep -E "CA |State"

# Set HCA list (adjust per hardware — exclude management/down ports)
export NCCL_IB_HCA=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12
```

## 3. Bare-metal test

```bash
sudo prlimit --pid=$$ --memlock=unlimited:unlimited

NCCL_DEBUG=INFO \
NCCL_NET_GDR_LEVEL=3 \
NCCL_DMABUF_ENABLE=1 \
NCCL_SOCKET_IFNAME=eth0 \
GLOO_SOCKET_IFNAME=eth0 \
NCCL_IB_HCA="$NCCL_IB_HCA" \
OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=8 --nnodes=2 \
  --master_addr="$MASTER_ADDR" --master_port=29541 \
  --node_rank=0 ./torch_mnist_train.py
```

Set `--node_rank=1` on Node B.

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

sudo docker run --runtime=runc --rm --gpus all $DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v $HOME/.local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages \
  -v $HOME/.local/bin:/usr/local/bin \
  -v $HOME/gvisor:/workspace \
  -e NCCL_DEBUG=INFO \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=1 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
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
sudo docker run --runtime=runsc-rdma --rm --gpus all $DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v $HOME/.local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages \
  -v $HOME/.local/bin:/usr/local/bin \
  -v $HOME/gvisor:/workspace \
  -v /tmp/nccl_topo.xml:/topo.xml:ro \
  -e NCCL_DEBUG=INFO \
  -e NCCL_NET_GDR_LEVEL=0 \
  -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_TOPO_FILE=/topo.xml \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e OMP_NUM_THREADS=1 \
  -w /workspace \
  torch-slim \
  torchrun --nproc_per_node=8 --nnodes=2 \
    --master_addr="$MASTER_ADDR" --master_port=29541 \
    --node_rank=0 ./torch_mnist_train.py
```

**Important:** Use `NCCL_NET_GDR_LEVEL=0` with gVisor. GPUDirect RDMA (GDR=3)
is not yet supported because GPU device memory VAs have no VMA in the sentry's
address space. With GDR=0 (CPU-staged), expect ~90 GB/s busbw (vs ~386 bare metal).

Also requires `NCCL_TOPO_FILE` pointing to a topo XML generated on bare metal
(gVisor hides PCI topology from the container).

## 7. Verify RDMA with port counters

Read IB port counters before and after a test to confirm bytes flow over the wire:

```bash
# Before test
for dev in mlx5_5 mlx5_6 mlx5_9 mlx5_10 mlx5_11; do
  rcv=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data)
  xmit=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_xmit_data)
  echo "$dev: rcv=$rcv xmit=$xmit"
done

# ... run test ...

# After test (same loop — compute deltas)
```

## 8. Verify RDMA bandwidth with ib_write_bw

```bash
# Node A (server):
ib_write_bw -d mlx5_5 --report_gbits

# Node B (client):
ib_write_bw -d mlx5_5 <NODE_A_IP> --report_gbits
```

**Expected:** ~386 Gb/s (400G line rate). Works identically through gVisor rdmaproxy.
