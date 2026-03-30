# gVisor RDMA Proxy — End-to-End Runbook

Complete walkthrough: set up **SSH from node A to node B** first, then clone
and build on node A only, copy artifacts to node B, deploy, validate RDMA,
run NCCL and PyTorch multi-node benchmarks comparing runc vs gVisor.

Assumes two comparable GPU nodes with RDMA-capable NICs (commonly mlx5),
Docker, and passwordless sudo.

---

## SSH access between nodes

On **node A**, substitute node B’s IP and SSH username wherever
`NODE_B_IP` and `NODE_B_USER` appear below, or `export` them once for your
session. Section 7 lists the same names together with `NODE_A_IP` for
benchmarks.

Generate a key if you do not already have one:

```bash
# Skip if ~/.ssh/id_ed25519 already exists
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
```

Someone who can open a session on **node B** (SSH with another key, cloud
console, etc.) grants access for node A’s key **on node B** as the **same
Unix account** node A will use later (`NODE_B_USER` — e.g. `opc`).

On **node A**, print the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

On **node B**, create `~/.ssh`, append that **entire single line** inside the
quotes (replace the placeholder), then fix permissions:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAA…paste-full-line-from-node-A…' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Use **single quotes** around the key so the line is appended exactly as on
node A.

From **node A**, run the **Verify** command at the end of this section.

If **`ssh … echo OK` still fails**, check **`NODE_B_USER`** (e.g. `export
NODE_B_USER=opc` on OCI). Paste the same public key into the provider’s
instance **SSH keys** UI if your cloud supports it (often takes a reboot).

If B already trusts **another** private key, append node A’s key in one shot:

```bash
ssh -i ~/.ssh/your_existing_key -o StrictHostKeyChecking=no modal@${NODE_B_IP} \
  "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys" \
  < ~/.ssh/id_ed25519.pub
```

If you **can** `ssh` with a password:

```bash
cat ~/.ssh/id_ed25519.pub | ssh modal@${NODE_B_IP} 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
```

Diagnose refusals:

```bash
ssh -v -o StrictHostKeyChecking=no modal@${NODE_B_IP}
```

### Verify (from node A)

```bash
ssh -o StrictHostKeyChecking=no modal@${NODE_B_IP} echo OK
```

---

**Clone, build, deploy, and benchmarks.** The numbered steps below assume
**SSH access between nodes** is already working.

## 1. Clone and build (node A only)

On **node A**:

```bash
git clone git@github.com:modal-labs/gvisor.git
cd gvisor
git checkout alessio/development

# Build runsc (~7 min first time, ~30s incremental)
sudo make copy TARGETS=runsc DESTINATION=/tmp
```

Do **not** clone or build the gVisor tree on node B; node B receives the
`runsc` binary from node A in section 2.

## 2. Copy runsc and docker to node B (from node A)

Still on **node A**, after section 1, copy the built runtime and the Docker
CLI so node B matches node A without a local build:

```bash
# runsc artifact from section 1 (root-owned after sudo make copy)
sudo chmod +r /tmp/runsc
scp /tmp/runsc modal@${NODE_B_IP}:/tmp/runsc

# Docker client binary (path may differ; use the same client as on A)
DOCKER_BIN="$(command -v docker)"
scp "$DOCKER_BIN" modal@${NODE_B_IP}:/tmp/docker
```

On **node B**, install the copied Docker CLI (optional if B already has a
compatible `docker`; use this when you want the exact same binary as A):

```bash
sudo cp /tmp/docker /usr/bin/docker
sudo chmod +x /usr/bin/docker
```

## 3. Deploy runsc-rdma (both nodes)

```bash
sudo pkill -f "runsc-rdma" 2>/dev/null; sleep 1
sudo rm -f /usr/local/bin/runsc-rdma
sudo cp /tmp/runsc /usr/local/bin/runsc-rdma
sudo chmod +x /usr/local/bin/runsc-rdma
```

## 4. Register Docker runtime (both nodes, once)

```bash
sudo python3 -c "
import json, os
p = '/etc/docker/daemon.json'
d = json.load(open(p)) if os.path.exists(p) else {}
d.setdefault('runtimes', {})['runsc-rdma'] = {
    'path': '/usr/local/bin/runsc-rdma',
    'runtimeArgs': [
        '--debug',
        '--debug-log=/tmp/runsc-rdma/logs/',
        '--rdmaproxy',
        '--nvproxy',
        '--network=host',
        '--rdma-expected-ipoib=-1'
    ]
}
json.dump(d, open(p,'w'), indent=2)
"
sudo systemctl restart docker
sleep 2
```

## 5. Load nvidia-peermem (both nodes)

```bash
sudo modprobe nvidia-peermem
```

## 6. Build Docker images

### NCCL test image (node A, then load on node B)

On **node A** (from the gVisor clone):

```bash
cd ~/gvisor
sudo docker build -f Dockerfile.nccl -t nccl-test .
```

Transfer the image to **node B** (same `NODE_B_IP` / `NODE_B_USER` as in
**SSH access between nodes**):

```bash
sudo docker save nccl-test | ssh -o StrictHostKeyChecking=no modal@${NODE_B_IP} 'sudo docker load'
```

### PyTorch image (both nodes)

The PyTorch image does not use the gVisor tree; run the same commands on
**both** nodes:

```bash
cat > /tmp/Dockerfile.pytorch <<'EOF'
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN PIP_INDEX_URL="https://download.pytorch.org/whl/cu124" && \
    python3 -m pip install --ignore-installed \
        torch torchvision torchaudio \
        --index-url "$PIP_INDEX_URL"
EOF
sudo docker build -t gvisor-pytorch -f /tmp/Dockerfile.pytorch /tmp
```

## 7. Identify node IPs and set environment

`NODE_B_IP` and `NODE_B_USER` should match **SSH access between nodes** and
section 2 (copy to node B).

Run on both nodes to find IPs:

```bash
ip -4 addr show | grep -E 'inet (172\.|10\.)' | grep -v docker | grep -v 127
```

Set these on node A (replace with your actual IPs):

```bash
export NODE_A_IP=<node-a-ip>    # e.g. 172.29.5.7
export NODE_B_IP=<node-b-ip>    # e.g. 172.29.11.158
export NODE_B_USER=<same-as-on-B>   # e.g. opc
export DOCKER_CPUS=$(($(sudo docker info --format '{{.NCPU}}') - 2))
export DEVS=$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
```

## 8. Generate NCCL topology file (node A, once)

gVisor doesn't expose PCI device sysfs, so NCCL can't discover the
NUMA/PCIe topology. Without this file, NCCL uses Ring (64 channels)
instead of NVLS Tree (16 channels) — an 8.6x performance hit.

```bash
sudo mkdir -p /tmp/nccl_shared
sudo docker run --runtime=runc --rm --gpus all $DEVS \
  --cpus="$DOCKER_CPUS" --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -e NCCL_TOPO_DUMP_FILE=/shared/topo.xml \
  -e RANK=0 -e NRANKS=1 -e NGPUS=8 \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29517 \
  -v /tmp/nccl_shared:/shared \
  nccl-test /usr/local/bin/nccl_multinode_bench

# Copy to node B
scp /tmp/nccl_shared/topo.xml modal@${NODE_B_IP}:/tmp/nccl_topo.xml
cp /tmp/nccl_shared/topo.xml /tmp/nccl_topo.xml
```

---

## 9. NCCL multinode benchmark

### Identify which HCAs to use

Some HCAs are management-only. Check which are active data-path devices:

```bash
for d in mlx5_{0..11}; do
  ibv_devinfo -d $d -v 2>/dev/null | grep -E 'hca_id|state:|active_speed' | head -3
  echo ---
done
```

Exclude management NICs (typically 50 Gbps) and any PORT_DOWN devices.
On OCI H200 nodes, mlx5_1/mlx5_7 are 50 Gbps management and mlx5_2/mlx5_8
are PORT_DOWN — leaving 8 data-path HCAs at 100 Gbps:

```bash
# Adjust for your hardware
export NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11
```

### 9a. runc baseline

**Node B (rank 1) — run first:**

```bash
ssh -o StrictHostKeyChecking=no modal@${NODE_B_IP} "
DOCKER_CPUS=\$((\$(sudo docker info --format '{{.NCPU}}') - 2))
DEVS=\$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
sudo docker run --runtime=runc --rm --gpus all \$DEVS \
  --cpus=\"\$DOCKER_CPUS\" --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -e RANK=1 -e NRANKS=2 -e NGPUS=8 \
  -e MASTER_ADDR=$NODE_A_IP -e MASTER_PORT=29500 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  nccl-test /usr/local/bin/nccl_multinode_bench
"
```

**Node A (rank 0):**

```bash
sudo docker run --runtime=runc --rm --gpus all $DEVS \
  --cpus="$DOCKER_CPUS" --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -e RANK=0 -e NRANKS=2 -e NGPUS=8 \
  -e MASTER_ADDR=$NODE_A_IP -e MASTER_PORT=29500 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  nccl-test /usr/local/bin/nccl_multinode_bench
```

### 9b. runsc-rdma (gVisor)

**Node B (rank 1) — run first:**

```bash
ssh -o StrictHostKeyChecking=no modal@${NODE_B_IP} "
DOCKER_CPUS=\$((\$(sudo docker info --format '{{.NCPU}}') - 2))
DEVS=\$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
sudo rm -rf /tmp/runsc-rdma/logs && sudo mkdir -p /tmp/runsc-rdma/logs
sudo docker run --runtime=runsc-rdma --rm --gpus all \$DEVS \
  --cpus=\"\$DOCKER_CPUS\" --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v /tmp/nccl_topo.xml:/topo.xml:ro \
  -e RANK=1 -e NRANKS=2 -e NGPUS=8 \
  -e MASTER_ADDR=$NODE_A_IP -e MASTER_PORT=29501 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_IB_GID_INDEX=0 \
  -e NCCL_TOPO_FILE=/topo.xml \
  nccl-test /usr/local/bin/nccl_multinode_bench
"
```

**Node A (rank 0):**

```bash
sudo rm -rf /tmp/runsc-rdma/logs && sudo mkdir -p /tmp/runsc-rdma/logs
sudo docker run --runtime=runsc-rdma --rm --gpus all $DEVS \
  --cpus="$DOCKER_CPUS" --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v /tmp/nccl_topo.xml:/topo.xml:ro \
  -e RANK=0 -e NRANKS=2 -e NGPUS=8 \
  -e MASTER_ADDR=$NODE_A_IP -e MASTER_PORT=29501 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_IB_GID_INDEX=0 \
  -e NCCL_TOPO_FILE=/topo.xml \
  nccl-test /usr/local/bin/nccl_multinode_bench
```

---

## 10. PyTorch all-reduce benchmark

### Copy the benchmark script (node A, then copy to B)

The benchmark script lives at `torch_allreduce_bench.py` in the repo root.
It includes optional **NCCL Flight Recorder** support (env-gated, zero
overhead when off) that dumps per-rank collective metadata — call stacks,
sizes, states, and nanosecond timestamps — as JSON. Useful for diagnosing
hangs or comparing collective behavior between runc and gVisor.

```bash
cp ~/gvisor/torch_allreduce_bench.py /tmp/torch_allreduce_bench.py
scp /tmp/torch_allreduce_bench.py modal@${NODE_B_IP}:/tmp/torch_allreduce_bench.py
```

### 10a. runc baseline

**Node B (rank 1) — run first:**

```bash
ssh -o StrictHostKeyChecking=no modal@${NODE_B_IP} "
DOCKER_CPUS=\$((\$(sudo docker info --format '{{.NCPU}}') - 2))
DEVS=\$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
sudo docker run --runtime=runc --rm --gpus all \$DEVS \
  --cpus=\"\$DOCKER_CPUS\" --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -e NCCL_DEBUG=WARN \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  -v /tmp/torch_allreduce_bench.py:/tmp/torch_allreduce_bench.py:ro \
  gvisor-pytorch torchrun \
    --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=$NODE_A_IP --master_port=29530 \
    /tmp/torch_allreduce_bench.py
"
```

**Node A (rank 0):**

```bash
sudo docker run --runtime=runc --rm --gpus all $DEVS \
  --cpus="$DOCKER_CPUS" --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -e NCCL_DEBUG=WARN \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  -v /tmp/torch_allreduce_bench.py:/tmp/torch_allreduce_bench.py:ro \
  gvisor-pytorch torchrun \
    --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=$NODE_A_IP --master_port=29530 \
    /tmp/torch_allreduce_bench.py
```

### 10b. runsc-rdma (gVisor)

**Node B (rank 1) — run first:**

```bash
ssh -o StrictHostKeyChecking=no modal@${NODE_B_IP} "
DOCKER_CPUS=\$((\$(sudo docker info --format '{{.NCPU}}') - 2))
DEVS=\$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
sudo rm -rf /tmp/runsc-rdma/logs && sudo mkdir -p /tmp/runsc-rdma/logs
sudo docker run --runtime=runsc-rdma --rm --gpus all \$DEVS \
  --cpus=\"\$DOCKER_CPUS\" --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v /tmp/nccl_topo.xml:/topo.xml:ro \
  -e NCCL_DEBUG=WARN \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_IB_GID_INDEX=0 \
  -e NCCL_TOPO_FILE=/topo.xml \
  -v /tmp/torch_allreduce_bench.py:/tmp/torch_allreduce_bench.py:ro \
  gvisor-pytorch torchrun \
    --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=$NODE_A_IP --master_port=29531 \
    /tmp/torch_allreduce_bench.py
"
```

**Node A (rank 0):**

```bash
sudo rm -rf /tmp/runsc-rdma/logs && sudo mkdir -p /tmp/runsc-rdma/logs
sudo docker run --runtime=runsc-rdma --rm --gpus all $DEVS \
  --cpus="$DOCKER_CPUS" --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v /tmp/nccl_topo.xml:/topo.xml:ro \
  -e NCCL_DEBUG=WARN \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_IB_GID_INDEX=0 \
  -e NCCL_TOPO_FILE=/topo.xml \
  -v /tmp/torch_allreduce_bench.py:/tmp/torch_allreduce_bench.py:ro \
  gvisor-pytorch torchrun \
    --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=$NODE_A_IP --master_port=29531 \
    /tmp/torch_allreduce_bench.py
```

### 10c. NCCL Flight Recorder (optional)

To capture Flight Recorder dumps, re-run any §10 benchmark with these
extra flags on **every** `docker run` command (both nodes):

```
  -e TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
  -e TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
  -e FR_DUMP=1 -e FR_DIR=/tmp/fr_dumps \
  -v /tmp/fr_dumps:/tmp/fr_dumps \
```

Create the host directory first (`sudo mkdir -p /tmp/fr_dumps` on both
nodes). `TORCH_NCCL_TRACE_BUFFER_SIZE` enables the c10d ring buffer;
`TORCH_NCCL_DUMP_ON_TIMEOUT` auto-dumps if a collective times out;
`FR_DUMP=1` tells the script to also dump on successful completion.

Each rank writes `rank<N>_gpu<G>.json` into `/tmp/fr_dumps/` — one file
per rank, per node. Each file contains:

- **`entries`** — every NCCL collective (all_reduce, barrier, reduce)
  with `profiling_name`, `input_sizes`, `output_sizes`, `state`,
  `time_created_ns`, `time_discovered_completed_ns`, and full Python
  call stacks (`frames`).
- **`pg_config`** — process group membership.
- **`pg_status`** — last-seen sequence IDs.

Inspect a dump:

```bash
python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
for e in data['entries']:
    ns = e.get('time_discovered_completed_ns') or 0
    cr = e.get('time_created_ns', 0)
    dt_ms = (ns - cr) / 1e6 if ns else '?'
    print(f\"{e['collective_seq_id']:3d}  {e['profiling_name']:30s}  {e['state']:12s}  {dt_ms} ms\")
" /tmp/fr_dumps/rank0_gpu0.json
```

---

## 11. Expected results (2x8 H200 nodes, 112 CPUs each)

### NCCL multinode bench (128 MiB message)

| Runtime | busbw | Channels | Algorithm |
|---------|-------|----------|-----------|
| runc | ~125 GB/s | 16 | NVLS Tree |
| runsc-rdma + TOPO_FILE | ~79 GB/s | 16 | NVLS Tree |
| runsc-rdma (no TOPO_FILE) | ~15 GB/s | 64 | Ring |

### PyTorch all-reduce (4 GB payload, 50 trials)

| Runtime | busbw |
|---------|-------|
| runc | ~4.1 GBps |
| runsc-rdma + TOPO_FILE | ~3.0 GBps |

The ~1.4x gap with TOPO_FILE is general gVisor systrap/sentry overhead,
not RDMA-specific. Without TOPO_FILE the gap is ~8.6x due to NCCL
selecting Ring instead of NVLS Tree.

---

## 12. Troubleshooting

**Port in use**: If you get `EADDRINUSE`, change `--master_port` to a
different value. Each test pair (runc vs runsc) should use different ports.

**gVisor logs**: Check `/tmp/runsc-rdma/logs/` on each node for sentry
boot logs and RDMA proxy activity.

**NCCL debug**: Change `-e NCCL_DEBUG=WARN` to `-e NCCL_DEBUG=INFO` for
verbose NCCL output (noisy but useful for diagnosing transport issues).

**Verify NCCL topology**: With `NCCL_DEBUG=INFO`, check for:
- `16 coll channels` (not 64)
- `Connected NVLS tree` (not `Connected all rings`)
- `Symmetric VA size=140GB` without "Symmetric memory is not supported"
