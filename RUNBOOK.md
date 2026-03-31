# gVisor RDMA

Two nodes, mlx5, Docker, sudo, L3 for NCCL. Commented lines = node / one-off only.

**A dev / B peer:** A runs clone, `make copy`, topo generation, and drives `curl`. B only needs what’s in the table — copy **`runsc`** and **`/tmp/nccl_topo.xml`** from A when they change; no clone / `make` / `nccl-test` on B unless you want B self-contained.

| Need on B | Why |
|-----------|-----|
| `runsc` / `runsc-rdma` from what A built | Same runtime under test |
| `daemon.json` + `nvidia-peermem` + PyTorch image pull | Same Docker + GPU/IB stack |
| `/tmp/nccl_topo.xml` | Copy from A for `runsc-rdma` (same file both sides) |
| `/tmp/torch_allreduce_bench.py` | Same bench |
| Exports (`NODE_A_IP`, `NODE_B_IP`, `NCCL_IB_HCA`, `DOCKER_CPUS`, …) | Same NCCL view of the job |
| `rdma_job_agent` on **B only** | Rank **1** via `POST`; rank **0** on A is plain `docker run` (no agent on A) |

**Skip on B** if A does the work: clone, `make copy` (B only receives the binary), topo generation on A (B only receives `/tmp/nccl_topo.xml`).

```bash
# --- 1) Node A: clone, build, copy /tmp/runsc to B ---
# git clone git@github.com:modal-labs/gvisor.git && cd gvisor && git checkout alessio/development
# sudo make copy TARGETS=runsc DESTINATION=/tmp && sudo chmod +r /tmp/runsc

# --- 2) Both nodes ---
sudo apt-get update && sudo apt-get install -y jq

sudo pkill -f "runsc-rdma" 2>/dev/null || true; sleep 1
sudo rm -f /usr/local/bin/runsc-rdma
sudo cp /tmp/runsc /usr/local/bin/runsc-rdma && sudo chmod +x /usr/local/bin/runsc-rdma

sudo test -f /etc/docker/daemon.json || echo '{}' | sudo tee /etc/docker/daemon.json >/dev/null
sudo jq '(.runtimes //= {}) | .runtimes["runsc-rdma"] = {
  "path": "/usr/local/bin/runsc-rdma",
  "runtimeArgs": [
    "--debug", "--debug-log=/tmp/runsc-rdma/logs/",
    "--rdmaproxy", "--nvproxy", "--network=host", "--rdma-expected-ipoib=-1"
  ]
}' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp >/dev/null \
  && sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
sudo systemctl restart docker && sleep 2
sudo modprobe nvidia-peermem

export PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
sudo docker login nvcr.io
sudo docker pull "$PYTORCH_IMAGE"

export NODE_A_IP=<a> NODE_B_IP=<b>
export NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11
export DOCKER_CPUS=112
export DEVS=$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
export DOCKER_RUN="--rm --gpus all $DEVS --cpus=$DOCKER_CPUS --ulimit memlock=-1:-1 --shm-size=1g --network=host"
```

```bash
# --- 3) Both nodes ---
cp ~/gvisor/torch_allreduce_bench.py /tmp/torch_allreduce_bench.py
```

```bash
# --- 4) Node A: /tmp/nccl_topo.xml for runsc-rdma (same path on B after copy) ---
# One-time per hardware SKU (or when NICs/GPUs change). Reuse the same XML for every
# gVisor/runsc iteration — no need to rerun this docker command each build.
# NCCL writes the file when it initializes if NCCL_TOPO_DUMP_FILE is set. Use runc here
# (full sysfs); gVisor hides PCI — dumps under runsc are wrong. Stop after the XML exists.
sudo docker run --runtime=runc --rm --gpus all $DOCKER_RUN \
  -e NCCL_DEBUG=WARN -e NCCL_SOCKET_IFNAME=eth0 -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml \
  -v /tmp/torch_allreduce_bench.py:/tmp/torch_allreduce_bench.py:ro \
  "$PYTORCH_IMAGE" torchrun --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29599 \
  /tmp/torch_allreduce_bench.py
# Copy topo to B: `scp /tmp/nccl_topo.xml modal@${NODE_B_IP}:/tmp/nccl_topo.xml`
# or POST the file to the job agent: `curl -sS -X POST --data-binary @/tmp/nccl_topo.xml "http://${NODE_B_IP}:8756/v1/nccl_topo"`
# (see rdma_job_agent/README.md).
```

```bash
# --- 5) Node B only: agent (DOCKER_CPUS exported on B). A does not run the agent. ---
cd ~/gvisor/rdma_job_agent
python3 agent.py --host 0.0.0.0 --port 8756
```

```bash
# --- 6) Node A: POST rank 1 on B, then rank 0 locally (runsc-rdma). ---
# Allow A→B:8756 (security group / firewall) so curl reaches B’s agent.
export MASTER_PORT=29541
export AGENT_B=http://${NODE_B_IP}:8756

body_rank1() {
  local rt="${1:-runsc-rdma}"
  jq -n \
    --arg img "$PYTORCH_IMAGE" \
    --arg ma "$NODE_A_IP" \
    --arg hca "$NCCL_IB_HCA" \
    --argjson mp "$MASTER_PORT" \
    --arg rt "$rt" \
    '{kind:"torch",runtime:$rt,async:true,nnodes:2,node_rank:1,nproc_per_node:8,
      master_addr:$ma, master_port:$mp, image:$img,
      script_host_path:"/tmp/torch_allreduce_bench.py", topo_host_path:"/tmp/nccl_topo.xml",
      env:{NCCL_DEBUG:"WARN",NCCL_SOCKET_IFNAME:"eth0",NCCL_IB_HCA:$hca,
           NCCL_NET_GDR_LEVEL:"3",NCCL_DMABUF_ENABLE:"0"}}'
}

R1=$(curl -sS -X POST "$AGENT_B/v1/jobs" -H 'Content-Type: application/json' \
  -d "$(body_rank1)" | jq -r .job_id)
sleep 2

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
  "$PYTORCH_IMAGE" torchrun \
    --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=$NODE_A_IP --master_port=$MASTER_PORT \
    /tmp/torch_allreduce_bench.py

curl -sS "$AGENT_B/v1/jobs/$R1" | jq .

# runc: `body_rank1 runc` and use --runtime=runc on the docker run; drop -v topo,
# NCCL_TOPO_FILE, NCCL_IB_GID_INDEX for rank 0.
```
