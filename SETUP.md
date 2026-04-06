Begin by cloning the repo, checking out the `alessio/development` branch, and building gVisor.

```bash
git clone git@github.com:modal-labs/gvisor.git
cd gvisor
git checkout alessio/development
```

On Node A

```bash
git config user.name "atoniolo76"
git config user.email "alessio@modal.com"
```

Build gVisor binary

```bash
sudo pkill -f "runsc-rdma" 2>/dev/null; sleep 1
sudo make copy TARGETS=runsc DESTINATION=/tmp
sudo rm -f /usr/local/bin/runsc-rdma
sudo cp /tmp/runsc /usr/local/bin/runsc-rdma
sudo chmod +x /usr/local/bin/runsc-rdma
```

On Node A
```bash
export MASTER_ADDR=$(ip -4 addr show | grep -e "ens7" -e "eth0" | grep inet | awk '{ print $2 }' | awk -F'/' '{ print $1}')
echo "export MASTER_ADDR=$MASTER_ADDR"
```
**Copy and paste this command on Node B.**

Run MNIST Training

```bash
bash rdma_job_agent/run_mnist_train.sh
```

Set `RUNTIME=runc` (or `RUNTIME=runsc-rdma`) to run via Docker instead of the
default `RUNTIME=torchrun`. See the script header for all optional env vars.

---

### Multi-node via agent

### On Node A

Export Node A’s IP after running `ip -4 addr show` and selecting the entry with `eth0`/ `ens7`

```bash
export NODE_A_IP=xxx
```

Export Node B’s IP

```bash
export NODE_B_IP=xxx
```

### On Node B

Run the iterative development agent

```bash
cd ~/gvisor/rdma_job_agent
sudo --preserve-env=SSH_AUTH_SOCK python3 agent.py --host 0.0.0.0 --port 8756
```

### On Node A

Curl Node B’s agent server

```bash
curl http://${NODE_B_IP}:8756/health
```

> Make sure you get an `{"ok": true}` response
> 

Copy the `nccl_topo.xml` file to Node B

```bash
curl -sS -X POST --data-binary @/tmp/nccl_topo.xml   "http://${NODE_B_IP}:8756/v1/nccl_topo"
```

Run the PyTorch training test — start rank 1 on Node B via the agent:

```bash
curl -sS -X POST "http://${NODE_B_IP}:8756/v1/jobs" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n \
    --arg ma "$NODE_A_IP" \
    --arg ib "$NCCL_IB_HCA" \
    --arg script "/home/modal/gvisor/torch_mnist_train.py" \
    '{
      kind: "train",
      runtime: "runsc-rdma",
      master_addr: $ma,
      master_port: 29500,
      nproc_per_node: 8,
      script_host_path: $script,
      env: {
        "NCCL_DEBUG": "INFO",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_HCA": $ib,
        "NCCL_NET_GDR_LEVEL": "3",
        "NCCL_DMABUF_ENABLE": "0"
      }
    }')" | jq .
```

Run the runsc RDMA test
```
sudo docker run --runtime=runsc-rdma --rm --gpus all $DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v /tmp/nccl_topo.xml:/topo.xml:ro \
  -v /home/modal/gvisor/torch_mnist_train.py:/tmp/train_script.py:ro \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_IB_GID_INDEX=0 \
  -e NCCL_TOPO_FILE=/topo.xml \
  nvcr.io/nvidia/pytorch:26.03-py3 torchrun \
    --nproc_per_node=8 --nnodes=2 \
    --master_addr="$NODE_A_IP" --master_port=29500 \
    --node_rank=0 /tmp/train_script.py
```

Then run rank 0 on Node A. With `runc` (no topo file needed — NCCL discovers PCIe topology directly):

```bash
sudo docker run --runtime=runc --rm --gpus all $DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v /home/modal/gvisor/torch_mnist_train.py:/tmp/train_script.py:ro \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  nvcr.io/nvidia/pytorch:26.03-py3 torchrun \
    --nproc_per_node=8 --nnodes=2 \
    --master_addr="$NODE_A_IP" --master_port=29500 \
    --node_rank=0 /tmp/train_script.py
```

To use `runsc-rdma` on Node A instead, add `--runtime=runsc-rdma`, the topo bind mount (`-v /tmp/nccl_topo.xml:/topo.xml:ro`), and the extra env vars (`-e NCCL_IB_GID_INDEX=0 -e NCCL_TOPO_FILE=/topo.xml`).

### Agent API (from Node A)

Poll a specific job's status and output:

```bash
curl -sS "http://${NODE_B_IP}:8756/v1/jobs/${JOB_ID}" | jq .
```

Cancel a running job:

```bash
curl -sS -X POST "http://${NODE_B_IP}:8756/v1/jobs/${JOB_ID}/cancel" | jq .
```

Deploy runsc-rdma on Node B (git pull, build, install, restart docker):

```bash
curl -sS -X POST "http://${NODE_B_IP}:8756/v1/admin/deploy_runsc" \
  -H 'Content-Type: application/json' -d '{"async": true}' | jq .
```