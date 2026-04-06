Begin by cloning the repo, checking out the `alessio/development` branch, and building gVisor.

```bash
git clone git@github.com:modal-labs/gvisor.git
cd gvisor
git checkout alessio/development
```

If pushing changes to branch:

```bash
git config user.name "atoniolo76"
git config user.email "alessio@modal.com"
```

Kill any old build processes, remove old binaries, and build `runsc-rdma` into `/usr/local/bin/`

```bash
sudo pkill -f "runsc-rdma" 2>/dev/null; sleep 1
sudo make copy TARGETS=runsc DESTINATION=/tmp
sudo rm -f /usr/local/bin/runsc-rdma
sudo cp /tmp/runsc /usr/local/bin/runsc-rdma
sudo chmod +x /usr/local/bin/runsc-rdma
```

Update Docker’s `daemon.json` to point to our `runsc-rdma` binary and pass necessary flags.

```bash
sudo apt-get update && sudo apt-get install -y jq

sudo pkill -f "runsc-rdma" 2>/dev/null || true; sleep 1
sudo rm -f /usr/local/bin/runsc-rdma
sudo cp /tmp/runsc /usr/local/bin/runsc-rdma && sudo chmod +x /usr/local/bin/runsc-rdma

sudo python3 -c "
import json, pathlib
p = pathlib.Path('/etc/docker/daemon.json')
raw = p.read_text().strip() if p.exists() else ''
cfg = json.loads(raw) if raw else {}
cfg.setdefault('runtimes', {})['runsc-rdma'] = {
    'path': '/usr/local/bin/runsc-rdma',
    'runtimeArgs': [
        '--debug', '--debug-log=/tmp/runsc-rdma/logs/',
        '--rdmaproxy', '--nvproxy',
        '--nvproxy-allowed-driver-capabilities=compute,utility,video',
        '--network=host', '--rdma-expected-ipoib=-1',
        '--strace'
    ],
}
p.write_text(json.dumps(cfg, indent=2) + '\n')
"
sudo systemctl restart docker && sleep 2
```

Load the NVIDIA-peermem kernel module

```bash
sudo modprobe nvidia-peermem
```

Export the RDMA-enabled Mellanox devices

```bash
export NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11
```

Export uverbs devices

```bash
export DEVS=$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
```

Run this rather long NCCL test with `runc` to get the topology XML file

```bash
sudo docker run --runtime=runc --rm --gpus all $DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -e NCCL_DEBUG=WARN -e NCCL_SOCKET_IFNAME=eth0 -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml \
  -v /tmp:/tmp \
  "nvcr.io/nvidia/pytorch:26.03-py3" torchrun --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29599 \
  /tmp/torch_allreduce_bench.py
```

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