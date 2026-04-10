---
name: nccl-test
description: Run NCCL allreduce bandwidth test on 2 remote B200 GPU nodes via SSH
user-invocable: true
allowed-tools: Bash
argument-hint: [node0-host node1-host]
---

# NCCL Multi-Node Bandwidth Test

Run the torch_mnist_train.py allreduce benchmark on two remote B200 GPU nodes.

## Arguments

$ARGUMENTS should be two SSH hostnames, e.g.: `wo-abc123 wo-def456`
- If no arguments given, ask the user for the node hostnames.
- Node 0 = first argument, Node 1 = second argument
- SSH user is `modal`

## Steps

### 1. Verify connectivity and InfiniBand
SSH to both nodes in parallel and verify they respond AND have InfiniBand:
```
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 modal@<node0> 'echo ok && ls /dev/infiniband/uverbs* && ibstat | grep -E "CA |State"'
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 modal@<node1> 'echo ok && ls /dev/infiniband/uverbs* && ibstat | grep -E "CA |State"'
```
**IMPORTANT:** If `/dev/infiniband/` does not exist on a node, STOP and tell the user.
These nodes lack InfiniBand hardware and cannot run RDMA tests. Do NOT proceed with setup.

### 2. Fix git remotes and pull
Both nodes need HTTPS git remote (SSH keys may not work for GitHub):
```
ssh modal@<node> 'cd ~/gvisor && git remote set-url origin https://github.com/modal-labs/gvisor.git && git pull'
```
Run on both nodes in parallel.

### 3. Get master address
Get node 0's ens7 IP to use as MASTER_ADDR:
```
ssh modal@<node0> "ip -4 addr show ens7 | grep inet | awk '{print \$2}' | cut -d/ -f1"
```

### 4. Setup prerequisites on both nodes
Run in parallel on both:
```
ssh modal@<node> 'sudo modprobe nvidia_peermem; sudo prlimit --pid=$$ --memlock=unlimited:unlimited'
```

### 5. Run the test
Start node 1 first (it waits for master), then node 0. Use a unique port to avoid conflicts with stale processes.

**Node 1** (background):
```
ssh modal@<node1> 'sudo prlimit --pid=$$ --memlock=unlimited:unlimited && export PATH="$HOME/.local/bin:$PATH" && cd ~/gvisor && \
  NCCL_NET_MERGE_LEVEL=LOC \
  NCCL_GRAPH_FILE=/home/modal/gvisor/nccl_graph.xml \
  NCCL_TOPO_FILE=/home/modal/gvisor/nccl_topo.xml \
  NCCL_SOCKET_IFNAME=ens7 \
  GLOO_SOCKET_IFNAME=ens7 \
  NCCL_IB_HCA=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12 \
  OMP_NUM_THREADS=1 \
  torchrun --nproc_per_node=8 --nnodes=2 --master_addr=<MASTER_IP> --master_port=<PORT> --node_rank=1 ./torch_mnist_train.py'
```

**Node 0** (foreground, capture output):
```
ssh modal@<node0> 'sudo prlimit --pid=$$ --memlock=unlimited:unlimited && export PATH="$HOME/.local/bin:$PATH" && cd ~/gvisor && \
  NCCL_NET_MERGE_LEVEL=LOC \
  NCCL_GRAPH_FILE=/home/modal/gvisor/nccl_graph.xml \
  NCCL_TOPO_FILE=/home/modal/gvisor/nccl_topo.xml \
  NCCL_SOCKET_IFNAME=ens7 \
  GLOO_SOCKET_IFNAME=ens7 \
  NCCL_IB_HCA=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12 \
  OMP_NUM_THREADS=1 \
  torchrun --nproc_per_node=8 --nnodes=2 --master_addr=<MASTER_IP> --master_port=<PORT> --node_rank=0 ./torch_mnist_train.py'
```

Sleep 3 seconds between starting node 1 and node 0.

### 6. Report results
Grep for `busbw` and `algbw` in node 0's output. Report:
- algbw (GB/s and Gb/s)
- busbw (GB/s and Gb/s)  
- Whether training completed successfully
- Any FATAL errors

If there are errors, also check node 1's log output.

### Optional: Add NCCL_DEBUG=INFO
If the user asks for debug output, add `NCCL_DEBUG=INFO` to the env vars. Grep for `NET/` and `channels loaded` lines.

### Optional: Graph dump
If the user asks to dump the graph, add `NCCL_GRAPH_DUMP_FILE=/tmp/nccl_graph_dump.xml NCCL_GRAPH_DUMP_FILE_RANK=0` to node 0's env vars. After the run, `cat /tmp/nccl_graph_dump.xml` from node 0.
