#!/usr/bin/env bash
# DDP MNIST training (single-node or multi-node).
#
# Single-node (standalone):
#   bash rdma_job_agent/run_mnist_train.sh
#
# Multi-node (run on each node):
#   MASTER_ADDR=<node0 IP> NODE_RANK=0 NNODES=2 bash rdma_job_agent/run_mnist_train.sh
#   MASTER_ADDR=<node0 IP> NODE_RANK=1 NNODES=2 bash rdma_job_agent/run_mnist_train.sh
#
# Required env for multi-node:
#   MASTER_ADDR   - IPv4 of the rank-0 node
#   NODE_RANK     - 0 on master, 1 on worker
#   NNODES        - total number of nodes (default: 1 = standalone)
#
# Optional env:
#   MASTER_PORT   - rendezvous port (default: 29500)
#   RUNTIME       - Docker runtime (default: runc)
#   NUM_GPUS      - GPUs per node (default: all, detected via nvidia-smi)
#   PYTORCH_IMAGE - base image (default: nvcr.io/nvidia/pytorch:26.03-py3)
#   NCCL_IB_HCA   - IB HCA filter (default: hardcoded list)
#   DEVS          - IB device flags (default: auto-detected)
#   NCCL_DEBUG    - NCCL log level (default: WARN)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNTIME="${RUNTIME:-runc}"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"
if [[ -n "${MASTER_ADDR:-}" ]]; then
  NNODES="${NNODES:-2}"
else
  NNODES="${NNODES:-1}"
fi
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11}"

if [[ -z "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
  if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "No GPUs detected; set NUM_GPUS manually." >&2
    exit 1
  fi
fi

if [[ -z "${DEVS:-}" ]]; then
  DEVS=""
  for d in /dev/infiniband/uverbs*; do
    [[ -e "$d" ]] && DEVS="${DEVS} --device=${d}"
  done
fi

if [[ "$RUNTIME" == "runsc-rdma" ]]; then
  sudo rm -rf /tmp/runsc-rdma/logs
  sudo mkdir -p /tmp/runsc-rdma/logs
fi

EXTRA_DOCKER_ARGS=()
EXTRA_ENV=()

if [[ "$RUNTIME" == "runsc-rdma" ]]; then
  EXTRA_DOCKER_ARGS+=(-v /tmp/nccl_topo.xml:/topo.xml:ro)
  EXTRA_ENV+=(-e NCCL_IB_GID_INDEX=0 -e NCCL_TOPO_FILE=/topo.xml)
  EXTRA_ENV+=(-e "NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-0}")
else
  EXTRA_DOCKER_ARGS+=(--privileged)
fi

EXTRA_ENV+=(-e "NCCL_IB_HCA=${NCCL_IB_HCA}")
EXTRA_ENV+=(-e "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}")

if [[ "$NNODES" -gt 1 ]]; then
  TORCHRUN_ARGS=(
    --nproc_per_node="$NUM_GPUS"
    --nnodes="$NNODES"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    --node_rank="$NODE_RANK"
  )
  echo "=== MNIST DDP: runtime=$RUNTIME gpus=$NUM_GPUS nnodes=$NNODES rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT ==="
else
  TORCHRUN_ARGS=(--standalone --nproc_per_node="$NUM_GPUS")
  echo "=== MNIST DDP: runtime=$RUNTIME gpus=$NUM_GPUS standalone ==="
fi

sudo docker run --runtime="$RUNTIME" --rm --gpus all ${DEVS} \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  "${EXTRA_DOCKER_ARGS[@]}" \
  -v "${REPO_ROOT}/torch_mnist_train.py:/tmp/train_script.py:ro" \
  -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}" \
  -e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}" \
  "${EXTRA_ENV[@]}" \
  "$PYTORCH_IMAGE" torchrun "${TORCHRUN_ARGS[@]}" /tmp/train_script.py
