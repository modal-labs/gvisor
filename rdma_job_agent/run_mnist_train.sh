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
#   NODE_RANK     - 0 on master, 1 on worker (optional; auto-detected if unset)
#   NNODES        - total number of nodes (default: 1 = standalone)
#
# Optional env:
#   MASTER_PORT   - rendezvous port (default: 29500)
#   RUNTIME       - "torchrun" (default) or Docker runtime name (e.g. runc, runsc-rdma)
#   NUM_GPUS      - GPUs per node (default: all, detected via nvidia-smi)
#   PYTORCH_IMAGE - base image (default: nvcr.io/nvidia/pytorch:26.03-py3)
#   NCCL_IB_HCA   - IB HCA filter (default: auto-detected via ibdev2netdev)
#   DEVS          - IB device flags (default: auto-detected from /dev/infiniband/uverbs*)
#   NCCL_DEBUG    - NCCL log level (default: WARN)
#   MASTER_ADDR_IFACE_REGEX         - interfaces to consider for local IP detection (default: ens7|eth0|gpu)
#   MASTER_ADDR_EXCLUDE_IFACE_REGEX - interfaces to exclude from local IP detection (default: ibs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

RUNTIME="${RUNTIME:-torchrun}"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"
if [[ -n "${MASTER_ADDR:-}" ]]; then
  NNODES="${NNODES:-2}"
else
  NNODES="${NNODES:-1}"
fi
MASTER_PORT="${MASTER_PORT:-29500}"
NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11}"

cleanup_children() {
  # Best-effort: ensure a Ctrl-C kills torchrun + all worker processes.
  # (torchrun sometimes leaves worker python procs behind.)
  pkill -TERM -P "$$" 2>/dev/null || true
  sleep 1
  pkill -KILL -P "$$" 2>/dev/null || true
}

trap cleanup_children INT TERM

detect_local_ipv4s() {
  local iface_re="${MASTER_ADDR_IFACE_REGEX:-ens7|eth0|gpu}"
  local exclude_re="${MASTER_ADDR_EXCLUDE_IFACE_REGEX:-ibs}"
  ip -o -4 addr show \
    | awk -v iface_re="$iface_re" -v exclude_re="$exclude_re" '
      $2 ~ iface_re && $2 !~ exclude_re {
        split($4, a, "/")
        print a[1]
      }' \
    | sort -u
}

detect_nccl_ib_hca() {
  # Best-effort: build a comma-separated mlx5 list via ibdev2netdev.
  # Example output:
  #   mlx5_10,mlx5_11,mlx5_12,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
  if command -v ibdev2netdev >/dev/null 2>&1; then
    ibdev2netdev \
      | awk '!/ibs/ && $1 ~ /(ib|rdma|gpu)/ {print $1}' \
      | paste -sd, - \
      | sed 's/,$//'
  fi
}

NODE_RANK_SOURCE=""
if [[ -n "${NODE_RANK:-}" ]]; then
  : # use NODE_RANK as-is
  NODE_RANK_SOURCE="NODE_RANK"
elif [[ -n "${RANK:-}" ]]; then
  NODE_RANK="$RANK"
  NODE_RANK_SOURCE="RANK"
elif [[ "$NNODES" -gt 1 ]]; then
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    echo "MASTER_ADDR is required when NNODES > 1." >&2
    exit 2
  fi
  mapfile -t LOCAL_IPV4S < <(detect_local_ipv4s)
  if [[ "${#LOCAL_IPV4S[@]}" -eq 0 ]]; then
    mapfile -t LOCAL_IPV4S < <(ip -o -4 addr show scope global | awk '{split($4,a,"/"); if (a[1] !~ /^127\\./) print a[1]}' | sort -u)
  fi
  if [[ "${#LOCAL_IPV4S[@]}" -eq 0 ]]; then
    echo "Couldn't detect a local IPv4 for rank auto-detection; set NODE_RANK explicitly." >&2
    exit 2
  fi
  NODE_RANK=1
  for ip in "${LOCAL_IPV4S[@]}"; do
    if [[ "$ip" == "$MASTER_ADDR" ]]; then
      NODE_RANK=0
      break
    fi
  done
  NODE_RANK_SOURCE="auto"
  echo "Auto-detected NODE_RANK=$NODE_RANK (MASTER_ADDR=$MASTER_ADDR, local IPv4s: ${LOCAL_IPV4S[*]})"
else
  NODE_RANK=0
  NODE_RANK_SOURCE="default"
fi

# Always require MASTER_ADDR for multi-node; keep it explicit to avoid
# accidental split-brain rendezvous.
if [[ "$NNODES" -gt 1 && -z "${MASTER_ADDR:-}" ]]; then
  echo "MASTER_ADDR is required when NNODES > 1." >&2
  exit 2
fi

# Auto-detect NCCL_IB_HCA if omitted.
if [[ -z "${NCCL_IB_HCA:-}" ]]; then
  NCCL_IB_HCA="$(detect_nccl_ib_hca || true)"
fi
if [[ -z "${NCCL_IB_HCA:-}" ]]; then
  NCCL_IB_HCA="mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11"
fi

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
  echo "=== MNIST DDP: runtime=$RUNTIME gpus=$NUM_GPUS nnodes=$NNODES rank=$NODE_RANK ($NODE_RANK_SOURCE) master=$MASTER_ADDR:$MASTER_PORT ==="
else
  TORCHRUN_ARGS=(--standalone --nproc_per_node="$NUM_GPUS")
  echo "=== MNIST DDP: runtime=$RUNTIME gpus=$NUM_GPUS standalone ==="
fi

if [[ "$RUNTIME" == "torchrun" ]]; then
  torchrun "${TORCHRUN_ARGS[@]}" ./torch_mnist_train.py
else
  sudo docker run --runtime="$RUNTIME" --rm --gpus all ${DEVS} \
    --ulimit memlock=-1:-1 --shm-size=1g --network=host \
    "${EXTRA_DOCKER_ARGS[@]}" \
    -v "${REPO_ROOT}/torch_mnist_train.py:/tmp/train_script.py:ro" \
    -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}" \
    -e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}" \
    "${EXTRA_ENV[@]}" \
    "$PYTORCH_IMAGE" torchrun "${TORCHRUN_ARGS[@]}" /tmp/train_script.py
fi
