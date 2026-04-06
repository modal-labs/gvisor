#!/usr/bin/env bash
# DDP MNIST training (2 nodes only; run on each node).
#
# On each node:
#   MASTER_ADDR=<node0 IP> bash rdma_job_agent/run_mnist_train.sh
#
# Required env:
#   MASTER_ADDR   - IPv4 of the rank-0 node
#   NODE_RANK     - 0 on master, 1 on worker (optional; auto-detected if unset)
#   NNODES        - must be 2 (default: 2)
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
NNODES="${NNODES:-2}"
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

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "MASTER_ADDR is required (this script is 2-node only)." >&2
  exit 2
fi
if [[ "$NNODES" -ne 2 ]]; then
  echo "NNODES must be 2 (this script is 2-node only)." >&2
  exit 2
fi

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

default_socket_ifname() {
  # Prefer eth0/ens7 if present; otherwise fall back to first non-lo interface.
  for iface in eth0 ens7; do
    if ip link show "$iface" >/dev/null 2>&1; then
      echo "$iface"
      return 0
    fi
  done
  ip -o link show | awk -F': ' '$2 != "lo" {print $2; exit}'
}

NODE_RANK_SOURCE=""
if [[ -n "${NODE_RANK:-}" ]]; then
  : # use NODE_RANK as-is
  NODE_RANK_SOURCE="NODE_RANK"
elif [[ -n "${RANK:-}" ]]; then
  NODE_RANK="$RANK"
  NODE_RANK_SOURCE="RANK"
else
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
fi

# Auto-detect NCCL_IB_HCA if omitted.
if [[ -z "${NCCL_IB_HCA:-}" ]]; then
  NCCL_IB_HCA="$(detect_nccl_ib_hca || true)"
fi
if [[ -z "${NCCL_IB_HCA:-}" ]]; then
  NCCL_IB_HCA="mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11"
fi

NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$(default_socket_ifname)}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Optional: turn on verbose distributed/NCCL logging to debug hangs.
if [[ "${DEBUG_DDP:-0}" == "1" ]]; then
  TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
  NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
  NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
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
EXTRA_ENV+=(-e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}")
EXTRA_ENV+=(-e "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}")

TORCHRUN_ARGS=(
  --nproc_per_node="$NUM_GPUS"
  --nnodes="$NNODES"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
  --node_rank="$NODE_RANK"
)
echo "=== MNIST DDP: runtime=$RUNTIME gpus=$NUM_GPUS nnodes=$NNODES rank=$NODE_RANK ($NODE_RANK_SOURCE) master=$MASTER_ADDR:$MASTER_PORT ==="

if [[ "$RUNTIME" == "torchrun" ]]; then
  SUDO_ENV=(
    "MASTER_ADDR=$MASTER_ADDR" "MASTER_PORT=$MASTER_PORT"
    "NCCL_IB_HCA=$NCCL_IB_HCA"
    "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}"
    "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
    "NCCL_DEBUG=$NCCL_DEBUG"
  )
  [[ -n "${NCCL_DEBUG_SUBSYS:-}" ]] && SUDO_ENV+=("NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS")
  [[ -n "${NCCL_ASYNC_ERROR_HANDLING:-}" ]] && SUDO_ENV+=("NCCL_ASYNC_ERROR_HANDLING=$NCCL_ASYNC_ERROR_HANDLING")
  [[ -n "${NCCL_BLOCKING_WAIT:-}" ]] && SUDO_ENV+=("NCCL_BLOCKING_WAIT=$NCCL_BLOCKING_WAIT")
  [[ -n "${TORCH_DISTRIBUTED_DEBUG:-}" ]] && SUDO_ENV+=("TORCH_DISTRIBUTED_DEBUG=$TORCH_DISTRIBUTED_DEBUG")

  # sudo gives us unlimited memlock (RDMA verbs pin pages for CQs/MRs).
  sudo env "${SUDO_ENV[@]}" torchrun "${TORCHRUN_ARGS[@]}" ./torch_mnist_train.py
else
  sudo docker run --runtime="$RUNTIME" --rm --gpus all ${DEVS} \
    --ulimit memlock=-1:-1 --shm-size=1g --network=host \
    "${EXTRA_DOCKER_ARGS[@]}" \
    -v "${REPO_ROOT}/torch_mnist_train.py:/tmp/train_script.py:ro" \
    -e "NCCL_DEBUG=${NCCL_DEBUG}" \
    "${EXTRA_ENV[@]}" \
    "$PYTORCH_IMAGE" torchrun "${TORCHRUN_ARGS[@]}" /tmp/train_script.py
fi
