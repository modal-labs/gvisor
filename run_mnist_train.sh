#!/usr/bin/env bash
# DDP MNIST training over 2 nodes via torchrun (bare metal).
#
# Usage:
#   MASTER_ADDR=<node0 IP> bash run_mnist_train.sh
#
# Optional env: NODE_RANK, MASTER_PORT, NUM_GPUS, NCCL_DEBUG

set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"

MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required}"
MASTER_PORT="${MASTER_PORT:-29500}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ -z "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
  if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "No GPUs detected; set NUM_GPUS manually." >&2
    exit 1
  fi
fi

# Auto-detect node rank from local IPs.
if [[ -z "${NODE_RANK:-}" ]]; then
  if ip -o -4 addr show 2>/dev/null | grep -qw "$MASTER_ADDR"; then
    NODE_RANK=0
  else
    NODE_RANK=1
  fi
  echo "Auto-detected NODE_RANK=$NODE_RANK"
fi

# Auto-detect IB HCAs, excluding IPoIB-only devices (ibs* interfaces).
if [[ -z "${NCCL_IB_HCA:-}" ]] && command -v ibdev2netdev &>/dev/null; then
  NCCL_IB_HCA="$(ibdev2netdev 2>/dev/null | awk '$5 !~ /^ibs/ {print $1}' | paste -sd, - || true)"
fi

# Auto-detect socket interface for NCCL OOB.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  NCCL_SOCKET_IFNAME="$(ip -o link show 2>/dev/null | awk -F': ' '$2 != "lo" {print $2; exit}' || true)"
fi

export MASTER_ADDR MASTER_PORT NCCL_DEBUG
export NCCL_IB_HCA="${NCCL_IB_HCA:-}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"

echo "=== MNIST DDP: gpus=$NUM_GPUS rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT hca=${NCCL_IB_HCA:-auto} ifname=$NCCL_SOCKET_IFNAME ==="

exec torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes=2 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --node_rank="$NODE_RANK" \
  ./torch_mnist_train.py
