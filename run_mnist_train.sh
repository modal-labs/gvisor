#!/usr/bin/env bash
# DDP MNIST training over 2 nodes via torchrun (bare metal).
#
# Usage:
#   MASTER_ADDR=<node0 IP> NODE_RANK=0 bash run_mnist_train.sh   # on master
#   MASTER_ADDR=<node0 IP> NODE_RANK=1 bash run_mnist_train.sh   # on worker
#
# Optional env: MASTER_PORT, NUM_GPUS, NCCL_DEBUG

set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"

MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required}"
NODE_RANK="${NODE_RANK:?NODE_RANK is required (0 on master, 1 on worker)}"
MASTER_PORT="${MASTER_PORT:-29500}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

NUM_GPUS="${NUM_GPUS:-8}"

# Auto-detect IB HCAs. Exclude IPoIB-only interfaces and mlx devices whose
# active port is Ethernet-only (for example, RoCE/Ethernet ports that NCCL
# should not use for this setup).
if [[ -z "${NCCL_IB_HCA:-}" ]] && command -v ibdev2netdev &>/dev/null; then
  hc_as=()
  while IFS= read -r dev; do
    [[ -n "$dev" ]] || continue
    if command -v ibv_devinfo >/dev/null 2>&1; then
      if ! ibv_devinfo -d "$dev" 2>/dev/null | awk '
        /state:[[:space:]]+PORT_ACTIVE/ { active=1 }
        /link_layer:[[:space:]]+InfiniBand/ { ib=1 }
        END { exit !(active && ib) }'
      then
        continue
      fi
    fi
    hc_as+=("$dev")
  done < <(ibdev2netdev 2>/dev/null | awk '$5 !~ /^ibs/ {print $1}')
  if [[ "${#hc_as[@]}" -gt 0 ]]; then
    NCCL_IB_HCA="$(IFS=,; echo "${hc_as[*]}")"
  fi
fi

export MASTER_ADDR MASTER_PORT NCCL_DEBUG
export NCCL_IB_HCA="${NCCL_IB_HCA:-}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-3}"
# Force NCCL/Gloo to use the data-plane interface, not tailscale0.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens7,eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens7,eth0}"

echo "=== MNIST DDP: gpus=$NUM_GPUS rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT hca=${NCCL_IB_HCA:-auto} ==="

exec torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes=2 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --node_rank="$NODE_RANK" \
  ./torch_mnist_train.py
