#!/usr/bin/env bash
# Copyright 2025 The gVisor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Single-node DDP MNIST training under runsc-rdma.
# Launches torchrun with all local GPUs (--nproc_per_node=NUM_GPUS).
#
# Usage:
#   bash rdma_job_agent/run_mnist_train.sh
#
# Optional env:
#   RUNTIME        — Docker runtime (default: runsc-rdma)
#   NUM_GPUS       — GPUs to use (default: all, detected via nvidia-smi)
#   PYTORCH_IMAGE  — base image (default: nvcr.io/nvidia/pytorch:24.07-py3)
#   DEVS           — IB device flags (default: auto-detected from /dev/infiniband)
#   EPOCHS         — override EPOCHS in the training script via env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUNTIME="${RUNTIME:-runsc-rdma}"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"

if [[ -z "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
  if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "No GPUs detected; set NUM_GPUS manually." >&2
    exit 1
  fi
fi

if [[ -z "${DEVS:-}" ]]; then
  DEVS="$(ls /dev/infiniband/uverbs* 2>/dev/null | sed 's/^/--device=/' | tr '\n' ' ')" || true
fi

if [[ "$RUNTIME" == "runsc-rdma" ]]; then
  sudo rm -rf /tmp/runsc-rdma/logs
  sudo mkdir -p /tmp/runsc-rdma/logs
fi

echo "=== MNIST DDP training: runtime=$RUNTIME  gpus=$NUM_GPUS  image=$PYTORCH_IMAGE ==="

sudo docker run --runtime="$RUNTIME" --rm --gpus all $DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v "$REPO_ROOT/torch_mnist_train.py:/tmp/torch_mnist_train.py:ro" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}" \
  "$PYTORCH_IMAGE" torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    /tmp/torch_mnist_train.py
