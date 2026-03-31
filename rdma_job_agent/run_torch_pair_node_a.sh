#!/usr/bin/env bash
# Copyright 2025 The gVisor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Node A: POST rank-1 torch job to the rdma_job_agent on node B, wait briefly, run
# rank-0 under runsc-rdma, then GET the job on B for status/output.
#
# Requires: jq, curl, sudo docker; §2-style env on A (see RUNBOOK.md).
#
# Usage:
#   export NODE_A_IP=... NODE_B_IP=... NCCL_IB_HCA=...
#   bash rdma_job_agent/run_torch_pair_node_a.sh
#
# Optional env: PYTORCH_IMAGE, DEVS
# (if unset, DEVS is derived from /dev/infiniband/uverbs*).
#
# NCCL_IB_DISABLE=1 — multi-node over socket (no IB verbs). Use to verify
# torchrun + two nodes + runsc-rdma when IB/RoCE shows ibv_modify_qp EINVAL
# (often fabric / GID / partition mismatch between nodes; not gVisor-specific).

set -euo pipefail

: "${NODE_A_IP:?NODE_A_IP not set}"
: "${NODE_B_IP:?NODE_B_IP not set}"

PYTORCH_IMAGE="${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:24.07-py3}"
if [[ -z "${DEVS:-}" ]]; then
  DEVS="$(ls /dev/infiniband/uverbs* 2>/dev/null | sed 's/^/--device=/' | tr '\n' ' ')"
fi

MASTER_PORT="${MASTER_PORT:-29541}"
RANK1_RUNTIME="${RANK1_RUNTIME:-runsc-rdma}"

if [[ "${NCCL_IB_DISABLE:-0}" == "1" ]]; then
  POST_BODY="$(jq -n \
    --arg ma "$NODE_A_IP" \
    --argjson mp "$MASTER_PORT" \
    '{kind:"torch",master_addr:$ma,master_port:$mp,env:{"NCCL_IB_DISABLE":"1"}}')"
else
  : "${NCCL_IB_HCA:?NCCL_IB_HCA not set (or set NCCL_IB_DISABLE=1 for socket-only smoke test)}"
  POST_BODY="$(jq -n \
    --arg ma "$NODE_A_IP" \
    --arg ib "$NCCL_IB_HCA" \
    --arg rt "$RANK1_RUNTIME" \
    --argjson mp "$MASTER_PORT" \
    '{kind:"torch",runtime:$rt,master_addr:$ma,master_port:$mp,env:{"NCCL_IB_HCA":$ib}}')"
fi

POST_JSON="$(curl -sS -X POST "http://${NODE_B_IP}:8756/v1/jobs" \
  -H 'Content-Type: application/json' \
  -d "$POST_BODY")"
R1="$(echo "$POST_JSON" | jq -r .job_id)"
if [[ -z "$R1" || "$R1" == "null" ]]; then
  echo "run_torch_pair_node_a: POST did not return job_id. Response:" >&2
  echo "$POST_JSON" | jq . >&2 || echo "$POST_JSON" >&2
  exit 1
fi

sleep 2

sudo rm -rf /tmp/runsc-rdma/logs
sudo mkdir -p /tmp/runsc-rdma/logs

TORCH_ARGS=(--nnodes=2 --nproc_per_node=8 --node_rank=0
  --master_addr="$NODE_A_IP" --master_port="$MASTER_PORT"
  /tmp/torch_allreduce_bench.py)

DOCKER_EC=0
if [[ "${NCCL_IB_DISABLE:-0}" == "1" ]]; then
  sudo docker run --runtime=runsc-rdma --rm --gpus all $DEVS \
    --ulimit memlock=-1:-1 --shm-size=1g --network=host \
    -e NCCL_DEBUG=WARN \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_IB_DISABLE=1 \
    -v /tmp/torch_allreduce_bench.py:/tmp/torch_allreduce_bench.py:ro \
    "$PYTORCH_IMAGE" torchrun "${TORCH_ARGS[@]}" || DOCKER_EC=$?
else
  sudo docker run --runtime=runsc-rdma --rm --gpus all $DEVS \
    --ulimit memlock=-1:-1 --shm-size=1g --network=host \
    -v /tmp/nccl_topo.xml:/topo.xml:ro \
    -e NCCL_DEBUG=WARN \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_IB_HCA="$NCCL_IB_HCA" \
    -e NCCL_NET_GDR_LEVEL=3 \
    -e NCCL_DMABUF_ENABLE=0 \
    -e NCCL_IB_GID_INDEX=0 \
    -e NCCL_TOPO_FILE=/topo.xml \
    -v /tmp/torch_allreduce_bench.py:/tmp/torch_allreduce_bench.py:ro \
    "$PYTORCH_IMAGE" torchrun "${TORCH_ARGS[@]}" || DOCKER_EC=$?
fi

curl -sS "http://${NODE_B_IP}:8756/v1/jobs/${R1}" | jq .

exit "${DOCKER_EC:-0}"
