# nanochat Two-Node Test Notes

Date: 2026-04-17

## Summary

The two-node `nanochat` test did not fail because either worker died. Both nodes remained reachable over SSH after the run.

The failure happened during distributed initialization, before training began:

- module launched: `-m scripts.base_train`
- repo path on workers: `/home/modal/nanochat`
- script file: `/home/modal/nanochat/scripts/base_train.py`
- failing init path:
  - `scripts/base_train.py:86`
  - `nanochat/common.py:201`

## Machine Information

Worker 0:

- SSH host: `wo-fiuot2i3n343royqbcuvi4cmk`
- runtime hostname seen on the machine: `61483872-7edd-4718-85cb-9dfe0ff517d4`
- GPUs: `8x NVIDIA H100 80GB HBM3`
- Python: `3.10.12`
- PyTorch: `2.11.0+cu130`
- `torchrun`: `/home/modal/.local/bin/torchrun`
- `nvidia_peermem`: loaded successfully during earlier setup

Worker 1:

- SSH host: `wo-1skvo5smnlpb47mgtigvo9g0w`
- runtime hostname seen on the machine: `8dbf093a-76af-4789-b40d-ab4ce15707ab`
- GPUs: `8x NVIDIA H100 80GB HBM3`
- Python: `3.10.12`
- PyTorch: `2.11.0+cu130`
- `torchrun`: `/home/modal/.local/bin/torchrun`
- `nvidia_peermem`: loaded successfully during earlier setup

## Network / Fabric Information

Worker 0 primary interfaces:

- `ens10f0np0` -> `93.119.168.24`
- `ens10f0v0` -> `10.100.2.169`
- `ib0..ib7` active
- `ib0` IP used for bootstrap side: `192.168.245.80`

Worker 1 primary interfaces:

- `ens10f0np0` -> `93.119.168.190`
- `ens10f0v0` -> `10.100.0.148`
- `ib0..ib7` active
- `ib0` IP observed earlier: `192.168.244.149`

RDMA devices:

- active HCAs on each node: `mlx5_ib0, mlx5_ib1, mlx5_ib2, mlx5_ib3, mlx5_ib4, mlx5_ib5, mlx5_ib6, mlx5_ib7`

Bootstrap / socket settings used for the `nanochat` test:

- `MASTER_ADDR=192.168.245.80`
- `NCCL_SOCKET_IFNAME=ib0`
- `GLOO_SOCKET_IFNAME=ib0`

Notes from earlier connectivity validation:

- management ping from worker 0 to worker 1 `10.100.0.148` failed
- IB-side ping from worker 0 to worker 1 `ib0` IP `192.168.244.149` succeeded
- because of that, IB bootstrap via `ib0` was used for these tests

## Relevant Source Files

- local clone entrypoint: [`/tmp/nanochat/scripts/base_train.py`](/tmp/nanochat/scripts/base_train.py)
- local clone distributed init: [`/tmp/nanochat/nanochat/common.py`](/tmp/nanochat/nanochat/common.py)

## Setup Performed

On both nodes:

```bash
git clone https://github.com/karpathy/nanochat.git ~/nanochat
cd ~/nanochat
python3 -m pip install --user datasets fastapi kernels psutil rustbpe tiktoken tokenizers uvicorn wandb pyarrow
python3 -m nanochat.dataset -n 1 -w 2
```

On node 0 only:

```bash
cd ~/nanochat
python3 -m scripts.tok_train --max-chars 2000000 --doc-cap 4000 --vocab-size 8192
```

Then copied:

- from node 0: `~/.cache/nanochat/tokenizer`
- to node 1: `~/.cache/nanochat/tokenizer`

## Main Launch Commands

Node 1:

```bash
cd ~/nanochat
env OMP_NUM_THREADS=1 \
  NCCL_SOCKET_IFNAME=ib0 \
  GLOO_SOCKET_IFNAME=ib0 \
  NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7 \
  NCCL_DEBUG=INFO \
  /home/modal/.local/bin/torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=192.168.245.80 \
    --master_port=29673 \
    -m scripts.base_train -- \
    --run=dummy \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=8192 \
    --num-iterations=5 \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --eval-tokens=512
```

Node 0:

```bash
cd ~/nanochat
env OMP_NUM_THREADS=1 \
  NCCL_SOCKET_IFNAME=ib0 \
  GLOO_SOCKET_IFNAME=ib0 \
  NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7 \
  NCCL_DEBUG=INFO \
  /home/modal/.local/bin/torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=192.168.245.80 \
    --master_port=29673 \
    -m scripts.base_train -- \
    --run=dummy \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=8192 \
    --num-iterations=5 \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --eval-tokens=512
```

## Reduced-Footprint Retry

The retry used the same commands except:

```bash
NCCL_IB_HCA=mlx5_ib0
NCCL_IB_QPS_PER_CONNECTION=1
NCCL_MIN_NCHANNELS=1
NCCL_MAX_NCHANNELS=1
```

and:

- `--master_port=29674`
- `--num-iterations=2`

This retry still failed with the same root cause.

## Full Relevant Trace

```text
[rank13]: Traceback (most recent call last):
[rank13]:   File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
[rank13]:     return _run_code(code, main_globals, None,
[rank13]:   File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
[rank13]:     exec(code, run_globals)
[rank13]:   File "/home/modal/nanochat/scripts/base_train.py", line 86, in <module>
[rank13]:     ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
[rank13]:   File "/home/modal/nanochat/nanochat/common.py", line 201, in compute_init
[rank13]:     dist.barrier()
[rank13]:   File "/home/modal/.local/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 83, in wrapper
[rank13]:     return func(*args, **kwargs)
[rank13]:   File "/home/modal/.local/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 5030, in barrier
[rank13]:     work = group.barrier(opts=opts)
[rank13]: torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:3780, unhandled system error (run with NCCL_DEBUG=INFO for details), NCCL version 2.28.9
[rank13]: ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error.
[rank13]: Last error:
[rank13]: Call to ibv_create_cq failed with error Cannot allocate memory
```

Additional NCCL lines seen around the failure:

```text
NCCL WARN Call to ibv_create_cq failed with error Cannot allocate memory
NCCL WARN [Service thread] Error encountered progressing operation=Connect, res=3, closing connection
```

## Interpretation

- `torchrun` multi-node rendezvous worked.
- The script started and reached distributed initialization.
- The failure was not a Python import or tokenizer/data-path issue.
- The failure occurred in NCCL process-group setup during the first `dist.barrier()`.
- The concrete root cause reported by NCCL was RDMA CQ allocation failure:

```text
Call to ibv_create_cq failed with error Cannot allocate memory
```

## Log Locations

- node 1 log: `/tmp/nanochat-node1.log`
- node 0 was streamed live during launch; the reduced retry showed the same CQ allocation failure on rank 0

## Node Reachability After Failure

Both nodes were still alive and reachable over SSH after the failed runs.
