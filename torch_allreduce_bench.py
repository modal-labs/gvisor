import json
import os
import pickle

import torch
import torch.distributed as dist

WARMUP_ITERS, TRIALS = 5, 50
N = 500000
M = 2000

FR_DUMP = os.environ.get("FR_DUMP", "0") == "1"
FR_DIR = os.environ.get("FR_DIR", "/tmp/fr_dumps")

def sync_all():
    torch.cuda.synchronize()
    dist.barrier()

def timed_allreduce(mat, start_event, end_event, warmup_iters, iters):
    sync_all()
    for _ in range(warmup_iters):
        dist.all_reduce(mat)
    sync_all()
    start_event.record()
    for _ in range(iters):
        dist.all_reduce(mat)
    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000
    avg_duration = duration / iters
    n = dist.get_world_size()
    size = M * N * 4
    algbw = torch.tensor([size / avg_duration]).cuda(local_rank)
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= n
    return algbw.item()

def dump_flight_recorder(local_rank):
    """Dump the NCCL Flight Recorder ring buffer to disk as JSON.

    The C++ API returns bytes wrapping a nested pickle; we unpack it
    and write a human-readable JSON alongside the raw pickle.
    """
    rank = dist.get_rank()
    os.makedirs(FR_DIR, exist_ok=True)
    try:
        raw = torch._C._distributed_c10d._dump_nccl_trace()
        pkl_path = os.path.join(FR_DIR, f"rank{rank}_gpu{local_rank}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(raw, f)
        data = pickle.loads(raw) if isinstance(raw, bytes) else raw
        json_path = os.path.join(FR_DIR, f"rank{rank}_gpu{local_rank}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[rank {rank}] Flight Recorder -> {json_path}")
    except Exception as e:
        print(f"[rank {rank}] Flight Recorder dump failed: {e}")

def run(local_rank):
    is_global_rank_0 = dist.get_rank() == 0
    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    algbw = timed_allreduce(mat, start_event, end_event, warmup_iters=WARMUP_ITERS, iters=TRIALS)
    n = dist.get_world_size()
    busbw = algbw * (2 * (n - 1) / n)
    if is_global_rank_0:
        print(
            f"The average bandwidth of all_reduce with a {M*N*4/1e9}GB payload ({TRIALS} trials, {n} ranks):\n",
            f"algbw: {algbw/1e9:.3f} GBps ({algbw*8/1e9:.1f} Gbps)\n",
            f"busbw: {busbw/1e9:.3f} GBps ({busbw*8/1e9:.1f} Gbps)\n",
        )
    if FR_DUMP:
        dump_flight_recorder(local_rank)

def init_processes(local_rank, fn, backend="nccl"):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend, device_id=torch.device(f"cuda:{local_rank}"))
    if dist.get_rank() == 0:
        print("Starting benchmark...")
    fn(local_rank)
    sync_all()
    dist.destroy_process_group()

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_processes(local_rank=local_rank, fn=run)
