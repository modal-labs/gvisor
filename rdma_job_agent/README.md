# RDMA job agent (node B)

Long-running HTTP service that **starts** `docker run` workloads (NCCL bench,
PyTorch `torchrun`, etc.) on the machine where it runs.

**Only node B runs the agent.** Node A does **not** need it: the master / rank 0
side runs with a normal **`docker run`** on A (see `RUNBOOK.md`).

**Multi-node is still two peers:** Typical flow:

1. **POST** a job to the agent on **B** with `"async": true`, `node_rank: 1`,
   `master_addr` = node A’s IP.
2. **Start rank 0 on node A** with `docker run` + `torchrun --node_rank=0`
   (same `MASTER_PORT`, image, and env as the agent would use).

## Security

- Default bind is **`127.0.0.1:8756`** — not reachable from other hosts.
- If B binds **`127.0.0.1`** only, reach from A with SSH local forward:  
  `ssh -L 8756:127.0.0.1:8756 user@NODE_B_IP`  
  then `curl http://127.0.0.1:8756/...` from A.  
  If B uses **`0.0.0.0:8756`**, curl **`http://${NODE_B_IP}:8756`** from A (open the
  port in the cloud security group / firewall).
- This API runs **`sudo docker`** as the user running the agent. Treat it like
  root-equivalent; do not expose it on untrusted networks without auth/TLS.

## Run on node B

```bash
cd ~/gvisor/rdma_job_agent
python3 agent.py --host 127.0.0.1 --port 8756
```

Requires `sudo docker` to work non-interactively for that user (e.g. sudoers).

Optional: `export DOCKER_CPUS=<n>` before starting the agent to set `docker run --cpus` for spawned jobs (default: all CPUs Docker reports on the host).

Optional: `export RDMA_TOPO_PATH=/tmp/nccl_topo.xml` to change where `POST /v1/nccl_topo` writes.

### Push topology from node A to B (instead of `scp`)

After generating `/tmp/nccl_topo.xml` on A, upload it to the agent on B (same path the job runner uses for `-v …:/topo.xml`):

```bash
curl -sS -X POST --data-binary @/tmp/nccl_topo.xml \
  "http://${NODE_B_IP}:8756/v1/nccl_topo"
```

Use SSH port-forward (`-L 8756:127.0.0.1:8756`) if B’s agent binds to loopback only.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | `{"ok": true}` |
| GET | `/v1/jobs` | List job ids with `state` and `kind` (debugging). |
| POST | `/v1/jobs` | Submit a job (see below). JSON body; `Content-Type: application/json` recommended. |
| POST | `/v1/jobs/<uuid>/cancel` | `SIGTERM` the spawned `docker run` process while it is running (requires a live process handle). |
| POST | `/v1/nccl_topo` | Upload NCCL topology XML (raw body; `Content-Length` required). Writes to `RDMA_TOPO_PATH` or `/tmp/nccl_topo.xml`. |
| GET | `/v1/jobs/<uuid>` | Poll job status and output (`proc` is never returned). |

Completed jobs are evicted after **50** terminal (`done` / `error`) records to bound memory.

### POST `/v1/jobs` body

Common fields:

- **`kind`**: `"nccl"` or `"torch"`.
- **`runtime`**: `"runc"` or `"runsc-rdma"`.
- **`async`**: if `true`, returns `202` immediately; poll `GET /v1/jobs/<id>` for
  `state` (`queued` → `running` → `done`) and `output`.

**NCCL** (`kind: "nccl"`):

- **`rank`**, **`nranks`**, **`master_addr`**, **`master_port`** (required).
- **`ngpus`** (default 8), **`image`** (default `nccl-test`),
  **`bench_binary`** (default `/usr/local/bin/nccl_multinode_bench`).
- **`topo_host_path`**: host path for `-v …:/topo.xml:ro` when using
  `runsc-rdma` (default `/tmp/nccl_topo.xml`).
- **`env`**: object of extra `-e` variables (e.g. `NCCL_DEBUG`, `NCCL_IB_HCA`,
  `NCCL_SOCKET_IFNAME`, …). Match the runbook and your hardware.

**PyTorch** (`kind: "torch"`):

- **`nnodes`**, **`node_rank`**, **`master_addr`**, **`master_port`** (required).
- **`nproc_per_node`** (default 8), **`image`** (default `nvcr.io/nvidia/pytorch:24.07-py3`).
- **`script_host_path`**: host path mounted as `/tmp/torch_allreduce_bench.py`
  (default `/tmp/torch_allreduce_bench.py`).
- **`topo_host_path`**, **`env`**: same idea as NCCL.

### Example: NCCL rank 1 on B (async), then rank 0 on A

From A (with SSH `-L 8756:127.0.0.1:8756` to B), after exporting `NODE_A_IP` and
`NCCL_IB_HCA` on both sides:

```bash
curl -sS -X POST http://127.0.0.1:8756/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "nccl",
    "runtime": "runsc-rdma",
    "async": true,
    "rank": 1,
    "nranks": 2,
    "ngpus": 8,
    "master_addr": "'"$NODE_A_IP"'",
    "master_port": 29501,
    "env": {
      "NCCL_DEBUG": "INFO",
      "NCCL_SOCKET_IFNAME": "eth0",
      "NCCL_IB_HCA": "'"$NCCL_IB_HCA"'",
      "NCCL_NET_GDR_LEVEL": "3",
      "NCCL_DMABUF_ENABLE": "0"
    }
  }'
```

Then on **node A**, run the rank-0 `docker run` + `torchrun` (same `MASTER_PORT`
and env) — see **`RUNBOOK.md`** section 6.

Poll B:

```bash
curl -sS http://127.0.0.1:8756/v1/jobs/<job_id>
```

### Other training programs

Extend `agent.py`: add a `kind` and a `build_*_command` function, or reuse
`kind: "torch"` with a different `script_host_path` and image after you build
a suitable Docker image on B.
