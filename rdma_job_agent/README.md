# RDMA job agent (node B)

Long-running HTTP service that **starts** `docker run` workloads (NCCL bench,
PyTorch `torchrun`, etc.) on the machine where it runs.

**Only node B runs the agent.** Node A does **not** need it: the master / rank 0
side runs with a normal **`docker run`** on A (see `RUNBOOK.md`).

**Multi-node is still two peers:** Typical flow:

1. **POST** a job to the agent on **B** with `"async": true`, `node_rank: 1`,
   `master_addr` = node A’s IP.
2. **Start rank 0 on node A** with `docker run` + `torchrun --node_rank=0`
   (same **`master_port`** as rank 1 — **29541** in `RUNBOOK.md` / `run_torch_pair_node_a.sh`, image, and env).

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

Optional: `export RDMA_TOPO_PATH=/tmp/nccl_topo.xml` to change where `POST /v1/nccl_topo` writes.

Spawned `docker run` commands do not pass `--cpus`; CPU use follows Docker’s cgroup (same as omitting `--cpus` on the CLI).

**Torch defaults (short POST bodies):** If you `export NCCL_IB_HCA=…` (and optionally `NCCL_SOCKET_IFNAME`, `RDMA_PYTORCH_IMAGE`, `RDMA_JOB_RUNTIME`) on **B** before starting the agent, a minimal job only needs `kind` and `master_addr` (`master_port` defaults to **29541**). The agent fills in `runtime` (default `runsc-rdma`), `async: true`, `nnodes: 2`, `node_rank: 1`, `nproc_per_node: 8`, image and script paths, and a standard `env` block (merged with any `env` you send in JSON).

### Push topology from node A to B (instead of `scp`)

After generating `/tmp/nccl_topo.xml` on A, upload it to the agent on B (same path the job runner uses for `-v …:/topo.xml`):

```bash
curl -sS -X POST --data-binary @/tmp/nccl_topo.xml \
  "http://${NODE_B_IP}:8756/v1/nccl_topo"
```

If you see `Operation not permitted` replacing `/tmp/nccl_topo.xml` on **B**, that path is often **root-owned** (e.g. left over from `sudo docker`). On B: `sudo rm -f /tmp/nccl_topo.xml`, then POST again (or `sudo chown "$USER" /tmp/nccl_topo.xml`).

Use SSH port-forward (`-L 8756:127.0.0.1:8756`) if B’s agent binds to loopback only.

## One-shot node A (automation-friendly)

From **node A**, after §2 env and with the agent listening on B, one script runs POST → sleep → rank-0 `docker run` → poll (same as `RUNBOOK.md` §6):

```bash
export NODE_A_IP=... NODE_B_IP=... NCCL_IB_HCA=...
bash ~/gvisor/rdma_job_agent/run_torch_pair_node_a.sh
```

Optional: `PYTORCH_IMAGE`, `DEVS` (if unset, `DEVS` is built from `/dev/infiniband/uverbs*`). Torch jobs use **`master_port` 29541** in the one-shot script and runbook.

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
- **`cgroup_parent`**: optional Docker `--cgroup-parent` value. Useful on
  Modal workers to force jobs into `modal-containers.slice` so they inherit the
  100-CPU cpuset instead of the default 12-CPU `system.slice` scope. You can
  also set `RDMA_JOB_CGROUP_PARENT` in the agent environment to apply a default.

**NCCL** (`kind: "nccl"`):

- **`rank`**, **`nranks`**, **`master_addr`**, **`master_port`** (required).
- **`ngpus`** (default 8), **`image`** (default `nccl-test`),
  **`bench_binary`** (default `/usr/local/bin/nccl_multinode_bench`).
- **`topo_host_path`**: host path for `-v …:/topo.xml:ro` when using
  `runsc-rdma` (default `/tmp/nccl_topo.xml`).
- **`env`**: object of extra `-e` variables (e.g. `NCCL_DEBUG`, `NCCL_IB_HCA`,
  `NCCL_SOCKET_IFNAME`, …). Match the runbook and your hardware.

**PyTorch** (`kind: "torch"`):

- **`master_addr`** (required). **`master_port`** defaults to **29541** if omitted. **`nnodes`**, **`node_rank`** optional if you use defaults below.
- Defaults applied when omitted: **`runtime`** `runsc-rdma` (override with `RDMA_JOB_RUNTIME` on the agent host), **`async`** `true`, **`nnodes`** `2`, **`node_rank`** `1`, **`nproc_per_node`** `8`, **`image`** `nvcr.io/nvidia/pytorch:24.07-py3` (override with `RDMA_PYTORCH_IMAGE`), **`script_host_path`** `/tmp/torch_allreduce_bench.py`, **`topo_host_path`** `/tmp/nccl_topo.xml`.
- **`env`**: merged with defaults (`NCCL_DEBUG`, `NCCL_SOCKET_IFNAME`, `NCCL_NET_GDR_LEVEL`, `NCCL_DMABUF_ENABLE`, and **`NCCL_IB_HCA` from the agent process environment** if set). Your JSON `env` keys override.

Minimal example from node A (agent on B has `NCCL_IB_HCA` exported):

```bash
curl -sS -X POST "http://${NODE_B_IP}:8756/v1/jobs" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg ma "$NODE_A_IP" --argjson mp "29541" \
      '{kind:"torch",master_addr:$ma,master_port:$mp}')"
```

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

Then on **node A**, run the rank-0 `docker run` + `torchrun` (same **`master_port`**
**29541** and env) — see **`RUNBOOK.md`** section 6.

Poll B:

```bash
curl -sS http://127.0.0.1:8756/v1/jobs/<job_id>
```

### Other training programs

Extend `agent.py`: add a `kind` and a `build_*_command` function, or reuse
`kind: "torch"` with a different `script_host_path` and image after you build
a suitable Docker image on B.
