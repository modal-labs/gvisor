# gVisor RDMA

Two nodes, mlx5, Docker, sudo, L3 for NCCL. Commented lines = node / one-off only.

**A dev / B peer:** A runs clone, `make copy`, topo generation, and drives `curl`. B only needs what’s in the table — copy **`runsc`** and **`/tmp/nccl_topo.xml`** from A when they change; no clone / `make` / `nccl-test` on B unless you want B self-contained.

| Need on B | Why |
|-----------|-----|
| `runsc` / `runsc-rdma` from what A built | Same runtime under test |
| `daemon.json` + `nvidia-peermem` + PyTorch image pull | Same Docker + GPU/IB stack |
| `/tmp/nccl_topo.xml` | Copy from A for `runsc-rdma` (same file both sides) |
| `/tmp/torch_allreduce_bench.py` | Same bench |
| Exports (`NODE_A_IP`, `NODE_B_IP`, `NCCL_IB_HCA`, …) | Same NCCL view of the job |
| `rdma_job_agent` on **B only** | Rank **1** via `POST`; rank **0** on A is plain `docker run` (no agent on A) |

**Skip on B** if A does the work: clone, `make copy` (B only receives the binary), topo generation on A (B only receives `/tmp/nccl_topo.xml`).

```bash
# --- 1) Node A: clone, build, copy /tmp/runsc to B ---
# git clone git@github.com:modal-labs/gvisor.git && cd gvisor && git checkout alessio/development
# sudo make copy TARGETS=runsc DESTINATION=/tmp && sudo chmod +r /tmp/runsc

# --- 2) Both nodes ---
sudo apt-get update && sudo apt-get install -y jq

sudo pkill -f "runsc-rdma" 2>/dev/null || true; sleep 1
sudo rm -f /usr/local/bin/runsc-rdma
sudo cp /tmp/runsc /usr/local/bin/runsc-rdma && sudo chmod +x /usr/local/bin/runsc-rdma

sudo test -f /etc/docker/daemon.json || echo '{}' | sudo tee /etc/docker/daemon.json >/dev/null
sudo jq '(.runtimes //= {}) | .runtimes["runsc-rdma"] = {
  "path": "/usr/local/bin/runsc-rdma",
  "runtimeArgs": [
    "--debug", "--debug-log=/tmp/runsc-rdma/logs/",
    "--rdmaproxy", "--nvproxy",
    "--nvproxy-allowed-driver-capabilities=compute,utility,video",
    "--network=host", "--rdma-expected-ipoib=-1"
  ]
}' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp >/dev/null \
  && sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
sudo systemctl restart docker && sleep 2
sudo modprobe nvidia-peermem

# If `runsc-rdma` was never configured, a merge-only jq can create `runtimeArgs` without `"path"` → Docker reports
# “unknown or invalid runtime name”. Re-run the full `jq` assignment above (lines 31–40), or merge only when
# `sudo jq '.runtimes["runsc-rdma"].path' /etc/docker/daemon.json` already prints the runsc binary path.

export PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
sudo docker login nvcr.io
sudo docker pull "$PYTORCH_IMAGE"

export NODE_A_IP=<a> NODE_B_IP=<b>
export NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11
export DEVS=$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
# No `--cpus` here: use Docker’s cgroup CPU set (see Modal note below).
export DOCKER_RUN="--rm --gpus all $DEVS --ulimit memlock=-1:-1 --shm-size=1g --network=host"
```

**Modal workers — Docker stuck at 12 CPUs:** The host has more CPUs (`nproc` e.g. 112), but `systemctl show system.slice -p AllowedCPUs` is **`0-11`**. `dockerd` lives in **`system.slice`**, so it inherits that **cpuset** (`docker info` → `CPUs: 12`; `docker run --cpus=100` fails). Workloads are meant to use **`modal-containers.slice`**, which has **`12-111`**. Move Docker into that slice, reload, restart:

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/50-modal-slice.conf <<'EOF'
[Service]
Slice=modal-containers.slice
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo docker info | grep '^ CPUs:'
```

Container runs omit `--cpus`; the effective limit is whatever CPUs Docker’s slice allows (`sudo docker info` → **CPUs:**, about **100** after moving Docker to `modal-containers.slice`).

```bash
# --- 3) Both nodes ---
cp ~/gvisor/torch_allreduce_bench.py /tmp/torch_allreduce_bench.py
```

```bash
# --- 4) Node A: /tmp/nccl_topo.xml for runsc-rdma (same path on B after copy) ---
# One-time per hardware SKU (or when NICs/GPUs change). Reuse the same XML for every
# gVisor/runsc iteration — no need to rerun this docker command each build.
# NCCL writes the file when it initializes if NCCL_TOPO_DUMP_FILE is set. Use runc here
# (full sysfs); gVisor hides PCI — dumps under runsc are wrong. Stop after the XML exists.
# Bind-mount host /tmp so the dump is on the host (otherwise it only exists in the
# container and disappears with --rm).
sudo docker run --runtime=runc --rm --gpus all $DOCKER_RUN \
  -e NCCL_DEBUG=WARN -e NCCL_SOCKET_IFNAME=eth0 -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml \
  -v /tmp:/tmp \
  "nvcr.io/nvidia/pytorch:24.07-py3" torchrun --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29599 \
  /tmp/torch_allreduce_bench.py
# Copy topo to B: `scp /tmp/nccl_topo.xml modal@${NODE_B_IP}:/tmp/nccl_topo.xml`
# or POST the file to the job agent: `curl -sS -X POST --data-binary @/tmp/nccl_topo.xml "http://${NODE_B_IP}:8756/v1/nccl_topo"`
# (see rdma_job_agent/README.md).
```

```bash
# --- 5) Node B only: agent. A does not run the agent. ---
cd ~/gvisor/rdma_job_agent
python3 agent.py --host 0.0.0.0 --port 8756
```

```bash
# --- 6) Node A: POST rank 1 on B, then rank 0 locally (runsc-rdma). ---
# One composable command (same sequence as below; good for automation / agents):
#   export NODE_A_IP=... NODE_B_IP=... NCCL_IB_HCA=...   # plus §2 as needed (PYTORCH_IMAGE, DEVS)
#   bash ~/gvisor/rdma_job_agent/run_torch_pair_node_a.sh
#
# Agent on B applies torch defaults; export NCCL_IB_HCA on B before agent.py — see rdma_job_agent/README.md.
# Manual equivalent (POST → sleep → rank-0 docker → poll). Master port fixed at 29541 (change in both POST and torchrun if needed):
POST_BODY="$(jq -n --arg ma "$NODE_A_IP" --argjson mp "29541" '{kind:"torch",master_addr:$ma,master_port:$mp}')"
R1=$(curl -sS -X POST "http://${NODE_B_IP}:8756/v1/jobs" -H 'Content-Type: application/json' -d "$POST_BODY" | jq -r .job_id)
# No jq for POST_BODY (IPv4): POST_BODY='{"kind":"torch","master_addr":"'"$NODE_A_IP"'","master_port":29541}'
sleep 2

sudo rm -rf /tmp/runsc-rdma/logs && sudo mkdir -p /tmp/runsc-rdma/logs
sudo docker run --runtime=runsc-rdma --rm --gpus all $DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -v /tmp/nccl_topo.xml:/topo.xml:ro \
  -e NCCL_DEBUG=WARN \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 \
  -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_IB_GID_INDEX=0 \
  -e NCCL_TOPO_FILE=/topo.xml \
  -v /tmp/torch_allreduce_bench.py:/tmp/torch_allreduce_bench.py:ro \
  "$PYTORCH_IMAGE" torchrun \
    --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr="${NODE_A_IP}" --master_port=29541 \
    /tmp/torch_allreduce_bench.py

curl -sS "http://${NODE_B_IP}:8756/v1/jobs/$R1" | jq .

# runc: --arg rt 'runc'; rank-0 docker: --runtime=runc, drop topo mount + NCCL_TOPO_FILE + NCCL_IB_GID_INDEX
```

---

## Optional: syscall timeline (gVisor strace → Perfetto)

gVisor **`--strace`** logs **guest** syscalls to the same **`--debug-log`** tree as `--debug` (not Linux `ptrace` on `runsc`). Use it for a **timeline** by converting logs to Chrome Trace JSON and opening them in **Perfetto** (or any Chrome-trace viewer).

### 1. Enable strace on `runsc-rdma`

Add to **`runtimes.runsc-rdma.runtimeArgs`** in `/etc/docker/daemon.json` (same array as §2), then **`sudo systemctl restart docker`**:

- **`--strace`** — trace syscalls (verbose; omit filters only for short runs).
- Optionally **`--strace-syscalls=ioctl,mmap,munmap,openat,close,...`** — comma list (see `runsc --help`).

Example (merge manually or extend the §2 `jq` so these two strings appear in `runtimeArgs`):

```json
"--strace",
"--strace-syscalls=ioctl,mmap,munmap,openat,close"
```

### 2. Capture logs

Run your workload (§6). Strace lines go into files under **`--debug-log`** (e.g. `/tmp/runsc-rdma/logs/`). Names look like **`runsc.log.<timestamp>.<command>.txt`**.

### 3. Convert to Chrome trace JSON

From the gVisor repo (or any path with **`tools/gvisor_strace_to_chrome_trace.py`**):

```bash
LOG="$(ls -t /tmp/runsc-rdma/logs/runsc.log.*.txt 2>/dev/null | head -1)"
test -n "$LOG" && python3 ~/gvisor/tools/gvisor_strace_to_chrome_trace.py "$LOG" -o /tmp/strace.json
```

Glog lines omit the year in the date prefix; pass **`--year YYYY`** if needed (default: current UTC year). For **`--debug-log-format=json`**, the same script accepts JSON lines.

### 4. View

Open **`https://ui.perfetto.dev`** → **Open trace file** → choose **`/tmp/strace.json`**.

(Eclipse Trace Compass may import Chrome-style JSON depending on build; Perfetto is the path we test.)

### 5. Tests

```bash
cd ~/gvisor/tools && python3 gvisor_strace_to_chrome_trace_test.py -v
```
