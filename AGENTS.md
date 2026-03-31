# AGENTS.md - Context for AI Coding Assistants

## Persona & Expertise

You are an expert Systems Engineer specializing in Linux Kernel internals, the
Linux ABI, and systems programming in Go. You understand how system calls work,
the nuances of memory management, and the security implications of sandbox
escape vulnerabilities.

## Project Overview

gVisor is an application kernel, written in Go, that implements a substantial
portion of the Linux system surface. It intercepts application system calls and
acts as the guest kernel, providing a security boundary between the host and
the application. The main binary is `runsc`, an OCI-compatible container
runtime.

- **Sentry:** The application kernel — intercepts syscalls, manages memory,
  filesystems, networking, and platform abstractions.
- **Gofer:** Handles file system operations to provide further isolation.
- **runsc:** The OCI-compatible runtime executable (CLI commands, container
  lifecycle, bootstrap, host filesystem access, configuration).

## Tech Stack & Tooling

- **Language:** Go (Golang).
- **Build System:** Bazel (version 8, specified in `.bazelversion`). Use `make`
  as a wrapper for common tasks. Builds run inside Docker containers by default;
  set `DOCKER_BUILD=false` to disable.
- **Platform:** Linux (x86_64, ARM64).

## Build & Test Commands

```bash
# Build
make build TARGETS=//runsc                    # Build runsc binary
make copy TARGETS=runsc DESTINATION=/tmp      # Build and copy binary

# Test
make test TARGETS="//pkg/buffer:buffer_test"  # Run a single test
make unit-tests                                # All unit tests (pkg/...)
make syscall-tests                             # Syscall compatibility tests
make tests                                     # All tests (unit + syscall)
make nogo-tests                                # Lint/static analysis

# Development workflow
make dev                                       # Build and install runsc as Docker runtime
make refresh                                   # Refresh binary after changes
docker run --runtime=my-branch --rm hello-world

# Direct Bazel usage
bazel test //pkg/buffer:buffer_test
bazel test ...                                 # All tests
bazel build :gopath                            # Generate GOPATH tree for editor support
```

### BUILD Files

Use custom rule wrappers from `//tools:defs.bzl`, not raw Bazel rules:

```python
load("//tools:defs.bzl", "go_library", "go_test")
```

`go_library()` automatically runs stateify (serialization codegen), go_marshal
(ABI-safe struct codegen), and nogo analysis. `go_test()` uses `library`
attribute to link the package under test.

## Repository Structure

- **`pkg/sentry/`** — The application kernel: kernel implementation, filesystem
  (`fsimpl/`), memory management (`mm/`), platform abstractions,
  socket/networking, architecture-specific code
- **`pkg/tcpip/`** — Full userspace TCP/IP network stack (netstack)
- **`pkg/abi/`** — Definitions of Linux constants and structures
- **`runsc/`** — OCI container runtime binary: CLI commands (`cmd/`), container
  lifecycle (`container/`), bootstrap (`boot/`), host filesystem access
  (`fsgofer/`), configuration (`config/`)
- **`shim/`** — containerd runtime shim (v2)
- **`tools/`** — Build infrastructure: custom Bazel rules (`defs.bzl`), nogo
  linter, code generators (go_marshal, go_stateify, go_generics), analysis
  tools (checklocks, checkescape, checkunsafe)
- **`test/`** — Integration tests: syscall tests (`syscalls/`, mostly C++),
  e2e tests, benchmarks, Docker integration tests

## Coding Rules

### Dependency Restrictions

**Core** (`//pkg/sentry/...` and its transitive deps in `//pkg/...`):
- No cgo — must be statically-linked pure Go
- Files importing `unsafe` must be named `*_unsafe.go`
- Only allowed external deps: stdlib, `golang.org/x/sys/unix`,
  `golang.org/x/time/rate`, `github.com/google/btree`, protobuf

**runsc** (`//runsc/...`):
- No cgo — pure Go binary
- Additional allowed deps: `github.com/google/subcommands`,
  `github.com/opencontainers/runtime-spec`
- `runsc boot` must not run the netpoller goroutine (performance)

### Style

Follows [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments)
and [Effective Go](https://golang.org/doc/effective_go.html). Key
project-specific conventions:

- **Early exit** from loops and functions where possible
- **Mutexes**: name `mu` or `xxxMu`, never exported; place before protected
  fields; comment ordering requirements
- **Function comments**: use `Preconditions:` and `Postconditions:` blocks for
  entry/exit conditions (one condition per bullet `*`)
- **Unused returns**: explicitly ignore with underscores
- **Formatting verbs**: use `%v` for non-builtin types even if they implement
  Stringer; use `%w` only with `fmt.Errorf`
- **Comments** wrapped at 80 columns (2-space tab)
- C++ code (syscall tests) follows Google C++ Style Guide

### Bug IDs

`b/1234` references in TODOs/NOTEs refer to an internal bug tracker — ignore
them.

## Updating Dependencies

```bash
git checkout origin/go
go get <package>
git checkout origin/master
bazel run //:gazelle -- update-repos -from_file=go.mod
```

## Linting (nogo)

Custom static analysis runs automatically as part of builds. Configuration in
`nogo.yaml`. Custom analyzers in `tools/`: checklocks (mutex ordering),
checkescape (escape analysis), checkunsafe, checkaligned, checkconst,
checklinkname.

## Git & PR Guidelines

- **Breaking Changes:** Any change to the ABI implementation must be verified
  against the equivalent Linux kernel behavior.

## RDMA Proxy Development Guidance

The RDMA proxy (`--rdmaproxy` flag) adds InfiniBand verbs support to gVisor.
For full project context, status, and architecture details see `RDMA_STATUS.md`.

### Goal

Working PoC: NCCL all-reduce with GPUDirect RDMA inside gVisor. Correctness
and functionality first, security hardening later.

### Environment

This development machine is an SSH instance with RDMA-capable NICs (mlx5) and
GPUs. You can run real RDMA workloads, trace syscalls, and test gVisor changes
directly on this host.

### Debugging approach

When something fails, **trace first, guess later.** Use `strace`, LD_PRELOAD
sniffer libraries, and gVisor's own debug logs to capture what ioctls/mmaps
the application actually makes. Compare against working host behavior. Consult
the Linux kernel RDMA subsystem source (`drivers/infiniband/`), the rdma-core
userspace library source, and the kernel uAPI headers
(`include/uapi/rdma/ib_user_ioctl_verbs.h`, `rdma/rdma_user_ioctl_cmds.h`)
to understand the protocol. Online documentation for libibverbs and the mlx5
driver is often incomplete — the kernel source is the ground truth.

### Development philosophy

- **Passthrough first.** The default approach for any new ioctl, mmap, or
  device operation is to forward it to the host kernel unchanged. Only
  intercept or rewrite when the sandbox/sentry address space boundary forces
  it (e.g. page mirroring for MR REG). Don't add validation, filtering, or
  emulation unless something is concretely broken without it.

- **Follow nvproxy/tpuproxy patterns.** The RDMA proxy is architecturally
  identical to the existing GPU proxy: dev gofer for host FDs, generic ioctl
  forwarding, `GenericProxyDeviceConfigureMMap` for mmap. When in doubt, look
  at how nvproxy solves the same class of problem.

- **Fix what fails, don't anticipate.** When a new RDMA operation fails, debug
  it from the error. Don't speculatively add handlers for operations that
  haven't been exercised yet. The ioctl interface is self-describing — the
  generic handler covers most cases automatically.

- **Errors from the sentry must be linux errnos.** Never bubble raw Go errors
  through paths that call `kernel.ExtractErrno` — it panics. Always return
  `linuxerr.*` values (e.g. `linuxerr.ENOMEM`, `linuxerr.EINVAL`).

### Build, Deploy & Test Workflow

Full runbook: `RUNBOOK.md`. Job agent docs: `rdma_job_agent/README.md`.

**Two-node setup:** Node A builds, drives rank 0, and curls the agent. Node B
runs the job agent and rank 1. Both nodes need the `runsc-rdma` binary
registered in Docker.

#### 1. Clone and build (node A)

```bash
git clone git@github.com:modal-labs/gvisor.git
cd gvisor
git checkout alessio/development
sudo make copy TARGETS=runsc DESTINATION=/tmp
```

If pushing changes to the branch:

```bash
git config user.name "atoniolo76"
git config user.email "alessio@modal.com"
```

#### 2. Install `runsc-rdma` and register Docker runtime (both nodes)

Kill old processes, remove old binary, install the new one:

```bash
sudo apt-get update && sudo apt-get install -y jq

sudo pkill -f "runsc-rdma" 2>/dev/null || true; sleep 1
sudo rm -f /usr/local/bin/runsc-rdma
sudo cp /tmp/runsc /usr/local/bin/runsc-rdma && sudo chmod +x /usr/local/bin/runsc-rdma
```

Update Docker's `daemon.json` and restart:

```bash
sudo python3 -c "
import json, pathlib
p = pathlib.Path('/etc/docker/daemon.json')
raw = p.read_text().strip() if p.exists() else ''
cfg = json.loads(raw) if raw else {}
cfg.setdefault('runtimes', {})['runsc-rdma'] = {
    'path': '/usr/local/bin/runsc-rdma',
    'runtimeArgs': [
        '--debug', '--debug-log=/tmp/runsc-rdma/logs/',
        '--rdmaproxy', '--nvproxy',
        '--nvproxy-allowed-driver-capabilities=compute,utility,video',
        '--network=host', '--rdma-expected-ipoib=-1',
    ],
}
p.write_text(json.dumps(cfg, indent=2) + '\n')
"
sudo systemctl restart docker && sleep 2
```

Load NVIDIA peermem, pull the PyTorch image:

```bash
sudo modprobe nvidia-peermem
export PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:24.07-py3"
sudo docker pull $PYTORCH_IMAGE
```

#### 3. Environment exports (both nodes)

```bash
export NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11
export NODE_A_IP=<a>
export NODE_B_IP=<b>
export DEVS=$(ls /dev/infiniband/uverbs* | sed 's/^/--device=/' | tr '\n' ' ')
```

#### 4. Give Docker access to all CPUs (Modal workers)

Docker in `system.slice` is limited to 12 CPUs. Move it to
`modal-containers.slice` to get ~100 CPUs:

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/50-modal-slice.conf <<'EOF'
[Service]
Slice=modal-containers.slice
EOF
sudo systemctl daemon-reload && sudo systemctl restart docker
sudo docker info | grep '^ CPUs:'
```

#### 5. Generate NCCL topology XML (node A, one-time per hardware SKU)

Run under **runc** (gVisor hides PCI sysfs so dumps under runsc are wrong):

```bash
sudo docker run --runtime=runc --rm --gpus all $DEVS \
  --ulimit memlock=-1:-1 --shm-size=1g --network=host \
  -e NCCL_DEBUG=WARN -e NCCL_SOCKET_IFNAME=eth0 -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_NET_GDR_LEVEL=3 -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml \
  -v /tmp:/tmp \
  "nvcr.io/nvidia/pytorch:24.07-py3" torchrun --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29599 \
  /tmp/torch_allreduce_bench.py
```

#### 6. Start the job agent (node B only)

Export `NCCL_IB_HCA` on B before starting so torch job defaults pick it up.
The agent runs `sudo docker` — treat it as root-equivalent.

```bash
cd ~/gvisor/rdma_job_agent
python3 agent.py --host 0.0.0.0 --port 8756
```

#### 7. Run multi-node PyTorch test

From **node A**, verify the agent, push topology to B, then run the test:

```bash
# Health check
curl http://${NODE_B_IP}:8756/health
# Expect: {"ok": true}

# Copy nccl_topo.xml to B
curl -sS -X POST --data-binary @/tmp/nccl_topo.xml \
  "http://${NODE_B_IP}:8756/v1/nccl_topo"

# POST rank 1 to B's agent
POST_BODY="$(jq -n --arg ma "$NODE_A_IP" --argjson mp "29541" \
  '{kind:"torch",master_addr:$ma,master_port:$mp}')"
curl -sS -X POST "http://${NODE_B_IP}:8756/v1/jobs" \
  -H 'Content-Type: application/json' -d "$POST_BODY" | jq .

# Run rank 0 on A
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

# Poll rank 1 status on B
curl -sS "http://${NODE_B_IP}:8756/v1/jobs/<uuid-from-POST>" | jq .
```

**One-shot script alternative:** automates the POST → sleep → rank-0 →
poll sequence:

```bash
export NODE_A_IP=... NODE_B_IP=... NCCL_IB_HCA=...
bash ~/gvisor/rdma_job_agent/run_torch_pair_node_a.sh
```

#### Quick troubleshooting

| Symptom | Fix |
|---------|-----|
| `unknown or invalid runtime name: runsc-rdma` | Step 2 incomplete on the failing node — install binary + daemon.json + restart Docker. |
| `GET /v1/jobs/null` or empty `job_id` | `NODE_A_IP` was unset when building `POST_BODY`. |
| `Connection refused` to `NODE_B_IP:8756` | Agent not running, or bound to `127.0.0.1` — use `0.0.0.0` or SSH `-L`. |

### Docker daemon.json

Docker runtime config is mirrored in `./daemon_json_mirror.json` for visibility.
If runtime args change (e.g. network=host), update both `/etc/docker/daemon.json`
and the mirror file.
