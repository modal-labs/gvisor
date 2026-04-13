---
name: node-setup
description: Set up remote GPU nodes for RDMA testing (clone repo, install PyTorch, discover IB)
user-invocable: true
allowed-tools: Bash
argument-hint: [node0-host node1-host]
---

# Node Setup

Provision two remote GPU nodes for multi-node RDMA testing.

## Arguments

$ARGUMENTS should be two SSH hostnames, e.g.: `wo-abc123 wo-def456`
- If no arguments given, ask the user for the node hostnames.
- SSH user is `modal`
- Try primary hostname first, then append `-1` as fallback.

## SSH convention

All SSH commands in this skill (and all other RDMA skills) MUST use:
```
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 modal@<host> '...' 2>&1 | grep -v "^Warning"
```
- `UserKnownHostsFile=/dev/null` prevents stale host key errors (Modal recycles nodes).
- `grep -v "^Warning"` suppresses the `Warning: Permanently added...` line that
  `/dev/null` known hosts produces. Without this, every SSH call adds noise.

## Steps

### 1. Verify connectivity
SSH to both nodes in parallel. If the primary hostname fails (timeout or
connection refused), retry with `<hostname>-1` appended.

**Once a working hostname is found for each node, use that hostname for ALL
subsequent steps.** Do not re-resolve on every step.

### 2. Verify InfiniBand
```
ssh ... modal@<node> 'ls /dev/infiniband/uverbs* && ibstat | grep -E "CA |State"'
```
**IMPORTANT:** If `/dev/infiniband/` does not exist, STOP and tell the user.
These nodes lack InfiniBand hardware and cannot run RDMA tests.

### 3. Check hardware
```
ssh ... modal@<node> 'nvidia-smi -L && echo "---" && python3 --version'
```
Report GPU type (H200, B200, A100) and count. A100 nodes have no IB — reject them.

### 4. Clone gvisor repo
```
ssh ... modal@<node> 'cd ~ && if [ -d gvisor ]; then cd gvisor && git remote set-url origin https://github.com/modal-labs/gvisor.git && git pull; else git clone https://github.com/modal-labs/gvisor.git && cd gvisor && git checkout alessio/development; fi'
```
Run on both nodes in parallel.

### 5. Install PyTorch
pip3 may not be preinstalled on Modal workers. Run on both nodes in parallel.

The install produces hundreds of lines of output. Pipe the install to `tail -3`,
then run the verification separately so it doesn't get lost:
```
ssh ... modal@<node> 'export PATH="$HOME/.local/bin:$PATH" && pip3 --version 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - 2>&1 | tail -2 && pip3 install torch torchvision 2>&1 | tail -3'
```
Then verify:
```
ssh ... modal@<node> 'export PATH="$HOME/.local/bin:$PATH" && python3 -c "import torch; print(torch.__version__, torch.cuda.device_count(), \"GPUs\")" && which torchrun'
```

### 6. Load nvidia_peermem and expand Docker cpuset
```
ssh ... modal@<node> 'sudo modprobe nvidia_peermem 2>&1; echo "exit: $?"'
```
May fail with EINVAL if already loaded — that's fine. Run on both in parallel.

Then expand the Docker cgroup cpuset so containers can use all CPUs.
Modal workers restrict `system.slice` to a small CPU set (e.g. 0-11), which
starves NCCL proxy threads and causes ~4-5x bandwidth regression in containers.
```
ssh ... modal@<node> 'sudo bash -c "echo 0-\$(( \$(nproc) - 1 )) > /sys/fs/cgroup/system.slice/cpuset.cpus" && cat /sys/fs/cgroup/system.slice/docker.service/cpuset.cpus.effective'
```
Run on both nodes in parallel. The output should match `nproc` (e.g. `0-111`).

### 7. Discover network and IB
```
ssh ... modal@<node> 'echo IFNAME=$(ip -4 addr show ens7 &>/dev/null && echo ens7 || echo eth0) && echo MASTER_IP=$(ip -4 addr show eth0 2>/dev/null | grep inet | awk "{print \$2}" | cut -d/ -f1 || ip -4 addr show ens7 | grep inet | awk "{print \$2}" | cut -d/ -f1) && echo NCCL_IB_HCA=$(for dev in /sys/class/infiniband/mlx5_*; do d=$(basename $dev); grep -q "4: ACTIVE" $dev/ports/1/state 2>/dev/null && echo -n "$d,"; done | sed "s/,$//") && for dev in /sys/class/infiniband/mlx5_*; do d=$(basename $dev); state=$(cat $dev/ports/1/state); echo "$d: $state"; done'
```
Run on node 0. Parse IFNAME, MASTER_IP, NCCL_IB_HCA from the output.
Report which HCAs are DOWN (e.g. mlx5_2, mlx5_8 — expected on some hardware).

### 8. Verify cross-node connectivity
```
ssh ... modal@<node0> 'ping -c 2 -W 2 <NODE1_IP>'
```

### 9. Report summary
Report for each node:
- GPU type and count
- PyTorch version
- IP address and interface name (eth0 or ens7)
- Active HCA count and list
- nvidia_peermem status
- Docker cpuset (should show all CPUs, e.g. `0-111`)
- Working SSH hostname (may differ from the one provided if `-1` fallback was used)
