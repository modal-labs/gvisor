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

## Steps

### 1. Verify connectivity
SSH to both nodes in parallel:
```
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 modal@<node> 'echo ok'
```
If the primary hostname fails, try `<node>-1`. Clean stale host keys with
`ssh-keygen -R <node>` if you get HOST IDENTIFICATION CHANGED warnings.

### 2. Verify InfiniBand
```
ssh modal@<node> 'ls /dev/infiniband/uverbs* && ibstat | grep -E "CA |State"'
```
**IMPORTANT:** If `/dev/infiniband/` does not exist, STOP and tell the user.
These nodes lack InfiniBand hardware and cannot run RDMA tests.

### 3. Check hardware
```
ssh modal@<node> 'nvidia-smi -L && echo "---" && python3 --version'
```
Report GPU type (H200, B200, A100) and count. A100 nodes have no IB — reject them.

### 4. Clone gvisor repo
```
ssh modal@<node> 'cd ~ && if [ -d gvisor ]; then cd gvisor && git remote set-url origin https://github.com/modal-labs/gvisor.git && git pull; else git clone https://github.com/modal-labs/gvisor.git && cd gvisor && git checkout alessio/development; fi'
```
Run on both nodes in parallel.

### 5. Install PyTorch
pip3 may not be preinstalled on Modal workers:
```
ssh modal@<node> 'export PATH="$HOME/.local/bin:$PATH" && pip3 --version 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - && pip3 install torch torchvision 2>&1 | tail -5 && python3 -c "import torch; print(torch.__version__, torch.cuda.device_count(), \"GPUs\")"'
```
Run on both nodes in parallel.

### 6. Load nvidia_peermem
```
ssh modal@<node> 'sudo modprobe nvidia_peermem 2>&1; echo "exit: $?"'
```
May fail with EINVAL if already loaded — that's fine.

### 7. Discover network and IB
```
# Find the right network interface
ssh modal@<node> 'ip -4 addr show eth0 2>/dev/null || ip -4 addr show ens7 2>/dev/null'

# Discover active HCAs
ssh modal@<node> 'for dev in /sys/class/infiniband/mlx5_*; do d=$(basename $dev); state=$(cat $dev/ports/1/state); echo "$d: $state"; done'
```
Get node 0's IP as MASTER_ADDR. List active HCAs (state "4: ACTIVE").
Report which HCAs are DOWN (e.g. mlx5_2, mlx5_8 — expected on some hardware).

### 8. Verify cross-node connectivity
```
ssh modal@<node0> 'ping -c 2 -W 2 <NODE1_IP>'
```

### 9. Report summary
Report for each node:
- GPU type and count
- PyTorch version
- IP address and interface name (eth0 or ens7)
- Active HCA count and list
- nvidia_peermem status
