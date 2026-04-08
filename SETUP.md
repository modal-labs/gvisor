Begin by cloning the repo, checking out the `alessio/development` branch, and building gVisor.

```bash
git clone git@github.com:modal-labs/gvisor.git
cd gvisor
git checkout alessio/development
```

Load nvidia_peermem for GPUDirect RDMA and raise the locked-memory limit

```bash
sudo modprobe nvidia_peermem
sudo prlimit --pid=$$ --memlock=unlimited:unlimited
```

Install Torch

```bash
sudo apt install python3-pip
pip3 install torch torchvision
export PATH="$HOME/.local/bin:$PATH"
```

On Node A only

```bash
export MASTER_ADDR=$(ip -4 addr show | grep -e "ens7" -e "eth0" | grep inet | awk '{ print $2 }' | awk -F'/' '{ print $1}')
echo "export MASTER_ADDR=$MASTER_ADDR"
```

**Copy and paste this command on Node B.**

Export the correct NCCL IB devices used.

```bash
export NCCL_IB_HCA=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12
```

Run the test. Need to set memory unlimited in the current shell.
```
sudo prlimit --pid=$$ --memlock=unlimited:unlimited && 
NCCL_NET_GDR_LEVEL=PHB \
NCCL_NET_MERGE_LEVEL=LOC \
NCCL_GRAPH_FILE=/home/modal/gvisor/nccl_graph.xml \
NCCL_TOPO_FILE=/home/modal/gvisor/nccl_topo.xml \
NCCL_SOCKET_IFNAME=ens7 \
GLOO_SOCKET_IFNAME=ens7 \
NCCL_IB_HCA=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12 \
OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=8 --nnodes=2 \
--master_addr="$MASTER_ADDR" --master_port=29500 \
--node_rank=0 ./torch_mnist_train.py
```

-- gVisor testing --
Build gVisor binary

```bash
sudo make copy TARGETS=runsc DESTINATION=/usr/local/bin/runsc
sudo chmod +x /usr/local/bin/runsc
```

