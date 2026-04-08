#!/bin/bash
# NCCL Flag Sweep - B200 2-node + Port Counters
# Usage: ./nccl_sweep.sh <node0_ssh> <node1_ssh> <master_ip>
# Example: ./nccl_sweep.sh modal@wo-ob88 modal@wo-nmaq 172.27.58.23

NODE0=${1:?Usage: $0 <node0_ssh> <node1_ssh> <master_ip>}
NODE1=${2:?}
MASTER=${3:?}
PORT=29550

cleanup() {
    ssh $NODE0 "pkill -f torchrun 2>/dev/null" &>/dev/null &
    ssh $NODE1 "pkill -f torchrun 2>/dev/null" &>/dev/null &
    wait; sleep 5
}

run_test() {
    local flags="$1"
    PORT=$((PORT + 1))
    cleanup

    ssh $NODE1 "sudo prlimit --pid=\$\$ --memlock=unlimited:unlimited && export PATH=\"\$HOME/.local/bin:\$PATH\" && cd ~/gvisor && $flags OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --nnodes=2 --master_addr=$MASTER --master_port=$PORT --node_rank=1 ./allreduce_bench.py" &>/tmp/node1_sweep.log &
    sleep 5
    result=$(ssh $NODE0 "sudo prlimit --pid=\$\$ --memlock=unlimited:unlimited && export PATH=\"\$HOME/.local/bin:\$PATH\" && cd ~/gvisor && $flags OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --nnodes=2 --master_addr=$MASTER --master_port=$PORT --node_rank=0 ./allreduce_bench.py" 2>&1)
    wait 2>/dev/null

    algbw_gbs=$(echo "$result" | grep "algbw:" | awk '{print $2}')
    algbw_gbps=$(echo "$result" | grep "algbw:" | sed 's/.*(\(.*\) Gbps).*/\1/')
    busbw_gbs=$(echo "$result" | grep "busbw:" | awk '{print $2}')
    busbw_gbps=$(echo "$result" | grep "busbw:" | sed 's/.*(\(.*\) Gbps).*/\1/')

    if [ -z "$busbw_gbs" ]; then
        echo "FAILED,FAILED,FAILED,FAILED"
    else
        echo "$algbw_gbs,$algbw_gbps,$busbw_gbs,$busbw_gbps"
    fi
}

OUTFILE="nccl_sweep_results.csv"
echo "test,algbw_GBps,algbw_Gbps,busbw_GBps,busbw_Gbps" > $OUTFILE

tests=(
    "1_baseline|NCCL_SOCKET_IFNAME=ens7 GLOO_SOCKET_IFNAME=ens7 NCCL_IB_HCA=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12"
    "2_topo_hca|NCCL_TOPO_FILE=/home/modal/gvisor/nccl_topo.xml NCCL_IB_HCA=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12 NCCL_SOCKET_IFNAME=ens7 GLOO_SOCKET_IFNAME=ens7"
    "3_graph_merge|NCCL_TOPO_FILE=/home/modal/gvisor/nccl_topo.xml NCCL_GRAPH_FILE=/home/modal/gvisor/nccl_graph.xml NCCL_NET_MERGE_LEVEL=LOC NCCL_IB_HCA=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12 NCCL_SOCKET_IFNAME=ens7 GLOO_SOCKET_IFNAME=ens7"
    "4_gdr|NCCL_TOPO_FILE=/home/modal/gvisor/nccl_topo.xml NCCL_GRAPH_FILE=/home/modal/gvisor/nccl_graph.xml NCCL_NET_MERGE_LEVEL=LOC NCCL_NET_GDR_LEVEL=PHB NCCL_IB_HCA=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12 NCCL_SOCKET_IFNAME=ens7 GLOO_SOCKET_IFNAME=ens7"
    "5_ring|NCCL_ALGO=Ring NCCL_TOPO_FILE=/home/modal/gvisor/nccl_topo.xml NCCL_GRAPH_FILE=/home/modal/gvisor/nccl_graph.xml NCCL_NET_MERGE_LEVEL=LOC NCCL_NET_GDR_LEVEL=PHB NCCL_IB_HCA=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12 NCCL_SOCKET_IFNAME=ens7 GLOO_SOCKET_IFNAME=ens7"
)

for t in "${tests[@]}"; do
    name="${t%%|*}"
    flags="${t##*|}"
    echo -n "Running $name... "
    r=$(run_test "$flags")
    echo "$name,$r" >> $OUTFILE
    echo "$r"
done

echo ""
echo "Results saved to $OUTFILE"
cat $OUTFILE
