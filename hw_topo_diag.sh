#!/bin/bash
# Hardware topology diagnostic for NCCL multi-rail analysis
# Run on both B200 and H100 nodes, diff the output

echo "============================================"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "============================================"

echo ""
echo "=== 1. GPU INFO ==="
nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv 2>/dev/null

echo ""
echo "=== 2. NVIDIA DRIVER ==="
cat /proc/driver/nvidia/version 2>/dev/null | head -1

echo ""
echo "=== 3. NCCL VERSION ==="
python3 -c "import torch; print('torch:', torch.__version__); print('nccl:', torch.cuda.nccl.version()); print('cuda:', torch.version.cuda)" 2>/dev/null

echo ""
echo "=== 4. FULL PCI DEVICE LIST ==="
lspci -D 2>/dev/null

echo ""
echo "=== 5. PCI TREE (hierarchy) ==="
lspci -tv 2>/dev/null

echo ""
echo "=== 6. PCI LINK CAPS (GPUs + NICs) ==="
for dev in $(lspci -D 2>/dev/null | grep -E "3D controller|Mellanox|NVIDIA.*Bridge" | awk '{print $1}'); do
    echo "$dev $(lspci -s $dev 2>/dev/null | cut -d: -f3-)"
    lspci -vvs $dev 2>/dev/null | grep -E "LnkCap|LnkSta|LnkCtl" | head -4
    echo ""
done

echo ""
echo "=== 7. GPU TOPOLOGY MATRIX ==="
nvidia-smi topo -m 2>/dev/null

echo ""
echo "=== 8. NVLINK STATUS ==="
nvidia-smi nvlink -s 2>/dev/null

echo ""
echo "=== 9. NVLINK REMOTE PEER BUS IDS (GPU0) ==="
for link in $(seq 0 17); do
    nvidia-smi nvlink -i 0 -l $link -p 2>/dev/null | grep "Link $link"
done

echo ""
echo "=== 10. NVSWITCH CHECK ==="
lspci -D 2>/dev/null | grep -i -E "bridge.*nvidia|nvswitch"
ls /dev/nvidia-nvswitch* 2>/dev/null || echo "No NVSwitch device files"
nvidia-smi nvswitch -ls 2>/dev/null || echo "No nvswitch subcommand or no switches"

echo ""
echo "=== 11. IB DEVICES ==="
ibstat 2>/dev/null | grep -E "CA |CA type|Port |State|Rate|Base lid"

echo ""
echo "=== 12. IB DEVICE TO NETDEV MAPPING ==="
ibdev2netdev 2>/dev/null

echo ""
echo "=== 13. IB DEVICE PCI BUS IDS ==="
for dev in $(ls /sys/class/infiniband/ 2>/dev/null); do
    echo "$dev: $(cat /sys/class/infiniband/$dev/device/uevent 2>/dev/null | grep PCI_SLOT)"
done

echo ""
echo "=== 14. PEERMEM / GDR STATUS ==="
cat /sys/module/nvidia_peermem/initstate 2>/dev/null || echo "nvidia_peermem NOT loaded"
lsmod | grep -E "peer|gdr" 2>/dev/null

echo ""
echo "=== 15. MEMLOCK LIMIT ==="
ulimit -l

echo ""
echo "=== 16. NUMA TOPOLOGY ==="
numactl -H 2>/dev/null || lscpu 2>/dev/null | grep -E "NUMA|Socket|Thread|Core"

echo ""
echo "=== 17. KERNEL / CMDLINE ==="
uname -r
cat /proc/cmdline

echo ""
echo "=== 18. SYSFS LINK WIDTH (PCIe config space) ==="
for dev in $(lspci -D 2>/dev/null | grep -E "3D controller|Mellanox" | awk '{print $1}'); do
    busid=$(echo $dev | tr ':.' '/')
    echo -n "$dev: width="
    cat /sys/bus/pci/devices/$dev/current_link_width 2>/dev/null || echo -n "n/a"
    echo -n " speed="
    cat /sys/bus/pci/devices/$dev/current_link_speed 2>/dev/null || echo "n/a"
done

echo ""
echo "=== 19. NCCL TOPO FILE CHECK ==="
ls -la /var/run/nvidia-topologyd/virtualTopology.xml 2>/dev/null || echo "No topology file"

echo ""
echo "=== 20. NCCL AUTO-DISCOVERED TOPOLOGY ==="
echo "(run separately: NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml torchrun --nproc_per_node=8 --nnodes=1 ...)"

echo ""
echo "============================================"
echo "DONE"
echo "============================================"
