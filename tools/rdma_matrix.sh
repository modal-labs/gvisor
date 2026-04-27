#!/usr/bin/env bash

set -euo pipefail

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"

RUN_DIR="$(pwd -P)"

CONTAINER_NAME="${CONTAINER_NAME:-nccl-test}"
ROCE="${ROCE:-0}"
RDMA_MOVE_NETDEVS="${RDMA_MOVE_NETDEVS:-${ROCE}}"
USE_NETNS_HOLDER_EXPLICIT="${USE_NETNS_HOLDER+x}"
USE_NETNS_HOLDER="${USE_NETNS_HOLDER:-${RDMA_MOVE_NETDEVS}}"
BOOTSTRAP_MODE="${BOOTSTRAP_MODE:-gre}"
NETNS_HOLDER_NAME="${NETNS_HOLDER_NAME:-${CONTAINER_NAME}-netns}"
NETNS_HOLDER_RUNTIME="${NETNS_HOLDER_RUNTIME:-runc}"
NETWORK_NAME="${NETWORK_NAME:-gre-net}"
BRIDGE_NAME="${BRIDGE_NAME:-br-gre}"
GRETAP_NAME="${GRETAP_NAME:-gretap1}"
IMAGE="${IMAGE:-atoniolo76/torch-ib-slim:latest}"
RUNTIME="${RUNTIME:-runsc-rdma}"
RUNSC_PATH="${RUNSC_PATH:-/usr/local/bin/runsc-rdma}"
DOCKER_DAEMON_JSON="${DOCKER_DAEMON_JSON:-/etc/docker/daemon.json}"
BOOTSTRAP_IFNAME_EXPLICIT="${BOOTSTRAP_IFNAME+x}"
BOOTSTRAP_IFNAME="${BOOTSTRAP_IFNAME:-eth0}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDMA_SNAPSHOT_PATH="${RDMA_SNAPSHOT_PATH:-/etc/rdma-snapshot.txt}"
RDMA_ROUTES_PATH="${RDMA_ROUTES_PATH:-${RDMA_SNAPSHOT_PATH}.routes}"
WORKSPACE_MOUNT="${WORKSPACE_MOUNT:-$RUN_DIR}"
TORCH_SCRIPT="${TORCH_SCRIPT:-./torch_mnist_train.py}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-2}"
SUBNET="${SUBNET:-10.89.0.0/24}"

NODE_A_BR_IP="${NODE_A_BR_IP:-10.89.0.1/24}"
NODE_A_GW="${NODE_A_GW:-10.89.0.1}"
NODE_A_IP_RANGE="${NODE_A_IP_RANGE:-10.89.0.64/26}"
NODE_A_CONTAINER_IP="${NODE_A_CONTAINER_IP:-10.89.0.64}"

NODE_B_BR_IP="${NODE_B_BR_IP:-10.89.0.2/24}"
NODE_B_GW="${NODE_B_GW:-10.89.0.2}"
NODE_B_IP_RANGE="${NODE_B_IP_RANGE:-10.89.0.128/26}"
NODE_B_CONTAINER_IP="${NODE_B_CONTAINER_IP:-10.89.0.128}"

TORCH_MASTER_ADDR_EXPLICIT="${TORCH_MASTER_ADDR+x}"
TORCH_MASTER_ADDR="${TORCH_MASTER_ADDR:-$NODE_A_CONTAINER_IP}"
REQUIRE_RDMA_IPV4="${REQUIRE_RDMA_IPV4:-1}"
ALLOW_BOOTSTRAP_RDMA_IFACE="${ALLOW_BOOTSTRAP_RDMA_IFACE:-0}"
RESTORE_RDMA_ROUTES="${RESTORE_RDMA_ROUTES:-1}"
RDMA_ROUTE_PREFIX_LEN="${RDMA_ROUTE_PREFIX_LEN:-24}"
REQUIRE_SANDBOX_RDMA_GIDS="${REQUIRE_SANDBOX_RDMA_GIDS:-1}"
REQUIRE_SANDBOX_RDMA_NDEVS="${REQUIRE_SANDBOX_RDMA_NDEVS:-auto}"

RUNSC_RDMA_PROXY="${RUNSC_RDMA_PROXY:-1}"
RUNSC_NVPROXY="${RUNSC_NVPROXY:-1}"
RUNSC_DEBUG="${RUNSC_DEBUG:-0}"
RUNSC_DEBUG_LOG="${RUNSC_DEBUG_LOG:-/tmp/runsc-rdma/logs/}"
RUNSC_DEBUG_COMMAND="${RUNSC_DEBUG_COMMAND:-}"
RUNSC_STRACE="${RUNSC_STRACE:-0}"
RUNSC_STRACE_SYSCALLS="${RUNSC_STRACE_SYSCALLS:-}"
RUNSC_STRACE_LOG_SIZE="${RUNSC_STRACE_LOG_SIZE:-}"
RUNSC_EXTRA_ANNOTATIONS="${RUNSC_EXTRA_ANNOTATIONS:-}"
RUNSC_NVPROXY_ALLOWED_DRIVER_CAPABILITIES="${RUNSC_NVPROXY_ALLOWED_DRIVER_CAPABILITIES:-compute,utility,video}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--roce|--no-roce] [--host-network|--gre] <subcommand> [args]

Per-node helper for the TCP bootstrap + RDMA test harness.
Assumes LOCAL_IP and REMOTE_IP are exported on each node for GRE or host-network mode.

Subcommands:
  preflight A|B             Validate Docker/runtime/RDMA candidates before setup.
  runtime-show              Show Docker's configured ${RUNTIME} runtime.
  runtime-install           Install/update ${RUNTIME} with annotation-friendly base args.
  setup-net A|B             Create the GRE bridge/docker network for node A or B.
  cleanup-net               Remove the GRE bridge/docker network on this node.
  inspect-rdma              Show RDMA HCA link layers and RoCE netdev candidates.
  snapshot-rdma             Snapshot RoCE netdev IPv4/MTU to ${RDMA_SNAPSHOT_PATH}.
  start-container A|B       Start the detached test container for node A or B.
  start-app-container A|B   Start only the runsc app container (holder mode).
  move-rdma                 Move the snapped RDMA netdevs into the container netns.
  restore-rdma-config       Reapply RDMA IP addresses and routes in netns.
  fix-rdma-routes           Reinstall RDMA routes in an already prepared container.
  validate-container        Validate RDMA sysfs and, with ROCE=1, moved netdevs.
  ping-peer A|B             Ping the opposite container over the GRE bootstrap net.
  wait-master A|B           Wait for TORCH_MASTER_ADDR:MASTER_PORT from the container.
  diagnose A|B              Dump bootstrap network, workspace, and master reachability.
  prepare A|B               Run preflight, setup-net, start, and validate.
  run-test A|B              Run torchrun inside the detached container as node A or B.
  kill-test                 Stop torchrun/python test processes inside the container.
  counters                  Print per-HCA port counters in GB.
  logs                      Show the latest runsc boot log and grep RDMA markers.
  cleanup-container         Remove the test container on this node.
  shell                     Open an interactive shell in the detached container.

Environment:
  Required for network setup:
    LOCAL_IP, REMOTE_IP      Underlay IPs used by gretap between the two hosts.

  Common optional overrides:
    DEVS                    Space-separated docker device args. If unset, uverbs* is autodetected.
    NCCL_IB_HCA             Passed through to NCCL and used as an RDMA HCA filter.
    RDMA_IB_DEVS            Explicit comma/space-separated HCA filter for snapshot-rdma.
    RDMA_IFACES             Explicit space-separated RoCE netdevs; bypasses sysfs autodetection.
    ROCE                    Set to 1 to enable RoCE netdev snapshot/move/route setup (default: ${ROCE})
    RDMA_MOVE_NETDEVS       Low-level override for ROCE netdev movement (default: ${RDMA_MOVE_NETDEVS})
    BOOTSTRAP_MODE          gre or host. host skips br-gre/gretap and uses --network=host (default: ${BOOTSTRAP_MODE})
    IMAGE                   Container image (default: ${IMAGE})
    RUNTIME                 Docker runtime (default: ${RUNTIME})
    USE_NETNS_HOLDER        Start app with --network=container:<holder> so runsc
                            snapshots RDMA sysfs after RoCE netdevs move (default: ${USE_NETNS_HOLDER})
    NETNS_HOLDER_NAME       Holder container name (default: ${NETNS_HOLDER_NAME})
    NETNS_HOLDER_RUNTIME    Holder runtime (default: ${NETNS_HOLDER_RUNTIME})
    RUNSC_PATH              runsc binary path for runtime-install (default: ${RUNSC_PATH})
    RUNSC_RDMA_PROXY        Set rdmaproxy via runsc flag annotation (default: ${RUNSC_RDMA_PROXY})
    RUNSC_NVPROXY           Set nvproxy via runsc flag annotation (default: ${RUNSC_NVPROXY})
    RUNSC_DEBUG             Set dev.gvisor.flag.debug annotation (default: ${RUNSC_DEBUG})
    RUNSC_DEBUG_LOG         Debug log dir when RUNSC_DEBUG=1 (default: ${RUNSC_DEBUG_LOG})
    RUNSC_STRACE            Add dev.gvisor.flag.strace=true annotation (default: ${RUNSC_STRACE})
    RUNSC_STRACE_SYSCALLS   Optional comma-separated strace allowlist.
    RUNSC_EXTRA_ANNOTATIONS Space-separated key=value OCI annotations.
    BOOTSTRAP_IFNAME        Container bootstrap NIC for NCCL/Gloo (default: ${BOOTSTRAP_IFNAME})
    REQUIRE_RDMA_IPV4       Require every moved RoCE netdev to have IPv4 (default: ${REQUIRE_RDMA_IPV4})
    RESTORE_RDMA_ROUTES     Restore/infer connected RDMA routes after moving links (default: ${RESTORE_RDMA_ROUTES})
    RDMA_ROUTE_PREFIX_LEN   Prefix to infer for /32 RDMA IPs if no route snapshot exists (default: ${RDMA_ROUTE_PREFIX_LEN})
    ALLOW_BOOTSTRAP_RDMA_IFACE
                            Allow moving the host iface used for LOCAL_IP/REMOTE_IP (default: 0)
    NCCL_DEBUG, NCCL_DEBUG_SUBSYS, TORCH_DISTRIBUTED_DEBUG, NCCL_IB_GID_INDEX

Examples:
  # Node A
  $(basename "$0") prepare A
  $(basename "$0") run-test A

  # Node B
  $(basename "$0") prepare B
  $(basename "$0") run-test B

  # RoCE nodes only: opt into netdev movement.
  $(basename "$0") --roce prepare A

  # Native InfiniBand only: use host networking for torch/NCCL bootstrap.
  $(basename "$0") --host-network prepare A

Run run-test on node B first so it waits for node A's master.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

warn() {
  echo "warning: $*" >&2
}

require_env() {
  local missing=()
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      missing+=("$name")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    die "missing required environment variable(s): ${missing[*]}"
  fi
}

require_cmd() {
  local missing=()
  local name
  for name in "$@"; do
    if ! command -v "$name" >/dev/null 2>&1; then
      missing+=("$name")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    die "missing required command(s): ${missing[*]}"
  fi
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

falsey() {
  case "${1:-}" in
    0|false|FALSE|no|NO|off|OFF)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

rdma_move_netdevs_enabled() {
  truthy "${RDMA_MOVE_NETDEVS}"
}

bootstrap_host_mode() {
  [[ "${BOOTSTRAP_MODE}" == "host" || "${BOOTSTRAP_MODE}" == "host-network" ]]
}

bootstrap_gre_mode() {
  [[ "${BOOTSTRAP_MODE}" == "gre" ]]
}

validate_bootstrap_mode() {
  if bootstrap_host_mode; then
    BOOTSTRAP_MODE="host"
    return 0
  fi
  if bootstrap_gre_mode; then
    return 0
  fi
  die "BOOTSTRAP_MODE must be 'gre' or 'host', got ${BOOTSTRAP_MODE}"
}

apply_roce_mode_defaults() {
  if [[ -z "${USE_NETNS_HOLDER_EXPLICIT}" ]]; then
    if rdma_move_netdevs_enabled; then
      USE_NETNS_HOLDER=1
    else
      USE_NETNS_HOLDER=0
    fi
  fi
}

runtime_show() {
  require_cmd sudo python3
  sudo env DOCKER_DAEMON_JSON="${DOCKER_DAEMON_JSON}" RUNTIME="${RUNTIME}" python3 - <<'PY'
import json
import os
from pathlib import Path

p = Path(os.environ["DOCKER_DAEMON_JSON"])
runtime = os.environ["RUNTIME"]
if not p.exists():
    print(f"{p} does not exist")
    raise SystemExit(1)
cfg = json.loads(p.read_text() or "{}")
r = cfg.get("runtimes", {}).get(runtime)
if not r:
    print(f"runtime {runtime!r} is not configured in {p}")
    raise SystemExit(1)
print(json.dumps(r, indent=2))
PY
}

runtime_install() {
  require_cmd sudo python3 systemctl

  sudo env \
    DOCKER_DAEMON_JSON="${DOCKER_DAEMON_JSON}" \
    RUNTIME="${RUNTIME}" \
    RUNSC_PATH="${RUNSC_PATH}" \
    RUNSC_NVPROXY_ALLOWED_DRIVER_CAPABILITIES="${RUNSC_NVPROXY_ALLOWED_DRIVER_CAPABILITIES}" \
    python3 - <<'PY'
import json
import os
from pathlib import Path

p = Path(os.environ["DOCKER_DAEMON_JSON"])
raw = p.read_text().strip() if p.exists() else ""
cfg = json.loads(raw) if raw else {}
args = [
    "--allow-flag-override",
    "--nvproxy-allowed-driver-capabilities=" + os.environ["RUNSC_NVPROXY_ALLOWED_DRIVER_CAPABILITIES"],
]
cfg.setdefault("runtimes", {})[os.environ["RUNTIME"]] = {
    "path": os.environ["RUNSC_PATH"],
    "runtimeArgs": args,
}
p.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"configured {os.environ['RUNTIME']} in {p}")
print(json.dumps(cfg["runtimes"][os.environ["RUNTIME"]], indent=2))
PY
  sudo systemctl restart docker
  sleep 2
}

set_role() {
  local role="${1:-}"
  role="${role^^}"
  case "$role" in
    A)
      BR_IP="${NODE_A_BR_IP}"
      GW="${NODE_A_GW}"
      IP_RANGE="${NODE_A_IP_RANGE}"
      CONTAINER_IP="${NODE_A_CONTAINER_IP}"
      PEER_CONTAINER_IP="${NODE_B_CONTAINER_IP}"
      MASTER_ADDR="${TORCH_MASTER_ADDR}"
      NODE_RANK="0"
      ROLE="A"
      ;;
    B)
      BR_IP="${NODE_B_BR_IP}"
      GW="${NODE_B_GW}"
      IP_RANGE="${NODE_B_IP_RANGE}"
      CONTAINER_IP="${NODE_B_CONTAINER_IP}"
      PEER_CONTAINER_IP="${NODE_A_CONTAINER_IP}"
      MASTER_ADDR="${TORCH_MASTER_ADDR}"
      NODE_RANK="1"
      ROLE="B"
      ;;
    *)
      die "role must be A or B"
      ;;
  esac

  if bootstrap_host_mode; then
    CONTAINER_IP="${LOCAL_IP:-host}"
    PEER_CONTAINER_IP="${REMOTE_IP:-host}"
    if [[ -z "${TORCH_MASTER_ADDR_EXPLICIT}" ]]; then
      if [[ "${ROLE}" == "A" ]]; then
        MASTER_ADDR="${LOCAL_IP:-}"
      else
        MASTER_ADDR="${REMOTE_IP:-}"
      fi
    fi
    [[ -n "${MASTER_ADDR}" ]] || die "host bootstrap mode requires TORCH_MASTER_ADDR or LOCAL_IP/REMOTE_IP"
  fi
}

read_sysfs() {
  local path="$1"
  local val=""
  if [[ -r "${path}" ]]; then
    IFS= read -r val <"${path}" 2>/dev/null || true
  fi
  printf '%s' "${val}"
}

docker_rm_container() {
  sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  sudo docker rm -f "${NETNS_HOLDER_NAME}" >/dev/null 2>&1 || true
}

container_pid() {
  local pid
  pid="$(sudo docker inspect -f '{{.State.Pid}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
  [[ -n "${pid}" && "${pid}" != "0" ]] || die "failed to determine PID for ${CONTAINER_NAME}"
  printf '%s\n' "${pid}"
}

container_running() {
  local name="$1"
  local state
  state="$(sudo docker inspect -f '{{.State.Running}}' "${name}" 2>/dev/null || true)"
  [[ "${state}" == "true" ]]
}

netns_pid() {
  local name="${CONTAINER_NAME}"
  if truthy "${USE_NETNS_HOLDER}" && container_running "${NETNS_HOLDER_NAME}"; then
    name="${NETNS_HOLDER_NAME}"
  fi

  local pid
  pid="$(sudo docker inspect -f '{{.State.Pid}}' "${name}" 2>/dev/null || true)"
  [[ -n "${pid}" && "${pid}" != "0" ]] || die "failed to determine netns PID for ${name}"
  printf '%s\n' "${pid}"
}

cleanup_net() {
  sudo ip link del "${GRETAP_NAME}" 2>/dev/null || true
  sudo ip link del "${BRIDGE_NAME}" 2>/dev/null || true
  sudo docker network rm "${NETWORK_NAME}" >/dev/null 2>&1 || true
}

setup_net() {
  require_cmd sudo docker ip
  require_env LOCAL_IP REMOTE_IP
  set_role "${1:-}"

  if bootstrap_host_mode; then
    cat <<EOF
bootstrap_mode=${BOOTSTRAP_MODE}
role=${ROLE}
host_local_ip=${LOCAL_IP}
host_remote_ip=${REMOTE_IP}
master_addr=${MASTER_ADDR}
network=host
EOF
    return 0
  fi

  cleanup_net

  sudo ip link add "${BRIDGE_NAME}" type bridge
  sudo ip addr add "${BR_IP}" dev "${BRIDGE_NAME}"
  sudo ip link set "${BRIDGE_NAME}" up

  sudo ip link add "${GRETAP_NAME}" type gretap local "${LOCAL_IP}" remote "${REMOTE_IP}" ttl 255
  sudo ip link set "${GRETAP_NAME}" up
  sudo ip link set "${GRETAP_NAME}" master "${BRIDGE_NAME}"

  sudo docker network create \
    --driver bridge \
    --subnet "${SUBNET}" \
    --gateway "${GW}" \
    --ip-range "${IP_RANGE}" \
    -o com.docker.network.bridge.name="${BRIDGE_NAME}" \
    -o com.docker.network.bridge.enable_ip_masquerade=false \
    "${NETWORK_NAME}" >/dev/null

  cat <<EOF
network=${NETWORK_NAME}
role=${ROLE}
bridge=${BRIDGE_NAME}
bridge_ip=${BR_IP}
gateway=${GW}
ip_range=${IP_RANGE}
container_ip=${CONTAINER_IP}
master_addr=${MASTER_ADDR}
EOF
}

hca_selected() {
  local ibdev="$1"
  local filter="${RDMA_IB_DEVS:-${NCCL_IB_HCA:-}}"
  [[ -z "${filter}" ]] && return 0

  local mode="include"
  if [[ "${filter}" == ^* ]]; then
    mode="exclude"
    filter="${filter#^}"
  fi

  filter="${filter//,/ }"
  local token
  local matched=1
  for token in ${filter}; do
    token="${token#=}"
    token="${token%%:*}"
    [[ -z "${token}" ]] && continue
    if [[ "${token}" == "${ibdev}" ]]; then
      matched=0
      break
    fi
  done

  if [[ "${mode}" == "exclude" ]]; then
    (( matched == 0 )) && return 1
    return 0
  fi
  return "${matched}"
}

print_hca_link_layers() {
  local d p
  shopt -s nullglob
  for d in /sys/class/infiniband/*; do
    hca_selected "$(basename "${d}")" || continue
    for p in "${d}"/ports/*; do
      [[ -e "${p}" ]] || continue
      printf '%s port %s link_layer=%s\n' \
        "$(basename "${d}")" "$(basename "${p}")" "$(read_sysfs "${p}/link_layer")"
    done
  done
  shopt -u nullglob
}

native_ib_hcas_csv() {
  local d p ibdev layer
  local -a hcas=()
  shopt -s nullglob
  for d in /sys/class/infiniband/*; do
    ibdev="$(basename "${d}")"
    hca_selected "${ibdev}" || continue
    for p in "${d}"/ports/*; do
      [[ -e "${p}" ]] || continue
      layer="$(read_sysfs "${p}/link_layer")"
      if [[ "${layer}" == "InfiniBand" ]]; then
        hcas+=("${ibdev}")
        break
      fi
    done
  done
  shopt -u nullglob
  (( ${#hcas[@]} == 0 )) && return 0
  local IFS=,
  printf '%s' "${hcas[*]}"
}

derive_nccl_ib_hca() {
  if rdma_move_netdevs_enabled && [[ -f "${RDMA_SNAPSHOT_PATH}" ]]; then
    sudo awk -F'|' '
      $1 !~ /^#/ && $5 != "" && $5 != "?" {
        split($5, devs, ",")
        for (i in devs) {
          if (devs[i] != "" && !seen[devs[i]]++) {
            out = out ? out "," devs[i] : devs[i]
          }
        }
      }
      END { print out }
    ' "${RDMA_SNAPSHOT_PATH}"
    return 0
  fi

  native_ib_hcas_csv
}

rdma_records_from_sysfs() {
  local f ibdev rest port idx ndev gid gid_type
  shopt -s nullglob
  for f in /sys/class/infiniband/*/ports/*/gid_attrs/ndevs/*; do
    ibdev="${f#/sys/class/infiniband/}"
    ibdev="${ibdev%%/*}"
    hca_selected "${ibdev}" || continue

    rest="${f#*/ports/}"
    port="${rest%%/*}"
    idx="${f##*/}"
    ndev="$(read_sysfs "${f}")"
    ndev="${ndev//$'\n'/}"
    ndev="${ndev//$'\t'/}"
    ndev="${ndev// /}"
    [[ -n "${ndev}" ]] || continue

    gid="$(read_sysfs "/sys/class/infiniband/${ibdev}/ports/${port}/gids/${idx}")"
    gid_type="$(read_sysfs "/sys/class/infiniband/${ibdev}/ports/${port}/gid_attrs/types/${idx}")"
    printf '%s|%s|%s|%s|%s|%s\n' "${ndev}" "${ibdev}" "${port}" "${idx}" "${gid}" "${gid_type}"
  done
  shopt -u nullglob
}

rdma_records_from_ibdev2netdev() {
  command -v ibdev2netdev >/dev/null 2>&1 || return 0

  local ibdev word port arrow netdev state rest
  while read -r ibdev word port arrow netdev state rest; do
    [[ "${word:-}" == "port" && "${arrow:-}" == "==>" ]] || continue
    [[ -n "${netdev:-}" ]] || continue
    hca_selected "${ibdev}" || continue
    printf '%s|%s|%s|?|?|%s\n' "${netdev}" "${ibdev}" "${port}" "${state:-?}"
  done < <(ibdev2netdev)
}

rdma_netdev_records() {
  if [[ -n "${RDMA_IFACES:-}" ]]; then
    local dev
    for dev in ${RDMA_IFACES}; do
      printf '%s|manual|?|?|?|\n' "${dev}"
    done
    return 0
  fi

  local records
  records="$(rdma_records_from_sysfs | awk -F'|' '$1 != "" && !seen[$1]++ {print}')"
  if [[ -z "${records}" ]]; then
    records="$(rdma_records_from_ibdev2netdev | awk -F'|' '$1 != "" && !seen[$1]++ {print}')"
  fi
  [[ -n "${records}" ]] && printf '%s\n' "${records}"
  return 0
}

detect_rdma_ifaces() {
  rdma_netdev_records | awk -F'|' '$1 != "" {print $1}'
}

bootstrap_host_ifaces() {
  [[ -n "${LOCAL_IP:-}" ]] || return 0

  local local_addr="${LOCAL_IP%%/*}"
  ip -o -4 addr show 2>/dev/null | awk -v ip="${local_addr}" '
    $4 ~ "^" ip "/" {
      gsub(/@.*/, "", $2)
      print $2
    }'

  if [[ -n "${REMOTE_IP:-}" ]]; then
    ip route get "${REMOTE_IP}" 2>/dev/null | awk '
      {
        for (i = 1; i <= NF; i++) {
          if ($i == "dev" && (i + 1) <= NF) {
            dev = $(i + 1)
            gsub(/@.*/, "", dev)
            print dev
          }
        }
      }' || true
  fi
}

effective_bootstrap_ifname() {
  if bootstrap_host_mode && [[ -z "${BOOTSTRAP_IFNAME_EXPLICIT}" ]]; then
    local iface
    iface="$(bootstrap_host_ifaces | awk 'NF {print; exit}')"
    if [[ -n "${iface}" ]]; then
      printf '%s\n' "${iface}"
      return 0
    fi
  fi
  printf '%s\n' "${BOOTSTRAP_IFNAME}"
}

guard_not_moving_bootstrap() {
  local records_file="$1"
  [[ "${ALLOW_BOOTSTRAP_RDMA_IFACE}" == "1" ]] && return 0
  [[ -n "${LOCAL_IP:-}" && -n "${REMOTE_IP:-}" ]] || return 0

  local -a bootstrap_ifaces=()
  local iface
  while IFS= read -r iface; do
    [[ -n "${iface}" ]] && bootstrap_ifaces+=("${iface}")
  done < <(bootstrap_host_ifaces | awk '!seen[$0]++')

  (( ${#bootstrap_ifaces[@]} > 0 )) || return 0

  local dev bad=()
  while IFS= read -r dev; do
    [[ -n "${dev}" ]] || continue
    for iface in "${bootstrap_ifaces[@]}"; do
      if [[ "${dev}" == "${iface}" ]]; then
        bad+=("${dev}")
      fi
    done
  done < <(awk -F'|' '$1 != "" && !seen[$1]++ {print $1}' "${records_file}")

  if (( ${#bad[@]} > 0 )); then
    die "refusing to move bootstrap underlay iface(s): ${bad[*]}; choose a non-RDMA LOCAL_IP/REMOTE_IP path or set ALLOW_BOOTSTRAP_RDMA_IFACE=1"
  fi
}

ip_addrs_csv() {
  local dev="$1"
  local -a addrs=()
  local addr
  while IFS= read -r addr; do
    [[ -n "${addr}" ]] && addrs+=("${addr}")
  done < <(ip -o -4 addr show dev "${dev}" scope global 2>/dev/null | awk '{print $4}')

  (( ${#addrs[@]} == 0 )) && return 0
  local IFS=,
  printf '%s' "${addrs[*]}"
}

inspect_rdma() {
  require_cmd ip awk mktemp

  echo "RDMA HCA link layers:"
  print_hca_link_layers | awk '{print "  " $0}'

  local records_tmp
  records_tmp="$(mktemp)"
  trap "rm -f -- '${records_tmp}'; trap - RETURN" RETURN

  rdma_netdev_records >"${records_tmp}"
  if [[ ! -s "${records_tmp}" ]]; then
    if rdma_move_netdevs_enabled; then
      die "ROCE=1 requires RDMA netdev candidates, but none were found from gid_attrs/ndevs or ibdev2netdev"
    fi
    echo
    echo "RDMA netdev candidates: none (ok for native InfiniBand; use ROCE=1 only for Ethernet/RoCE HCAs)"
    return 0
  fi

  echo
  echo "RDMA netdev candidates:"
  printf '%-20s %-12s %-6s %-6s %-40s %s\n' "netdev" "ibdev" "port" "gid" "gid_value" "gid_type"
  awk -F'|' '{
    printf "%-20s %-12s %-6s %-6s %-40s %s\n", $1, $2, $3, $4, $5, $6
  }' "${records_tmp}"

  if [[ -n "${LOCAL_IP:-}" ]]; then
    echo
    echo "bootstrap host iface candidates:"
    bootstrap_host_ifaces | awk '!seen[$0]++ {print "  " $0}'
  fi
}

snapshot_rdma() {
  require_cmd sudo ip awk mktemp

  local tmp routes_tmp records_tmp
  tmp="$(mktemp)"
  routes_tmp="$(mktemp)"
  records_tmp="$(mktemp)"
  trap "rm -f -- '${tmp}' '${routes_tmp}' '${records_tmp}'; trap - RETURN" RETURN

  rdma_netdev_records >"${records_tmp}"
  [[ -s "${records_tmp}" ]] || die "no RDMA interfaces were found"
  guard_not_moving_bootstrap "${records_tmp}"

  printf '# netdev|ipv4_csv|mtu|mac|ibdevs\n' >"${tmp}"
  printf '# netdev|route\n' >"${routes_tmp}"

  local found=0
  local dev addr_csv mtu mac ibdevs
  while IFS= read -r dev; do
    [[ -n "${dev}" ]] || continue
    if [[ ! -e "/sys/class/net/${dev}" ]]; then
      die "RDMA netdev ${dev} is not present in the current host netns"
    fi

    addr_csv="$(ip_addrs_csv "${dev}")"
    if [[ -z "${addr_csv}" ]]; then
      if [[ "${REQUIRE_RDMA_IPV4}" == "1" ]]; then
        die "RDMA netdev ${dev} has no IPv4 address; assign one or set REQUIRE_RDMA_IPV4=0"
      fi
      addr_csv="-"
      warn "snapshotting ${dev} without IPv4 address"
    fi

    mtu="$(read_sysfs "/sys/class/net/${dev}/mtu")"
    mac="$(read_sysfs "/sys/class/net/${dev}/address")"
    ibdevs="$(awk -F'|' -v d="${dev}" '
      $1 == d && $2 != "" && !seen[$2]++ {
        out = out ? out "," $2 : $2
      }
      END { print out }
    ' "${records_tmp}")"

    printf '%s|%s|%s|%s|%s\n' "${dev}" "${addr_csv}" "${mtu:-?}" "${mac:-?}" "${ibdevs:-?}" >>"${tmp}"
    ip -4 route show table all dev "${dev}" 2>/dev/null | awk -v d="${dev}" '
      /^(local|broadcast|multicast|unreachable|throw|prohibit|blackhole) / { next }
      { print d "|" $0 }
    ' >>"${routes_tmp}"
    found=1
  done < <(awk -F'|' '$1 != "" && !seen[$1]++ {print $1}' "${records_tmp}")

  (( found == 1 )) || die "no RDMA interfaces with usable state were found"

  sudo install -m 0644 "${tmp}" "${RDMA_SNAPSHOT_PATH}"
  sudo install -m 0644 "${routes_tmp}" "${RDMA_ROUTES_PATH}"
  echo "snapshot written to ${RDMA_SNAPSHOT_PATH}:"
  sudo cat "${RDMA_SNAPSHOT_PATH}"
  echo "route snapshot written to ${RDMA_ROUTES_PATH}:"
  sudo cat "${RDMA_ROUTES_PATH}"
}

auto_device_args() {
  local -n out="$1"

  if [[ -n "${DEVS:-}" ]]; then
    read -r -a out <<<"${DEVS}"
    return
  fi

  shopt -s nullglob
  local dev
  for dev in /dev/infiniband/uverbs*; do
    out+=("--device=${dev}")
  done
  shopt -u nullglob

  if (( ${#out[@]} == 0 )); then
    warn "no DEVS set and no /dev/infiniband/uverbs* devices found"
  fi
}

runsc_annotation_args() {
  local -n out="$1"
  local ann

  if truthy "${RUNSC_RDMA_PROXY}"; then
    out+=(--annotation "dev.gvisor.flag.rdmaproxy=true")
    out+=(--annotation "dev.gvisor.internal.rdmaproxy=true")
  elif falsey "${RUNSC_RDMA_PROXY}"; then
    out+=(--annotation "dev.gvisor.flag.rdmaproxy=false")
    out+=(--annotation "dev.gvisor.internal.rdmaproxy=false")
  fi
  if truthy "${RUNSC_NVPROXY}"; then
    out+=(--annotation "dev.gvisor.flag.nvproxy=true")
    out+=(--annotation "dev.gvisor.internal.nvproxy=true")
  elif falsey "${RUNSC_NVPROXY}"; then
    out+=(--annotation "dev.gvisor.flag.nvproxy=false")
    out+=(--annotation "dev.gvisor.internal.nvproxy=false")
  fi
  if truthy "${RUNSC_DEBUG}"; then
    out+=(--annotation "dev.gvisor.flag.debug=true")
    if [[ -n "${RUNSC_DEBUG_LOG}" ]]; then
      out+=(--annotation "dev.gvisor.flag.debug-log=${RUNSC_DEBUG_LOG}")
    fi
  elif falsey "${RUNSC_DEBUG}"; then
    out+=(--annotation "dev.gvisor.flag.debug=false")
  fi
  if [[ -n "${RUNSC_DEBUG_COMMAND}" ]]; then
    out+=(--annotation "dev.gvisor.flag.debug-command=${RUNSC_DEBUG_COMMAND}")
  fi
  if truthy "${RUNSC_STRACE}"; then
    out+=(--annotation "dev.gvisor.flag.strace=true")
  elif falsey "${RUNSC_STRACE}"; then
    out+=(--annotation "dev.gvisor.flag.strace=false")
  fi
  if [[ -n "${RUNSC_STRACE_SYSCALLS}" ]]; then
    out+=(--annotation "dev.gvisor.flag.strace-syscalls=${RUNSC_STRACE_SYSCALLS}")
  fi
  if [[ -n "${RUNSC_STRACE_LOG_SIZE}" ]]; then
    out+=(--annotation "dev.gvisor.flag.strace-log-size=${RUNSC_STRACE_LOG_SIZE}")
  fi
  for ann in ${RUNSC_EXTRA_ANNOTATIONS}; do
    out+=(--annotation "${ann}")
  done
}

start_netns_holder() {
  require_cmd sudo docker
  set_role "${1:-}"

  if bootstrap_gre_mode; then
    sudo docker network inspect "${NETWORK_NAME}" >/dev/null
  fi
  sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  sudo docker rm -f "${NETNS_HOLDER_NAME}" >/dev/null 2>&1 || true

  local -a runtime_args=()
  if [[ -n "${NETNS_HOLDER_RUNTIME}" ]]; then
    runtime_args+=(--runtime="${NETNS_HOLDER_RUNTIME}")
  fi

  local -a network_args=()
  if bootstrap_host_mode; then
    network_args+=(--network host)
  else
    network_args+=(--network "${NETWORK_NAME}" --ip "${CONTAINER_IP}")
  fi

  sudo docker run -d --name "${NETNS_HOLDER_NAME}" \
    "${runtime_args[@]}" \
    "${network_args[@]}" \
    "${IMAGE}" \
    sleep infinity >/dev/null

  echo "netns_holder=${NETNS_HOLDER_NAME} runtime=${NETNS_HOLDER_RUNTIME:-default} role=${ROLE} ip=${CONTAINER_IP} bootstrap_mode=${BOOTSTRAP_MODE}"
}

start_app_container() {
  require_cmd sudo docker
  set_role "${1:-}"

  if [[ ! -f "${WORKSPACE_MOUNT}/${TORCH_SCRIPT#./}" ]]; then
    warn "torch script not found at ${WORKSPACE_MOUNT}/${TORCH_SCRIPT#./}"
  fi
  if [[ ! -f "${WORKSPACE_MOUNT}/nccl_topo.xml" && -z "${NCCL_TOPO_FILE:-}" ]]; then
    warn "nccl_topo.xml not found under ${WORKSPACE_MOUNT}; generate it before performance runs"
  fi

  if bootstrap_gre_mode; then
    sudo docker network inspect "${NETWORK_NAME}" >/dev/null
  fi
  sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

  local -a dev_args=()
  auto_device_args dev_args
  local -a annotation_args=()
  runsc_annotation_args annotation_args
  local bootstrap_ifname
  bootstrap_ifname="$(effective_bootstrap_ifname)"
  local -a env_args=(
    -e "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}"
    -e "NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-1}"
    -e "NCCL_TOPO_FILE=${NCCL_TOPO_FILE:-/workspace/nccl_topo.xml}"
    -e "NCCL_SOCKET_IFNAME=${bootstrap_ifname}"
    -e "GLOO_SOCKET_IFNAME=${bootstrap_ifname}"
    -e "BW_ONLY=${BW_ONLY:-1}"
    -e "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
  )
  local effective_nccl_ib_hca="${NCCL_IB_HCA:-}"
  if [[ -z "${effective_nccl_ib_hca}" ]]; then
    effective_nccl_ib_hca="$(derive_nccl_ib_hca)"
  fi
  if [[ -n "${effective_nccl_ib_hca}" ]]; then
    env_args+=(-e "NCCL_IB_HCA=${effective_nccl_ib_hca}")
  fi
  if [[ -n "${NCCL_IB_GID_INDEX:-}" ]]; then
    env_args+=(-e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}")
  fi

  local -a network_args=()
  if bootstrap_host_mode; then
    network_args+=(--network host)
  elif truthy "${USE_NETNS_HOLDER}"; then
    if ! container_running "${NETNS_HOLDER_NAME}"; then
      die "netns holder ${NETNS_HOLDER_NAME} is not running; run start-container/move-rdma first or prepare ${ROLE}"
    fi
    network_args+=(--network "container:${NETNS_HOLDER_NAME}")
  else
    network_args+=(--network "${NETWORK_NAME}" --ip "${CONTAINER_IP}")
  fi

  sudo docker run -d --name "${CONTAINER_NAME}" \
    --runtime="${RUNTIME}" "${network_args[@]}" \
    "${annotation_args[@]}" \
    --gpus all "${dev_args[@]}" \
    --ulimit memlock=-1:-1 --shm-size=1g \
    -v "${WORKSPACE_MOUNT}:/workspace" \
    -w /workspace \
    "${env_args[@]}" \
    "${IMAGE}" \
    sleep infinity >/dev/null

  echo "container=${CONTAINER_NAME} runtime=${RUNTIME} role=${ROLE} ip=${CONTAINER_IP}"
  if bootstrap_host_mode; then
    echo "network=host"
  elif truthy "${USE_NETNS_HOLDER}"; then
    echo "network=container:${NETNS_HOLDER_NAME}"
  else
    echo "network=${NETWORK_NAME}"
  fi
  echo "bootstrap_ifname=${bootstrap_ifname}"
  echo "workspace_mount=${WORKSPACE_MOUNT}:/workspace"
  if [[ -n "${effective_nccl_ib_hca}" ]]; then
    echo "NCCL_IB_HCA=${effective_nccl_ib_hca}"
  fi
  if (( ${#annotation_args[@]} > 0 )); then
    echo "runsc_annotations=${annotation_args[*]}"
  fi
}

start_container() {
  set_role "${1:-}"
  if truthy "${USE_NETNS_HOLDER}" && rdma_move_netdevs_enabled; then
    start_netns_holder "${ROLE}"
    echo "holder ready; move RDMA netdevs, then run start-app-container ${ROLE} (prepare does this automatically)"
  elif truthy "${USE_NETNS_HOLDER}"; then
    start_netns_holder "${ROLE}"
    start_app_container "${ROLE}"
  else
    start_app_container "${ROLE}"
  fi
}

SNAP_DEV=""
SNAP_IPV4=""
SNAP_MTU=""
SNAP_MAC=""
SNAP_IBDEVS=""

parse_snapshot_line() {
  local line="$1"
  [[ -z "${line}" || "${line:0:1}" == "#" ]] && return 1

  SNAP_DEV=""
  SNAP_IPV4=""
  SNAP_MTU=""
  SNAP_MAC=""
  SNAP_IBDEVS=""

  if [[ "${line}" == *"|"* ]]; then
    IFS='|' read -r SNAP_DEV SNAP_IPV4 SNAP_MTU SNAP_MAC SNAP_IBDEVS <<<"${line}"
  else
    read -r SNAP_DEV SNAP_IPV4 <<<"${line}"
    SNAP_MTU=""
    SNAP_MAC=""
    SNAP_IBDEVS=""
  fi

  [[ -n "${SNAP_DEV}" ]] || return 1
  [[ -n "${SNAP_IPV4}" ]] || SNAP_IPV4="-"
  return 0
}

restore_ipv4_addrs() {
  local pid="$1"
  local dev="$2"
  local csv="$3"
  [[ -z "${csv}" || "${csv}" == "-" ]] && return 0

  local -a addrs=()
  local addr
  IFS=',' read -r -a addrs <<<"${csv}"
  for addr in "${addrs[@]}"; do
    [[ -n "${addr}" ]] || continue
    sudo nsenter -t "${pid}" -n ip addr replace "${addr}" dev "${dev}"
  done
}

route_cidr_for_addr() {
  local cidr="$1"
  local ip="${cidr%%/*}"
  local prefix="${RDMA_ROUTE_PREFIX_LEN}"
  local o1 o2 o3 o4
  IFS=. read -r o1 o2 o3 o4 <<<"${ip}"

  case "${prefix}" in
    8)
      printf '%s.0.0.0/8\n' "${o1}"
      ;;
    16)
      printf '%s.%s.0.0/16\n' "${o1}" "${o2}"
      ;;
    24)
      printf '%s.%s.%s.0/24\n' "${o1}" "${o2}" "${o3}"
      ;;
    32)
      printf '%s/32\n' "${ip}"
      ;;
    *)
      python3 -c 'import ipaddress, sys; print(ipaddress.ip_network(f"{sys.argv[1]}/{sys.argv[2]}", strict=False))' "${ip}" "${prefix}"
      ;;
  esac
}

restore_snapshot_routes_for_dev() {
  local pid="$1"
  local dev="$2"
  [[ -f "${RDMA_ROUTES_PATH}" ]] || return 1

  local restored=0
  local route
  while IFS= read -r route; do
    [[ -n "${route}" ]] || continue
    if [[ "${route}" != *" dev "* ]]; then
      route="${route} dev ${dev}"
    fi
    if sudo nsenter -t "${pid}" -n env ROUTE="${route}" bash -lc 'ip -4 route replace $ROUTE'; then
      restored=1
    else
      warn "failed to restore route for ${dev}: ${route}"
    fi
  done < <(sudo awk -F'|' -v d="${dev}" '$1 == d {print $2}' "${RDMA_ROUTES_PATH}" | awk '
    / scope link / { print "0|" $0; next }
    { print "1|" $0 }
  ' | sort -t'|' -k1,1n | cut -d'|' -f2-)

  (( restored == 1 ))
}

infer_connected_routes_for_dev() {
  local pid="$1"
  local dev="$2"
  local csv="$3"
  [[ -z "${csv}" || "${csv}" == "-" ]] && return 0

  local -a addrs=()
  local addr ip route
  IFS=',' read -r -a addrs <<<"${csv}"
  for addr in "${addrs[@]}"; do
    [[ -n "${addr}" ]] || continue
    ip="${addr%%/*}"
    route="$(route_cidr_for_addr "${addr}")"
    sudo nsenter -t "${pid}" -n ip -4 route replace "${route}" dev "${dev}" src "${ip}" || \
      warn "failed to infer RDMA route ${route} dev ${dev} src ${ip}"
  done
}

restore_rdma_routes_for_dev() {
  local pid="$1"
  local dev="$2"
  local csv="$3"
  [[ "${RESTORE_RDMA_ROUTES}" == "1" ]] || return 0

  if restore_snapshot_routes_for_dev "${pid}" "${dev}"; then
    return 0
  fi
  infer_connected_routes_for_dev "${pid}" "${dev}" "${csv}"
}

restore_bootstrap_config() {
  require_cmd sudo docker nsenter ip
  set_role "${1:-${ROLE:-}}"

  local pid prefix
  pid="$(netns_pid)"
  prefix="${SUBNET#*/}"

  if sudo nsenter -t "${pid}" -n ip link show dev "${BOOTSTRAP_IFNAME}" >/dev/null 2>&1; then
    sudo nsenter -t "${pid}" -n ip link set dev "${BOOTSTRAP_IFNAME}" up || true
    sudo nsenter -t "${pid}" -n ip addr replace "${CONTAINER_IP}/${prefix}" dev "${BOOTSTRAP_IFNAME}" || \
      warn "failed to restore bootstrap address ${CONTAINER_IP}/${prefix} on ${BOOTSTRAP_IFNAME}"
    sudo nsenter -t "${pid}" -n ip route replace "${SUBNET}" dev "${BOOTSTRAP_IFNAME}" src "${CONTAINER_IP}" || true
    sudo nsenter -t "${pid}" -n ip route replace default via "${GW}" dev "${BOOTSTRAP_IFNAME}" || true
  fi
}

restore_rdma_config() {
  require_cmd sudo docker nsenter ip awk
  [[ -f "${RDMA_SNAPSHOT_PATH}" ]] || die "snapshot file not found: ${RDMA_SNAPSHOT_PATH}"

  local pid
  pid="$(netns_pid)"

  local line
  while IFS= read -r line; do
    parse_snapshot_line "${line}" || continue
    if ! sudo nsenter -t "${pid}" -n ip link show dev "${SNAP_DEV}" >/dev/null 2>&1; then
      warn "skipping ${SNAP_DEV}; not present in ${CONTAINER_NAME}'s host netns"
      continue
    fi
    if [[ -n "${SNAP_MTU}" && "${SNAP_MTU}" != "?" ]]; then
      sudo nsenter -t "${pid}" -n ip link set dev "${SNAP_DEV}" mtu "${SNAP_MTU}" || true
    fi
    restore_ipv4_addrs "${pid}" "${SNAP_DEV}" "${SNAP_IPV4}"
    sudo nsenter -t "${pid}" -n ip link set dev "${SNAP_DEV}" up || true
    restore_rdma_routes_for_dev "${pid}" "${SNAP_DEV}" "${SNAP_IPV4}"
    sudo nsenter -t "${pid}" -n ip -br addr show dev "${SNAP_DEV}" || true
  done < <(sudo cat "${RDMA_SNAPSHOT_PATH}")
}

fix_rdma_routes() {
  restore_rdma_config

  echo "RDMA routes in ${CONTAINER_NAME}'s host netns:"
  local pid
  pid="$(netns_pid)"
  sudo nsenter -t "${pid}" -n ip -4 route show table all | grep -E 'gpu[0-9]+rdma|mlx|rdma' || true
}

move_rdma() {
  require_cmd sudo docker nsenter ip
  [[ -f "${RDMA_SNAPSHOT_PATH}" ]] || die "snapshot file not found: ${RDMA_SNAPSHOT_PATH}"

  local pid
  pid="$(netns_pid)"

  local line
  while IFS= read -r line; do
    parse_snapshot_line "${line}" || continue

    if sudo ip link show dev "${SNAP_DEV}" >/dev/null 2>&1; then
      sudo ip link set dev "${SNAP_DEV}" down || true
      sudo ip link set dev "${SNAP_DEV}" netns "${pid}"
    elif sudo nsenter -t "${pid}" -n ip link show dev "${SNAP_DEV}" >/dev/null 2>&1; then
      echo "${SNAP_DEV} is already in container netns"
    else
      die "${SNAP_DEV} is neither in the host netns nor in ${CONTAINER_NAME}'s netns"
    fi

    if [[ -n "${SNAP_MTU}" && "${SNAP_MTU}" != "?" ]]; then
      sudo nsenter -t "${pid}" -n ip link set dev "${SNAP_DEV}" mtu "${SNAP_MTU}" || true
    fi
    restore_ipv4_addrs "${pid}" "${SNAP_DEV}" "${SNAP_IPV4}"
    sudo nsenter -t "${pid}" -n ip link set dev "${SNAP_DEV}" up
    restore_rdma_routes_for_dev "${pid}" "${SNAP_DEV}" "${SNAP_IPV4}"
    sudo nsenter -t "${pid}" -n ip -br addr show dev "${SNAP_DEV}"
  done < <(sudo cat "${RDMA_SNAPSHOT_PATH}")

  validate_container
}

validate_sandbox_rdma_sysfs() {
  require_cmd sudo docker

  if ! container_running "${CONTAINER_NAME}"; then
    echo
    echo "sandbox RDMA sysfs view: skipped; ${CONTAINER_NAME} is not running yet"
    return 0
  fi

  echo
  echo "sandbox RDMA sysfs view:"
  if ! sudo docker exec \
    -e "REQUIRE_SANDBOX_RDMA_GIDS=${REQUIRE_SANDBOX_RDMA_GIDS}" \
    -e "REQUIRE_SANDBOX_RDMA_NDEVS=${REQUIRE_SANDBOX_RDMA_NDEVS}" \
    -e "RDMA_MOVE_NETDEVS=${RDMA_MOVE_NETDEVS}" \
    "${CONTAINER_NAME}" \
    bash -lc '
    set -e
    zero_gid="0000:0000:0000:0000:0000:0000:0000:0000"
    nonzero=0
    with_ndev=0
    printed=0
    if [ -d /sys/class/infiniband ]; then
      for d in /sys/class/infiniband/*; do
        [ -e "$d" ] || continue
        printf "  hca=%s\n" "$(basename "$d")"
      done
    fi
    for f in /sys/class/infiniband/*/ports/*/gids/*; do
      [ -e "$f" ] || continue
      gid="$(cat "$f" 2>/dev/null || true)"
      [ -n "$gid" ] || continue
      [ "$gid" != "$zero_gid" ] || continue
      nonzero=$((nonzero + 1))
      idx="${f##*/}"
      port_dir="${f%/gids/*}"
      type="$(cat "$port_dir/gid_attrs/types/$idx" 2>/dev/null || true)"
      ndev="$(cat "$port_dir/gid_attrs/ndevs/$idx" 2>/dev/null || true)"
      [ -z "$ndev" ] || with_ndev=$((with_ndev + 1))
      if [ "$printed" -lt 32 ]; then
        printf "  %s gid=%s type=%s ndev=%s\n" "$f" "$gid" "$type" "$ndev"
        printed=$((printed + 1))
      fi
    done
    printf "  nonzero_gids=%s with_ndev=%s\n" "$nonzero" "$with_ndev"
    [ "${REQUIRE_SANDBOX_RDMA_GIDS:-1}" != "1" ] || [ "$nonzero" -gt 0 ]
    require_ndev="${REQUIRE_SANDBOX_RDMA_NDEVS:-auto}"
    if [ "$require_ndev" = "auto" ]; then
      case "${RDMA_MOVE_NETDEVS:-0}" in
        1|true|TRUE|yes|YES|on|ON) require_ndev=1 ;;
        *) require_ndev=0 ;;
      esac
    fi
    [ "$require_ndev" != "1" ] || [ "$with_ndev" -gt 0 ]
  '; then
    if [[ "${REQUIRE_SANDBOX_RDMA_GIDS}" == "1" ]]; then
      return 1
    fi
    warn "could not validate sandbox RDMA sysfs via docker exec"
  fi
}

validate_container() {
  require_cmd sudo docker

  if ! rdma_move_netdevs_enabled; then
    echo "ROCE=${ROCE} RDMA_MOVE_NETDEVS=${RDMA_MOVE_NETDEVS}; skipping moved-netdev validation"
    validate_sandbox_rdma_sysfs || die "sandbox RDMA sysfs validation failed"
    return 0
  fi

  require_cmd nsenter ip awk grep readlink
  [[ -f "${RDMA_SNAPSHOT_PATH}" ]] || die "snapshot file not found: ${RDMA_SNAPSHOT_PATH}"

  local pid
  pid="$(netns_pid)"

  echo "netns_pid=${pid}"
  if container_running "${CONTAINER_NAME}"; then
    echo "container_pid=$(container_pid)"
  else
    echo "container_pid=not-running"
  fi
  if truthy "${USE_NETNS_HOLDER}"; then
    echo "netns_holder=${NETNS_HOLDER_NAME}"
  fi
  echo "container_netns=$(sudo readlink "/proc/${pid}/ns/net")"
  echo "host view of container netns:"
  sudo nsenter -t "${pid}" -n ip -br addr

  local failed=0
  local line actual_addrs actual_mtu addr
  local -a expected_addrs=()
  while IFS= read -r line; do
    parse_snapshot_line "${line}" || continue

    if ! sudo nsenter -t "${pid}" -n ip link show dev "${SNAP_DEV}" >/dev/null 2>&1; then
      echo "missing in container netns: ${SNAP_DEV}" >&2
      failed=1
      continue
    fi

    if [[ -n "${SNAP_MTU}" && "${SNAP_MTU}" != "?" ]]; then
      actual_mtu="$(sudo nsenter -t "${pid}" -n cat "/sys/class/net/${SNAP_DEV}/mtu" 2>/dev/null || true)"
      if [[ -n "${actual_mtu}" && "${actual_mtu}" != "${SNAP_MTU}" ]]; then
        echo "MTU mismatch for ${SNAP_DEV}: expected ${SNAP_MTU}, got ${actual_mtu}" >&2
        failed=1
      fi
    fi

    if [[ -n "${SNAP_IPV4}" && "${SNAP_IPV4}" != "-" ]]; then
      actual_addrs="$(sudo nsenter -t "${pid}" -n ip -o -4 addr show dev "${SNAP_DEV}" scope global | awk '{print $4}' || true)"
      expected_addrs=()
      IFS=',' read -r -a expected_addrs <<<"${SNAP_IPV4}"
      for addr in "${expected_addrs[@]}"; do
        [[ -n "${addr}" ]] || continue
        if ! printf '%s\n' "${actual_addrs}" | grep -Fxq "${addr}"; then
          echo "IPv4 mismatch for ${SNAP_DEV}: missing ${addr}" >&2
          failed=1
        fi
      done
    fi
  done < <(sudo cat "${RDMA_SNAPSHOT_PATH}")

  validate_sandbox_rdma_sysfs || failed=1

  (( failed == 0 )) || die "container RDMA netns validation failed"
}

preflight() {
  require_cmd sudo docker ip awk mktemp
  require_env LOCAL_IP REMOTE_IP
  set_role "${1:-}"

  inspect_rdma

  local records_tmp
  records_tmp="$(mktemp)"
  trap "rm -f -- '${records_tmp}'; trap - RETURN" RETURN
  rdma_netdev_records >"${records_tmp}"
  if rdma_move_netdevs_enabled; then
    [[ -s "${records_tmp}" ]] || die "ROCE=1 requires RDMA netdev candidates, but none were found"
    guard_not_moving_bootstrap "${records_tmp}"
  fi

  if ! sudo docker info --format '{{range $name, $_ := .Runtimes}}{{println $name}}{{end}}' | grep -Fxq "${RUNTIME}"; then
    die "Docker runtime ${RUNTIME} is not registered"
  fi
  if truthy "${USE_NETNS_HOLDER}" && [[ -n "${NETNS_HOLDER_RUNTIME}" ]]; then
    if ! sudo docker info --format '{{range $name, $_ := .Runtimes}}{{println $name}}{{end}}' | grep -Fxq "${NETNS_HOLDER_RUNTIME}"; then
      die "Docker runtime ${NETNS_HOLDER_RUNTIME} for netns holder is not registered"
    fi
  fi

  if [[ ! -f "${WORKSPACE_MOUNT}/${TORCH_SCRIPT#./}" ]]; then
    warn "torch script not found at ${WORKSPACE_MOUNT}/${TORCH_SCRIPT#./}"
  fi
  if [[ ! -f "${WORKSPACE_MOUNT}/nccl_topo.xml" && -z "${NCCL_TOPO_FILE:-}" ]]; then
    warn "nccl_topo.xml not found under ${WORKSPACE_MOUNT}; generate it before performance runs"
  fi

  echo
  echo "preflight ok for role=${ROLE} runtime=${RUNTIME} container_ip=${CONTAINER_IP} roce=${ROCE} move_netdevs=${RDMA_MOVE_NETDEVS}"
}

prepare() {
  local role="${1:-}"
  set_role "${role}"
  cleanup_container
  preflight "${role}"
  setup_net "${role}"
  if rdma_move_netdevs_enabled; then
    snapshot_rdma
    if truthy "${USE_NETNS_HOLDER}" && ! bootstrap_host_mode; then
      start_netns_holder "${role}"
      move_rdma
      start_app_container "${role}"
      restore_rdma_config
      validate_container
    else
      start_app_container "${role}"
      move_rdma
    fi
  else
    echo "ROCE=${ROCE} RDMA_MOVE_NETDEVS=${RDMA_MOVE_NETDEVS}; skipping RDMA netdev snapshot/move"
    if truthy "${USE_NETNS_HOLDER}"; then
      start_netns_holder "${role}"
    fi
    start_app_container "${role}"
    validate_sandbox_rdma_sysfs
  fi
}

run_test() {
  require_cmd sudo docker
  set_role "${1:-}"

  local workspace_source
  workspace_source="$(sudo docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
  if ! sudo docker exec "${CONTAINER_NAME}" bash -lc "test -f ${TORCH_SCRIPT@Q}"; then
    die "missing ${TORCH_SCRIPT} inside ${CONTAINER_NAME}; /workspace is mounted from ${workspace_source:-unknown}. Recreate the container from the directory containing torch_mnist_train.py."
  fi
  if [[ -z "${NCCL_TOPO_FILE:-}" ]] && ! sudo docker exec "${CONTAINER_NAME}" bash -lc "test -f /workspace/nccl_topo.xml"; then
    warn "missing /workspace/nccl_topo.xml inside ${CONTAINER_NAME}; NCCL_TOPO_FILE defaults to /workspace/nccl_topo.xml"
  fi
  if [[ "${NODE_RANK}" != "0" ]]; then
    echo "node_rank=${NODE_RANK} will wait for rank 0 at ${MASTER_ADDR}:${MASTER_PORT}"
    if ! container_tcp_probe "${MASTER_ADDR}" "${MASTER_PORT}" 1; then
      warn "master is not reachable yet; this is expected only while node A/rank 0 has not started"
    fi
  fi

  local effective_nccl_ib_hca="${NCCL_IB_HCA:-}"
  if [[ -z "${effective_nccl_ib_hca}" ]]; then
    effective_nccl_ib_hca="$(derive_nccl_ib_hca)"
    if [[ -n "${effective_nccl_ib_hca}" ]]; then
      echo "NCCL_IB_HCA=${effective_nccl_ib_hca} (derived from RDMA link-layer state)"
    fi
  fi

  local -a exec_args=(sudo docker exec)
  local name
  for name in NCCL_DEBUG NCCL_DEBUG_SUBSYS TORCH_DISTRIBUTED_DEBUG NCCL_IB_GID_INDEX BW_ONLY OMP_NUM_THREADS; do
    if [[ -n "${!name:-}" ]]; then
      exec_args+=(-e "${name}=${!name}")
    fi
  done
  if [[ -n "${effective_nccl_ib_hca}" ]]; then
    exec_args+=(-e "NCCL_IB_HCA=${effective_nccl_ib_hca}")
  fi

  local torchrun_cmd
  torchrun_cmd="torchrun ${TORCHRUN_ARGS:-} --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --node_rank=${NODE_RANK} ${TORCH_SCRIPT}"

  exec_args+=(
    "${CONTAINER_NAME}"
    bash
    -lc
    "${torchrun_cmd}"
  )

  "${exec_args[@]}"
}

ping_peer() {
  require_cmd sudo docker
  set_role "${1:-}"
  sudo docker exec "${CONTAINER_NAME}" bash -lc "ping -c 3 -W 2 ${PEER_CONTAINER_IP}"
}

container_tcp_probe() {
  local addr="$1"
  local port="$2"
  local timeout="${3:-2}"

  sudo docker exec \
    -e "PROBE_ADDR=${addr}" \
    -e "PROBE_PORT=${port}" \
    -e "PROBE_TIMEOUT=${timeout}" \
    "${CONTAINER_NAME}" \
    python3 -c '
import os
import socket
import sys

addr = os.environ["PROBE_ADDR"]
port = int(os.environ["PROBE_PORT"])
timeout = float(os.environ["PROBE_TIMEOUT"])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(timeout)
try:
    s.connect((addr, port))
except OSError as e:
    print(f"tcp_connect={addr}:{port} failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
else:
    print(f"tcp_connect={addr}:{port} ok", flush=True)
finally:
    s.close()
'
}

wait_master() {
  require_cmd sudo docker
  set_role "${1:-}"

  local timeout="${WAIT_MASTER_TIMEOUT:-120}"
  local deadline=$((SECONDS + timeout))
  echo "waiting for ${MASTER_ADDR}:${MASTER_PORT} from ${CONTAINER_NAME} (timeout=${timeout}s)"
  while (( SECONDS < deadline )); do
    if container_tcp_probe "${MASTER_ADDR}" "${MASTER_PORT}" 2; then
      return 0
    fi
    sleep 2
  done
  die "master ${MASTER_ADDR}:${MASTER_PORT} was not reachable from ${CONTAINER_NAME} within ${timeout}s"
}

diagnose() {
  require_cmd sudo docker
  set_role "${1:-}"

  local pid app_pid workspace_source
  pid="$(netns_pid)"
  app_pid="$(sudo docker inspect -f '{{.State.Pid}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
  workspace_source="$(sudo docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' "${CONTAINER_NAME}" 2>/dev/null || true)"

  cat <<EOF
role=${ROLE}
bootstrap_mode=${BOOTSTRAP_MODE}
container=${CONTAINER_NAME}
container_pid=${app_pid:-not-running}
netns_pid=${pid}
netns_holder=$(truthy "${USE_NETNS_HOLDER}" && echo "${NETNS_HOLDER_NAME}" || echo disabled)
container_ip=${CONTAINER_IP}
peer_container_ip=${PEER_CONTAINER_IP}
master=${MASTER_ADDR}:${MASTER_PORT}
bootstrap_ifname=$(effective_bootstrap_ifname)
workspace_source=${workspace_source:-unknown}
EOF

  echo
  echo "host netns view used by moved RDMA netdevs:"
  sudo nsenter -t "${pid}" -n ip -br addr || true
  sudo nsenter -t "${pid}" -n ip route || true

  echo
  echo "sandbox network view seen by docker exec:"
  sudo docker exec "${CONTAINER_NAME}" bash -lc '
    ip -br addr 2>/dev/null || true
    ip route 2>/dev/null || true
    printf "hostname=%s\n" "$(hostname 2>/dev/null || true)"
    printf "hostname_i=%s\n" "$(hostname -I 2>/dev/null || true)"
  '

  echo
  echo "workspace files:"
  sudo docker exec "${CONTAINER_NAME}" bash -lc 'pwd; ls -l /workspace/torch_mnist_train.py /workspace/nccl_topo.xml 2>&1 || true'

  echo
  echo "route probes:"
  sudo docker exec "${CONTAINER_NAME}" bash -lc "ip route get ${PEER_CONTAINER_IP} 2>/dev/null || true; ip route get ${MASTER_ADDR} 2>/dev/null || true"

  echo
  echo "master tcp probe:"
  container_tcp_probe "${MASTER_ADDR}" "${MASTER_PORT}" 2 || true

  echo
  echo "listening sockets:"
  sudo docker exec "${CONTAINER_NAME}" bash -lc 'ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null || true'

  echo
  echo "tcp sockets involving peer/master:"
  sudo docker exec \
    -e "PEER_CONTAINER_IP=${PEER_CONTAINER_IP}" \
    -e "MASTER_ADDR=${MASTER_ADDR}" \
    "${CONTAINER_NAME}" \
    bash -lc 'ss -tanp 2>/dev/null | awk -v peer="$PEER_CONTAINER_IP" -v master="$MASTER_ADDR" "NR == 1 || index(\$0, peer) || index(\$0, master)" || true'

  echo
  echo "python processes:"
  sudo docker exec "${CONTAINER_NAME}" bash -lc 'ps -eo pid,ppid,stat,etime,cmd | grep -E "torchrun|python3|torch_mnist" | grep -v grep || true'
}

kill_test() {
  require_cmd sudo docker
  sudo docker exec "${CONTAINER_NAME}" bash -lc '
    set +e
    pkill -TERM -f "torchrun|torch_mnist_train.py"
    sleep 2
    pkill -KILL -f "torchrun|torch_mnist_train.py"
    exit 0
  '
}

show_counters() {
  local dev port_dir rcv xmit found=0
  shopt -s nullglob
  for port_dir in /sys/class/infiniband/*/ports/1; do
    dev="$(basename "$(dirname "$(dirname "${port_dir}")")")"
    [[ -r "${port_dir}/counters/port_rcv_data" && -r "${port_dir}/counters/port_xmit_data" ]] || continue
    rcv="$(cat "${port_dir}/counters/port_rcv_data")"
    xmit="$(cat "${port_dir}/counters/port_xmit_data")"
    awk -v d="${dev}" -v rcv="${rcv}" -v xmit="${xmit}" 'BEGIN {
      printf "%s: rcv=%.6f GB xmit=%.6f GB\n", d, rcv*4/1e9, xmit*4/1e9
    }'
    found=1
  done
  shopt -u nullglob
  (( found == 1 )) || die "no readable RDMA port counters found"
}

show_logs() {
  local log
  log="$(ls -t /tmp/runsc-rdma/logs/runsc.log.*.boot.txt 2>/dev/null | head -1 || true)"
  [[ -n "${log}" ]] || die "no runsc boot logs found under /tmp/runsc-rdma/logs/"

  echo "latest_log=${log}"
  grep -E 'rdma collect|rdma sysfs|rdmaproxy|direct write|MODIFY_QP|EFAULT|MR REG|CQ/QP CREATE|DMABUF' "${log}" | tail -n 250 || true
}

cleanup_container() {
  docker_rm_container
  sleep 1
}

open_shell() {
  sudo docker exec -it "${CONTAINER_NAME}" bash
}

main() {
  while (( $# > 0 )); do
    case "${1:-}" in
      --roce)
        ROCE=1
        RDMA_MOVE_NETDEVS=1
        shift
        ;;
      --no-roce)
        ROCE=0
        RDMA_MOVE_NETDEVS=0
        shift
        ;;
      --host-network|--host-bootstrap)
        BOOTSTRAP_MODE=host
        shift
        ;;
      --gre)
        BOOTSTRAP_MODE=gre
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done
  validate_bootstrap_mode
  apply_roce_mode_defaults

  local subcommand="${1:-}"
  case "${subcommand}" in
    runtime-show)
      runtime_show
      ;;
    runtime-install)
      runtime_install
      ;;
    preflight)
      shift
      preflight "$@"
      ;;
    setup-net)
      shift
      setup_net "$@"
      ;;
    cleanup-net)
      cleanup_net
      ;;
    inspect-rdma)
      inspect_rdma
      ;;
    snapshot-rdma)
      snapshot_rdma
      ;;
    start-container)
      shift
      start_container "$@"
      ;;
    start-app-container)
      shift
      start_app_container "$@"
      ;;
    move-rdma)
      move_rdma
      ;;
    restore-rdma-config)
      shift
      restore_rdma_config
      ;;
    fix-rdma-routes)
      fix_rdma_routes
      ;;
    validate-container)
      validate_container
      ;;
    ping-peer)
      shift
      ping_peer "$@"
      ;;
    wait-master)
      shift
      wait_master "$@"
      ;;
    diagnose)
      shift
      diagnose "$@"
      ;;
    prepare)
      shift
      prepare "$@"
      ;;
    run-test)
      shift
      run_test "$@"
      ;;
    kill-test)
      kill_test
      ;;
    counters)
      show_counters
      ;;
    logs)
      show_logs
      ;;
    cleanup-container)
      cleanup_container
      ;;
    shell)
      open_shell
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      die "unknown subcommand: ${subcommand}"
      ;;
  esac
}

main "$@"
