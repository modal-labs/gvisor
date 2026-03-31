#!/usr/bin/env python3
# Copyright 2025 The gVisor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Long-running HTTP agent to spawn NCCL / PyTorch multi-node jobs on this node.

Multi-node collectives still require a second node (typically rank 0) to run the
matching workload. This agent runs the rank-1 container on node B; start rank 0
on node A after dispatching B (see README).

Security: bind to loopback by default; reach via SSH -L or a VPN. Do not expose
unauthenticated job execution on an untrusted network.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}

# Max upload size for NCCL topology XML (POST /v1/nccl_topo).
_MAX_NCCL_TOPO_BYTES = 10 * 1024 * 1024

# Keep at most this many terminal (done/error) jobs; evict oldest by finished_unix.
_MAX_COMPLETED_JOBS = 50


def _nccl_topo_host_path() -> str:
    return os.environ.get("RDMA_TOPO_PATH", "/tmp/nccl_topo.xml")


def _dev_flags() -> list[str]:
    try:
        names = os.listdir("/dev/infiniband")
    except FileNotFoundError:
        return []
    devs = sorted(
        os.path.join("/dev/infiniband", f) for f in names if f.startswith("uverbs")
    )
    return [f"--device={d}" for d in devs]


def _docker_run_base(body: dict[str, Any]) -> tuple[list[str], str]:
    """Shared `docker run` prefix: runtime, GPUs, IB devices, ulimit, shm, network, optional topo bind for runsc-rdma.

    No `--cpus`: containers use whatever CPU set Docker’s cgroup allows (avoids nproc vs docker.info mismatch).
    """
    runtime = str(body.get("runtime", "runc"))
    topo_host = str(body.get("topo_host_path", "/tmp/nccl_topo.xml"))
    parts: list[str] = [
        "sudo",
        "docker",
        "run",
        "--rm",
        f"--runtime={runtime}",
        "--gpus",
        "all",
    ]
    parts.extend(_dev_flags())
    parts.extend(
        [
            "--ulimit",
            "memlock=-1:-1",
            "--shm-size=1g",
            "--network=host",
        ]
    )
    if runtime == "runsc-rdma":
        parts.extend(["-v", f"{topo_host}:/topo.xml:ro"])
    return parts, runtime


def build_nccl_command(body: dict[str, Any]) -> list[str]:
    parts, runtime = _docker_run_base(body)
    rank = int(body["rank"])
    nranks = int(body["nranks"])
    ngpus = int(body.get("ngpus", 8))
    master_addr = str(body["master_addr"])
    master_port = int(body["master_port"])
    image = str(body.get("image", "nccl-test"))
    bench = str(body.get("bench_binary", "/usr/local/bin/nccl_multinode_bench"))
    env_extra = body.get("env", {}) or {}

    def e(name: str, value: str) -> None:
        parts.extend(["-e", f"{name}={value}"])

    e("RANK", str(rank))
    e("NRANKS", str(nranks))
    e("NGPUS", str(ngpus))
    e("MASTER_ADDR", master_addr)
    e("MASTER_PORT", str(master_port))
    for k, v in env_extra.items():
        e(str(k), str(v))
    if runtime == "runsc-rdma":
        e("NCCL_IB_GID_INDEX", "0")
        e("NCCL_TOPO_FILE", "/topo.xml")

    parts.extend([image, bench])
    return parts


def _merge_torch_defaults(body: dict[str, Any]) -> None:
    """Fill omitted torch fields so POST bodies can be minimal (master_addr only).

    Defaults match the usual gVisor RDMA two-node PyTorch bench. Override any field in
    the JSON body; use env on the agent host for image/runtime/NCCL (see README).
    """
    body.setdefault("master_port", 29541)
    body.setdefault(
        "runtime",
        os.environ.get("RDMA_JOB_RUNTIME", "runsc-rdma"),
    )
    body.setdefault("async", True)
    body.setdefault("nnodes", 2)
    body.setdefault("node_rank", 1)
    body.setdefault("nproc_per_node", 8)
    body.setdefault(
        "image",
        os.environ.get("RDMA_PYTORCH_IMAGE", "nvcr.io/nvidia/pytorch:24.07-py3"),
    )
    body.setdefault("script_host_path", "/tmp/torch_allreduce_bench.py")
    body.setdefault("topo_host_path", "/tmp/nccl_topo.xml")

    merged_env: dict[str, str] = {
        "NCCL_DEBUG": "WARN",
        "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "eth0"),
        "NCCL_NET_GDR_LEVEL": "3",
        "NCCL_DMABUF_ENABLE": "0",
    }
    ib = os.environ.get("NCCL_IB_HCA", "").strip()
    if ib:
        merged_env["NCCL_IB_HCA"] = ib

    user_env = body.get("env")
    if isinstance(user_env, dict):
        merged_env.update({str(k): str(v) for k, v in user_env.items()})
    body["env"] = merged_env


def build_torch_command(body: dict[str, Any]) -> list[str]:
    parts, runtime = _docker_run_base(body)
    nnodes = int(body["nnodes"])
    nproc = int(body.get("nproc_per_node", 8))
    node_rank = int(body["node_rank"])
    master_addr = str(body["master_addr"])
    master_port = int(body["master_port"])
    image = str(body.get("image", "nvcr.io/nvidia/pytorch:24.07-py3"))
    script = str(body.get("script_host_path", "/tmp/torch_allreduce_bench.py"))
    env_extra = body.get("env", {}) or {}

    for k, v in env_extra.items():
        parts.extend(["-e", f"{k}={v}"])
    if runtime == "runsc-rdma":
        parts.extend(["-e", "NCCL_IB_GID_INDEX=0", "-e", "NCCL_TOPO_FILE=/topo.xml"])

    parts.extend(
        [
            "-v",
            f"{script}:/tmp/torch_allreduce_bench.py:ro",
            image,
            "torchrun",
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc}",
            f"--node_rank={node_rank}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
            "/tmp/torch_allreduce_bench.py",
        ]
    )
    return parts


def _runsc_prep_logs() -> None:
    subprocess.run(["sudo", "rm", "-rf", "/tmp/runsc-rdma/logs"], check=False)
    subprocess.run(["sudo", "mkdir", "-p", "/tmp/runsc-rdma/logs"], check=True)


def _evict_completed_jobs_locked() -> None:
    assert _JOB_LOCK.locked()
    terminal = [
        jid
        for jid, j in _JOBS.items()
        if j.get("state") in ("done", "error")
    ]
    if len(terminal) <= _MAX_COMPLETED_JOBS:
        return
    terminal.sort(key=lambda jid: _JOBS[jid].get("finished_unix", 0))
    for jid in terminal[: len(terminal) - _MAX_COMPLETED_JOBS]:
        del _JOBS[jid]


def _job_public_view(job: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy; omit subprocess handle."""
    return {k: v for k, v in job.items() if k != "proc"}


def _run_job(job_id: str, argv: list[str], runtime: str) -> None:
    with _JOB_LOCK:
        _JOBS[job_id]["state"] = "running"
        _JOBS[job_id]["started_unix"] = time.time()

    if runtime == "runsc-rdma":
        try:
            _runsc_prep_logs()
        except (OSError, subprocess.CalledProcessError) as ex:
            with _JOB_LOCK:
                _JOBS[job_id]["state"] = "error"
                _JOBS[job_id]["error"] = str(ex)
                _JOBS[job_id]["finished_unix"] = time.time()
                _evict_completed_jobs_locked()
            return

    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as ex:
        with _JOB_LOCK:
            _JOBS[job_id]["state"] = "error"
            _JOBS[job_id]["error"] = str(ex)
            _JOBS[job_id]["finished_unix"] = time.time()
            _evict_completed_jobs_locked()
        return

    with _JOB_LOCK:
        _JOBS[job_id]["proc"] = proc

    out_chunks: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            out_chunks.append(line)
    code = proc.wait()
    out = "".join(out_chunks)

    with _JOB_LOCK:
        _JOBS[job_id].pop("proc", None)
        _JOBS[job_id]["state"] = "done"
        _JOBS[job_id]["exit_code"] = code
        _JOBS[job_id]["output"] = out
        _JOBS[job_id]["finished_unix"] = time.time()
        _evict_completed_jobs_locked()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _norm_path(self) -> str:
        """Path without query string; trailing slash stripped (except root)."""
        p = self.path.split("?", 1)[0].rstrip("/")
        return p if p else "/"

    def _json(self, code: int, obj: dict[str, Any]) -> None:
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:
        path = self._norm_path()
        if path == "/health":
            self._json(200, {"ok": True})
            return
        if path == "/v1/jobs":
            with _JOB_LOCK:
                brief = [
                    {"id": jid, "state": j.get("state"), "kind": j.get("kind")}
                    for jid, j in _JOBS.items()
                ]
            self._json(200, {"jobs": brief})
            return
        m = re.match(r"^/v1/jobs/([a-f0-9-]+)$", path)
        if m:
            jid = m.group(1)
            with _JOB_LOCK:
                raw = _JOBS.get(jid)
                snap = _job_public_view(dict(raw)) if raw else None
            if snap is None:
                self._json(404, {"error": "unknown job"})
                return
            self._json(200, snap)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self._norm_path()
        if path == "/v1/nccl_topo":
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0:
                self._json(400, {"error": "Content-Length required"})
                return
            if n > _MAX_NCCL_TOPO_BYTES:
                self._json(413, {"error": "body too large"})
                return
            data = self.rfile.read(n)
            if len(data) != n:
                self._json(400, {"error": "short read"})
                return
            topo_path = _nccl_topo_host_path()
            parent = os.path.dirname(topo_path)
            if parent:
                try:
                    os.makedirs(parent, exist_ok=True)
                except OSError as ex:
                    self._json(500, {"error": str(ex)})
                    return
            tmp = topo_path + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, topo_path)
            except OSError as ex:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                err = str(ex)
                if getattr(ex, "errno", None) in (errno.EACCES, errno.EPERM):
                    err += (
                        f" If {topo_path} already exists and is root-owned (e.g. created by "
                        f"sudo docker), run: sudo rm -f {topo_path}  or  sudo chown $USER {topo_path}"
                    )
                self._json(500, {"error": err})
                return
            self._json(200, {"ok": True, "path": topo_path, "bytes": len(data)})
            return

        m_cancel = re.match(r"^/v1/jobs/([a-f0-9-]+)/cancel$", path)
        if m_cancel:
            jid = m_cancel.group(1)
            with _JOB_LOCK:
                job = _JOBS.get(jid)
                if not job:
                    proc = None
                    st = None
                    missing = True
                else:
                    missing = False
                    proc = job.get("proc")
                    st = job.get("state")
            if missing:
                self._json(404, {"error": "unknown job"})
                return
            if st not in ("running", "queued"):
                self._json(400, {"error": f"job not cancelable (state={st})"})
                return
            if proc is None:
                self._json(
                    400,
                    {"error": "no process handle yet (retry shortly or job finished)"},
                )
                return
            proc.terminate()
            self._json(200, {"ok": True, "job_id": jid, "signal": "SIGTERM"})
            return

        if path != "/v1/jobs":
            self._json(404, {"error": "not found"})
            return
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return

        kind = body.get("kind")
        if kind not in ("nccl", "torch"):
            self._json(400, {"error": 'kind must be "nccl" or "torch"'})
            return

        if kind == "torch" and not str(body.get("master_addr", "")).strip():
            self._json(
                400,
                {
                    "error": "torch job requires non-empty master_addr (set NODE_A_IP on A before building POST body)",
                },
            )
            return

        try:
            if kind == "nccl":
                argv = build_nccl_command(body)
            else:
                _merge_torch_defaults(body)
                argv = build_torch_command(body)
        except (KeyError, TypeError, ValueError) as ex:
            self._json(400, {"error": str(ex)})
            return

        runtime = str(body.get("runtime", "runc"))
        async_flag = bool(body.get("async", False))
        job_id = str(uuid.uuid4())
        display = shlex.join(argv)
        with _JOB_LOCK:
            _JOBS[job_id] = {
                "id": job_id,
                "kind": kind,
                "state": "queued",
                "command": display,
            }

        def work() -> None:
            try:
                _run_job(job_id, argv, runtime)
            except Exception as ex:  # pylint: disable=broad-except
                with _JOB_LOCK:
                    j = _JOBS.get(job_id)
                    if j is not None:
                        j.pop("proc", None)
                        j["state"] = "error"
                        j["error"] = str(ex)
                        j["finished_unix"] = time.time()
                        _evict_completed_jobs_locked()

        if async_flag:
            threading.Thread(target=work, daemon=True).start()
            self._json(
                202,
                {
                    "job_id": job_id,
                    "async": True,
                    "command": display,
                },
            )
            return

        work()
        with _JOB_LOCK:
            raw = _JOBS.get(job_id)
            if not raw:
                self._json(500, {"error": "job disappeared"})
                return
            job = _job_public_view(dict(raw))
        if job.get("state") == "error":
            self._json(
                500,
                {
                    "job_id": job_id,
                    "error": job.get("error", "unknown error"),
                    "command": display,
                },
            )
            return
        self._json(
            200,
            {
                "job_id": job_id,
                "exit_code": job.get("exit_code"),
                "output": job.get("output", ""),
                "command": display,
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="RDMA job agent (spawn docker benchmarks)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (use 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8756)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"rdma_job_agent listening on http://{args.host}:{args.port}", file=sys.stderr)
    print(
        "POST /v1/jobs (JSON); POST /v1/nccl_topo; POST /v1/jobs/<id>/cancel; "
        "GET /v1/jobs; GET /v1/jobs/<id>; GET /health",
        file=sys.stderr,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
