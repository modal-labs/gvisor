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

"""Long-running HTTP agent to spawn NCCL / torchrun multi-node jobs on this node.

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
import pathlib
import pwd
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


def _default_topo_host_path() -> str:
    """Default NCCL topology path on the agent host.

    Uses ~/.cache/... instead of /tmp so uploads are not blocked by sticky-bit
    /tmp when a root-owned /tmp/nccl_topo.xml exists (e.g. from sudo docker).
    """
    return os.path.join(
        os.path.expanduser("~"), ".cache", "rdma_job_agent", "nccl_topo.xml"
    )


def _nccl_topo_host_path() -> str:
    return os.environ.get("RDMA_TOPO_PATH", _default_topo_host_path())


def _atomic_write_nccl_topo(topo_path: str, data: bytes) -> None:
    """Write data to topo_path via temp file + replace."""
    parent = os.path.dirname(topo_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = topo_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, topo_path)


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

    No `--cpus`: containers use whatever CPU set Docker’s cgroup allows
    (avoids nproc vs docker.info mismatch). Callers may optionally set
    `cgroup_parent` in the job body or `RDMA_JOB_CGROUP_PARENT` in the
    agent environment to place the container under a specific systemd slice,
    and may set `cpuset_cpus` to further restrict the visible CPUs.
    """
    runtime = str(body.get("runtime", "runc"))
    topo_host = str(body.get("topo_host_path", _default_topo_host_path()))
    cgroup_parent = str(
        body.get("cgroup_parent", os.environ.get("RDMA_JOB_CGROUP_PARENT", ""))
    ).strip()
    cpuset_cpus = str(body.get("cpuset_cpus", "")).strip()
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
    if cgroup_parent:
        parts.append(f"--cgroup-parent={cgroup_parent}")
    if cpuset_cpus:
        parts.append(f"--cpuset-cpus={cpuset_cpus}")
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


def build_train_command(body: dict[str, Any]) -> list[str]:
    """Build a `docker run … torchrun` command for multi-node DDP training.

    The agent always runs node_rank=1; the POSTing node runs node_rank=0.
    master_addr is the POSTing node's IPv4 address.
    """
    parts, runtime = _docker_run_base(body)
    master_addr = str(body["master_addr"])
    master_port = int(body.get("master_port", 29500))
    nproc_per_node = int(body.get("nproc_per_node", 8))
    node_rank = int(body.get("node_rank", 1))
    nnodes = int(body.get("nnodes", 2))
    image = str(body.get("image", "nvcr.io/nvidia/pytorch:26.03-py3"))
    script_host_path = str(body["script_host_path"])
    env_extra = body.get("env", {}) or {}

    for k, v in env_extra.items():
        parts.extend(["-e", f"{k}={v}"])
    if runtime == "runsc-rdma":
        parts.extend(["-e", "NCCL_IB_GID_INDEX=0", "-e", "NCCL_TOPO_FILE=/topo.xml"])

    parts.extend([
        "-v", f"{script_host_path}:/tmp/train_script.py:ro",
        image,
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--nnodes={nnodes}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        f"--node_rank={node_rank}",
        "/tmp/train_script.py",
    ])
    return parts


def _runsc_prep_logs() -> None:
    subprocess.run(["sudo", "rm", "-rf", "/tmp/runsc-rdma/logs"], check=False)
    subprocess.run(["sudo", "mkdir", "-p", "/tmp/runsc-rdma/logs"], check=True)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _run_capture(
    argv: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None
) -> dict[str, Any]:
    proc = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {
        "command": shlex.join(argv),
        "cwd": cwd,
        "exit_code": proc.returncode,
        "output": proc.stdout,
    }


def _git_pull_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_SSH_COMMAND", "ssh -o StrictHostKeyChecking=accept-new")

    sudo_user = os.environ.get("SUDO_USER", "").strip()
    if sudo_user:
        try:
            pw = pwd.getpwnam(sudo_user)
        except KeyError:
            return env
        env["HOME"] = pw.pw_dir
        env["USER"] = sudo_user
        env["LOGNAME"] = sudo_user
    return env


def _configure_runsc_docker_runtime() -> dict[str, Any]:
    daemon_json = pathlib.Path("/etc/docker/daemon.json")
    raw = daemon_json.read_text().strip() if daemon_json.exists() else ""
    cfg = json.loads(raw) if raw else {}
    cfg.setdefault("runtimes", {})["runsc-rdma"] = {
        "path": "/usr/local/bin/runsc-rdma",
        "runtimeArgs": [
            "--debug",
            "--debug-log=/tmp/runsc-rdma/logs/",
            "--strace",
            "--rdmaproxy",
            "--nvproxy",
            "--nvproxy-allowed-driver-capabilities=compute,utility,video",
            "--network=host",
            "--rdma-expected-ipoib=-1",
        ],
    }
    daemon_json.write_text(json.dumps(cfg, indent=2) + "\n")
    return {
        "command": "configure docker runtime runsc-rdma",
        "cwd": None,
        "exit_code": 0,
        "output": f"updated {daemon_json}\n",
    }


def _deploy_runsc_runtime(progress_cb: Any | None = None) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise PermissionError("deploy_runsc requires the agent to run as root (start agent.py with sudo)")

    repo_root = _repo_root()
    steps: list[dict[str, Any]] = []
    git_env = _git_pull_env()
    cmds: list[tuple[list[str], str | None, dict[str, str] | None]] = [
        (["git", "fetch", "origin"], repo_root, git_env),
        (["git", "reset", "--hard", "origin/alessio/development"], repo_root, git_env),
        (["make", "copy", "TARGETS=runsc", "DESTINATION=/tmp"], repo_root, None),
        (["rm", "-f", "/usr/local/bin/runsc-rdma"], None, None),
        (["cp", "/tmp/runsc", "/usr/local/bin/runsc-rdma"], None, None),
        (["chmod", "+x", "/usr/local/bin/runsc-rdma"], None, None),
        (["systemctl", "restart", "docker"], None, None),
        (["modprobe", "nvidia-peermem"], None, None),
        (["/usr/local/bin/runsc-rdma", "--version"], None, None),
    ]
    for argv, cwd, env in cmds:
        if argv == ["systemctl", "restart", "docker"]:
            step = _configure_runsc_docker_runtime()
            steps.append(step)
            if progress_cb is not None:
                progress_cb(
                    {
                        "current_step": step["command"],
                        "output": "".join(
                            f"$ {prior['command']}\n{prior['output']}\n" for prior in steps
                        ),
                    }
                )
        if progress_cb is not None:
            progress_cb(
                {
                    "current_step": shlex.join(argv),
                    "output": "".join(
                        f"$ {step['command']}\n{step['output']}\n" for step in steps
                    ),
                }
            )
        step = _run_capture(argv, cwd=cwd, env=env)
        steps.append(step)
        if step["exit_code"] != 0:
            return {
                "ok": False,
                "repo_root": repo_root,
                "failed_command": step["command"],
                "steps": steps,
            }
    return {"ok": True, "repo_root": repo_root, "steps": steps}


def _build_nccl_test_image(progress_cb: Any | None = None) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise PermissionError("build_nccl_test requires the agent to run as root (start agent.py with sudo)")

    repo_root = _repo_root()
    steps: list[dict[str, Any]] = []
    cmds: list[tuple[list[str], str | None, dict[str, str] | None]] = [
        (["docker", "build", "-f", "Dockerfile.nccl", "-t", "nccl-test", "."], repo_root, None),
        (["docker", "image", "inspect", "nccl-test"], None, None),
    ]
    for argv, cwd, env in cmds:
        if progress_cb is not None:
            progress_cb(
                {
                    "current_step": shlex.join(argv),
                    "output": "".join(
                        f"$ {step['command']}\n{step['output']}\n" for step in steps
                    ),
                }
            )
        step = _run_capture(argv, cwd=cwd, env=env)
        steps.append(step)
        if step["exit_code"] != 0:
            return {
                "ok": False,
                "repo_root": repo_root,
                "failed_command": step["command"],
                "steps": steps,
            }
    return {"ok": True, "repo_root": repo_root, "steps": steps}


def _run_deploy_job(job_id: str) -> None:
    def progress(update: dict[str, Any]) -> None:
        with _JOB_LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                return
            job.update(update)

    with _JOB_LOCK:
        _JOBS[job_id]["state"] = "running"
        _JOBS[job_id]["started_unix"] = time.time()

    try:
        result = _deploy_runsc_runtime(progress_cb=progress)
    except Exception as ex:  # pylint: disable=broad-except
        with _JOB_LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["state"] = "error"
                job["error"] = str(ex)
                job["finished_unix"] = time.time()
                _evict_completed_jobs_locked()
        return

    output = "".join(f"$ {step['command']}\n{step['output']}\n" for step in result["steps"])
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["repo_root"] = result["repo_root"]
        job["steps"] = result["steps"]
        job["output"] = output
        job["finished_unix"] = time.time()
        job.pop("current_step", None)
        if result.get("ok"):
            job["state"] = "done"
            job["exit_code"] = 0
        else:
            job["state"] = "error"
            job["exit_code"] = 1
            job["error"] = result.get("failed_command", "deploy_runsc failed")
            job["failed_command"] = result.get("failed_command")
        _evict_completed_jobs_locked()


def _run_build_nccl_test_job(job_id: str) -> None:
    def progress(update: dict[str, Any]) -> None:
        with _JOB_LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                return
            job.update(update)

    with _JOB_LOCK:
        _JOBS[job_id]["state"] = "running"
        _JOBS[job_id]["started_unix"] = time.time()

    try:
        result = _build_nccl_test_image(progress_cb=progress)
    except Exception as ex:  # pylint: disable=broad-except
        with _JOB_LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["state"] = "error"
                job["error"] = str(ex)
                job["finished_unix"] = time.time()
                _evict_completed_jobs_locked()
        return

    output = "".join(f"$ {step['command']}\n{step['output']}\n" for step in result["steps"])
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["repo_root"] = result["repo_root"]
        job["steps"] = result["steps"]
        job["output"] = output
        job["finished_unix"] = time.time()
        job.pop("current_step", None)
        if result.get("ok"):
            job["state"] = "done"
            job["exit_code"] = 0
        else:
            job["state"] = "error"
            job["exit_code"] = 1
            job["error"] = result.get("failed_command", "build_nccl_test failed")
            job["failed_command"] = result.get("failed_command")
        _evict_completed_jobs_locked()


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
        m_bad = re.match(r"^/v1/jobs/([^/]+)$", path)
        if m_bad and m_bad.group(1) in ("null", "undefined"):
            self._json(
                400,
                {
                    "error": "invalid job id path",
                    "hint": (
                        "POST /v1/jobs did not return job_id; jq -r .job_id became null. "
                        "curl POST with same body and jq . for error. On A: export NODE_A_IP "
                        "before building POST_BODY."
                    ),
                },
            )
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
            primary = _nccl_topo_host_path()
            fallback = _default_topo_host_path()
            topo_path = primary
            used_fallback = False
            try:
                _atomic_write_nccl_topo(topo_path, data)
            except OSError as ex:
                try:
                    tmp = topo_path + ".tmp"
                    os.unlink(tmp)
                except OSError:
                    pass
                if (
                    getattr(ex, "errno", None) in (errno.EACCES, errno.EPERM)
                    and os.path.abspath(fallback) != os.path.abspath(primary)
                ):
                    try:
                        _atomic_write_nccl_topo(fallback, data)
                        topo_path = fallback
                        used_fallback = True
                    except OSError as ex2:
                        err = str(ex2)
                        if getattr(ex2, "errno", None) in (errno.EACCES, errno.EPERM):
                            err += (
                                f" If {fallback} is not writable, or {primary} is root-owned "
                                f"(e.g. sudo docker), run: sudo rm -f {primary} {primary}.tmp"
                            )
                        self._json(500, {"error": err})
                        return
                else:
                    err = str(ex)
                    if getattr(ex, "errno", None) in (errno.EACCES, errno.EPERM):
                        err += (
                            f" If {topo_path} already exists and is root-owned (e.g. created by "
                            f"sudo docker), run: sudo rm -f {topo_path}  or  sudo chown $USER {topo_path}"
                        )
                    self._json(500, {"error": err})
                    return
            body: dict[str, Any] = {"ok": True, "path": topo_path, "bytes": len(data)}
            if used_fallback:
                body["used_fallback_from"] = primary
                body["note"] = (
                    "Wrote to default cache path because RDMA_TOPO_PATH (or primary) was not "
                    "writable; runsc-rdma jobs use topo_host_path default matching this path."
                )
            self._json(200, body)
            return

        if path == "/v1/admin/deploy_runsc":
            try:
                body = self._read_json()
            except json.JSONDecodeError:
                body = {}
            async_flag = bool(body.get("async", True))
            job_id = str(uuid.uuid4())
            with _JOB_LOCK:
                _JOBS[job_id] = {
                    "id": job_id,
                    "kind": "deploy_runsc",
                    "state": "queued",
                    "command": "deploy_runsc",
                }
            if async_flag:
                def work() -> None:
                    _run_deploy_job(job_id)

                threading.Thread(target=work, daemon=True).start()
                self._json(202, {"job_id": job_id, "async": True, "kind": "deploy_runsc"})
                return
            try:
                _run_deploy_job(job_id)
            except PermissionError as ex:
                self._json(403, {"error": str(ex)})
                return
            with _JOB_LOCK:
                raw = _JOBS.get(job_id)
                result = _job_public_view(dict(raw)) if raw else None
            if not result:
                self._json(500, {"error": "deploy job disappeared"})
                return
            code = 200 if result.get("state") == "done" else 500
            self._json(code, result)
            return

        if path == "/v1/admin/build_nccl_test":
            try:
                body = self._read_json()
            except json.JSONDecodeError:
                body = {}
            async_flag = bool(body.get("async", True))
            job_id = str(uuid.uuid4())
            with _JOB_LOCK:
                _JOBS[job_id] = {
                    "id": job_id,
                    "kind": "build_nccl_test",
                    "state": "queued",
                    "command": "build_nccl_test",
                }
            if async_flag:
                def work() -> None:
                    _run_build_nccl_test_job(job_id)

                threading.Thread(target=work, daemon=True).start()
                self._json(202, {"job_id": job_id, "async": True, "kind": "build_nccl_test"})
                return
            try:
                _run_build_nccl_test_job(job_id)
            except PermissionError as ex:
                self._json(403, {"error": str(ex)})
                return
            with _JOB_LOCK:
                raw = _JOBS.get(job_id)
                result = _job_public_view(dict(raw)) if raw else None
            if not result:
                self._json(500, {"error": "build job disappeared"})
                return
            code = 200 if result.get("state") == "done" else 500
            self._json(code, result)
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
        if kind not in ("nccl", "train"):
            self._json(400, {"error": 'kind must be "nccl" or "train"'})
            return

        if kind == "train":
            if not str(body.get("master_addr", "")).strip():
                self._json(400, {"error": "train job requires master_addr (the POSTing node's IPv4)"})
                return
            if not str(body.get("script_host_path", "")).strip():
                self._json(400, {"error": "train job requires script_host_path (host path to the training .py)"})
                return

        try:
            if kind == "nccl":
                argv = build_nccl_command(body)
            else:
                argv = build_train_command(body)
        except (KeyError, TypeError, ValueError) as ex:
            self._json(400, {"error": str(ex)})
            return

        runtime = str(body.get("runtime", "runc"))
        async_flag = bool(body.get("async", True))
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
        "POST /v1/jobs (JSON); POST /v1/nccl_topo; POST /v1/admin/deploy_runsc; "
        "POST /v1/admin/build_nccl_test; "
        "POST /v1/jobs/<id>/cancel; "
        "GET /v1/jobs; GET /v1/jobs/<id>; GET /health",
        file=sys.stderr,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
