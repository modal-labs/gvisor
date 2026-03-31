#!/usr/bin/env python3
# Copyright 2025 The gVisor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Convert gVisor `--strace` log lines (Google glog text or JSON lines) into
# Chrome Trace Event Format (JSON) for timeline viewers:
#
#   - Chrome / Perfetto: chrome://tracing or https://ui.perfetto.dev
#   - Eclipse Trace Compass: import as Chrome / JSON trace (depends on build;
#     org.eclipse.tracecompass.jsontrace supports JSON traces)
#
# Log formats (see pkg/log/glog.go and pkg/sentry/strace/strace.go):
#   Text: I0331 15:04:05.123456   12345 strace.go:596] taskname E ioctl(...)
#   Exit: ... taskname X ioctl(...) = 123 (0x7b) (1.234µs)
#   JSON: one JSON object per line with "msg" and RFC3339 "time" (pkg/log/json.go)
#
# Usage:
#   python3 tools/gvisor_strace_to_chrome_trace.py /tmp/runsc-rdma/logs/runsc.log.0 \
#     -o strace.json
#   # Then open strace.json in https://ui.perfetto.dev (Open trace file)

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from typing import Any

# glog: Lmmdd hh:mm:ss.uuuuuu threadid file:line] message
_GLOG_HEAD = re.compile(
    r"^[IWED](?P<mm>\d{2})(?P<dd>\d{2})\s+"
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<us>\d{6})\s+"
    r"\d+\s+[^:]+:\d+\]\s+(?P<msg>.*)$"
)

# Message body: taskname E syscall(args)  /  taskname X syscall(args) = ret
# Split E vs X so ret may contain parentheses.
_STRACE_PREFIX = r"(?:\[\s*\d+:\s*\d+\]\s+)?"
_STRACE_E = re.compile(
    rf"^{_STRACE_PREFIX}(?P<task>\S+)\s+E\s+(?P<sc>[a-zA-Z0-9_]+)\((?P<args>.*)\)\s*$"
)
_STRACE_X = re.compile(
    rf"^{_STRACE_PREFIX}(?P<task>\S+)\s+X\s+(?P<sc>[a-zA-Z0-9_]+)\((?P<args>.*)\)\s*=\s*(?P<ret>.+)$"
)

def _extract_duration_from_ret(ret: str) -> int | None:
    """Return duration in microseconds from exit rval string."""
    if not ret:
        return None
    # Last parenthetical often is elapsed, e.g. ... (1.234µs)
    parts = re.findall(r"\(([^)]*)\)", ret)
    if not parts:
        return None
    last = parts[-1].strip()
    # Could be hex addr — skip if looks like 0x
    if last.lower().startswith("0x"):
        if len(parts) >= 2:
            last = parts[-2].strip()
        else:
            return None
    # Try full Go duration string (may include spaces)
    ns = _parse_simple_go_duration(last)
    if ns is not None:
        # strace prints elapsed as Go duration; convert ns -> µs for Chrome dur field
        return max(1, ns // 1000)
    return None


def _parse_simple_go_duration(s: str) -> int | None:
    """Return duration in nanoseconds."""
    s = s.strip()
    if not s:
        return None
    total = 0
    pat = re.compile(r"(\d+(?:\.\d+)?)(ns|µs|us|μs|ms|s|m|h)")
    for m in pat.finditer(s):
        val = float(m.group(1))
        u = m.group(2).lower().replace("μ", "µ")
        mult = {
            "ns": 1,
            "µs": 1000,
            "us": 1000,
            "ms": 1_000_000,
            "s": 1_000_000_000,
            "m": 60_000_000_000,
            "h": 3600_000_000_000,
        }.get(u)
        if mult:
            total += int(val * mult)
        pos = m.end()
    return total if total > 0 else None


def _parse_glog_ts(line: str, year: int) -> tuple[datetime.datetime | None, str]:
    m = _GLOG_HEAD.match(line.rstrip("\n"))
    if not m:
        return None, line
    mm = int(m.group("mm"))
    dd = int(m.group("dd"))
    h, mi, sec = int(m.group("h")), int(m.group("m")), int(m.group("s"))
    us = int(m.group("us"))
    try:
        dt = datetime.datetime(year, mm, dd, h, mi, sec, us, tzinfo=datetime.timezone.utc)
    except ValueError:
        return None, m.group("msg")
    return dt, m.group("msg")


def _parse_json_line(line: str) -> tuple[datetime.datetime | None, str] | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        o = json.loads(line)
    except json.JSONDecodeError:
        return None
    msg = o.get("msg")
    if not isinstance(msg, str):
        return None
    t = o.get("time")
    dt: datetime.datetime | None = None
    if isinstance(t, str):
        try:
            dt = datetime.datetime.fromisoformat(t.replace("Z", "+00:00"))
        except ValueError:
            dt = None
    # Strip file:line] prefix from msg (json emitter)
    msg_body = msg
    br = msg.find("] ")
    if br != -1 and ":" in msg[:br]:
        msg_body = msg[br + 2 :]
    return (dt, msg_body)


def _ts_us(dt: datetime.datetime | None, base: list[float]) -> float:
    """Microseconds since trace start; monotonic fallback."""
    if dt is None:
        base[0] += 0.001  # 1ns synthetic step
        return base[0]
    epoch_us = dt.timestamp() * 1e6
    if base[1] < 0:
        base[1] = epoch_us
    return epoch_us - base[1]


def _tid(task: str) -> int:
    h = 0
    for c in task:
        h = (h * 31 + ord(c)) & 0x7FFFFFFF
    return h or 1


def convert_lines(
    lines: list[str],
    year: int,
) -> dict[str, Any]:
    """Return Chrome trace dict with traceEvents."""
    base = [0.0, -1.0]  # monotonic, first_epoch_us
    events: list[dict[str, Any]] = []
    stacks: dict[str, list[tuple[str, float]]] = {}

    def process_msg(msg: str, dt: datetime.datetime | None) -> None:
        msg = msg.rstrip("\n")
        ts = _ts_us(dt, base)

        m_e = _STRACE_E.match(msg)
        if m_e:
            task = m_e.group("task")
            sc = m_e.group("sc")
            stacks.setdefault(task, []).append((sc, ts))
            return

        m_x = _STRACE_X.match(msg)
        if not m_x:
            return
        task = m_x.group("task")
        sc = m_x.group("sc")
        args = m_x.group("args")
        ret = m_x.group("ret")

        st = stacks.get(task)
        if not st:
            return
        if st[-1][0] != sc:
            while st and st[-1][0] != sc:
                st.pop()
            if not st:
                return
        _, t_enter = st.pop()
        dur = max(0.0, ts - t_enter)
        d = _extract_duration_from_ret(ret)
        if d is not None and d > 0:
            dur = float(d)

        tid = _tid(task)
        events.append(
            {
                "name": sc,
                "cat": "strace",
                "ph": "X",
                "ts": t_enter,
                "dur": dur,
                "pid": 1,
                "tid": tid,
                "args": {
                    "task": task,
                    "args": args[:512] + ("..." if len(args) > 512 else ""),
                    "ret": ret[:512],
                },
            }
        )

    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        j = _parse_json_line(line)
        if j is not None:
            dt, msg = j
            process_msg(msg, dt)
            continue
        dt, msg = _parse_glog_ts(line, year)
        if dt is not None:
            process_msg(msg, dt)
        else:
            # Raw strace line without glog header
            process_msg(line, None)

    return {
        "traceEvents": events,
        "metadata": {
            "source": "gvisor_strace_to_chrome_trace.py",
            "note": "ts and dur use Chrome convention (microseconds).",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert gVisor strace logs to Chrome Trace JSON (Perfetto / Trace Compass)."
    )
    ap.add_argument("input", type=argparse.FileType("r"), help="runsc log file (glog text or JSON lines)")
    ap.add_argument("-o", "--output", required=True, help="output .json path")
    ap.add_argument(
        "--year",
        type=int,
        default=datetime.datetime.now(datetime.timezone.utc).year,
        help="year for glog mmdd (default: current UTC year)",
    )
    args = ap.parse_args()
    try:
        data = convert_lines(args.input.readlines(), args.year)
    finally:
        args.input.close()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    n = len(data["traceEvents"])
    print(f"wrote {n} complete events to {args.output}", file=sys.stderr)
    print("Open in https://ui.perfetto.dev (Open trace file)", file=sys.stderr)


if __name__ == "__main__":
    main()
