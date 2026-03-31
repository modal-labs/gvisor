#!/usr/bin/env python3
# Copyright 2025 The gVisor Authors.

"""Tests for gvisor_strace_to_chrome_trace.py (stdlib unittest)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import gvisor_strace_to_chrome_trace as conv


class TestConvert(unittest.TestCase):
    def test_ioctl_pair(self) -> None:
        lines = [
            "I0331 12:00:00.000001   12345 strace.go:596] t1 E ioctl(9)",
            "I0331 12:00:00.000101   12345 strace.go:596] t1 X ioctl(9) = 0 (0x0) (100µs)",
        ]
        d = conv.convert_lines(lines, 2026)
        ev = d["traceEvents"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["name"], "ioctl")
        self.assertEqual(ev[0]["ph"], "X")
        self.assertEqual(ev[0]["dur"], 100.0)

    def test_sample_file(self) -> None:
        p = Path(__file__).resolve().parent / "testdata" / "strace_sample.log"
        text = p.read_text(encoding="utf-8")
        d = conv.convert_lines(text.splitlines(keepends=True), 2026)
        ev = d["traceEvents"]
        self.assertEqual(len(ev), 3, [e["name"] for e in ev])
        names = [e["name"] for e in ev]
        self.assertEqual(names, ["ioctl", "read", "futex"])  # worker-1 x2, worker-2 x1

    def test_json_log_line(self) -> None:
        jl = json.dumps(
            {
                "msg": "strace.go:596] t2 E openat(9)",
                "level": "info",
                "time": "2026-03-31T12:00:00.000001Z",
            }
        )
        jl2 = json.dumps(
            {
                "msg": "strace.go:612] t2 X openat(9) = 3 (0x3) (5µs)",
                "level": "info",
                "time": "2026-03-31T12:00:00.000010Z",
            }
        )
        d = conv.convert_lines([jl + "\n", jl2 + "\n"], 2026)
        self.assertEqual(len(d["traceEvents"]), 1)
        self.assertEqual(d["traceEvents"][0]["name"], "openat")


if __name__ == "__main__":
    unittest.main()
