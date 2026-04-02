#!/usr/bin/env python3
"""Parse nccl_multinode_bench output from stdin and update a comparison CSV."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

ROW_RE = re.compile(r"^\s*(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*$")
BASE_COLUMNS = ["size_bytes", "size_human"]
LABEL_ORDER = ["native", "runc", "gvisor"]

LABEL_ALIASES = {
    "native": "native",
    "runc": "runc",
    "gvisor": "gvisor",
    "runsc": "gvisor",
    "runsc-rdma": "gvisor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read nccl_multinode_bench output from stdin and update a CSV."
    )
    parser.add_argument(
        "--label",
        required=True,
        choices=sorted(LABEL_ALIASES),
        help="Which result set to update in the CSV.",
    )
    parser.add_argument(
        "--csv",
        default="nccl_allreduce_comparison.csv",
        help="Path to the comparison CSV to create or update.",
    )
    return parser.parse_args()


def size_human(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{int(size)}{unit}"
    if size.is_integer():
        return f"{int(size)}{unit}"
    return f"{size:.1f}{unit}"


def runtime_columns(label: str) -> list[str]:
    return [f"{label}_busbw_gbps", f"{label}_time_us"]


def labels_from_headers(fieldnames: list[str] | None) -> set[str]:
    labels: set[str] = set()
    if not fieldnames:
        return labels

    prefixes = tuple(f"{label}_" for label in LABEL_ORDER)
    for fieldname in fieldnames:
        if not fieldname.startswith(prefixes):
            continue
        prefix = fieldname.split("_", 1)[0]
        if prefix in LABEL_ORDER:
            labels.add(prefix)
    return labels


def ordered_labels(labels: set[str]) -> list[str]:
    return [label for label in LABEL_ORDER if label in labels]


def output_columns(labels: set[str]) -> list[str]:
    columns = list(BASE_COLUMNS)
    for label in ordered_labels(labels):
        columns.append(f"{label}_busbw_gbps")
    for label in ordered_labels(labels):
        columns.append(f"{label}_time_us")
    if {"gvisor", "runc"}.issubset(labels):
        columns.append("gvisor_vs_runc_ratio")
    if {"gvisor", "native"}.issubset(labels):
        columns.append("gvisor_vs_native_ratio")
    return columns


def load_rows(csv_path: pathlib.Path) -> tuple[dict[int, dict[str, str]], set[str]]:
    rows: dict[int, dict[str, str]] = {}
    labels: set[str] = set()
    if not csv_path.exists():
        return rows, labels

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        labels = labels_from_headers(reader.fieldnames)
        for raw_row in reader:
            size = int(raw_row["size_bytes"])
            row = {
                key: value
                for key, value in raw_row.items()
                if key is not None and value is not None
            }
            rows[size] = row
    return rows, labels


def format_ratio(numerator: str, denominator: str) -> str:
    if not numerator or not denominator:
        return ""
    denom = float(denominator)
    if denom == 0:
        return ""
    ratio = float(numerator) / denom
    return f"{ratio:.2f}x"


def write_rows(csv_path: pathlib.Path, rows: dict[int, dict[str, str]], labels: set[str]) -> None:
    columns = output_columns(labels)
    for row in rows.values():
        if "gvisor_vs_runc_ratio" in columns:
            row["gvisor_vs_runc_ratio"] = format_ratio(
                row.get("gvisor_time_us", ""), row.get("runc_time_us", "")
            )
        else:
            row.pop("gvisor_vs_runc_ratio", None)
        if "gvisor_vs_native_ratio" in columns:
            row["gvisor_vs_native_ratio"] = format_ratio(
                row.get("gvisor_time_us", ""), row.get("native_time_us", "")
            )
        else:
            row.pop("gvisor_vs_native_ratio", None)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for size in sorted(rows):
            writer.writerow({column: rows[size].get(column, "") for column in columns})


def main() -> int:
    args = parse_args()
    label = LABEL_ALIASES[args.label]
    csv_path = pathlib.Path(args.csv)
    rows, labels = load_rows(csv_path)
    labels.add(label)

    parsed_rows = 0
    for line in sys.stdin:
        sys.stdout.write(line)
        match = ROW_RE.match(line)
        if match is None:
            continue

        size_bytes = int(match.group(1))
        time_us = match.group(2)
        busbw = match.group(4)

        row = rows.setdefault(size_bytes, {})
        row["size_bytes"] = str(size_bytes)
        row["size_human"] = size_human(size_bytes)
        row[f"{label}_time_us"] = time_us
        row[f"{label}_busbw_gbps"] = busbw
        parsed_rows += 1

    if parsed_rows == 0:
        print("no NCCL benchmark rows found on stdin", file=sys.stderr)
        return 1

    write_rows(csv_path, rows, labels)
    print(
        f"updated {csv_path} with {parsed_rows} rows for {label}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
