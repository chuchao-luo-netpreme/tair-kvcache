#!/usr/bin/env python3
"""Report L2-load transfer-vs-fragmentation summary."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DRAM_BW_RE = re.compile(r"DramBw(\d+(?:\.\d+)?)gB", re.IGNORECASE)
NUMERIC_TEXT_RE = re.compile(r"-?\d+(?:\.\d+)?%?")


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    iteration_count: int
    l2_load_iteration_count: int
    l2_load_latency_s: float
    ttft_excluding_queue_total_s: float | None
    l2_load_pct_ttft_excluding_queue: float | None
    transfer_time_s: float
    fragmentation_overhead_s: float
    mean_iter_transfer_pct: float
    mean_iter_fragmentation_overhead_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For iteration.jsonl files, compute how much L2-load latency comes "
            "from bytes/bandwidth transfer time versus segment fragmentation "
            "overhead."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("codex-bench-bw"),
        help="Benchmark directory containing runs/<run-name>/iteration.jsonl.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        nargs="+",
        default=[64.0, 512.0],
        help="DRAM bandwidth values parsed from run names. Defaults to 64 512.",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Explicit iteration.jsonl paths or run directories. If set, "
            "--bandwidth is ignored."
        ),
    )
    return parser.parse_args()


def pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator * 100.0


def optional_pct(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator * 100.0


def parse_bandwidth(run_name: str) -> float | None:
    match = DRAM_BW_RE.search(run_name)
    return float(match.group(1)) if match else None


def resolve_iteration_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "iteration.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def find_iteration_paths(input_dir: Path, bandwidths: Iterable[float]) -> list[Path]:
    wanted = {float(value) for value in bandwidths}
    paths = []
    for path in sorted(input_dir.glob("runs/*/iteration.jsonl")):
        bandwidth = parse_bandwidth(path.parent.name)
        if bandwidth in wanted:
            paths.append(path)
    missing = wanted - {parse_bandwidth(path.parent.name) for path in paths}
    if missing:
        missing_text = ", ".join(f"{value:g}" for value in sorted(missing))
        raise FileNotFoundError(
            f"Did not find iteration.jsonl for bandwidth(s): {missing_text}"
        )
    return sorted(paths, key=lambda path: (parse_bandwidth(path.parent.name) or 0.0))


def load_ttft_excluding_queue_total_from_metrics(run_dir: Path) -> float | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        metrics = json.load(f)
    mean_ttft_excluding_queue_ms = metrics.get("mean_ttft_excluding_queue_ms")
    num_requests = metrics.get("num_requests")
    if mean_ttft_excluding_queue_ms is None or num_requests is None:
        return None
    mean_ttft_excluding_queue_ms = float(mean_ttft_excluding_queue_ms)
    if mean_ttft_excluding_queue_ms < 0:
        return None
    return mean_ttft_excluding_queue_ms / 1000.0 * float(num_requests)


def load_ttft_excluding_queue_total_from_requests(run_dir: Path) -> float | None:
    request_path = run_dir / "request.jsonl"
    if not request_path.exists():
        return None
    total = 0.0
    count = 0
    with request_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            latencies = row.get("gen_token_latencies") or []
            if not latencies:
                continue
            queue_duration = float(row.get("queue_end") or 0.0) - float(
                row.get("queue_start") or 0.0
            )
            total += float(latencies[0]) - queue_duration
            count += 1
    return total if count else None


def load_ttft_excluding_queue_total(run_dir: Path) -> float | None:
    return load_ttft_excluding_queue_total_from_metrics(
        run_dir
    ) or load_ttft_excluding_queue_total_from_requests(run_dir)


def load_summary(path: Path) -> RunSummary:
    run_name = path.parent.name
    iteration_count = 0
    l2_load_iteration_count = 0
    total_latency = 0.0
    total_transfer = 0.0
    total_overhead = 0.0
    transfer_pct_total = 0.0
    fragmentation_overhead_pct_total = 0.0

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iteration_count += 1

            transfer = float(row.get("l2_load_bytes_bw_latency") or 0.0)
            overhead = float(row.get("l2_load_segment_overhead_latency") or 0.0)
            latency = float(row.get("l2_load_latency") or (transfer + overhead))

            total_latency += latency
            total_transfer += transfer
            total_overhead += overhead

            if latency > 0:
                l2_load_iteration_count += 1
                transfer_pct_total += pct(transfer, latency)
                fragmentation_overhead_pct_total += pct(overhead, latency)

    ttft_excluding_queue_total = load_ttft_excluding_queue_total(path.parent)
    return RunSummary(
        run_name=run_name,
        iteration_count=iteration_count,
        l2_load_iteration_count=l2_load_iteration_count,
        l2_load_latency_s=total_latency,
        ttft_excluding_queue_total_s=ttft_excluding_queue_total,
        l2_load_pct_ttft_excluding_queue=optional_pct(
            total_latency,
            ttft_excluding_queue_total,
        ),
        transfer_time_s=total_transfer,
        fragmentation_overhead_s=total_overhead,
        mean_iter_transfer_pct=safe_mean(
            transfer_pct_total, l2_load_iteration_count
        ),
        mean_iter_fragmentation_overhead_pct=safe_mean(
            fragmentation_overhead_pct_total, l2_load_iteration_count
        ),
    )


def safe_mean(total: float, count: int) -> float:
    return total / count if count else 0.0


def is_numeric_text(value: str) -> bool:
    return NUMERIC_TEXT_RE.fullmatch(value) is not None


def print_aligned_csv(rows: list[list[str]], file=sys.stdout) -> None:
    if not rows:
        return

    column_count = max(len(row) for row in rows)
    widths = [
        max(len(row[index]) if index < len(row) else 0 for row in rows)
        for index in range(column_count)
    ]
    numeric_columns = [
        all(
            index < len(row) and is_numeric_text(row[index])
            for row in rows[1:]
            if index < len(row)
        )
        for index in range(column_count)
    ]

    for row_index, row in enumerate(rows):
        cells = []
        for index in range(column_count):
            value = row[index] if index < len(row) else ""
            if row_index > 0 and numeric_columns[index]:
                cells.append(value.rjust(widths[index]))
            else:
                cells.append(value.ljust(widths[index]))
        print(", ".join(cells), file=file)


def print_summary(summaries: list[RunSummary]) -> None:
    rows = [
        [
            "run",
            "count(iterations)",
            "count(l2_load_iterations)",
            "sum(l2_load_s)",
            "sum(ttft_excl_queue_s)",
            "sum(l2_load_s)/sum(ttft_excl_queue_s)",
            "sum(transfer_s)",
            "mean(iter_transfer_pct)",
            "sum(fragmentation_overhead_s)",
            "mean(iter_fragmentation_overhead_pct)",
        ]
    ]
    for row in summaries:
        rows.append(
            [
                row.run_name,
                str(row.iteration_count),
                str(row.l2_load_iteration_count),
                f"{row.l2_load_latency_s:.6f}",
                (
                    f"{row.ttft_excluding_queue_total_s:.6f}"
                    if row.ttft_excluding_queue_total_s is not None
                    else ""
                ),
                (
                    f"{row.l2_load_pct_ttft_excluding_queue:.2f}%"
                    if row.l2_load_pct_ttft_excluding_queue is not None
                    else ""
                ),
                f"{row.transfer_time_s:.6f}",
                f"{row.mean_iter_transfer_pct:.2f}%",
                f"{row.fragmentation_overhead_s:.6f}",
                f"{row.mean_iter_fragmentation_overhead_pct:.2f}%",
            ]
        )
    print_aligned_csv(rows)


def main() -> None:
    args = parse_args()
    if args.runs:
        paths = [resolve_iteration_path(path) for path in args.runs]
    else:
        paths = find_iteration_paths(args.input_dir, args.bandwidth)

    summaries = []
    for path in paths:
        summaries.append(load_summary(path))
    print_summary(summaries)


if __name__ == "__main__":
    main()
