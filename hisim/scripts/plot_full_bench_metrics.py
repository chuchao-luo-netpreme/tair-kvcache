#!/usr/bin/env python3
"""Plot random benchmark metrics against a selected DRAM variable."""

from __future__ import annotations

import argparse
from pathlib import Path

from bench_plots.analyze_data import X_AXIS_CONFIG, Row, load_rows
from bench_plots.plot import render_charts
from bench_plots.presets import DEFAULT_CHARTS, chart_specs


RANDOM_BENCH_X_AXES = ("dram-bandwidth", "dram-size")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot TTFT, TPOT, ITL, prefix cache reuse, throughput, and "
            "iteration timing against DRAM size or DRAM bandwidth."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("random_bench_metrics"),
        help=(
            "Benchmark output directory. The script first reads "
            "runs/<run-name>/metrics.json, then falls back to top-level "
            "DramSize*gB*.json files."
        ),
    )
    parser.add_argument(
        "--x-axis",
        choices=sorted(RANDOM_BENCH_X_AXES),
        default="dram-size",
        help="Independent variable to parse from benchmark file names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to a PNG under --input-dir.",
    )
    return parser.parse_args()


def plot(rows: list[Row], x_axis: str, output: Path) -> None:
    render_charts(
        rows,
        x_axis,
        chart_specs(DEFAULT_CHARTS),
        output,
        title="Random Benchmark Metrics",
    )


def main() -> None:
    args = parse_args()
    output = args.output or args.input_dir / X_AXIS_CONFIG[args.x_axis]["output_name"]
    rows = load_rows(args.input_dir, args.x_axis)
    plot(rows, args.x_axis, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
