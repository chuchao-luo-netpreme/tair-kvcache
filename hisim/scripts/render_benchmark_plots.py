#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from bench_plots.analyze_data import X_AXIS_CONFIG, load_rows
from bench_plots.plot import render_charts
from bench_plots.presets import CHART_REGISTRY, DEFAULT_CHARTS, chart_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render selected benchmark metric charts from parsed run data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("random_bench_metrics"),
        help="Directory containing metrics JSON files or runs/*/metrics.json.",
    )
    parser.add_argument(
        "--x-axis",
        choices=sorted(X_AXIS_CONFIG),
        default="dram-size",
        help="Experiment parameter to use as the x-axis.",
    )
    parser.add_argument(
        "--charts",
        default=",".join(DEFAULT_CHARTS),
        help="Comma-separated chart names. Use --list-charts to inspect choices.",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument(
        "--list-charts",
        action="store_true",
        help="Print available chart names and exit.",
    )
    return parser.parse_args()


def parse_chart_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise SystemExit("No charts selected")

    unknown = [name for name in names if name not in CHART_REGISTRY]
    if unknown:
        known = ", ".join(sorted(CHART_REGISTRY))
        raise SystemExit(f"Unknown chart(s): {', '.join(unknown)}. Known charts: {known}")

    return names


def default_output(input_dir: Path, x_axis: str, chart_names: list[str]) -> Path:
    x_name = x_axis.replace("-", "_")
    if tuple(chart_names) == DEFAULT_CHARTS:
        return input_dir / f"modular_{X_AXIS_CONFIG[x_axis]['output_name']}"
    return input_dir / f"{'_'.join(chart_names)}_vs_{x_name}.png"


def main() -> None:
    args = parse_args()
    if args.list_charts:
        for name in sorted(CHART_REGISTRY):
            print(name)
        return

    chart_names = parse_chart_names(args.charts)
    rows = load_rows(args.input_dir, args.x_axis)
    charts = chart_specs(chart_names)
    output = args.output or default_output(args.input_dir, args.x_axis, chart_names)
    render_charts(rows, args.x_axis, charts, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
