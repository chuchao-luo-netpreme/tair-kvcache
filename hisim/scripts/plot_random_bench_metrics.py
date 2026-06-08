#!/usr/bin/env python3
"""Plot random benchmark metrics against a selected DRAM variable."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


DRAM_SIZE_RE = re.compile(r"DramSize(\d+)gB", re.IGNORECASE)
DRAM_BW_RE = re.compile(r"DramBw(\d+)gB", re.IGNORECASE)
X_AXIS_CONFIG = {
    "dram-size": {
        "regex": DRAM_SIZE_RE,
        "row_key": "dram_size_gb",
        "label": "DRAM size (GB)",
        "title": "DRAM Size",
        "output_name": "dram_size_metrics.png",
    },
    "dram-bandwidth": {
        "regex": DRAM_BW_RE,
        "row_key": "dram_bandwidth_gb",
        "label": "DRAM bandwidth (GB/s)",
        "title": "DRAM Bandwidth",
        "output_name": "dram_bandwidth_metrics.png",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot TTFT, TPOT, ITL, and prefix cache reuse against DRAM size "
            "or DRAM bandwidth."
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
        choices=sorted(X_AXIS_CONFIG),
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


def load_rows(input_dir: Path, x_axis: str) -> list[dict[str, float]]:
    x_config = X_AXIS_CONFIG[x_axis]
    rows: list[dict[str, float]] = []

    metrics_paths = sorted(input_dir.glob("runs/*/metrics.json"))
    source = "per-run metrics.json under runs/"
    if not metrics_paths:
        metrics_paths = sorted(input_dir.glob("*.json"))
        source = "top-level JSON"

    for path in metrics_paths:
        run_name = path.parent.name if path.name == "metrics.json" else path.name
        match = x_config["regex"].search(run_name)
        if not match:
            continue

        with path.open() as f:
            data = json.load(f)

        rows.append(
            {
                x_config["row_key"]: float(match.group(1)),
                "mean_ttft_s": data["mean_ttft_ms"] / 1000,
                "mean_tpot_ms": data["mean_tpot_ms"],
                "mean_itl_ms": data["mean_itl_ms"],
                "prefix_cache_reused_pct": data["prefix_cache_reused_ratio"] * 100,
            }
        )

    rows.sort(key=lambda row: row[x_config["row_key"]])
    if not rows:
        raise FileNotFoundError(
            f"No {source} files with {x_config['title']} in the run/file name "
            f"found in {input_dir}. Expected either "
            f"{input_dir}/runs/DramSize*gB_DramBw*gB_*/metrics.json "
            f"or {input_dir}/DramSize*gB_DramBw*gB_*.json."
        )
    return rows


def plot(rows: list[dict[str, float]], x_axis: str, output: Path) -> None:
    x_config = X_AXIS_CONFIG[x_axis]
    x_values = [row[x_config["row_key"]] for row in rows]
    charts = [
        ("mean_ttft_s", "Mean TTFT", "seconds", "{:.2f}"),
        ("mean_tpot_ms", "Mean TPOT", "milliseconds", "{:.2f}"),
        ("mean_itl_ms", "Mean ITL", "milliseconds", "{:.2f}"),
        ("prefix_cache_reused_pct", "Prefix Cache Reused", "percent", "{:.2f}%"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"Random Benchmark Metrics vs {x_config['title']}", fontsize=15)

    for ax, (key, title, ylabel, value_fmt) in zip(axes.flat, charts):
        values = [row[key] for row in rows]
        ax.plot(x_values, values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(x_config["label"])
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
        ax.set_xticks(x_values)
        if key == "prefix_cache_reused_pct":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))

        for x, y in zip(x_values, values):
            ax.annotate(
                value_fmt.format(y),
                xy=(x, y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output or args.input_dir / X_AXIS_CONFIG[args.x_axis]["output_name"]
    rows = load_rows(args.input_dir, args.x_axis)
    plot(rows, args.x_axis, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
