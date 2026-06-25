from __future__ import annotations

import math
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .analyze_data import Row, X_AXIS_CONFIG
from .chart_spec import ChartSpec


def get_fixed_value(rows: list[Row], key: str) -> float | None:
    """Return a field value only when it is constant across all rows."""
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    rounded_values = {round(float(value), 6) for value in values}
    if len(rounded_values) != 1:
        return None
    return float(values[0])


def format_number(value: float) -> str:
    """Format axis and parameter values for compact plot labels."""
    if value.is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def build_fixed_param_label(rows: list[Row], x_axis: str) -> str:
    """Build the subtitle that lists fixed benchmark parameters."""
    params = []
    candidates = [
        ("dram_size_gb", "DRAM size", "GB"),
        ("dram_bandwidth_gb", "DRAM bandwidth", "GB/s"),
        ("request_rate", "request rate", "RPS"),
        ("num_requests", "requests", ""),
        ("avg_input_tokens", "avg input", "tokens"),
        ("avg_output_tokens", "avg output", "tokens"),
        ("groups", "groups", ""),
        ("prompts_per_group", "prompts/group", ""),
    ]
    x_key = X_AXIS_CONFIG[x_axis]["row_key"]
    for key, label, unit in candidates:
        if key == x_key:
            continue
        value = get_fixed_value(rows, key)
        if value is None:
            continue
        suffix = f" {unit}" if unit else ""
        params.append(f"{label}={format_number(value)}{suffix}")

    if not params:
        return ""
    return "Parameters: " + "; ".join(params)


def set_zero_based_ylim(ax, values: list[float], is_percent: bool = False) -> None:
    """Set a zero-based y-axis with padding for metric values."""
    if not values:
        ax.set_ylim(bottom=0, top=1)
        return
    max_value = max(values)
    if is_percent:
        top = 100 if max_value <= 100 else max_value * 1.05
    else:
        top = max_value * 1.15 if max_value > 0 else 1
    ax.set_ylim(bottom=0, top=top)


def plot_chart(ax, rows: list[Row], x_axis: str, chart: ChartSpec) -> None:
    """Draw one configured benchmark chart onto an axes object."""
    x_config = X_AXIS_CONFIG[x_axis]
    x_key = x_config["row_key"]
    x_values = [float(row[x_key]) for row in rows if row.get(x_key) is not None]
    x_labels = [format_number(value) for value in x_values]

    plotted = False
    plotted_values = []
    for series_index, (label, key) in enumerate(chart.series):
        values = [row.get(key) for row in rows]
        if any(value is None for value in values):
            continue

        plot_values = [float(value) for value in values]
        ax.plot(x_values, plot_values, marker="o", linewidth=1.8, label=label)
        plotted_values.extend(plot_values)

        y_offset = 7 if series_index % 2 == 0 else -13
        for x_value, y_value in zip(x_values, plot_values):
            text = f"{y_value:.3f}%" if chart.is_percent else f"{y_value:.2f}"
            ax.annotate(
                text,
                xy=(x_value, y_value),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
    else:
        set_zero_based_ylim(ax, plotted_values, is_percent=chart.is_percent)
        ax.legend(fontsize=8)

    ax.set_title(chart.title)
    ax.set_xlabel(x_config["label"])
    ax.set_ylabel(chart.ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels)
    ax.margins(x=0.04)
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(90)
        tick_label.set_horizontalalignment("center")
    ax.tick_params(axis="x", labelsize=8)
    if chart.is_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))


def render_charts(
    rows: list[Row],
    x_axis: str,
    charts: list[ChartSpec],
    output: Path,
    title: str = "Benchmark Metrics",
) -> None:
    """Render a grid of benchmark charts to a PNG file."""
    if not charts:
        raise ValueError("At least one chart is required")

    x_config = X_AXIS_CONFIG[x_axis]
    fixed_param_label = build_fixed_param_label(rows, x_axis)
    ncols = 1 if len(charts) == 1 else 2
    nrows = math.ceil(len(charts) / ncols)
    width = 11 if ncols == 1 else 13
    height = max(5.6, nrows * 3.6 + 1.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    if fixed_param_label and ncols == 1:
        top = 0.70
        label_y = 0.88
        label_width = 90
    elif fixed_param_label:
        top = 0.84
        label_y = 0.93
        label_width = 120
    else:
        top = 0.89
        label_y = 0.93
        label_width = 120
    fig.subplots_adjust(top=top, hspace=0.5, wspace=0.22)
    fig.suptitle(f"{title} vs {x_config['title']}", fontsize=15, y=0.98)
    if fixed_param_label:
        fig.text(
            0.5,
            label_y,
            textwrap.fill(fixed_param_label, width=label_width),
            ha="center",
            va="top",
            fontsize=9,
        )

    for ax, chart in zip(axes_flat, charts):
        plot_chart(ax, rows, x_axis, chart)

    for ax in axes_flat[len(charts) :]:
        ax.set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
