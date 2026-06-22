from __future__ import annotations

from ..chart_spec import ChartSpec


def mean_median() -> ChartSpec:
    return ChartSpec(
        key="ttft_mean_median",
        title="Time to First Token",
        ylabel="milliseconds",
        series=(("mean", "mean_ttft_ms"), ("median", "median_ttft_ms")),
    )


def excluding_queue_mean_median_p99() -> ChartSpec:
    return ChartSpec(
        key="ttft_excluding_queue_mean_median_p99",
        title="Time to First Token Excluding Queue",
        ylabel="milliseconds",
        series=(
            ("mean", "mean_ttft_excluding_queue_ms"),
            ("median", "median_ttft_excluding_queue_ms"),
            ("p99", "p99_ttft_excluding_queue_ms"),
        ),
    )


def excluding_queue_mean_median() -> ChartSpec:
    return excluding_queue_mean_median_p99()


def median() -> ChartSpec:
    return ChartSpec(
        key="ttft_median",
        title="Median Time to First Token",
        ylabel="milliseconds",
        series=(("median", "median_ttft_ms"),),
    )


def mean() -> ChartSpec:
    return ChartSpec(
        key="ttft_mean",
        title="Mean Time to First Token",
        ylabel="milliseconds",
        series=(("mean", "mean_ttft_ms"),),
    )
