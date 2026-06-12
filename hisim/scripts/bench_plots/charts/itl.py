from __future__ import annotations

from ..chart_spec import ChartSpec


def mean_median() -> ChartSpec:
    return ChartSpec(
        key="itl_mean_median",
        title="Inter-token Latency",
        ylabel="milliseconds",
        series=(("mean", "mean_itl_ms"), ("median", "median_itl_ms")),
    )

