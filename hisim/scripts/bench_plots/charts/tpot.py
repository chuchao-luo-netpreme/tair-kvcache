from __future__ import annotations

from ..chart_spec import ChartSpec


def mean_median() -> ChartSpec:
    return ChartSpec(
        key="tpot_mean_median",
        title="Time per Output Token",
        ylabel="milliseconds",
        series=(("mean", "mean_tpot_ms"), ("median", "median_tpot_ms")),
    )

