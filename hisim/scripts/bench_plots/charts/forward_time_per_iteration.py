from __future__ import annotations

from ..chart_spec import ChartSpec


def chart() -> ChartSpec:
    return ChartSpec(
        key="forward_time_per_iteration",
        title="Forward Time per Iteration",
        ylabel="milliseconds",
        series=(("forward", "forward_time_per_iteration_ms"),),
    )

