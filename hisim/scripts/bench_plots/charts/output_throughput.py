from __future__ import annotations

from ..chart_spec import ChartSpec


def chart() -> ChartSpec:
    return ChartSpec(
        key="output_throughput",
        title="Output Throughput",
        ylabel="tokens/s",
        series=(("output", "output_throughput"),),
    )

