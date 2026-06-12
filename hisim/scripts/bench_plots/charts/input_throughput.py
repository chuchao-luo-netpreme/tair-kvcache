from __future__ import annotations

from ..chart_spec import ChartSpec


def chart() -> ChartSpec:
    return ChartSpec(
        key="input_throughput",
        title="Input Throughput",
        ylabel="tokens/s",
        series=(("input", "input_throughput"),),
    )

