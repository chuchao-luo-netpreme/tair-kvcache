from __future__ import annotations

from collections.abc import Callable, Iterable

from .chart_spec import ChartSpec
from .charts import (
    forward_time_per_iteration,
    input_throughput,
    itl,
    output_throughput,
    prefix_cache_reused,
    tpot,
    ttft,
)


ChartFactory = Callable[[], ChartSpec]

CHART_REGISTRY: dict[str, ChartFactory] = {
    "ttft_mean_median": ttft.mean_median,
    "ttft_median": ttft.median,
    "ttft_mean": ttft.mean,
    "tpot_mean_median": tpot.mean_median,
    "itl_mean_median": itl.mean_median,
    "prefix_cache_reused": prefix_cache_reused.chart,
    "forward_time_per_iteration": forward_time_per_iteration.chart,
    "input_throughput": input_throughput.chart,
    "output_throughput": output_throughput.chart,
}

DEFAULT_CHARTS = (
    "ttft_mean_median",
    "tpot_mean_median",
    "itl_mean_median",
    "prefix_cache_reused",
    "forward_time_per_iteration",
    "input_throughput",
    "output_throughput",
)


def chart_specs(names: Iterable[str]) -> list[ChartSpec]:
    return [CHART_REGISTRY[name]() for name in names]
