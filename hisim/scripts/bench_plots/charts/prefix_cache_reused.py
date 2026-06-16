from __future__ import annotations

from ..chart_spec import ChartSpec


def chart() -> ChartSpec:
    return ChartSpec(
        key="prefix_cache_reused",
        title="Prefix Cache Reused",
        ylabel="percent",
        series=(("reused", "prefix_cache_reused_pct"),),
        is_percent=True,
    )

