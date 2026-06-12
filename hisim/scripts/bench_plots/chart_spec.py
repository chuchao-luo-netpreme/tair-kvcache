from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChartSpec:
    key: str
    title: str
    ylabel: str
    series: tuple[tuple[str, str], ...]
    is_percent: bool = False

