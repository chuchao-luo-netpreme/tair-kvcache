from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


LATENCY_STATS = ("mean", "median", "p99")

DRAM_SIZE_RE = re.compile(r"DramSize(\d+)gB", re.IGNORECASE)
DRAM_BW_RE = re.compile(r"DramBw(\d+)gB", re.IGNORECASE)
REQUEST_RATE_RE = re.compile(r"_(\d+(?:\.\d+)?)RPS(?:_|\.|$)", re.IGNORECASE)
GROUPS_RE = re.compile(r"_Groups(\d+)(?:_|\.|$)", re.IGNORECASE)
PER_GROUP_RE = re.compile(r"_PerGroup(\d+)(?:_|\.|$)", re.IGNORECASE)

X_AXIS_CONFIG = {
    "dram-size": {
        "regex": DRAM_SIZE_RE,
        "row_key": "dram_size_gb",
        "label": "DRAM size (GB)",
        "title": "DRAM Size",
        "output_name": "dram_size_metrics.png",
    },
    "dram-bandwidth": {
        "regex": DRAM_BW_RE,
        "row_key": "dram_bandwidth_gb",
        "label": "DRAM bandwidth (GB/s)",
        "title": "DRAM Bandwidth",
        "output_name": "dram_bandwidth_metrics.png",
    },
    "request-rate": {
        "regex": REQUEST_RATE_RE,
        "row_key": "request_rate",
        "label": "Request rate (RPS)",
        "title": "Request Rate",
        "output_name": "request_rate_metrics.png",
    },
}

Row = dict[str, float | None]


def load_forward_time_per_iteration_ms(run_dir: Path) -> float | None:
    iteration_path = run_dir / "iteration.jsonl"
    if not iteration_path.exists():
        return None

    total = 0.0
    count = 0
    with iteration_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total += float(item.get("forward_latency") or 0.0)
            count += 1

    if count == 0:
        return None
    return total / count * 1000


def metric_paths(input_dir: Path) -> list[Path]:
    run_metrics = sorted(input_dir.glob("runs/*/metrics.json"))
    if run_metrics:
        return run_metrics
    return sorted(input_dir.glob("*.json"))


def _float_from_match(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    return float(match.group(1))


def _avg_tokens(total: Any, num_requests: Any) -> float | None:
    if total is None or not num_requests:
        return None
    return float(total) / float(num_requests)


def load_rows(input_dir: Path, x_axis: str) -> list[Row]:
    x_config = X_AXIS_CONFIG[x_axis]
    rows: list[Row] = []

    for path in metric_paths(input_dir):
        run_name = path.parent.name if path.name == "metrics.json" else path.stem
        match = x_config["regex"].search(run_name)
        if not match:
            continue

        with path.open() as f:
            data = json.load(f)

        num_requests = data.get("num_requests")
        total_input = data.get("total_input")
        total_output = data.get("total_output")
        row: Row = {
            x_config["row_key"]: float(match.group(1)),
            "dram_size_gb": _float_from_match(DRAM_SIZE_RE.search(run_name)),
            "dram_bandwidth_gb": _float_from_match(DRAM_BW_RE.search(run_name)),
            "request_rate": _float_from_match(REQUEST_RATE_RE.search(run_name)),
            "groups": _float_from_match(GROUPS_RE.search(run_name)),
            "prompts_per_group": _float_from_match(PER_GROUP_RE.search(run_name)),
            "num_requests": float(num_requests) if num_requests is not None else None,
            "avg_input_tokens": _avg_tokens(total_input, num_requests),
            "avg_output_tokens": _avg_tokens(total_output, num_requests),
            "prefix_cache_reused_pct": data["prefix_cache_reused_ratio"] * 100,
            "input_throughput": data.get("input_throughput"),
            "output_throughput": data.get("output_throughput"),
            "forward_time_per_iteration_ms": (
                load_forward_time_per_iteration_ms(path.parent)
                if path.name == "metrics.json"
                else None
            ),
        }

        for stat in LATENCY_STATS:
            for metric in ("ttft", "ttft_excluding_queue", "tpot", "itl"):
                key = f"{stat}_{metric}_ms"
                if key in data:
                    row[key] = data[key]

        rows.append(row)

    rows.sort(key=lambda row: row[x_config["row_key"]] or 0)
    if not rows:
        raise SystemExit(
            f"No metric files with {x_config['title']} metadata found under {input_dir}"
        )

    return rows

