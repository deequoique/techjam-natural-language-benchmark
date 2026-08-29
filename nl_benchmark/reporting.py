"""JSON and human-readable report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def format_report(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    lines = [
        f"samples: {summary.get('count', 0)}",
        f"exact Top-1: {float(summary.get('exact_top1', 0.0)):.3f}",
        f"Hit@10: {float(summary.get('hit_at_10', 0.0)):.3f}",
        f"MRR: {float(summary.get('mrr', 0.0)):.3f}",
        f"MTTC: {float(summary.get('mttc', 0.0)):.2f}",
        f"mean clarification turns: {float(summary.get('mean_clarification_turns', 0.0)):.2f}",
    ]
    by_scenario = summary.get("by_scenario")
    if isinstance(by_scenario, Mapping) and by_scenario:
        lines.append("scenario breakdown:")
        for name, values in by_scenario.items():
            if not isinstance(values, Mapping):
                continue
            lines.append(
                f"  {name}: n={values.get('count', 0)} "
                f"Hit@10={float(values.get('hit_at_10', 0.0)):.3f} "
                f"Top-1={float(values.get('exact_top1', 0.0)):.3f}"
            )
    simulator = summary.get("simulator")
    if isinstance(simulator, Mapping):
        lines.append(f"simulator statuses: {json.dumps(simulator.get('status_counts', {}), ensure_ascii=False, sort_keys=True)}")
    if summary.get("error_count"):
        lines.append(f"samples with protocol errors: {summary['error_count']}")
    return "\n".join(lines)
