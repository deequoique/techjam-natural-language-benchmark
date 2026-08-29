"""Exact-target metrics and deterministic aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


def _observations(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    traces = result.get("trace", result.get("turns", []))
    return [item for item in traces if isinstance(item, Mapping)] if isinstance(traces, list) else []


def first_target_rank(result: Mapping[str, Any], target: str, *, start_turn: int = 1) -> tuple[int | None, int | None]:
    """Return (turn, rank) for the first exact target appearance."""

    target = str(target)
    for observation in _observations(result):
        try:
            observation_turn = int(observation.get("turn", 0))
        except (TypeError, ValueError):
            observation_turn = 0
        if observation_turn < int(start_turn):
            continue
        recommendations = observation.get("recommendations", [])
        if not isinstance(recommendations, list):
            continue
        for index, value in enumerate(recommendations, 1):
            candidate = value.get("parent_asin") if isinstance(value, Mapping) else value
            if str(candidate) == target:
                try:
                    turn = int(observation.get("turn", index))
                except (TypeError, ValueError):
                    turn = index
                return turn, index
    return None, None


def _one(result: Mapping[str, Any], *, max_turns: int) -> dict[str, Any]:
    target = str(result.get("target_parent_asin", ""))
    try:
        start_turn = max(int(result.get("metrics_start_turn", 1)), 1)
    except (TypeError, ValueError):
        start_turn = 1
    turn, rank = first_target_rank(result, target, start_turn=start_turn)
    hit = rank is not None and rank <= 10
    top1 = rank == 1
    return {
        "hit_at_10": bool(hit),
        "exact_top1": bool(top1),
        "target_turn": turn,
        "target_rank": rank,
        "mrr": 1.0 / rank if hit and rank else 0.0,
        "mttc": turn if hit and turn is not None else max_turns + 1,
        "clarification_turns": sum(1 for item in _observations(result) if item.get("simulator_status") == "revealed"),
        "metrics_start_turn": start_turn,
    }


def summarize(results: Iterable[Mapping[str, Any]], *, max_turns: int = 10) -> dict[str, Any]:
    rows = list(results)
    scored = [dict(row.get("metrics") or _one(row, max_turns=max_turns), **({} if "metrics" in row else {})) for row in rows]
    # ``metrics`` may be present in a persisted result. Recompute from trace
    # when possible so a hand-edited rank cannot silently alter the report.
    scored = [_one(row, max_turns=max_turns) for row in rows]
    count = len(scored)

    def avg(name: str) -> float:
        return sum(float(row[name]) for row in scored) / count if count else 0.0

    summary: dict[str, Any] = {
        "count": count,
        "hit_at_10": sum(bool(row["hit_at_10"]) for row in scored) / count if count else 0.0,
        "exact_top1": sum(bool(row["exact_top1"]) for row in scored) / count if count else 0.0,
        "mrr": avg("mrr"),
        "mttc": avg("mttc"),
        "mean_clarification_turns": avg("clarification_turns"),
        "hits": sum(bool(row["hit_at_10"]) for row in scored),
        "misses": sum(not bool(row["hit_at_10"]) for row in scored),
    }
    scenario_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, metric in zip(rows, scored):
        scenario_rows[str(row.get("scenario_type", "unknown"))].append(metric)
    by_scenario: dict[str, Any] = {}
    for scenario in sorted(scenario_rows):
        entries = scenario_rows[scenario]
        n = len(entries)
        by_scenario[scenario] = {
            "count": n,
            "hit_at_10": sum(item["hit_at_10"] for item in entries) / n,
            "exact_top1": sum(item["exact_top1"] for item in entries) / n,
            "mrr": sum(item["mrr"] for item in entries) / n,
            "mttc": sum(item["mttc"] for item in entries) / n,
        }
    summary["by_scenario"] = by_scenario
    status_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    question_reason_counts: Counter[str] = Counter()
    reply_reason_counts: Counter[str] = Counter()
    for row in rows:
        for observation in _observations(row):
            status = observation.get("simulator_status")
            if status:
                status_counts[str(status)] += 1
            diagnostics = observation.get("simulator_diagnostics")
            if isinstance(diagnostics, Mapping):
                if diagnostics.get("route_reason"):
                    route_counts[str(diagnostics["route_reason"])] += 1
                if diagnostics.get("reply_reason"):
                    reply_reason_counts[str(diagnostics["reply_reason"])] += 1
                interpretation = diagnostics.get("question_interpretation")
                if isinstance(interpretation, Mapping) and interpretation.get("reason"):
                    question_reason_counts[str(interpretation["reason"])] += 1
    summary["simulator"] = {
        "status_counts": dict(sorted(status_counts.items())),
        "route_reason_counts": dict(sorted(route_counts.items())),
        "question_reason_counts": dict(sorted(question_reason_counts.items())),
        "reply_reason_counts": dict(sorted(reply_reason_counts.items())),
    }
    return summary
