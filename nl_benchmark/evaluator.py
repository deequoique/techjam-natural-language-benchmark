"""Run the external TechJam Agent through frozen samples."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping
import uuid

from .catalog import Catalog
from .metrics import summarize
from .schema import MAX_TURNS, TOP_K, Sample, project_for_agent
from .simulator import IntelligentSimulator


ASK_WORDS_RE = re.compile(r"\?|\b(ask|tell me|which|what|could you|do you need|prefer)\b", re.I)


def normalize_recommendations(payload: object, catalog_ids: Iterable[str], *, limit: int = TOP_K) -> list[dict[str, Any]]:
    """Keep only valid, de-duplicated catalog IDs while retaining scores."""

    allowed = set(catalog_ids)
    if not isinstance(payload, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload:
        if isinstance(item, Mapping):
            parent_asin = str(item.get("parent_asin") or item.get("product_id") or "").strip()
            score = item.get("score")
        else:
            parent_asin = str(item or "").strip()
            score = None
        if not parent_asin or parent_asin in seen or parent_asin not in allowed:
            continue
        seen.add(parent_asin)
        row: dict[str, Any] = {"parent_asin": parent_asin}
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            row["score"] = float(score)
        normalized.append(row)
        if len(normalized) >= max(int(limit), 1):
            break
    return normalized


def _response_field(response: object, field: str, default: object = None) -> object:
    if isinstance(response, Mapping):
        return response.get(field, default)
    return getattr(response, field, default)


def _rank(parent_asin: str, values: object) -> int | None:
    if not isinstance(values, list):
        return None
    normalized = [str(value).strip() for value in values]
    try:
        return normalized.index(parent_asin) + 1
    except ValueError:
        return None


def _diagnostics_with_target_analysis(agent: object, target: str) -> dict[str, Any]:
    """Add target ranks in the evaluator parent, never in the Agent worker."""

    raw = getattr(agent, "last_diagnostics", {})
    diagnostics = dict(raw) if isinstance(raw, Mapping) else {}
    stages = diagnostics.get("stages")
    stages = dict(stages) if isinstance(stages, Mapping) else {}
    retrieval_rank = _rank(target, stages.get("retrieved_ids"))
    feature_input_rank = _rank(target, stages.get("feature_input_ids"))
    feature_rank = _rank(target, stages.get("feature_ranked_ids"))
    semantic_input_rank = _rank(target, stages.get("semantic_input_ids"))
    semantic_rank = _rank(target, stages.get("semantic_ranked_ids"))
    final_rank = _rank(target, stages.get("final_ids"))
    diagnostics["target_analysis"] = {
        "target_in_retrieval": retrieval_rank is not None,
        "target_retrieval_rank": retrieval_rank,
        "target_feature_input_rank": feature_input_rank,
        "target_feature_rank": feature_rank,
        "target_semantic_input_rank": semantic_input_rank,
        "target_semantic_rank": semantic_rank,
        "target_final_rank": final_rank,
    }
    return diagnostics


def evaluate_sample(agent: Any, catalog: Catalog, sample: Sample, *, max_turns: int = MAX_TURNS, top_k: int = TOP_K) -> dict[str, Any]:
    """Evaluate one sample, keeping the target outside every Agent call."""

    target = sample.target_parent_asin
    projection = project_for_agent(sample)
    sent_messages: list[str] = []
    traces: list[dict[str, Any]] = []
    errors: list[str] = []
    simulator = IntelligentSimulator(catalog, sample, max_turns=max_turns)
    session_id = f"nlb-{sample.sample_id}-{uuid.uuid5(uuid.NAMESPACE_URL, sample.sample_id).hex[:10]}"
    try:
        agent.reset(session_id, projection["user_profile"])
    except Exception as exc:
        errors.append(f"reset: {type(exc).__name__}: {exc}")
        return {
            "sample_id": sample.sample_id,
            "scenario_type": sample.scenario_type,
            "target_parent_asin": target,
            "trace": [],
            "errors": errors,
        }

    current_message = str(projection["query"])
    override = sample.override or {}
    override_turn = int(override.get("turn", -1)) if str(override.get("turn", "")).lstrip("-").isdigit() else -1
    for turn in range(1, max(int(max_turns), 1) + 1):
        if target.casefold() in current_message.casefold():
            errors.append(f"target ID leaked into Agent input at turn {turn}")
            break
        sent_messages.append(current_message)
        try:
            response = agent.respond(session_id, current_message, turn, top_k)
        except Exception as exc:
            errors.append(f"turn {turn}: {type(exc).__name__}: {exc}")
            break
        response_message = str(_response_field(response, "message", "") or "")
        ask_attribute = _response_field(response, "ask_attribute", None)
        recommendations = normalize_recommendations(_response_field(response, "recommendations", []), catalog.ids, limit=top_k)
        observation: dict[str, Any] = {
            "turn": turn,
            "user_message": current_message,
            "response_message": response_message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "diagnostics": _diagnostics_with_target_analysis(agent, target),
        }
        if target.casefold() in response_message.casefold():
            # This is a protocol failure even if the response happens to rank
            # the target correctly: the target was not allowed in Agent input,
            # output messages are also audited for accidental leakage.
            errors.append(f"target ID leaked into Agent response message at turn {turn}")
        next_message: str | None = None
        if turn + 1 == override_turn and isinstance(override.get("message"), str):
            # The old preference is a false decoy; the explicit new target
            # fact becomes disclosed to the simulated user at the same point
            # the override message is sent.  It is never passed as structured
            # data to the Agent.
            simulator_reply = simulator.apply_override(override)
            next_message = str(override["message"])
            observation["simulator_status"] = "intent_override_scheduled"
            observation["simulator_revealed_fact_ids"] = list(simulator_reply.revealed_fact_ids)
            observation["simulator_candidate_count"] = simulator_reply.candidate_count
            observation["simulator_diagnostics"] = simulator_reply.diagnostics
        elif ask_attribute is not None or ASK_WORDS_RE.search(response_message):
            simulator_reply = simulator.answer(ask_attribute, response_message, turn=turn)
            observation["simulator_status"] = simulator_reply.status
            observation["simulator_message"] = simulator_reply.message
            observation["simulator_revealed_fact_ids"] = list(simulator_reply.revealed_fact_ids)
            observation["simulator_candidate_count"] = simulator_reply.candidate_count
            observation["simulator_diagnostics"] = simulator_reply.diagnostics
            next_message = simulator_reply.message
        traces.append(observation)
        if next_message is None:
            break
        current_message = next_message

    return {
        "sample_id": sample.sample_id,
        "scenario_type": sample.scenario_type,
        "target_parent_asin": target,
        "trace": traces,
        "errors": errors,
        # Recommendations before an explicit intent override are useful in a
        # trace but do not count toward exact retrieval metrics.
        "metrics_start_turn": override_turn if override_turn > 0 else 1,
    }


def evaluate_dataset(agent: Any, catalog: Catalog, samples: Iterable[Sample], *, max_turns: int = MAX_TURNS, top_k: int = TOP_K) -> dict[str, Any]:
    results = [evaluate_sample(agent, catalog, sample, max_turns=max_turns, top_k=top_k) for sample in samples]
    report = {
        "protocol": {
            "max_turns": int(max_turns),
            "top_k": int(top_k),
            "scoring": "exact parent_asin only",
            "agent_receives": ["query", "user_profile", "simulator replies"],
            "agent_does_not_receive": ["target_parent_asin", "signature", "candidate pools", "rubric"],
        },
        "summary": summarize(results, max_turns=max_turns),
        "results": results,
    }
    report["summary"]["error_count"] = sum(bool(result.get("errors")) for result in results)
    return report
