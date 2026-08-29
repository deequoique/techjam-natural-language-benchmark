"""Private child-process endpoint for :class:`SubprocessAgent`."""

from __future__ import annotations

import argparse
import contextlib
from functools import wraps
import json
import sys
from types import MethodType
from typing import Mapping

from .agent_loader import AgentLoadError, load_agent


def _write(value: dict) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _candidate_id(value: object) -> str | None:
    if isinstance(value, Mapping):
        raw = value.get("parent_asin") or value.get("product_id")
    else:
        raw = getattr(value, "parent_asin", None)
    if raw is None:
        return None
    normalized = str(raw).strip()
    return normalized or None


def _candidate_ids(values: object, *, limit: int = 1000) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        parent_asin = _candidate_id(value)
        if parent_asin and parent_asin not in seen:
            seen.add(parent_asin)
            output.append(parent_asin)
        if len(output) >= limit:
            break
    return output


def _semantic_ids(result: object) -> list[str]:
    for name in ("ordered_parent_asins", "ordered_ids", "parent_asins"):
        values = getattr(result, name, None)
        if isinstance(values, (list, tuple)):
            return [str(value).strip() for value in values if str(value).strip()][:1000]
    if isinstance(result, Mapping):
        for name in ("ordered_parent_asins", "ordered_ids", "parent_asins"):
            values = result.get(name)
            if isinstance(values, (list, tuple)):
                return [str(value).strip() for value in values if str(value).strip()][:1000]
    return []


def _safe_value(value: object, *, depth: int = 0) -> object:
    """Keep diagnostics JSON-safe and bounded without serializing arbitrary objects."""

    if depth > 5:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value if not isinstance(value, str) else value[:2000]
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 200:
                break
            output[str(key)[:200]] = _safe_value(item, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_value(item, depth=depth + 1) for item in list(value)[:1000]]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        try:
            return _safe_value(as_dict(), depth=depth + 1)
        except Exception:
            return None
    return None


def _constraint(value: object) -> dict[str, object]:
    fields = (
        "attribute", "value", "polarity", "hardness", "source",
        "confidence", "active", "status", "epoch", "turn",
    )
    return {
        field: _safe_value(getattr(value, field, None))
        for field in fields
        if getattr(value, field, None) is not None
    }


def _state_diagnostics(agent: object, session_id: str) -> dict[str, object]:
    sessions = getattr(agent, "sessions", None)
    state = sessions.get(session_id) if isinstance(sessions, Mapping) else None
    if state is None:
        return {}
    active_constraints = getattr(state, "active_constraints", ())
    active_evidence = getattr(state, "active_query_evidence", ())
    return {
        "category_anchor": _safe_value(getattr(state, "category_anchor", None)),
        "intent_epoch": _safe_value(getattr(state, "intent_epoch", None)),
        "active_constraints": [
            _constraint(item) for item in list(active_constraints)[:100]
        ],
        "active_query_terms": _safe_value(getattr(state, "active_query_terms", [])),
        "active_query_evidence": [
            _safe_value(item.as_dict() if callable(getattr(item, "as_dict", None)) else {
                "text": getattr(item, "text", None),
                "kind": getattr(item, "kind", None),
                "attribute_hint": getattr(item, "attribute_hint", None),
                "confidence": getattr(item, "confidence", None),
                "source": getattr(item, "source", None),
                "status": getattr(item, "status", None),
            })
            for item in list(active_evidence)[:100]
        ],
        "no_preference": _safe_value(getattr(state, "no_preference", [])),
        "attribute_exhausted": _safe_value(getattr(state, "attribute_exhausted", [])),
        "asked_attributes": _safe_value(getattr(state, "asked_attributes", [])),
        "ask_counts": _safe_value(getattr(state, "ask_counts", {})),
        "global_exhausted": _safe_value(getattr(state, "global_exhausted", None)),
        "boundary_seen": _safe_value(getattr(state, "boundary_seen", None)),
    }


def _install_stage_probes(agent: object) -> None:
    """Wrap stable Agent hooks while preserving their outputs and signatures."""

    setattr(agent, "_nl_benchmark_stage_diagnostics", {})
    for method_name in ("_retrieve", "_feature_rank", "_semantic_rank"):
        method = getattr(agent, method_name, None)
        if not callable(method):
            continue

        @wraps(method)
        def wrapper(*args: object, __method=method, __name=method_name, **kwargs: object) -> object:
            stages = getattr(agent, "_nl_benchmark_stage_diagnostics", {})
            if __name == "_feature_rank" and len(args) >= 2:
                stages["feature_input_ids"] = _candidate_ids(args[1])
            elif __name == "_semantic_rank" and len(args) >= 2:
                stages["semantic_input_ids"] = _candidate_ids(args[1])
            result = __method(*args, **kwargs)
            if __name == "_retrieve":
                stages["retrieved_ids"] = _candidate_ids(result)
            elif __name == "_feature_rank":
                stages["feature_ranked_ids"] = _candidate_ids(result)
            elif __name == "_semantic_rank":
                stages["semantic_ranked_ids"] = _semantic_ids(result)
                stages["semantic_backend"] = _safe_value(getattr(result, "backend", None))
            setattr(agent, "_nl_benchmark_stage_diagnostics", stages)
            return result

        setattr(agent, method_name, wrapper)


def _turn_diagnostics(agent: object, session_id: str, response: object) -> dict[str, object]:
    stages = _safe_value(getattr(agent, "_nl_benchmark_stage_diagnostics", {}))
    facade = _safe_value(getattr(agent, "last_diagnostics", {}))
    if isinstance(response, Mapping):
        final_ids = _candidate_ids(response.get("recommendations", []))
    else:
        final_ids = _candidate_ids(getattr(response, "recommendations", []))
    return {
        "intent_and_policy": facade if isinstance(facade, dict) else {},
        "state": _state_diagnostics(agent, session_id),
        "stages": {**(stages if isinstance(stages, dict) else {}), "final_ids": final_ids},
    }


def _configure_intent_experiment(
    agent: object,
    *,
    force_model: bool,
    confidence: float | None,
) -> None:
    """Apply an evaluator-only intent ablation without editing the Agent checkout."""

    interpreter = getattr(agent, "intent_interpreter", None)
    if interpreter is None:
        return
    if force_model:
        setattr(interpreter, "enabled", True)

        def always_call_model(self: object, message: str, update: object) -> tuple[bool, str]:
            del self, message, update
            return True, "benchmark_forced_model"

        setattr(interpreter, "_should_call_model", MethodType(always_call_model, interpreter))
    if confidence is None:
        return
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("intent confidence must be between 0.0 and 1.0")
    original_merge = getattr(interpreter, "_merge", None)
    if not callable(original_merge):
        return

    def merge_with_confidence(
        self: object,
        message: str,
        deterministic: object,
        model: Mapping[str, object],
        *,
        turn: int,
    ) -> object:
        del self
        adjusted = dict(model)
        adjusted["confidence"] = confidence
        mutations = adjusted.get("mutations", [])
        if isinstance(mutations, list):
            adjusted["mutations"] = [
                {**item, "confidence": confidence} if isinstance(item, Mapping) else item
                for item in mutations
            ]
        return original_merge(message, deterministic, adjusted, turn=turn)

    setattr(interpreter, "_merge", MethodType(merge_with_confidence, interpreter))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-repo", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--force-intent-model", action="store_true")
    parser.add_argument("--intent-confidence", type=float)
    args = parser.parse_args(argv)
    try:
        # External imports/logging are redirected so stdout remains a strict
        # one-request/one-response JSONL channel.
        with contextlib.redirect_stdout(sys.stderr):
            agent = load_agent(args.agent_repo, args.catalog)
            _configure_intent_experiment(
                agent,
                force_model=bool(args.force_intent_model),
                confidence=args.intent_confidence,
            )
            _install_stage_probes(agent)
    except Exception as exc:
        _write({"ok": False, "error": f"load: {type(exc).__name__}: {exc}"})
        return 1
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            operation = request.get("op")
            with contextlib.redirect_stdout(sys.stderr):
                if operation == "reset":
                    agent.reset(str(request["session_id"]), dict(request.get("user_profile") or {}))
                    response = None
                elif operation == "respond":
                    setattr(agent, "_nl_benchmark_stage_diagnostics", {})
                    session_id = str(request["session_id"])
                    response = agent.respond(
                        session_id,
                        str(request.get("user_message") or ""),
                        int(request["turn"]),
                        int(request["top_k"]),
                    )
                    diagnostics = _turn_diagnostics(agent, session_id, response)
                    diagnostics["intent_experiment"] = {
                        "force_model": bool(args.force_intent_model),
                        "confidence_override": args.intent_confidence,
                    }
                else:
                    raise ValueError(f"unsupported operation: {operation!r}")
            payload = {"ok": True, "response": response}
            if operation == "respond":
                payload["diagnostics"] = diagnostics
            _write(payload)
        except Exception as exc:
            _write({"ok": False, "error": f"request: {type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
