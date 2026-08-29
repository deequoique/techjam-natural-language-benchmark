"""Independent validation for frozen target-exact samples."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable

from .catalog import Catalog, match_fact, values_for_field
from .facts import ID_RE, fact_display_matches_value, fact_evidence_in_text, is_identifier_like
from .schema import Fact, Sample, project_for_agent, stable_id


ALLOWED_SCENARIOS = {
    "direct_search",
    "multi_constraint",
    "profile_hidden",
    "clarification_required",
    "negative_constraint",
    "budget_rating",
    "intent_override",
}
ALLOWED_SOURCES = {
    "categories", "store", "price", "rating", "rating_count", "details",
    "features", "description", "title", "cross_catalog", "decoy",
}
IDISH_RE = re.compile(r"\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{8,}\b")


@dataclass(frozen=True)
class ValidationConfig:
    min_initial_candidates: int = 2
    max_initial_candidates: int = 200
    enforce_initial_bounds: bool = True
    reject_title_leakage: bool = True


@dataclass
class ValidationResult:
    sample_id: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
        }


def _fact_identity(fact: Fact) -> tuple[str, str, str]:
    return fact.field, fact.operator, str(fact.value).casefold()


def _visible_text(sample: Sample) -> str:
    projection = project_for_agent(sample)
    return json.dumps(projection, ensure_ascii=False, sort_keys=True)


def _validate_fact_identity(fact: Fact, errors: list[str], label: str) -> None:
    expected = stable_id(fact.field, fact.operator, fact.value, fact.polarity, fact.display)
    if fact.fact_id != expected:
        errors.append(f"{label} has an invalid fact_id")
    if not fact_display_matches_value(fact):
        errors.append(f"{label} display is not semantically consistent with value")
    if fact.source not in ALLOWED_SOURCES:
        errors.append(f"{label} uses unsupported source {fact.source!r}")
    if fact.operator not in {"eq", "exact", "contains", "has", "le", "max", "ge", "min", "neq", "not_contains", "not"}:
        errors.append(f"{label} uses unsupported operator {fact.operator!r}")
    if is_identifier_like(fact.display) or IDISH_RE.search(fact.display):
        errors.append(f"{label} contains an identifier-like display value")


def validate_sample(catalog: Catalog, sample: Sample, *, config: ValidationConfig | None = None) -> ValidationResult:
    config = config or ValidationConfig()
    errors: list[str] = []
    warnings: list[str] = []
    target = catalog.get(sample.target_parent_asin)
    if sample.schema_version != 1:
        errors.append(f"unsupported schema_version {sample.schema_version}")
    if not sample.sample_id:
        errors.append("sample_id is empty")
    if sample.scenario_type not in ALLOWED_SCENARIOS:
        errors.append(f"unsupported scenario_type {sample.scenario_type!r}")
    if target is None:
        errors.append("target_parent_asin is not present in catalog")
        return ValidationResult(sample.sample_id, False, errors, warnings)
    if not sample.query.strip():
        errors.append("query is empty")
    if not isinstance(sample.user_profile, dict):
        errors.append("user_profile is not an object")

    signature = list(sample.signature)
    query_facts = list(sample.query_facts)
    profile_facts = list(sample.profile_facts)
    clarification_facts = list(sample.clarification_facts)
    if not signature:
        errors.append("signature is empty")
    identities: set[tuple[str, str, str]] = set()
    by_id: dict[str, Fact] = {}
    for index, fact in enumerate(signature):
        label = f"signature[{index}]"
        _validate_fact_identity(fact, errors, label)
        identity = _fact_identity(fact)
        if identity in identities:
            errors.append(f"duplicate signature fact {fact.fact_id}")
        identities.add(identity)
        if fact.fact_id in by_id:
            errors.append(f"duplicate fact_id {fact.fact_id}")
        by_id[fact.fact_id] = fact
        if not match_fact(target, fact):
            errors.append(f"{label} is not true for target")
        if not fact_display_matches_value(fact):
            errors.append(f"{label} display does not match canonical value")

    partition_seen: set[tuple[str, str, str]] = set()
    for label, facts in (("query", query_facts), ("profile", profile_facts), ("clarification", clarification_facts)):
        for index, fact in enumerate(facts):
            _validate_fact_identity(fact, errors, f"{label}[{index}]")
            identity = _fact_identity(fact)
            if identity not in identities:
                errors.append(f"{label}[{index}] is not in signature")
            if identity in partition_seen:
                errors.append(f"fact appears in multiple partitions: {fact.fact_id}")
            partition_seen.add(identity)
    if partition_seen != identities:
        errors.append("signature facts are not partitioned exactly once")

    # Re-check that the prose actually carries every structured disclosure.
    # This catches tampering in a frozen JSONL file instead of trusting the
    # generator's original rendering call.
    query_text = sample.query
    profile_text = json.dumps(sample.user_profile, ensure_ascii=False, sort_keys=True)
    for index, fact in enumerate(query_facts):
        if not fact_evidence_in_text(fact, query_text):
            errors.append(f"query text does not render query[{index}] evidence")
    for index, fact in enumerate(profile_facts):
        if not fact_evidence_in_text(fact, profile_text):
            errors.append(f"profile text does not render profile[{index}] evidence")

    visible = [*query_facts, *profile_facts]
    visible_ids = catalog.candidate_ids(visible)
    full_ids = catalog.candidate_ids(signature)
    if full_ids != {sample.target_parent_asin}:
        errors.append(f"full signature is not unique: {len(full_ids)} candidates")
    if sample.target_parent_asin not in visible_ids:
        errors.append("visible facts do not retain the target")
    visible_count = len(visible_ids)
    if config.enforce_initial_bounds:
        if sample.scenario_type == "direct_search":
            if visible_count < 1 or visible_count > config.max_initial_candidates:
                errors.append(f"initial candidate count {visible_count} is outside direct-search bounds")
        elif not (config.min_initial_candidates <= visible_count <= config.max_initial_candidates):
            errors.append(f"initial candidate count {visible_count} is outside [{config.min_initial_candidates}, {config.max_initial_candidates}]")
    elif visible_count == 1 and sample.scenario_type != "direct_search":
        warnings.append("initial disclosure is already unique")

    projection = project_for_agent(sample)
    projection_text = json.dumps(projection, ensure_ascii=False, sort_keys=True)
    if sample.target_parent_asin.casefold() in projection_text.casefold():
        errors.append("target parent_asin leaks into Agent projection")
    if "signature" in projection or "target_parent_asin" in projection:
        errors.append("Agent projection contains evaluator-only keys")
    if IDISH_RE.search(projection_text):
        errors.append("Agent projection contains an identifier-like token")
    if config.reject_title_leakage:
        target_title = " ".join(target.title.casefold().split())
        visible_lower = projection_text.casefold()
        if len(target_title) >= 16 and target_title in visible_lower:
            errors.append("full target title leaks into Agent projection")
    if sample.override and isinstance(sample.override.get("message"), str):
        if sample.target_parent_asin.casefold() in sample.override["message"].casefold():
            errors.append("target parent_asin leaks into override")

    if sample.scenario_type == "intent_override":
        override = sample.override
        if not isinstance(override, dict):
            errors.append("intent_override has no structured override")
        else:
            if int(override.get("turn", -1)) != 2:
                errors.append("intent_override must occur at turn 2")
            old_data = override.get("old_fact")
            new_data = override.get("new_fact")
            try:
                old_fact = Fact.from_dict(old_data) if isinstance(old_data, dict) else None
                new_fact = Fact.from_dict(new_data) if isinstance(new_data, dict) else None
            except (TypeError, ValueError):
                old_fact = new_fact = None
            if old_fact is None or new_fact is None:
                errors.append("intent_override old_fact/new_fact are malformed")
            else:
                if old_fact.source != "decoy" or match_fact(target, old_fact):
                    errors.append("intent_override old_fact is not a false decoy")
                if not catalog.candidate_ids([old_fact]):
                    errors.append("intent_override old_fact is not observed in catalog")
                if new_fact.fact_id not in {fact.fact_id for fact in signature}:
                    errors.append("intent_override new_fact is not a signature fact")
                if not match_fact(target, new_fact):
                    errors.append("intent_override new_fact is not true for target")
                if new_fact.fact_id not in {fact.fact_id for fact in clarification_facts}:
                    errors.append("intent_override new_fact must start hidden")
                if not fact_evidence_in_text(old_fact, sample.query):
                    errors.append("query text does not render the old decoy")
                override_text = str(override.get("message") or "")
                if not fact_evidence_in_text(old_fact, override_text):
                    errors.append("override text does not render old decoy")
                if not fact_evidence_in_text(new_fact, override_text):
                    errors.append("override text does not render new target fact")
                post_override = [*query_facts, *profile_facts, new_fact]
                post_override.extend(fact for fact in clarification_facts if fact.fact_id != new_fact.fact_id)
                post_ids = catalog.candidate_ids(post_override)
                if post_ids != {sample.target_parent_asin}:
                    errors.append(f"post-override disclosure is not unique: {len(post_ids)} candidates")

    clarification_replies = sample.simulator.get("clarification_replies") if isinstance(sample.simulator, dict) else None
    replies_by_id = {
        str(item.get("fact_id")): str(item.get("message") or "")
        for item in clarification_replies
        if isinstance(item, dict) and item.get("fact_id")
    } if isinstance(clarification_replies, list) else {}
    if clarification_facts and len(replies_by_id) != len(clarification_facts):
        errors.append("simulator clarification replies do not cover every hidden fact")
    for fact in clarification_facts:
        reply_text = replies_by_id.get(fact.fact_id, "")
        if not reply_text:
            continue
        if not fact_evidence_in_text(fact, reply_text):
            errors.append(f"clarification reply does not render fact {fact.fact_id}")
        if sample.target_parent_asin.casefold() in reply_text.casefold() or IDISH_RE.search(reply_text):
            errors.append(f"clarification reply leaks an identifier for {fact.fact_id}")

    if sample.scenario_type == "profile_hidden" and not profile_facts:
        errors.append("profile_hidden has no profile facts")
    if sample.scenario_type == "clarification_required" and not clarification_facts:
        errors.append("clarification_required has no hidden facts")
    if sample.scenario_type == "negative_constraint" and not any(fact.polarity == "negative" or fact.operator in {"neq", "not_contains", "not"} for fact in signature):
        errors.append("negative_constraint has no negative fact")
    if sample.scenario_type == "budget_rating" and not any(fact.field in {"price", "rating", "rating_count"} for fact in signature):
        errors.append("budget_rating has no numeric fact")
    if sample.scenario_type == "intent_override" and not sample.override:
        errors.append("intent_override has no override message")

    diagnostics = {
        "catalog_size": len(catalog),
        "initial_candidate_count": visible_count,
        "full_candidate_count": len(full_ids),
        "signature_size": len(signature),
        "query_fact_count": len(query_facts),
        "profile_fact_count": len(profile_facts),
        "clarification_fact_count": len(clarification_facts),
        "target": sample.target_parent_asin,
    }
    return ValidationResult(sample.sample_id, not errors, errors, warnings, diagnostics)


def validate_dataset(catalog: Catalog, samples: Iterable[Sample], *, config: ValidationConfig | None = None) -> dict[str, Any]:
    results = [validate_sample(catalog, sample, config=config) for sample in samples]
    return {
        "valid": all(result.valid for result in results),
        "count": len(results),
        "valid_count": sum(result.valid for result in results),
        "invalid_count": sum(not result.valid for result in results),
        "results": [result.to_dict() for result in results],
    }
