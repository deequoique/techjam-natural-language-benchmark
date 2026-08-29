"""Deterministic target-exact sample generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Any, Iterable

from .catalog import Catalog, Product, match_fact, normalize_text
from .facts import extract_facts, fact_aliases, render_fact, render_facts
from .schema import Fact, Sample, stable_id


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratorConfig:
    min_initial_candidates: int = 2
    max_initial_candidates: int = 200
    max_signature_facts: int = 8
    max_attempts_per_sample: int = 250
    max_feature_facts_per_product: int = 80
    min_title_token_length: int = 4


SCENARIOS = (
    "direct_search",
    "multi_constraint",
    "profile_hidden",
    "clarification_required",
    "negative_constraint",
    "budget_rating",
    "intent_override",
)


def _fact_key(fact: Fact) -> tuple[str, str, str]:
    return fact.field, fact.operator, normalize_text(fact.value)


def _dedupe(facts: Iterable[Fact]) -> list[Fact]:
    result: list[Fact] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        key = _fact_key(fact)
        if key not in seen:
            seen.add(key)
            result.append(fact)
    return result


def _sort_fact_candidates(current_count: int, target_count: int, fact: Fact, gain: int) -> tuple[float, ...]:
    # Prefer useful but not answer-like reductions.  Singleton predicates are
    # still allowed when no less abrupt predicate can finish the signature.
    single_penalty = 0.45 if target_count == 1 else 1.0
    source_weight = {
        "categories": 1.15,
        "details": 1.08,
        "store": 0.94,
        "price": 0.88,
        "rating": 0.75,
        "rating_count": 0.7,
        "features": 0.82,
        "description": 0.6,
        "title": 0.53,
        "cross_catalog": 0.45,
    }.get(fact.source, 0.65)
    # Information gain per remaining item; deterministic tie breakers follow.
    ratio = (gain / max(current_count, 1)) * source_weight * single_penalty
    return (ratio, float(gain), -float(target_count), -float(len(fact.display)), fact.fact_id)


def _safe_negative_fact(catalog: Catalog, product: Product, base: list[Fact]) -> Fact | None:
    """Find a truthful, useful negative color/material constraint.

    The value is chosen from a current candidate pool, so the statement is
    grounded in the catalog rather than invented.  ``neq`` is interpreted as
    “the target does not contain this value”; this is checked by the validator.
    """

    current_ids = catalog.candidate_ids(base)
    if len(current_ids) < 3:
        return None
    # Prefer an explicit detail key when possible.  Unlike a guessed color in
    # marketing prose, a catalog detail gives a crisp, auditable negative
    # statement such as “not size 11” while retaining the target's true value.
    target_details = {normalize_text(key): (key, value) for key, value in product.details.items() if value}
    for normalized_key, (original_key, target_value) in sorted(target_details.items()):
        if any(term in normalized_key for term in ("asin", "model", "date", "dimension", "weight", "rank", "sku", "upc", "battery", "batteries", "country", "origin", "isbn", "manufacturer part")):
            continue
        key_attribute = ("color" if any(term in normalized_key for term in ("color", "colour", "shade")) else "material" if any(term in normalized_key for term in ("material", "fabric", "fiber")) else "size" if any(term in normalized_key for term in ("size", "sizing", "width", "length")) else "style" if any(term in normalized_key for term in ("style", "pattern", "shape", "closure", "neck", "sleeve")) else None)
        if key_attribute is None:
            continue
        values: dict[str, int] = {}
        for parent_asin in current_ids:
            candidate = catalog.products[parent_asin]
            for key, value in candidate.details.items():
                if normalize_text(key) == normalized_key and value:
                    value_key = normalize_text(value)
                    values[value_key] = values.get(value_key, 0) + 1
        for value, frequency in sorted(values.items(), key=lambda pair: (-pair[1], pair[0])):
            if value == normalize_text(target_value) or frequency < 1:
                continue
            fact = Fact(
                field=f"detail:{normalized_key}",
                operator="neq",
                value=value,
                source="cross_catalog",
                attribute=key_attribute,
                polarity="negative",
                display=value,
            )
            reduced = catalog.candidate_ids((*base, fact))
            if match_fact(product, fact) and reduced:
                if len(reduced) < len(current_ids):
                    return fact
    for field, attribute in (("color", "color"), ("material", "material")):
        alternatives: dict[str, int] = {}
        for parent_asin in current_ids:
            candidate = catalog.products[parent_asin]
            text = normalize_text(candidate.searchable_text)
            known = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "navy", "beige", "silver", "gold") if field == "color" else ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric", "linen", "suede", "denim")
            for value in known:
                if f" {value} " in f" {text} ":
                    alternatives[value] = alternatives.get(value, 0) + 1
        target_text = normalize_text(product.searchable_text)
        ranked = sorted(alternatives.items(), key=lambda pair: (-pair[1], pair[0]))
        for value, frequency in ranked:
            if frequency < 2 or f" {value} " in f" {target_text} ":
                continue
            fact = Fact(
                field=field,
                operator="neq",
                value=value,
                source="cross_catalog",
                attribute=attribute,
                polarity="negative",
                display=value,
            )
            if match_fact(product, fact) and catalog.candidate_ids((*base, fact)):
                reduced = catalog.candidate_ids((*base, fact))
                if len(reduced) < len(current_ids):
                    return fact
    return None


def _alternate_decoy_fact(catalog: Catalog, product: Product, signature: list[Fact]) -> Fact | None:
    """Build a false but catalog-backed preference for intent override.

    The decoy is deliberately *not* a signature fact and must be false for
    the target.  Its value is nevertheless observed in another catalog row,
    which lets the validator distinguish a genuine old preference from an
    invented string.
    """

    for original in signature:
        if original.field == "category":
            values = sorted({candidate.category_leaf for candidate in catalog.all() if candidate.category_leaf})
            candidates = [value for value in values if normalize_text(value) != normalize_text(original.value)]
        elif original.field == "store":
            values = sorted({candidate.store for candidate in catalog.all() if candidate.store})
            candidates = [value for value in values if normalize_text(value) != normalize_text(original.value)]
        elif original.field.startswith("detail:"):
            key = normalize_text(original.field.split(":", 1)[1])
            values: set[str] = set()
            for candidate in catalog.all():
                for detail_key, detail_value in candidate.details.items():
                    if normalize_text(detail_key) == key and detail_value:
                        values.add(detail_value)
            candidates = sorted(value for value in values if normalize_text(value) != normalize_text(original.value))
        elif original.field == "title_token":
            values = sorted({token for candidate in catalog.all() for token in candidate.title.casefold().split()})
            candidates = [value for value in values if normalize_text(value) != normalize_text(original.value) and len(normalize_text(value)) >= 4]
        else:
            continue
        for value in candidates:
            decoy = Fact(
                field=original.field,
                operator=original.operator,
                value=normalize_text(value),
                source="decoy",
                attribute=original.attribute,
                polarity="decoy",
                display=str(value),
            )
            if not match_fact(product, decoy) and catalog.candidate_ids([decoy]):
                return decoy
    return None


class TargetGenerator:
    def __init__(self, catalog: Catalog, *, seed: int = 42, config: GeneratorConfig | None = None):
        self.catalog = catalog
        self.seed = int(seed)
        self.config = config or GeneratorConfig()
        self.rng = random.Random(self.seed)

    def _product_facts(self, product: Product) -> list[Fact]:
        facts = extract_facts(product)
        if len(self.catalog) > 1000:
            # The full catalog has enough category/store/detail/title facts
            # for target identification.  Avoid trying hundreds of verbose
            # prose predicates against all 50k rows during greedy search.
            facts = [fact for fact in facts if fact.source not in {"features", "description"}]
        # A large catalog row can contain hundreds of verbose bullets. Keep a
        # deterministic bounded pool and preserve structured/numeric facts.
        structured = [fact for fact in facts if fact.source not in {"features", "description", "title"}]
        text_facts = [fact for fact in facts if fact.source in {"features", "description", "title"}]
        text_facts.sort(key=lambda fact: (fact.source != "title", len(fact.display), fact.fact_id))
        return _dedupe((*structured, *text_facts[: self.config.max_feature_facts_per_product]))

    def _signature(self, product: Product, *, scenario: str) -> tuple[list[Fact], list[Fact]] | None:
        facts = self._product_facts(product)
        category = next((fact for fact in facts if fact.field == "category"), None)
        selected: list[Fact] = [category] if category is not None else []
        selected = _dedupe(selected)
        if scenario == "negative_constraint":
            negative = _safe_negative_fact(self.catalog, product, selected)
            if negative is not None:
                # This scenario is required to contain a genuine negative
                # disclosure, even when the negative predicate is redundant
                # for uniqueness.  Keep it in the signature from the start;
                # minimization below deliberately preserves it.
                selected.append(negative)
                facts.append(negative)
        if scenario == "budget_rating":
            numeric = [fact for fact in facts if fact.field in {"price", "rating", "rating_count"}]
            # Put a useful numeric fact in the pool early without forcing both
            # price and rating when one is absent.
            facts = _dedupe((*numeric, *facts))
        selected_keys = {_fact_key(fact) for fact in selected}
        current = self.catalog.candidate_ids(selected)
        if product.parent_asin not in current:
            return None

        while len(current) > 1 and len(selected) < self.config.max_signature_facts:
            best: tuple[tuple[float, ...], Fact, set[str]] | None = None
            for fact in facts:
                key = _fact_key(fact)
                if key in selected_keys:
                    continue
                # Full title and identifier-like values are excluded by
                # extraction; this second check protects custom fixtures.
                if re.search(r"\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{8,}\b", fact.display):
                    continue
                candidates = self.catalog.candidate_ids((*selected, fact))
                if product.parent_asin not in candidates or len(candidates) >= len(current):
                    continue
                gain = len(current) - len(candidates)
                score = _sort_fact_candidates(len(current), len(candidates), fact, gain)
                if best is None or score > best[0]:
                    best = (score, fact, candidates)
            if best is None:
                return None
            _, fact, current = best
            selected.append(fact)
            selected_keys.add(_fact_key(fact))
        if len(current) != 1 or next(iter(current)) != product.parent_asin:
            return None

        # Remove redundant facts while preserving the ordered, auditable
        # signature.  Keep the category when it exists and is not redundant,
        # because it makes generated language much more natural.
        minimized = list(selected)
        for fact in list(selected):
            if len(minimized) <= 1:
                break
            if scenario == "negative_constraint" and (fact.polarity == "negative" or fact.operator in {"neq", "not_contains", "not"}):
                continue
            if scenario == "budget_rating" and fact.field in {"price", "rating", "rating_count"}:
                continue
            trial = [item for item in minimized if item != fact]
            trial_ids = self.catalog.candidate_ids(trial)
            if len(trial_ids) == 1 and next(iter(trial_ids)) == product.parent_asin:
                minimized = trial
        # Keep a copy of all candidate facts for partitioning and scenario
        # support.  The second return value includes only facts that are true
        # for target; it is not sent to the Agent.
        if scenario == "intent_override" and len(minimized) < 2:
            extras = [fact for fact in facts if _fact_key(fact) not in {_fact_key(item) for item in minimized}]
            extras.sort(key=lambda fact: (fact.source not in {"categories", "details", "store"}, len(fact.display), fact.fact_id))
            for extra in extras:
                if match_fact(product, extra):
                    minimized.append(extra)
                    break
            if len(minimized) < 2:
                return None
        return minimized, facts

    def _initial_prefix(self, signature: list[Fact]) -> tuple[int, int]:
        """Return (visible prefix length, candidate count) with ambiguity."""

        best: tuple[int, int] | None = None
        for length in range(1, len(signature) + 1):
            count = self.catalog.candidate_count(signature[:length])
            if self.config.min_initial_candidates <= count <= self.config.max_initial_candidates and count > 1:
                # Prefer the most informative visible prefix while retaining a
                # real dialogue.
                best = (length, count)
        if best is not None:
            return best
        # A direct-search item may legitimately be unique from the start. If
        # the first fact itself is broad, retain it and let hidden facts finish
        # the identification; validator records the relaxed bound.
        first_count = self.catalog.candidate_count(signature[:1]) if signature else 0
        if first_count > 1:
            return 1, first_count
        return len(signature), self.catalog.candidate_count(signature)

    def _profile(self, facts: list[Fact], *, scenario: str) -> dict[str, Any]:
        tags = [render_fact(fact, variant=index + 1).rstrip(".") for index, fact in enumerate(facts)]
        summary = "I usually shop with a few preferences in mind."
        if tags:
            summary += " " + " ".join(tags)
        average = None
        for fact in facts:
            if fact.field == "rating" and fact.operator in {"ge", "min"}:
                average = float(fact.value)
                break
        return {
            "purchase_frequency": "occasional" if scenario != "budget_rating" else "regular",
            "average_prior_rating": average,
            "rating_style": "quality-conscious" if average is not None else "balanced",
            "preference_tags": tags,
            "summary": summary,
        }

    def _partition(self, product: Product, signature: list[Fact], *, scenario: str) -> tuple[list[Fact], list[Fact], list[Fact], dict[str, Any] | None, int]:
        if not signature:
            raise GenerationError("empty signature")
        if scenario == "direct_search":
            return list(signature), [], [], None, self.catalog.candidate_count(signature)
        if scenario == "intent_override":
            if len(signature) < 2:
                raise GenerationError("intent_override requires at least two target facts")
            # Keep the new target fact hidden until the explicit override at
            # turn two.  The old preference is a separate false decoy.
            visible_length = min(max(self._initial_prefix(signature)[0], 1), len(signature) - 1)
            visible = list(signature[:visible_length])
            hidden = list(signature[visible_length:])
            new_fact = hidden[0]
            old_fact = _alternate_decoy_fact(self.catalog, product, signature)
            if old_fact is None:
                raise GenerationError("intent_override has no catalog-backed false decoy")
            profile: list[Fact] = []
            query = visible
            if not query:
                query = [hidden.pop(0)]
                new_fact = hidden[0] if hidden else query[-1]
            query_text = render_facts(query)
            # The extra decoy prose is appended by _sample_for, where the
            # final query variant/index is available.
            override = {
                "turn": 2,
                "old_fact": old_fact.to_dict(),
                "new_fact": new_fact.to_dict(),
                "replaces_fact_id": old_fact.fact_id,
                "message": f"Actually, ignore that earlier preference for {old_fact.display}. Please prioritize {new_fact.display} instead.",
            }
            initial_count = self.catalog.candidate_count((*query, *profile))
            return query, profile, hidden, override, initial_count
        visible_length, initial_count = self._initial_prefix(signature)
        visible_length = min(max(visible_length, 1), len(signature))
        visible = signature[:visible_length]
        hidden = signature[visible_length:]
        if not hidden and scenario in {"clarification_required", "profile_hidden", "multi_constraint", "negative_constraint", "budget_rating", "intent_override"}:
            # Make the last fact a clarification even when a minimized
            # signature is short. This does not change the full predicate.
            if len(visible) > 1:
                hidden = [visible.pop()]
                initial_count = self.catalog.candidate_count(visible)
        profile_count = 0
        if scenario in {"profile_hidden", "multi_constraint", "budget_rating", "intent_override"} and len(visible) > 1:
            profile_count = 1
        if scenario == "clarification_required" and len(visible) > 2:
            profile_count = 1
        query = visible[: len(visible) - profile_count]
        profile = visible[len(visible) - profile_count:] if profile_count else []
        if not query:
            query = profile[:1]
            profile = profile[1:]
        override: dict[str, Any] | None = None
        return query, profile, hidden, override, initial_count

    def _sample_for(self, product: Product, index: int, scenario: str) -> Sample | None:
        result = self._signature(product, scenario=scenario)
        if result is None:
            return None
        signature, _all_facts = result
        # The optional scenario must be supported by the actual signature.
        if scenario == "budget_rating" and not any(fact.field in {"price", "rating"} for fact in signature):
            return None
        if scenario == "negative_constraint" and not any(fact.polarity == "negative" for fact in signature):
            return None
        try:
            query_facts, profile_facts, clarification_facts, override, initial_count = self._partition(product, signature, scenario=scenario)
        except GenerationError:
            return None
        if scenario == "clarification_required" and not clarification_facts:
            return None
        if scenario == "profile_hidden" and not profile_facts:
            return None
        if scenario not in {"direct_search"} and not (self.config.min_initial_candidates <= initial_count <= self.config.max_initial_candidates):
            return None
        full_count = self.catalog.candidate_count((*query_facts, *profile_facts, *clarification_facts))
        if full_count != 1 or self.catalog.candidate_ids((*query_facts, *profile_facts, *clarification_facts)) != {product.parent_asin}:
            return None
        query = render_facts(query_facts, variant=index)
        if scenario == "intent_override" and override:
            old = Fact.from_dict(override["old_fact"])
            query = f"{query} I initially thought I wanted {old.display}, but I am still deciding."
        if not query:
            return None
        profile = self._profile(profile_facts, scenario=scenario)
        clarification_replies = [
            {"fact_id": fact.fact_id, "message": render_fact(fact, variant=index + offset, reply=True)}
            for offset, fact in enumerate(clarification_facts)
        ]
        simulator_config = {
            "max_turns": 10,
            "slot_count": len(clarification_facts),
            "routing": "structured_plus_semantic",
            "clarification_replies": clarification_replies,
        }
        if override:
            simulator_config["override_new_fact_id"] = override.get("new_fact", {}).get("fact_id")
        sample_seed = self.seed * 1000003 + index
        sample = Sample(
            sample_id=f"nlb-{index:05d}-{stable_id(self.seed, index, product.parent_asin)}",
            seed=sample_seed,
            scenario_type=scenario,
            target_parent_asin=product.parent_asin,
            query=query,
            user_profile=profile,
            signature=signature,
            query_facts=query_facts,
            profile_facts=profile_facts,
            clarification_facts=clarification_facts,
            simulator=simulator_config,
            override=override,
            metadata={
                "generator": "deterministic-v1",
                "initial_candidate_count": initial_count,
                "full_candidate_count": full_count,
                "target_category": product.category_leaf,
                "signature_size": len(signature),
                "visible_fact_count": len(query_facts) + len(profile_facts),
                "clarification_fact_count": len(clarification_facts),
            },
            schema_version=1,
        )
        return sample

    def generate(self, count: int, *, scenarios: Iterable[str] = SCENARIOS, require_coverage: bool | None = None) -> list[Sample]:
        count = int(count)
        if count < 1:
            return []
        requested = tuple(str(item) for item in scenarios)
        if not requested:
            requested = SCENARIOS
        if any(item not in SCENARIOS for item in requested):
            raise GenerationError(f"unsupported requested scenario(s): {sorted(set(requested) - set(SCENARIOS))}")
        if require_coverage is None:
            require_coverage = requested == SCENARIOS and count >= len(SCENARIOS)
        ids = list(self.catalog.ids)
        # Sorting before sampling means JSONL order is stable across Python
        # processes and hash randomization settings.
        ids.sort()
        self.rng.shuffle(ids)
        samples: list[Sample] = []
        used: set[str] = set()
        attempts = 0
        cursor = 0
        while len(samples) < count and attempts < count * self.config.max_attempts_per_sample:
            if cursor >= len(ids):
                self.rng.shuffle(ids)
                cursor = 0
            parent_asin = ids[cursor]
            cursor += 1
            attempts += 1
            if parent_asin in used:
                continue
            scenario = requested[len(samples) % len(requested)]
            product = self.catalog.products[parent_asin]
            sample = self._sample_for(product, len(samples), scenario)
            if sample is None:
                continue
            samples.append(sample)
            used.add(parent_asin)
        if len(samples) < count:
            raise GenerationError(
                f"could generate {len(samples)} of {count} samples after {attempts} attempts for requested scenarios {requested}; "
                "try a smaller count or wider initial-candidate bounds"
            )
        if require_coverage:
            missing = sorted(set(requested) - {sample.scenario_type for sample in samples})
            if missing:
                raise GenerationError(f"scenario coverage incomplete; missing {missing}")
        return samples


def generate_samples(catalog: Catalog, count: int, *, seed: int = 42, config: GeneratorConfig | None = None, scenarios: Iterable[str] = SCENARIOS, require_coverage: bool | None = None) -> list[Sample]:
    return TargetGenerator(catalog, seed=seed, config=config).generate(count, scenarios=scenarios, require_coverage=require_coverage)
