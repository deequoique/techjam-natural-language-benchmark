"""Versioned, JSON-serialisable contracts used by the benchmark.

The sample deliberately keeps evaluator-only fields beside the frozen
conversation.  Callers must use :func:`project_for_agent` when crossing the
Agent boundary; the projection contains no target, signature, or audit data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping


SCHEMA_VERSION = 1
MAX_TURNS = 10
TOP_K = 10


def normalize_text(value: object) -> str:
    """Return a stable comparison form for catalog and user text."""

    text = str(value or "").casefold()
    # A compiled regex is materially faster than a Python character loop for
    # the verbose feature/description fields in the 50k-row catalog.
    return " ".join(_NON_WORD_RE.sub(" ", text).split())


_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def compact_text(value: object, limit: int = 240) -> str:
    return " ".join(str(value or "").split()).strip()[:limit].rstrip()


def stable_id(*parts: object) -> str:
    payload = "\x1f".join(normalize_text(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Fact:
    """One deterministic, catalog-grounded predicate.

    ``field`` is a matcher field (for example ``category`` or
    ``detail:color``), while ``attribute`` is the coarse Agent-facing routing
    class.  ``value`` is the canonical value used by the matcher.  The
    optional ``display`` value is what natural-language rendering uses and
    never contains an identifier by construction/validation.
    """

    field: str
    operator: str
    value: str | float | int
    source: str
    attribute: str = "other"
    polarity: str = "positive"
    display: str = ""
    fact_id: str = ""

    def __post_init__(self) -> None:
        if not self.display:
            object.__setattr__(self, "display", compact_text(self.value))
        if not self.fact_id:
            object.__setattr__(
                self,
                "fact_id",
                # The rendered token is part of the identity as well as the
                # canonical predicate.  A serialized ``Rain`` fact whose
                # display is changed to ``Black Socks`` therefore cannot keep
                # its original ID and pass validation.
                stable_id(self.field, self.operator, self.value, self.polarity, self.display),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "source": self.source,
            "attribute": self.attribute,
            "polarity": self.polarity,
            "display": self.display,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fact":
        required = ("field", "operator", "value", "source")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"fact missing fields: {', '.join(missing)}")
        value = data["value"]
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError("fact value must be a string or finite number")
        return cls(
            fact_id=str(data.get("fact_id") or ""),
            field=str(data["field"]),
            operator=str(data["operator"]),
            value=value,
            source=str(data["source"]),
            attribute=str(data.get("attribute") or "other"),
            polarity=str(data.get("polarity") or "positive"),
            display=compact_text(data.get("display") or data["value"]),
        )


def _facts(values: object) -> list[Fact]:
    if not isinstance(values, list):
        return []
    return [Fact.from_dict(item) for item in values if isinstance(item, Mapping)]


@dataclass
class Sample:
    """A frozen benchmark item, including evaluator-only ground truth."""

    sample_id: str
    seed: int
    scenario_type: str
    target_parent_asin: str
    query: str
    user_profile: dict[str, Any]
    signature: list[Fact]
    query_facts: list[Fact]
    profile_facts: list[Fact]
    clarification_facts: list[Fact]
    simulator: dict[str, Any] = field(default_factory=dict)
    override: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "seed": self.seed,
            "scenario_type": self.scenario_type,
            "target_parent_asin": self.target_parent_asin,
            "query": self.query,
            "user_profile": self.user_profile,
            "signature": [fact.to_dict() for fact in self.signature],
            "partitions": {
                "query": [fact.to_dict() for fact in self.query_facts],
                "profile": [fact.to_dict() for fact in self.profile_facts],
                "clarification": [fact.to_dict() for fact in self.clarification_facts],
            },
            "simulator": self.simulator,
            "override": self.override,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sample":
        partitions = data.get("partitions")
        if not isinstance(partitions, Mapping):
            raise ValueError("sample partitions must be an object")
        profile = data.get("user_profile")
        if not isinstance(profile, Mapping):
            raise ValueError("sample user_profile must be an object")
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            sample_id=str(data.get("sample_id") or ""),
            seed=int(data.get("seed", 0)),
            scenario_type=str(data.get("scenario_type") or ""),
            target_parent_asin=str(data.get("target_parent_asin") or ""),
            query=str(data.get("query") or ""),
            user_profile=dict(profile),
            signature=_facts(data.get("signature")),
            query_facts=_facts(partitions.get("query")),
            profile_facts=_facts(partitions.get("profile")),
            clarification_facts=_facts(partitions.get("clarification")),
            simulator=dict(data.get("simulator") or {}),
            override=dict(data["override"]) if isinstance(data.get("override"), Mapping) else None,
            metadata=dict(data.get("metadata") or {}),
        )

    def visible_fact_ids(self) -> set[str]:
        return {fact.fact_id for fact in (*self.query_facts, *self.profile_facts)}

    def hidden_fact_ids(self) -> set[str]:
        return {fact.fact_id for fact in self.clarification_facts}


def project_for_agent(sample: Sample) -> dict[str, Any]:
    """Return the only sample view allowed to cross into an Agent.

    Do not add target or signature information here.  Keeping this function
    small and explicit makes the leakage boundary easy to audit and test.
    """

    profile = {
        "purchase_frequency": str(sample.user_profile.get("purchase_frequency", "occasional")),
        "average_prior_rating": sample.user_profile.get("average_prior_rating"),
        "rating_style": str(sample.user_profile.get("rating_style", "balanced")),
        "preference_tags": [str(tag) for tag in sample.user_profile.get("preference_tags", [])],
        "summary": str(sample.user_profile.get("summary", "")),
    }
    return {
        "scenario_type": sample.scenario_type,
        "query": sample.query,
        "user_profile": profile,
        "max_turns": MAX_TURNS,
    }


def write_jsonl(path: str, values: list[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(value)
    return rows
