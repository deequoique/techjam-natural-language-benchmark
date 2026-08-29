"""Deterministic, bounded interpretation of an Agent's clarification question."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .facts import ATTRIBUTE_ALIASES
from .schema import normalize_text


ATTRIBUTE_ORDER = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)
ALLOWED_ATTRIBUTES = frozenset(ATTRIBUTE_ORDER)

_QUESTION_RE = re.compile(
    r"\?|\b(?:what|which|who|whose|where|when|how|can\s+you|could\s+you|"
    r"would\s+you|do\s+you|are\s+you|tell\s+me|please\s+(?:tell|specify|share))\b",
    re.I,
)
_BROAD_RE = re.compile(
    r"\b(?:anything\s+else|something\s+else|what\s+else|other\s+preferences?|"
    r"more\s+(?:details?|requirements?|preferences?)|else\s+matters?|"
    r"any\s+other|additional\s+(?:detail|requirement|preference))\b",
    re.I,
)

# These patterns complement the public attribute vocabulary with ordinary
# questions a model is likely to produce.  They only resolve to the existing
# bounded Agent attributes; they never create catalog fields or values.
_ATTRIBUTE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "category": (
        re.compile(r"\b(?:kind|type|category|sort)\s+of\s+(?:item|product|thing)\b", re.I),
        re.compile(r"\bwhat\s+(?:are\s+you\s+)?(?:shopping|looking)\s+for\b", re.I),
    ),
    "material": (
        re.compile(r"\b(?:made\s+(?:of|from)|fabric|composition|material)\b", re.I),
    ),
    "color": (
        re.compile(r"\b(?:colour|color|shade|tone|what\s+look)\b", re.I),
    ),
    "size": (
        re.compile(r"\b(?:size|sizing|fit|width|measurement|how\s+(?:large|small|wide))\b", re.I),
    ),
    "style": (
        re.compile(r"\b(?:style|pattern|design|shape|silhouette|closure)\b", re.I),
    ),
    "brand": (
        re.compile(r"\b(?:brand|maker|manufacturer|label|which\s+company|who\s+makes|made\s+by)\b", re.I),
        re.compile(r"\b(?:where|who)\s+(?:should\s+it|does\s+it|would\s+it)?\s*(?:come|be)\s+from\b", re.I),
    ),
    "budget": (
        re.compile(r"\b(?:price|cost|budget|afford|spend|price\s+range|how\s+much)\b", re.I),
    ),
    "feature": (
        re.compile(r"\b(?:feature|function|specification|capability|rating|rated|stars?|reviews?|reputation|口碑)\b", re.I),
        re.compile(r"\bhow\s+well\s+(?:reviewed|rated)\b", re.I),
    ),
    "use_case": (
        re.compile(r"\b(?:use\s+case|occasion|activity|purpose|where\s+.*\s+use|what\s+.*\s+for|when\s+.*\s+wear)\b", re.I),
    ),
}

_COMPATIBLE = {
    ("feature", "use_case"),
    ("feature", "style"),
    ("use_case", "feature"),
}


def _contains_alias(text: str, alias: str) -> bool:
    normalized_alias = normalize_text(alias)
    return bool(
        normalized_alias
        and re.search(
            r"(?<!\w)" + re.escape(normalized_alias) + r"(?!\w)",
            text,
        )
    )


def _text_attributes(message: str) -> tuple[str, ...]:
    normalized = normalize_text(message)
    found: list[str] = []
    for attribute in ATTRIBUTE_ORDER:
        if attribute == "other":
            continue
        aliases = ATTRIBUTE_ALIASES.get(attribute, ())
        patterns = _ATTRIBUTE_PATTERNS.get(attribute, ())
        if any(_contains_alias(normalized, alias) for alias in aliases) or any(
            pattern.search(message) for pattern in patterns
        ):
            found.append(attribute)
    # Stable public-schema order makes signatures and tie breaking replayable.
    return tuple(item for item in ATTRIBUTE_ORDER if item in found)


@dataclass(frozen=True)
class QuestionInterpretation:
    structured_attribute: str | None
    text_attributes: tuple[str, ...]
    resolved_attributes: tuple[str, ...]
    broad: bool
    question: bool
    conflict: bool
    confidence: float
    reason: str
    semantic_signature: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "structured_attribute": self.structured_attribute,
            "text_attributes": list(self.text_attributes),
            "resolved_attributes": list(self.resolved_attributes),
            "broad": self.broad,
            "question": self.question,
            "conflict": self.conflict,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
            "semantic_signature": self.semantic_signature,
        }


def interpret_question(message: object, ask_attribute: object = None) -> QuestionInterpretation:
    """Resolve a free-form question into the existing bounded attributes."""

    text = str(message or "")[:1200]
    raw_structured = str(ask_attribute).casefold().strip() if isinstance(ask_attribute, str) else ""
    structured = raw_structured if raw_structured in ALLOWED_ATTRIBUTES else None
    invalid_structured = bool(raw_structured and structured is None)
    text_attributes = _text_attributes(text)
    broad = bool(_BROAD_RE.search(text))
    question = bool(structured or _QUESTION_RE.search(text))

    conflict = False
    reason = "unsupported"
    resolved: tuple[str, ...] = ()
    confidence = 0.0
    if invalid_structured:
        reason = "invalid_structured_attribute"
    elif structured == "other":
        if text_attributes:
            resolved = text_attributes
            confidence = 0.88
            reason = "natural_language_over_other"
        elif broad:
            confidence = 0.9
            reason = "broad_question"
    elif structured and text_attributes:
        compatible = tuple(
            item
            for item in text_attributes
            if item == structured or (structured, item) in _COMPATIBLE
        )
        incompatible = tuple(item for item in text_attributes if item not in compatible)
        if incompatible:
            conflict = True
            reason = "structured_text_conflict"
        else:
            resolved = tuple(dict.fromkeys((structured, *compatible)))
            confidence = 0.98
            reason = "structured_text_agree"
    elif structured:
        resolved = (structured,)
        confidence = 0.78
        reason = "structured_only"
    elif text_attributes:
        resolved = text_attributes
        confidence = 0.86 if len(text_attributes) == 1 else 0.74
        reason = "natural_language"
    elif broad:
        confidence = 0.82
        reason = "broad_question"

    if conflict:
        signature = "conflict:" + "+".join(sorted((structured or "none", *text_attributes)))
    elif resolved:
        signature = "ask:" + "+".join(sorted(resolved))
    elif broad:
        signature = "ask:broad"
    else:
        signature = "unsupported"
    return QuestionInterpretation(
        structured_attribute=structured,
        text_attributes=text_attributes,
        resolved_attributes=resolved,
        broad=broad,
        question=question,
        conflict=conflict,
        confidence=confidence,
        reason=reason,
        semantic_signature=signature,
    )


__all__ = [
    "ALLOWED_ATTRIBUTES",
    "QuestionInterpretation",
    "interpret_question",
]
