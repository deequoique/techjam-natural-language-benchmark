"""Target-safe intelligent simulated user for clarification turns."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .catalog import Catalog
from .facts import ATTRIBUTE_ALIASES, fact_aliases, render_fact
from .schema import Fact, Sample, normalize_text


ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
}
NO_PREFERENCE_RE = re.compile(r"\b(no preference|don't care|do not care|anything is fine|any is fine|up to you|whatever)\b", re.I)
QUESTION_RE = re.compile(r"\?|\b(what|which|can you|could you|would you|do you|tell me|how much|how many|is it|please tell|please specify)\b", re.I)
BROAD_RE = re.compile(r"\b(more|else|specific|detail|details|tell me|what about|which one|recommend|preference|anything else)\b", re.I)


@dataclass
class SimulatorReply:
    message: str
    status: str
    revealed_fact_ids: list[str] = field(default_factory=list)
    matched_attributes: list[str] = field(default_factory=list)
    candidate_count: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "status": self.status,
            "revealed_fact_ids": list(self.revealed_fact_ids),
            "matched_attributes": list(self.matched_attributes),
            "candidate_count": self.candidate_count,
            "diagnostics": dict(self.diagnostics),
        }


def _contains_alias(message: str, alias: str) -> bool:
    if not alias:
        return False
    normalized_message = normalize_text(message)
    normalized_alias = normalize_text(alias)
    if not normalized_alias:
        return False
    # Both values have already been converted to single-space forms, so a
    # boundary-aware regex prevents ``red`` matching ``redwood`` and avoids
    # accidentally routing a question on a substring of an unrelated word.
    return bool(re.search(r"(?<!\w)" + re.escape(normalized_alias) + r"(?!\w)", normalized_message))


def is_question(message: object, ask_attribute: object = None) -> bool:
    if isinstance(ask_attribute, str) and ask_attribute.strip():
        return True
    return bool(QUESTION_RE.search(str(message or "")))


class IntelligentSimulator:
    """Reveal only configured hidden facts in response to valid questions."""

    def __init__(self, catalog: Catalog, sample: Sample, *, max_turns: int = 10):
        self.catalog = catalog
        self.sample = sample
        self.max_turns = max(int(max_turns), 1)
        self.hidden = list(sample.clarification_facts)
        self.disclosed: list[Fact] = [*sample.query_facts, *sample.profile_facts]
        self._disclosed_ids = {fact.fact_id for fact in self.disclosed}
        self._revealed_ids: set[str] = set()
        self._questions: list[str] = []
        self._asked_signatures: set[tuple[str, str]] = set()
        self._turns = 0
        self._candidate_ids = self.catalog.candidate_ids(self.disclosed)
        self._last_reply: SimulatorReply | None = None

    @property
    def candidate_count(self) -> int:
        return len(self._candidate_ids)

    @property
    def exhausted(self) -> bool:
        return all(fact.fact_id in self._disclosed_ids for fact in self.hidden)

    def _boundary(self, status: str, message: str, *, attributes: list[str] | None = None) -> SimulatorReply:
        reply = SimulatorReply(
            message=message,
            status=status,
            matched_attributes=attributes or [],
            candidate_count=self.candidate_count,
            diagnostics={"turn": self._turns, "hidden_remaining": sum(fact.fact_id not in self._disclosed_ids for fact in self.hidden)},
        )
        self._last_reply = reply
        return reply

    def _route(self, ask_attribute: object, message: str) -> list[tuple[float, int, Fact, str]]:
        structured = str(ask_attribute).casefold().strip() if isinstance(ask_attribute, str) else ""
        if structured and structured not in ALLOWED_ATTRIBUTES:
            return []
        question = normalize_text(message)
        routed: list[tuple[float, int, Fact, str]] = []
        before = self.candidate_count
        for index, fact in enumerate(self.hidden):
            if fact.fact_id in self._disclosed_ids:
                continue
            score = 0.0
            reasons: list[str] = []
            if structured:
                if structured == fact.attribute:
                    score += 8.0
                    reasons.append("structured_attribute")
                elif structured == "other":
                    # ``other`` is a boundary value, not permission to leak
                    # an arbitrary hidden slot.  It only becomes useful when
                    # the free-form text is a genuine broad question.
                    if BROAD_RE.search(message):
                        score += 0.5
                elif structured == "use_case" and fact.attribute == "feature":
                    score += 2.5
                elif structured == "feature" and fact.attribute in {"feature", "use_case", "style"}:
                    score += 3.0
            for alias in fact_aliases(fact):
                if _contains_alias(question, alias):
                    # Attribute labels carry more weight than a coincidental
                    # value word (e.g. “black” in an unrelated question).
                    score += 3.0 if alias in ATTRIBUTE_ALIASES.get(fact.attribute, ()) else 1.2
                    reasons.append(alias)
            if score <= 0 and BROAD_RE.search(message):
                if fact.attribute in {"feature", "style", "use_case"} or fact.field in {"category", "store"}:
                    score = 0.7
                    reasons.append("broad_relevant")
            if score <= 0:
                continue
            after = len(self.catalog.candidate_ids((*self.disclosed, fact)))
            gain = max(before - after, 0)
            score += min(gain / max(before, 1), 1.0) * 2.0
            routed.append((score, gain, fact, ",".join(reasons)))
        routed.sort(key=lambda item: (-item[0], -item[1], item[2].fact_id))
        return routed

    def apply_override(self, override: object) -> SimulatorReply:
        """Apply an explicit old-decoy -> new-target transition.

        The override is evaluator-controlled serialized data, not Agent input.
        Only the configured new fact can be disclosed; the old fact is a
        known false preference and is never added to the target candidate
        predicate.
        """

        if not isinstance(override, dict):
            return self._boundary("unsupported", "I could not understand that preference change.")
        new_data = override.get("new_fact")
        if not isinstance(new_data, dict):
            return self._boundary("unsupported", "I could not understand that preference change.")
        try:
            new_fact = Fact.from_dict(new_data)
        except (TypeError, ValueError):
            return self._boundary("unsupported", "I could not understand that preference change.")
        hidden = next((fact for fact in self.hidden if fact.fact_id == new_fact.fact_id), None)
        if hidden is None:
            return self._boundary("unsupported", "That preference change is not available in this scenario.")
        if hidden.fact_id not in self._disclosed_ids:
            self.disclosed.append(hidden)
            self._disclosed_ids.add(hidden.fact_id)
            self._revealed_ids.add(hidden.fact_id)
            self._candidate_ids = self.catalog.candidate_ids(self.disclosed)
        reply = SimulatorReply(
            message=render_fact(hidden, variant=self._turns + 1, reply=True),
            status="intent_override",
            revealed_fact_ids=[hidden.fact_id],
            matched_attributes=[hidden.attribute],
            candidate_count=self.candidate_count,
            diagnostics={"turn": self._turns, "route_reason": "explicit_override", "hidden_remaining": sum(item.fact_id not in self._disclosed_ids for item in self.hidden)},
        )
        self._last_reply = reply
        return reply

    def answer(self, ask_attribute: object = None, message: str = "", *, turn: int | None = None) -> SimulatorReply:
        """Answer one Agent clarification request.

        The method accepts both the structured field from the TechJam
        response and the free-form message, so an Agent can be upgraded from a
        fixed attribute enum to natural-language questions without changing
        the benchmark protocol.
        """

        self._turns = max(self._turns + 1, int(turn or self._turns + 1))
        question = str(message or "")
        if self._turns > self.max_turns:
            return self._boundary("max_turns", "I have shared what I can within this conversation.")
        if NO_PREFERENCE_RE.search(question):
            return self._boundary("no_preference", "I don't have a preference on that point; please use your best judgment.")
        if not is_question(question, ask_attribute):
            return self._boundary("unsupported", "Please ask me a question about one specific product attribute.")
        normalized_question = normalize_text(question)
        structured_key = str(ask_attribute).casefold().strip() if isinstance(ask_attribute, str) else ""
        question_signature = (structured_key, normalized_question)
        if normalized_question and normalized_question in self._questions:
            return self._boundary("repeated", "I already answered that question; please use the detail I shared earlier.")
        if question_signature in self._asked_signatures:
            return self._boundary("repeated", "I already answered that question; please use the detail I shared earlier.")
        self._questions.append(normalized_question)
        self._asked_signatures.add(question_signature)
        if self.exhausted:
            return self._boundary("exhausted", "I have no additional preferences to add.")
        routed = self._route(ask_attribute, question)
        if not routed:
            if isinstance(ask_attribute, str) and ask_attribute and ask_attribute not in ALLOWED_ATTRIBUTES:
                return self._boundary("unsupported", "I can answer questions about category, material, color, size, style, brand, budget, or features.")
            return self._boundary("unsupported", "Could you ask about one specific product attribute, such as material, color, price, or a feature?")
        _, gain, fact, reason = routed[0]
        self.disclosed.append(fact)
        self._disclosed_ids.add(fact.fact_id)
        self._revealed_ids.add(fact.fact_id)
        self._candidate_ids = self.catalog.candidate_ids(self.disclosed)
        reply = SimulatorReply(
            message=render_fact(fact, variant=self._turns, reply=True),
            status="revealed",
            revealed_fact_ids=[fact.fact_id],
            matched_attributes=[fact.attribute],
            candidate_count=self.candidate_count,
            diagnostics={
                "turn": self._turns,
                "route_reason": reason,
                "information_gain": gain,
                "hidden_remaining": sum(item.fact_id not in self._disclosed_ids for item in self.hidden),
            },
        )
        self._last_reply = reply
        return reply

    def next_reply(self, ask_attribute: object = None, message: str = "", *, turn: int | None = None) -> dict[str, Any]:
        return self.answer(ask_attribute, message, turn=turn).to_dict()
