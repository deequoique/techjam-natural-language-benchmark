"""Target-safe intelligent simulated user for clarification turns."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .catalog import Catalog
from .facts import ATTRIBUTE_ALIASES, fact_aliases, render_fact
from .question_interpreter import (
    ALLOWED_ATTRIBUTES,
    QuestionInterpretation,
    interpret_question,
)
from .schema import Fact, Sample, normalize_text


NO_PREFERENCE_RE = re.compile(r"\b(no preference|don't care|do not care|anything is fine|any is fine|up to you|whatever)\b", re.I)


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
    return interpret_question(message, ask_attribute).question


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
        self._asked_signatures: set[str] = set()
        self._turns = 0
        self._candidate_ids = self.catalog.candidate_ids(self.disclosed)
        self._last_reply: SimulatorReply | None = None

    @property
    def candidate_count(self) -> int:
        return len(self._candidate_ids)

    @property
    def exhausted(self) -> bool:
        return all(fact.fact_id in self._disclosed_ids for fact in self.hidden)

    def _boundary(
        self,
        status: str,
        message: str,
        *,
        attributes: list[str] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> SimulatorReply:
        details = {
            "turn": self._turns,
            "hidden_remaining": sum(
                fact.fact_id not in self._disclosed_ids for fact in self.hidden
            ),
        }
        details.update(diagnostics or {})
        reply = SimulatorReply(
            message=message,
            status=status,
            matched_attributes=attributes or [],
            candidate_count=self.candidate_count,
            diagnostics=details,
        )
        self._last_reply = reply
        return reply

    def _route(
        self,
        facts: list[Fact],
        interpretation: QuestionInterpretation,
        message: str,
        *,
        undisclosed_only: bool,
    ) -> list[tuple[float, int, Fact, str]]:
        question = normalize_text(message)
        routed: list[tuple[float, int, Fact, str]] = []
        before = self.candidate_count
        for fact in facts:
            if undisclosed_only and fact.fact_id in self._disclosed_ids:
                continue
            score = 0.0
            reasons: list[str] = []
            if fact.attribute in interpretation.resolved_attributes:
                score += 8.0
                reasons.append("resolved_attribute")
            elif "use_case" in interpretation.resolved_attributes and fact.attribute == "feature":
                score += 2.5
                reasons.append("compatible_feature")
            elif "feature" in interpretation.resolved_attributes and fact.attribute in {"feature", "use_case", "style"}:
                score += 3.0
                reasons.append("compatible_feature")
            elif interpretation.broad:
                # A genuine "anything else?" question may disclose one
                # arbitrary remaining preference.  It is deliberately weak
                # so a specific natural-language/structured attribute always
                # wins, and semantic-repeat protection prevents an Agent from
                # draining the entire hidden signature by spamming ``other``.
                score += 0.5
                reasons.append("broad_fallback")
            for alias in fact_aliases(fact):
                if _contains_alias(question, alias):
                    # Attribute labels carry more weight than a coincidental
                    # value word (e.g. “black” in an unrelated question).
                    score += 3.0 if alias in ATTRIBUTE_ALIASES.get(fact.attribute, ()) else 1.2
                    reasons.append(alias)
            if score <= 0 and interpretation.broad:
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

    @staticmethod
    def _question_diagnostics(interpretation: QuestionInterpretation) -> dict[str, Any]:
        return {"question_interpretation": interpretation.as_dict()}

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
        interpretation = interpret_question(question, ask_attribute)
        question_diagnostics = self._question_diagnostics(interpretation)
        if self._turns > self.max_turns:
            return self._boundary(
                "max_turns",
                "I have shared what I can within this conversation.",
                diagnostics=question_diagnostics,
            )
        if NO_PREFERENCE_RE.search(question):
            return self._boundary(
                "no_preference",
                "I don't have a preference on that point; please use your best judgment.",
                diagnostics=question_diagnostics,
            )
        if not interpretation.question:
            return self._boundary(
                "unsupported",
                "Please ask me a question about one specific product attribute.",
                diagnostics=question_diagnostics,
            )
        if interpretation.conflict:
            return self._boundary(
                "ambiguous",
                "The question and requested attribute do not match; please ask one clear product question.",
                diagnostics={**question_diagnostics, "reply_reason": "structured_text_conflict"},
            )
        if not interpretation.resolved_attributes and not interpretation.broad:
            return self._boundary(
                "unsupported",
                "Could you ask about one specific product attribute, such as material, color, price, or a feature?",
                diagnostics=question_diagnostics,
            )
        signature = interpretation.semantic_signature
        if signature in self._asked_signatures:
            return self._boundary(
                "repeated",
                "I already answered that question; please use the detail I shared earlier.",
                diagnostics={**question_diagnostics, "reply_reason": "semantic_repeat"},
            )

        routed = self._route(
            self.hidden,
            interpretation,
            question,
            undisclosed_only=True,
        )
        if not routed and interpretation.resolved_attributes and not interpretation.broad:
            # Explicit questions may reconfirm facts that were already present
            # in the query/profile.  This is realistic but does not add new
            # evidence or change the benchmark candidate predicate.
            reconfirmed = self._route(
                self.disclosed,
                interpretation,
                question,
                undisclosed_only=False,
            )
            if reconfirmed:
                _score, _gain, fact, reason = reconfirmed[0]
                self._asked_signatures.add(signature)
                reply = SimulatorReply(
                    message=render_fact(fact, variant=self._turns, reply=True),
                    status="reconfirmed",
                    matched_attributes=[fact.attribute],
                    candidate_count=self.candidate_count,
                    diagnostics={
                        **question_diagnostics,
                        "turn": self._turns,
                        "route_reason": reason,
                        "reply_reason": "reconfirmed_disclosed_fact",
                        "hidden_remaining": sum(
                            item.fact_id not in self._disclosed_ids for item in self.hidden
                        ),
                    },
                )
                self._last_reply = reply
                return reply
        if not routed:
            if self.exhausted:
                return self._boundary(
                    "exhausted",
                    "I have no additional preferences to add.",
                    diagnostics={**question_diagnostics, "reply_reason": "no_hidden_facts"},
                )
            return self._boundary(
                "unsupported",
                "Could you ask about one specific product attribute, such as material, color, price, or a feature?",
                diagnostics={**question_diagnostics, "reply_reason": "no_matching_fact"},
            )
        _, gain, fact, reason = routed[0]
        self._asked_signatures.add(signature)
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
                "reply_reason": "revealed_hidden_fact",
                "information_gain": gain,
                "hidden_remaining": sum(item.fact_id not in self._disclosed_ids for item in self.hidden),
                **question_diagnostics,
            },
        )
        self._last_reply = reply
        return reply

    def next_reply(self, ask_attribute: object = None, message: str = "", *, turn: int | None = None) -> dict[str, Any]:
        return self.answer(ask_attribute, message, turn=turn).to_dict()
