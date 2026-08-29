"""Grounded fact extraction and deterministic natural-language rendering."""

from __future__ import annotations

import math
import re
from typing import Iterable

from .catalog import Product, parse_number, tokens
from .schema import Fact, compact_text, normalize_text


# Product identifiers such as ASINs normally mix letters and digits.  A plain
# long English word (for example “waterproof”) must not be treated as an ID.
ID_RE = re.compile(r"\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{8,}\b")
URL_RE = re.compile(r"(?:https?://|www\.)", re.I)
GENERIC_TOKENS = {
    "women", "woman", "womens", "men", "mens", "adult", "kids", "child",
    "clothing", "shoes", "jewelry", "fashion", "style", "classic", "new",
    "quality", "premium", "great", "best", "design", "comfortable", "comfort",
    "available", "imported", "machine", "wash", "closure", "pack", "piece",
}
UNSAFE_DETAIL_KEYS = (
    "asin", "model", "item model", "date first", "date last", "best sellers",
    "dimensions", "weight", "manufacturer part", "upc", "isbn", "sku", "rank",
    "batteries", "country of origin", "recommended age",
)
ATTRIBUTE_ALIASES = {
    "category": ("category", "type", "kind", "what is it"),
    "material": ("material", "fabric", "made of", "made from", "composition"),
    "color": ("color", "colour", "shade", "tone", "look"),
    "size": ("size", "sizing", "fit", "dimensions", "measurement"),
    "style": ("style", "pattern", "design", "shape", "closure"),
    "brand": ("brand", "maker", "manufacturer", "store", "label"),
    "budget": ("price", "cost", "budget", "under", "spend", "affordable"),
    "feature": ("feature", "specification", "detail", "function"),
    "use_case": ("use case", "occasion", "activity", "travel", "sport"),
    "other": ("anything else", "other", "preference"),
}


def is_identifier_like(value: object) -> bool:
    text = str(value or "")
    return bool(ID_RE.search(text) or URL_RE.search(text))


def _safe_detail(key: str, value: str) -> bool:
    lowered = normalize_text(key)
    if not lowered or not value or len(value) > 180 or is_identifier_like(value):
        return False
    return not any(term in lowered for term in UNSAFE_DETAIL_KEYS)


def is_safe_detail(key: str, value: str) -> bool:
    """Public wrapper used by generation and validation of detail facts."""

    return _safe_detail(key, value)


def classify_attribute(field: str, value: object = "") -> str:
    # Keep the delimiter while classifying detail fields.  Calling
    # normalize_text first turns ``detail:Color`` into ``detail color`` and
    # used to make every detail fall through to ``feature``.
    raw_field = str(field or "").casefold().strip()
    lowered = normalize_text(raw_field)
    text = normalize_text(value)
    if lowered in {"price", "rating", "rating_count"}:
        return "budget" if lowered == "price" else "feature"
    if lowered in {"category", "category_path"}:
        return "category"
    if lowered == "store" or lowered.startswith("detail:brand"):
        return "brand"
    if lowered in {"material"} or "material" in lowered or "fabric" in lowered:
        return "material"
    if lowered == "color" or "color" in lowered or "colour" in lowered:
        return "color"
    if lowered == "title_token":
        return "category" if text in {"shirt", "dress", "shoe", "shoes", "earring", "bracelet", "bag"} else "feature"
    if raw_field.startswith("detail:"):
        key = normalize_text(raw_field.split(":", 1)[1])
        if any(word in key for word in ("size", "fit", "length", "width")):
            return "size"
        if any(word in key for word in ("style", "pattern", "shape", "closure", "sole", "neck", "sleeve")):
            return "style"
        if any(word in key for word in ("material", "fabric", "fiber")):
            return "material"
        if any(word in key for word in ("color", "colour", "shade")):
            return "color"
        if any(word in key for word in ("brand", "manufacturer", "department")):
            return "brand"
        return "feature"
    if any(word in text for word in ("running", "hiking", "workout", "gym", "travel", "camp", "office", "outdoor")):
        return "use_case"
    return "feature"


def _fact(field: str, operator: str, value: str | float | int, source: str, *, display: str | None = None, attribute: str | None = None, polarity: str = "positive") -> Fact:
    return Fact(
        field=field,
        operator=operator,
        value=value,
        source=source,
        attribute=attribute or classify_attribute(field, value),
        polarity=polarity,
        display=compact_text(display if display is not None else value, 240),
    )


def _numeric_facts(product: Product) -> Iterable[Fact]:
    if product.price is not None and product.price >= 0:
        # The ceiling keeps the utterance natural and remains a truthful
        # constraint for the target.  Exact prices are retained only when the
        # amount is already an integer (useful in tiny fixtures).
        step = 1.0 if product.price < 20 else 5.0
        threshold = math.ceil((product.price - 1e-9) / step) * step
        yield _fact("price", "le", round(threshold, 2), "price", attribute="budget")
    if product.rating is not None and product.rating >= 0:
        threshold = math.floor((product.rating + 1e-9) * 10.0) / 10.0
        yield _fact("rating", "ge", round(threshold, 1), "rating", attribute="feature")
    if product.rating_count is not None and product.rating_count > 0:
        thresholds = (10, 25, 50, 100, 250, 500, 1000, 5000, 10000)
        threshold = max((item for item in thresholds if item <= product.rating_count), default=1)
        yield _fact("rating_count", "ge", threshold, "rating_count", attribute="feature")


def extract_facts(product: Product) -> list[Fact]:
    """Extract bounded, safe predicates that are true for ``product``.

    We intentionally expose only a handful of detail and text facts.  The
    generator can then choose a discriminative conjunction without putting a
    raw title, ASIN, model number, or URL in the user-visible conversation.
    """

    facts: list[Fact] = []
    if product.category_leaf:
        facts.append(_fact("category", "eq", normalize_text(product.category_leaf), "categories", attribute="category", display=product.category_leaf))
    if len(product.categories) >= 2:
        path = " / ".join(product.categories[-2:])
        facts.append(_fact("category_path", "eq", normalize_text(path), "categories", attribute="category", display=path))
    if product.store and not is_identifier_like(product.store):
        facts.append(_fact("store", "eq", normalize_text(product.store), "store", attribute="brand", display=product.store))

    for key, value in product.details.items():
        if not _safe_detail(key, value):
            continue
        # Very short administrative values do not make a useful natural
        # constraint and often produce an accidental universal predicate.
        if len(normalize_text(value)) < 2 or normalize_text(value) in {"yes", "no", "none", "n a"}:
            continue
        field = f"detail:{normalize_text(key)}"
        facts.append(_fact(field, "eq", normalize_text(value), "details", attribute=classify_attribute(field, value), display=value))

    for feature in product.features:
        cleaned = compact_text(feature, 220)
        normalized = normalize_text(cleaned)
        if len(normalized) < 8 or is_identifier_like(cleaned):
            continue
        # Skip boilerplate-only bullets but retain concrete specifications and
        # use-case language.
        feature_tokens = set(tokens(cleaned))
        if not feature_tokens or feature_tokens <= GENERIC_TOKENS:
            continue
        facts.append(_fact("feature", "contains", normalized, "features", attribute=classify_attribute("feature", cleaned), display=cleaned))

    for token in tokens(product.title):
        if token in GENERIC_TOKENS or len(token) < 4 or is_identifier_like(token):
            continue
        facts.append(_fact("title_token", "eq", token, "title", attribute=classify_attribute("title_token", token), display=token))

    for description in product.description[:3]:
        cleaned = compact_text(description, 180)
        normalized = normalize_text(cleaned)
        if len(normalized) >= 18 and not is_identifier_like(cleaned):
            # Full description phrases are deliberately lower priority than
            # structured fields, but can rescue a target with sparse metadata.
            facts.append(_fact("description", "contains", normalized, "description", attribute=classify_attribute("description", cleaned), display=cleaned))

    facts.extend(_numeric_facts(product))
    unique: dict[str, Fact] = {}
    for fact in facts:
        unique.setdefault(fact.fact_id, fact)
    return list(unique.values())


def fact_aliases(fact: Fact) -> tuple[str, ...]:
    """Terms used by the deterministic natural-language question router."""

    aliases = list(ATTRIBUTE_ALIASES.get(fact.attribute, ()))
    if fact.field.startswith("detail:"):
        key = fact.field.split(":", 1)[1].replace("_", " ")
        aliases.extend(key.split())
        aliases.append(key)
    elif fact.field == "category_path":
        aliases.extend(("category", "type", "kind"))
    elif fact.field == "store":
        aliases.extend(("brand", "maker", "store"))
    elif fact.field == "title_token":
        aliases.extend(("name", "listing", "title"))
    elif fact.field == "feature":
        aliases.extend(("feature", "specification", "detail"))
    elif fact.field == "description":
        aliases.extend(("description", "details", "feature"))
    elif fact.field == "price":
        aliases.extend(("price", "cost", "budget"))
    elif fact.field == "rating":
        aliases.extend(("rating", "stars", "reviews"))
    elif fact.field == "rating_count":
        aliases.extend(("review count", "number of reviews", "popularity"))
    routing_stopwords = {
        "have", "has", "include", "includes", "look", "looking", "use", "using",
        "work", "wear", "need", "want", "find", "tell", "get", "does", "do",
        "can", "could", "would", "please", "something", "thing", "item", "product",
    }
    cleaned: list[str] = []
    for alias in (*aliases, *tokens(fact.display)[:8]):
        normalized = normalize_text(alias)
        if not normalized or normalized in routing_stopwords:
            continue
        cleaned.append(normalized)
    return tuple(dict.fromkeys(cleaned))


def fact_display_matches_value(fact: Fact) -> bool:
    """Prove a display is a faithful presentation of the canonical value."""

    if not str(fact.display or "").strip():
        return False
    if fact.field in {"price", "rating", "rating_count"}:
        try:
            expected = float(fact.value)
            actual = parse_number(fact.display)
        except (TypeError, ValueError):
            return False
        if actual is None:
            return False
        return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
    # Catalog-grounded text facts are normalized before they are stored in
    # ``value``.  Case, punctuation, and path separators may vary, but the
    # semantic token sequence must be identical.
    return normalize_text(fact.display) == normalize_text(fact.value)


def render_fact(fact: Fact, *, variant: int = 0, reply: bool = False) -> str:
    """Render one fact with a deterministic paraphrase variant."""

    display = fact.display or str(fact.value)
    if fact.field == "category":
        forms = (f"I'm looking for {display}.", f"I need something in the {display} category.")
    elif fact.field == "category_path":
        forms = (f"It should be in the {display} area.", f"Please focus on {display}.")
    elif fact.field == "store":
        forms = (f"I usually prefer {display} as the brand.", f"Please prioritize the {display} label.")
    elif fact.field == "price" and fact.operator in {"le", "max"}:
        forms = (f"I'd like to keep the price under ${float(fact.value):.2f}.", f"My budget is no more than ${float(fact.value):.2f}.")
    elif fact.field == "rating" and fact.operator in {"ge", "min"}:
        forms = (f"I would like at least {float(fact.value):.1f} stars.", f"Please look for something rated {float(fact.value):.1f} stars or higher.")
    elif fact.field == "rating_count" and fact.operator in {"ge", "min"}:
        forms = (f"At least {int(float(fact.value))} reviews would make me more comfortable.", f"I prefer a product with {int(float(fact.value))}+ reviews.")
    elif fact.field.startswith("detail:") and fact.attribute == "color":
        forms = (f"A {display} color would be ideal.", f"I'd prefer it in {display}.")
    elif fact.attribute == "material":
        forms = (f"I prefer {display} material.", f"It should be made from {display}.")
    elif fact.field == "title_token":
        forms = (f"The listing mentions {display}.", f"I remember seeing {display} in the product name.")
    elif fact.field == "feature":
        forms = (f"One important detail is: {display}.", f"Please make sure it has this feature: {display}.")
    elif fact.field == "description":
        forms = (f"The description should mention {display}.", f"I am looking for something described as {display}.")
    else:
        forms = (f"I'd prefer {display}.", f"Please look for {display}.")
    rendered = forms[variant % len(forms)]
    if fact.operator in {"neq", "not_contains", "not"} or fact.polarity == "negative":
        rendered = f"I don't want {display}."
    if reply:
        if fact.operator in {"neq", "not_contains", "not"} or fact.polarity == "negative":
            return f"I would avoid {display}."
        return f"Yes — {rendered[0].lower() + rendered[1:]}"
    return rendered


def render_facts(facts: Iterable[Fact], *, variant: int = 0, prefix: str = "") -> str:
    rendered = [render_fact(fact, variant=variant + index) for index, fact in enumerate(facts)]
    body = " ".join(rendered).strip()
    return f"{prefix}{body}".strip()


def fact_evidence_in_text(fact: Fact, text: object) -> bool:
    """Check that the serialized rendering carries this fact's evidence.

    This is intentionally stricter than keyword routing: validator callers
    use it to detect a hand-edited query/profile/reply whose prose no longer
    corresponds to its structured fact.  Numeric renderings accept the
    formatting used by :func:`render_fact` while textual facts require their
    normalized display phrase.
    """

    haystack = normalize_text(text)
    if not haystack:
        return False
    if fact.field in {"price", "rating", "rating_count"}:
        try:
            number = float(fact.value)
        except (TypeError, ValueError):
            return False
        candidates = {normalize_text(str(fact.value)), normalize_text(f"{number:g}")}
        if fact.field == "price":
            candidates.add(normalize_text(f"{number:.2f}"))
        elif fact.field == "rating":
            candidates.add(normalize_text(f"{number:.1f}"))
        else:
            candidates.add(normalize_text(str(int(number))))
        return any(candidate and candidate in haystack for candidate in candidates)
    evidence = normalize_text(fact.display or fact.value)
    if not evidence:
        return False
    return evidence in haystack
