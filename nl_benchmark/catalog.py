"""Read-only JSONL catalog and deterministic fact matching.

The matching functions in this module are the benchmark's ground-truth
semantics.  Generation, validation, and candidate-pool diagnostics all call
the same code, while validation still performs a fresh pass over the loaded
records rather than trusting serialized candidate counts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .schema import Fact, normalize_text, read_jsonl


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "please", "some", "that", "the", "this", "to", "want", "with",
    "would", "you", "looking", "need", "prefer", "preference", "product",
    "item", "women", "woman", "men", "man", "new", "best", "set",
}


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(
            f"{key} {text_value(item)}"
            for key, item in value.items()
            if item not in (None, "", [])
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(text_value(item) for item in value if item not in (None, ""))
    return str(value)


def tokens(value: object) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in TOKEN_RE.findall(str(value or "").casefold()):
        token = normalize_text(raw)
        if len(token) < 3 or token in STOPWORDS or token in seen or token.isnumeric():
            continue
        seen.add(token)
        result.append(token)
    return result


def parse_number(value: object) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        match = re.search(r"-?\d+(?:[,.]\d+)*", str(value))
        if not match:
            return None
        try:
            parsed = float(match.group(0).replace(",", ""))
        except ValueError:
            return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: object) -> int | None:
    parsed = parse_number(value)
    return int(parsed) if parsed is not None else None


def clean_values(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(
            f"{key}: {text_value(item)}"
            for key, item in value.items()
            if item not in (None, "", [])
        )
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


@dataclass(frozen=True)
class Product:
    parent_asin: str
    title: str
    categories: tuple[str, ...]
    features: tuple[str, ...]
    description: tuple[str, ...]
    details: dict[str, str]
    store: str | None
    price: float | None
    rating: float | None
    rating_count: int | None
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Product":
        parent_asin = str(raw.get("parent_asin") or "").strip()
        if not parent_asin:
            raise ValueError("catalog product is missing parent_asin")
        details_raw = raw.get("details")
        details = {
            str(key): text_value(value).strip()
            for key, value in details_raw.items()
            if value not in (None, "", [])
        } if isinstance(details_raw, Mapping) else {}
        rating = parse_number(raw.get("average_rating", raw.get("rating")))
        rating_count = parse_int(raw.get("rating_number", raw.get("rating_count")))
        if rating_count is None:
            for key, value in details.items():
                lowered = key.casefold()
                if "rating" in lowered and ("count" in lowered or "review" in lowered):
                    rating_count = parse_int(value)
                    if rating_count is not None:
                        break
        return cls(
            parent_asin=parent_asin,
            title=text_value(raw.get("title")).strip(),
            categories=clean_values(raw.get("categories")),
            features=clean_values(raw.get("features")),
            description=clean_values(raw.get("description")),
            details=details,
            store=text_value(raw.get("store")).strip() or None,
            price=parse_number(raw.get("price")),
            rating=rating,
            rating_count=rating_count,
            raw=dict(raw),
        )

    @property
    def category_leaf(self) -> str:
        return self.categories[-1] if self.categories else ""

    @property
    def category_path(self) -> str:
        return " / ".join(self.categories[-3:])

    @property
    def searchable_text(self) -> str:
        return " ".join(
            part for part in (
                self.title,
                *self.categories,
                *self.features,
                *self.description,
                *[f"{key} {value}" for key, value in self.details.items()],
                self.store or "",
            ) if part
        )


def _values_for_field(product: Product, field: str) -> tuple[str, ...]:
    normalized = normalize_text
    if field == "category":
        return (normalized(product.category_leaf),) if product.category_leaf else ()
    if field == "category_path":
        return (normalized(product.category_path),) if product.category_path else ()
    if field == "store":
        return (normalized(product.store),) if product.store else ()
    if field == "title_token":
        return tuple(tokens(product.title))
    if field == "feature":
        values = tuple(normalized(value) for value in product.features)
        return tuple(value for value in values if value)
    if field == "description":
        values = tuple(normalized(value) for value in product.description)
        return tuple(value for value in values if value)
    if field == "material":
        corpus = normalize_text(product.searchable_text)
        known = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric", "linen", "suede", "denim")
        return tuple(value for value in known if f" {value} " in f" {corpus} ")
    if field == "color":
        corpus = normalize_text(product.searchable_text)
        known = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "navy", "beige", "silver", "gold")
        return tuple(value for value in known if f" {value} " in f" {corpus} ")
    if field.startswith("detail:"):
        wanted = normalize_text(field.split(":", 1)[1])
        return tuple(
            normalized(value)
            for key, value in product.details.items()
            if normalized(key) == wanted and normalized(value)
        )
    return ()


def values_for_field(product: Product, field: str) -> tuple[str, ...]:
    """Expose normalized field values for generator/validator audits."""

    return _values_for_field(product, field)


def match_fact(product: Product, fact: Fact) -> bool:
    """Apply one fact predicate to a product using benchmark semantics."""

    operator = fact.operator
    if fact.field == "price":
        actual = product.price
    elif fact.field == "rating":
        actual = product.rating
    elif fact.field == "rating_count":
        actual = product.rating_count
    else:
        actual = None
    if fact.field in {"price", "rating", "rating_count"}:
        expected = parse_number(fact.value)
        if actual is None or expected is None:
            return False if operator != "neq" else True
        if operator in {"eq", "exact"}:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6)
        if operator in {"le", "max"}:
            return float(actual) <= float(expected) + 1e-9
        if operator in {"ge", "min"}:
            return float(actual) + 1e-9 >= float(expected)
        if operator == "neq":
            return not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6)
        return False
    values = set(_values_for_field(product, fact.field))
    expected_text = normalize_text(fact.value)
    if operator in {"eq", "exact"}:
        return expected_text in values
    if operator in {"contains", "has"}:
        return any(expected_text in value for value in values)
    if operator in {"neq", "not_contains", "not"}:
        return expected_text not in values and all(expected_text not in value for value in values)
    return False


class Catalog:
    """Immutable in-memory index over the supplied JSONL catalog."""

    def __init__(self, path: str | Path | None = None, records: Iterable[Mapping[str, Any]] | None = None):
        if path is None and records is None:
            raise ValueError("Catalog requires path or records")
        raw_records = read_jsonl(str(path)) if path is not None else list(records or [])
        products: dict[str, Product] = {}
        for raw in raw_records:
            product = Product.from_mapping(raw)
            if product.parent_asin in products:
                raise ValueError(f"duplicate parent_asin: {product.parent_asin}")
            products[product.parent_asin] = product
        self.path = Path(path) if path is not None else None
        self.products = products
        self.ids = frozenset(products)
        self._index: dict[tuple[str, str, str], frozenset[str]] = {}
        self._build_index()

    @classmethod
    def from_products(cls, products: Iterable[Mapping[str, Any]]) -> "Catalog":
        return cls(records=products)

    def _build_index(self) -> None:
        buckets: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
        for parent_asin, product in self.products.items():
            # Structured fields and title tokens are cheap and cover the
            # normal signature path.  Verbose feature/description text is
            # matched by a bounded scan when requested; indexing every prose
            # bullet in the 58MB catalog makes startup needlessly expensive.
            # ``material`` and ``color`` are also used by generated negative
            # constraints.  Leaving them on the scan path makes a 100-sample
            # dataset repeatedly normalize the searchable text of all 50k
            # products.  Indexing the small, closed vocabularies once keeps
            # candidate semantics identical and makes generation scale.
            for field in (
                "category",
                "category_path",
                "store",
                "title_token",
                "material",
                "color",
            ):
                for value in _values_for_field(product, field):
                    buckets[(field, "eq", value)].add(parent_asin)
            for key, value in product.details.items():
                normalized_key = normalize_text(key)
                normalized_value = normalize_text(value)
                if normalized_key and normalized_value:
                    buckets[(f"detail:{normalized_key}", "eq", normalized_value)].add(parent_asin)
        self._index = {key: frozenset(value) for key, value in buckets.items()}

    def __len__(self) -> int:
        return len(self.products)

    def get(self, parent_asin: str) -> Product | None:
        return self.products.get(str(parent_asin))

    def all(self) -> list[Product]:
        return list(self.products.values())

    def candidate_ids(self, facts: Iterable[Fact]) -> set[str]:
        candidates = set(self.ids)
        for fact in facts:
            if not candidates:
                break
            expected = normalize_text(fact.value)
            if fact.operator in {"eq", "exact"} and fact.field not in {"price", "rating", "rating_count"}:
                candidates.intersection_update(self._index.get((fact.field, "eq", expected), frozenset()))
            elif fact.operator in {"neq", "not_contains", "not"} and fact.field not in {"price", "rating", "rating_count"} and (fact.field, "eq", expected) in self._index:
                # Negation over an indexed exact field is a set difference;
                # scanning verbose product records for every greedy candidate
                # made negative-constraint generation unnecessarily slow.
                candidates.difference_update(self._index[(fact.field, "eq", expected)])
            else:
                candidates = {parent_asin for parent_asin in candidates if match_fact(self.products[parent_asin], fact)}
        return candidates

    def candidate_count(self, facts: Iterable[Fact]) -> int:
        return len(self.candidate_ids(facts))

    def facts_for(self, parent_asin: str) -> list[Fact]:
        from .facts import extract_facts
        product = self.products.get(parent_asin)
        return extract_facts(product) if product is not None else []
