import unittest

from nl_benchmark.catalog import Catalog
from nl_benchmark.facts import extract_facts
from nl_benchmark.facts import classify_attribute
from nl_benchmark.schema import Fact


def fixture_products():
    return [
        {
            "parent_asin": "P000000001",
            "title": "Northstar Trail Runner Red",
            "features": ["Waterproof shell for rainy runs", "Reflective heel detail"],
            "description": ["A lightweight trail runner for wet weather."],
            "price": 79.0,
            "categories": ["Clothing", "Shoes", "Running Shoes"],
            "details": {"Color": "Red", "Material": "Nylon", "Size": "10", "Style": "Trail"},
            "average_rating": 4.6,
            "rating_number": 1200,
            "store": "Northstar",
        },
        {
            "parent_asin": "P000000002",
            "title": "Northstar Trail Runner Blue",
            "features": ["Waterproof shell for rainy runs", "Reflective heel detail"],
            "description": ["A lightweight trail runner for wet weather."],
            "price": 79.0,
            "categories": ["Clothing", "Shoes", "Running Shoes"],
            "details": {"Color": "Blue", "Material": "Nylon", "Size": "10", "Style": "Trail"},
            "average_rating": 4.6,
            "rating_number": 1200,
            "store": "Northstar",
        },
        {
            "parent_asin": "P000000003",
            "title": "Northstar Trail Runner Red Wide",
            "features": ["Waterproof shell for rainy runs", "Reflective heel detail"],
            "description": ["A lightweight trail runner for wet weather."],
            "price": 89.0,
            "categories": ["Clothing", "Shoes", "Running Shoes"],
            "details": {"Color": "Red", "Material": "Nylon", "Size": "11", "Style": "Trail"},
            "average_rating": 4.7,
            "rating_number": 1300,
            "store": "Northstar",
        },
        {
            "parent_asin": "P000000004",
            "title": "City Walker Black Sneaker",
            "features": ["Breathable mesh upper", "Everyday comfort"],
            "description": ["A casual sneaker for city walking."],
            "price": 49.0,
            "categories": ["Clothing", "Shoes", "Sneakers"],
            "details": {"Color": "Black", "Material": "Mesh", "Size": "10", "Style": "Casual"},
            "average_rating": 4.2,
            "rating_number": 200,
            "store": "City Walker",
        },
    ]


class CatalogAndFactsTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog.from_products(fixture_products())

    def test_text_and_numeric_matching(self):
        color = Fact("detail:color", "eq", "red", "details", attribute="color", display="Red")
        budget = Fact("price", "le", 80, "price", attribute="budget", display="80")
        self.assertEqual(self.catalog.candidate_ids([color]), {"P000000001", "P000000003"})
        self.assertEqual(self.catalog.candidate_ids([color, budget]), {"P000000001"})

    def test_feature_and_title_facts_are_grounded(self):
        product = self.catalog.get("P000000001")
        facts = extract_facts(product)
        self.assertTrue(any(fact.field == "feature" for fact in facts))
        self.assertTrue(any(fact.field == "title_token" and fact.value == "northstar" for fact in facts))
        self.assertFalse(any("P000000001" in fact.display for fact in facts))

    def test_catalog_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            Catalog.from_products([fixture_products()[0], fixture_products()[0]])

    def test_detail_attribute_classification_preserves_field_delimiter(self):
        self.assertEqual(classify_attribute("detail:Color", "red"), "color")
        self.assertEqual(classify_attribute("detail:Size", "10"), "size")
        self.assertEqual(classify_attribute("detail:Material", "nylon"), "material")
        self.assertEqual(classify_attribute("detail:Brand", "Northstar"), "brand")
