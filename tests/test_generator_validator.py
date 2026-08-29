import unittest

from nl_benchmark.catalog import Catalog, match_fact
from nl_benchmark.generator import GeneratorConfig, GenerationError, TargetGenerator, generate_samples
from nl_benchmark.schema import Fact, Sample, project_for_agent
from nl_benchmark.validator import ValidationConfig, validate_sample

from test_catalog_and_facts import fixture_products


class GeneratorValidatorTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog.from_products(fixture_products())

    def test_generator_builds_unique_target_signature(self):
        config = GeneratorConfig(min_initial_candidates=2, max_initial_candidates=4, max_attempts_per_sample=20)
        samples = generate_samples(self.catalog, 1, seed=7, config=config, scenarios=("clarification_required",))
        sample = samples[0]
        self.assertEqual(self.catalog.candidate_ids(sample.signature), {sample.target_parent_asin})
        result = validate_sample(self.catalog, sample, config=ValidationConfig(min_initial_candidates=2, max_initial_candidates=4))
        self.assertTrue(result.valid, result.errors)
        projection = project_for_agent(sample)
        self.assertNotIn("target_parent_asin", projection)
        self.assertNotIn(sample.target_parent_asin, str(projection))

    def test_ambiguous_signature_is_rejected(self):
        target = self.catalog.get("P000000001")
        category = Fact("category", "eq", "running shoes", "categories", attribute="category", display="Running Shoes")
        sample = Sample(
            sample_id="ambiguous",
            seed=1,
            scenario_type="clarification_required",
            target_parent_asin=target.parent_asin,
            query="I need running shoes.",
            user_profile={"purchase_frequency": "occasional", "average_prior_rating": None, "rating_style": "balanced", "preference_tags": [], "summary": ""},
            signature=[category],
            query_facts=[category],
            profile_facts=[],
            clarification_facts=[],
        )
        result = validate_sample(self.catalog, sample, config=ValidationConfig(enforce_initial_bounds=False))
        self.assertFalse(result.valid)
        self.assertTrue(any("not unique" in error for error in result.errors))

    def test_projection_does_not_include_audit_fields(self):
        samples = generate_samples(self.catalog, 1, seed=3, scenarios=("direct_search",))
        text = str(project_for_agent(samples[0]))
        for forbidden in ("signature", "clarification", "target_parent_asin"):
            self.assertNotIn(forbidden, text)

    def test_explicit_unsupported_scenario_fails_instead_of_relabeling(self):
        tiny = Catalog.from_products([fixture_products()[0]])
        with self.assertRaises(GenerationError):
            generate_samples(
                tiny,
                1,
                seed=1,
                config=GeneratorConfig(max_attempts_per_sample=2),
                scenarios=("clarification_required",),
            )

    def test_intent_override_has_false_old_fact_and_unique_new_transition(self):
        config = GeneratorConfig(min_initial_candidates=2, max_initial_candidates=4, max_attempts_per_sample=30)
        sample = generate_samples(self.catalog, 1, seed=7, config=config, scenarios=("intent_override",))[0]
        override = sample.override
        self.assertIsNotNone(override)
        self.assertEqual(override["turn"], 2)
        old = Fact.from_dict(override["old_fact"])
        new = Fact.from_dict(override["new_fact"])
        self.assertEqual(old.source, "decoy")
        self.assertFalse(match_fact(self.catalog.get(sample.target_parent_asin), old))
        self.assertIn(new.fact_id, {fact.fact_id for fact in sample.clarification_facts})
        self.assertEqual(self.catalog.candidate_ids(sample.signature), {sample.target_parent_asin})
        result = validate_sample(self.catalog, sample, config=ValidationConfig(min_initial_candidates=2, max_initial_candidates=4))
        self.assertTrue(result.valid, result.errors)

    def test_validator_catches_rendered_text_tampering(self):
        sample = generate_samples(self.catalog, 1, seed=7, scenarios=("intent_override",))[0]
        sample.query = "I need a product."
        result = validate_sample(self.catalog, sample, config=ValidationConfig(enforce_initial_bounds=False))
        self.assertFalse(result.valid)
        self.assertTrue(any("query text does not render" in error or "old decoy" in error for error in result.errors))
        sample = generate_samples(self.catalog, 1, seed=7, scenarios=("intent_override",))[0]
        sample.simulator["clarification_replies"][0]["message"] = "Sure, anything is fine."
        result = validate_sample(self.catalog, sample, config=ValidationConfig(enforce_initial_bounds=False))
        self.assertFalse(result.valid)
        self.assertTrue(any("clarification reply" in error for error in result.errors))
