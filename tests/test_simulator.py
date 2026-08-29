import unittest

from nl_benchmark.catalog import Catalog
from nl_benchmark.facts import render_fact
from nl_benchmark.generator import generate_samples
from nl_benchmark.simulator import IntelligentSimulator

from test_catalog_and_facts import fixture_products


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog.from_products(fixture_products())
        self.sample = generate_samples(self.catalog, 1, seed=5, scenarios=("clarification_required",))[0]

    def test_natural_language_question_routes_to_hidden_fact(self):
        simulator = IntelligentSimulator(self.catalog, self.sample)
        hidden = self.sample.clarification_facts[0] if self.sample.clarification_facts else self.sample.signature[-1]
        reply = simulator.answer(None, f"Could you tell me about the {hidden.attribute}, please?", turn=1)
        self.assertIn(reply.status, {"revealed", "unsupported"})
        if self.sample.clarification_facts:
            self.assertEqual(reply.status, "revealed")
            self.assertEqual(reply.revealed_fact_ids, [hidden.fact_id])
            self.assertNotIn(self.sample.target_parent_asin, reply.message)

    def test_structured_question_and_repeat_boundary(self):
        simulator = IntelligentSimulator(self.catalog, self.sample)
        attribute = self.sample.clarification_facts[0].attribute if self.sample.clarification_facts else "feature"
        reply = simulator.answer(attribute, f"What {attribute} matters most?", turn=1)
        if self.sample.clarification_facts:
            self.assertEqual(reply.status, "revealed")
            repeat = simulator.answer(attribute, f"What {attribute} matters most?", turn=2)
            self.assertEqual(repeat.status, "repeated")

    def test_no_preference_and_unsupported_do_not_reveal(self):
        simulator = IntelligentSimulator(self.catalog, self.sample)
        before = simulator.candidate_count
        no_preference = simulator.answer("color", "I don't care about color.", turn=1)
        self.assertEqual(no_preference.status, "no_preference")
        self.assertEqual(simulator.candidate_count, before)
        unsupported = simulator.answer("not-a-real-attribute", "Tell me the moon phase.", turn=2)
        self.assertEqual(unsupported.status, "unsupported")

    def test_boundaries_require_a_real_question_and_do_not_leak_for_other(self):
        simulator = IntelligentSimulator(self.catalog, self.sample)
        self.assertEqual(simulator.answer(None, "I prefer something nice.", turn=1).status, "unsupported")
        simulator = IntelligentSimulator(self.catalog, self.sample)
        self.assertEqual(simulator.answer("other", "", turn=1).status, "unsupported")
        simulator = IntelligentSimulator(self.catalog, self.sample)
        self.assertEqual(simulator.answer(None, "What redness should it have?", turn=1).status, "unsupported")

    def test_intent_override_updates_hidden_state(self):
        from nl_benchmark.generator import generate_samples
        override_sample = generate_samples(self.catalog, 1, seed=7, scenarios=("intent_override",))[0]
        simulator = IntelligentSimulator(self.catalog, override_sample)
        before = simulator.candidate_count
        reply = simulator.apply_override(override_sample.override)
        self.assertEqual(reply.status, "intent_override")
        self.assertLessEqual(simulator.candidate_count, before)
        self.assertIn(reply.revealed_fact_ids[0], {fact.fact_id for fact in override_sample.clarification_facts})

    def test_exhaustion_boundary(self):
        simulator = IntelligentSimulator(self.catalog, self.sample)
        for index, fact in enumerate(self.sample.clarification_facts, 1):
            simulator.answer(fact.attribute, f"Please tell me about {fact.attribute}.", turn=index)
        if self.sample.clarification_facts:
            reply = simulator.answer("other", "What else can you tell me?", turn=len(self.sample.clarification_facts) + 1)
            self.assertEqual(reply.status, "exhausted")
