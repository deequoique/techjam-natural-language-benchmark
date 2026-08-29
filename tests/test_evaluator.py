import unittest

from nl_benchmark.catalog import Catalog
from nl_benchmark.evaluator import evaluate_sample
from nl_benchmark.generator import generate_samples
from nl_benchmark.metrics import summarize

from test_catalog_and_facts import fixture_products


class SpyAgent:
    def __init__(self, target, wrong=None):
        self.target = target
        self.wrong = wrong or "P000000004"
        self.inputs = []
        self.profile = None

    def reset(self, session_id, user_profile):
        self.profile = user_profile

    def respond(self, session_id, user_message, turn, top_k):
        self.inputs.append(str(user_message))
        return {
            "message": "Here is a recommendation.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": self.target}, {"parent_asin": self.wrong}],
        }


class WrongAgent(SpyAgent):
    def respond(self, session_id, user_message, turn, top_k):
        self.inputs.append(str(user_message))
        return {
            "message": "No more questions.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": self.wrong}],
        }


class DiagnosticAgent(WrongAgent):
    def respond(self, session_id, user_message, turn, top_k):
        response = super().respond(session_id, user_message, turn, top_k)
        self.last_diagnostics = {
            "stages": {
                "retrieved_ids": [self.target, self.wrong],
                "feature_input_ids": [self.target, self.wrong],
                "feature_ranked_ids": [self.wrong, self.target],
                "semantic_input_ids": [self.wrong],
                "semantic_ranked_ids": [self.wrong],
                "final_ids": [self.wrong],
            }
        }
        return response


class PreOverrideHitAgent(SpyAgent):
    def respond(self, session_id, user_message, turn, top_k):
        self.inputs.append(str(user_message))
        return {
            "message": "Here is a recommendation.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": self.target}],
        }


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog.from_products(fixture_products())
        self.sample = generate_samples(self.catalog, 1, seed=9, scenarios=("direct_search",))[0]

    def test_exact_target_hit_and_isolation(self):
        agent = SpyAgent(self.sample.target_parent_asin)
        result = evaluate_sample(agent, self.catalog, self.sample)
        self.assertFalse(result["errors"], result["errors"])
        self.assertNotIn(self.sample.target_parent_asin, " ".join(agent.inputs))
        self.assertEqual(result["trace"][0]["recommendations"][0]["parent_asin"], self.sample.target_parent_asin)
        self.assertEqual(summarize([result])["hit_at_10"], 1.0)
        self.assertEqual(summarize([result])["exact_top1"], 1.0)

    def test_wrong_similar_product_is_an_exact_miss(self):
        agent = WrongAgent(self.sample.target_parent_asin)
        result = evaluate_sample(agent, self.catalog, self.sample)
        summary = summarize([result])
        self.assertEqual(summary["hit_at_10"], 0.0)
        self.assertEqual(summary["mrr"], 0.0)
        self.assertEqual(summary["mttc"], 11.0)

    def test_pre_override_hit_is_excluded_from_exact_metrics(self):
        override_sample = generate_samples(self.catalog, 1, seed=7, scenarios=("intent_override",))[0]
        agent = PreOverrideHitAgent(override_sample.target_parent_asin)
        result = evaluate_sample(agent, self.catalog, override_sample)
        self.assertEqual(result["metrics_start_turn"], 2)
        self.assertEqual(len(result["trace"]), 2)
        summary = summarize([result])
        self.assertEqual(summary["hit_at_10"], 1.0)
        self.assertEqual(summary["exact_top1"], 1.0)
        self.assertEqual(summary["mttc"], 2.0)

    def test_parent_adds_target_stage_ranks_without_sending_target(self):
        agent = DiagnosticAgent(self.sample.target_parent_asin)
        result = evaluate_sample(agent, self.catalog, self.sample)
        analysis = result["trace"][0]["diagnostics"]["target_analysis"]
        self.assertTrue(analysis["target_in_retrieval"])
        self.assertEqual(analysis["target_retrieval_rank"], 1)
        self.assertEqual(analysis["target_feature_rank"], 2)
        self.assertIsNone(analysis["target_semantic_input_rank"])
        self.assertIsNone(analysis["target_final_rank"])
        self.assertNotIn(self.sample.target_parent_asin, " ".join(agent.inputs))
