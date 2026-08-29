import tempfile
from pathlib import Path
import sys
import unittest

from nl_benchmark.agent_loader import AgentLoadError, AgentProcessError, SubprocessAgent, load_agent


class AgentLoaderTests(unittest.TestCase):
    def test_rejects_out_of_range_intent_confidence(self):
        with self.assertRaises(AgentProcessError):
            SubprocessAgent("/tmp", "/tmp/does-not-exist", intent_confidence=2.0)

    def test_in_process_external_loading_is_disabled(self):
        with self.assertRaises(AgentLoadError):
            load_agent("/tmp/does-not-matter", "/tmp/does-not-matter/catalog.jsonl")

    def test_loads_external_agent_in_subprocess_without_bytecode_or_state_pollution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            starter = root / "starter"
            starter.mkdir()
            (starter / "__init__.py").write_text("", encoding="utf-8")
            catalog = root / "catalog.jsonl"
            catalog.write_text("{\"parent_asin\":\"P000000001\"}\n", encoding="utf-8")
            (starter / "agent.py").write_text(
                "class Item:\n"
                "    def __init__(self, parent_asin): self.parent_asin = parent_asin\n"
                "class Agent:\n"
                "    def __init__(self, catalog_path): self.catalog_path = catalog_path; self.last_diagnostics = {}\n"
                "    def reset(self, session_id, user_profile): pass\n"
                "    def _retrieve(self): return [Item('P000000001')]\n"
                "    def _feature_rank(self, state, candidates): return candidates\n"
                "    def _semantic_rank(self, context, candidates): return None\n"
                "    def respond(self, session_id, user_message, turn, top_k):\n"
                "        found = self._retrieve(); ranked = self._feature_rank(None, found); self._semantic_rank(None, ranked)\n"
                "        self.last_diagnostics = {'intent_path': 'rules'}\n"
                "        return {'message': '', 'ask_attribute': None, 'recommendations': [{'parent_asin': ranked[0].parent_asin}]}\n",
                encoding="utf-8",
            )
            self.assertNotIn("starter", sys.modules)
            with SubprocessAgent(root, catalog, timeout=10) as agent:
                agent.reset("session", {})
                response = agent.respond("session", "hello", 1, 10)
                self.assertEqual(response["recommendations"], [{"parent_asin": "P000000001"}])
                self.assertEqual(agent.last_diagnostics["intent_and_policy"]["intent_path"], "rules")
                self.assertEqual(agent.last_diagnostics["stages"]["retrieved_ids"], ["P000000001"])
                self.assertEqual(agent.last_diagnostics["stages"]["feature_ranked_ids"], ["P000000001"])
                self.assertEqual(agent.last_diagnostics["stages"]["final_ids"], ["P000000001"])
            self.assertNotIn("starter", sys.modules)
            self.assertFalse(list(starter.rglob("__pycache__")))
