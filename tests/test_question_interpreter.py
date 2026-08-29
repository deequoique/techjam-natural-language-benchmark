import unittest

from nl_benchmark.question_interpreter import interpret_question


class QuestionInterpreterTests(unittest.TestCase):
    def test_model_style_questions_resolve_without_structured_attribute(self):
        cases = {
            "Do you have a maker in mind?": "brand",
            "How much would you be comfortable spending?": "budget",
            "How well reviewed should it be?": "feature",
            "What occasion are you planning to use it for?": "use_case",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                interpretation = interpret_question(message)
                self.assertTrue(interpretation.question)
                self.assertEqual(interpretation.resolved_attributes, (expected,))
                self.assertEqual(interpretation.reason, "natural_language")

    def test_other_fallback_yields_to_specific_natural_language(self):
        interpretation = interpret_question(
            "Which company should make it?",
            "other",
        )
        self.assertEqual(interpretation.resolved_attributes, ("brand",))
        self.assertFalse(interpretation.conflict)
        self.assertEqual(interpretation.reason, "natural_language_over_other")

    def test_structured_and_text_conflict_is_explicit(self):
        interpretation = interpret_question("How much should it cost?", "color")
        self.assertTrue(interpretation.conflict)
        self.assertEqual(interpretation.resolved_attributes, ())
        self.assertEqual(interpretation.reason, "structured_text_conflict")

    def test_paraphrases_share_a_semantic_signature(self):
        first = interpret_question("Which brand do you prefer?")
        second = interpret_question("Who makes the item you want?")
        self.assertEqual(first.semantic_signature, "ask:brand")
        self.assertEqual(second.semantic_signature, first.semantic_signature)

    def test_broad_question_is_bounded_and_does_not_invent_an_attribute(self):
        interpretation = interpret_question("Do you have anything else in mind?", "other")
        self.assertTrue(interpretation.broad)
        self.assertEqual(interpretation.resolved_attributes, ())
        self.assertEqual(interpretation.semantic_signature, "ask:broad")


if __name__ == "__main__":
    unittest.main()
