import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

from langchain_core.runnables import RunnableLambda
from studio_graph.agents import run_validator
from tests.test_storyboard_validation import VALID_FOUR_PANEL_STORYBOARD


class StoryboardValidatorRoutingTests(unittest.TestCase):
    def test_valid_four_panel_draft_reaches_user_review_without_art_director_or_retry(self):
        state = {
            "user_prompt": "Sam loses a coffee mug.",
            "current_schema": VALID_FOUR_PANEL_STORYBOARD,
            "validation_errors": [],
            "pageCount": 1,
            "retry_count": 0,
            "max_retries": 3,
            "next_step": "",
        }
        art_director = Mock()

        with patch.dict(os.environ, {"ART_DIRECTOR_ENABLED": "false"}, clear=False), patch("studio_graph.agents.llm", art_director):
            result = run_validator(state)

        self.assertEqual(result["next_step"], "renderer")
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["validation_errors"], [])
        art_director.assert_not_called()

    def test_opt_in_art_director_rejection_is_feedback_not_a_retry(self):
        state = {
            "user_prompt": "Sam loses a coffee mug.",
            "current_schema": VALID_FOUR_PANEL_STORYBOARD,
            "validation_errors": [],
            "pageCount": 1,
            "retry_count": 0,
            "max_retries": 3,
            "next_step": "",
        }
        art_director = RunnableLambda(
            lambda _: SimpleNamespace(content="This is creatively weak; regenerate it.")
        )

        with patch.dict(os.environ, {"ART_DIRECTOR_ENABLED": "true"}, clear=False), patch("studio_graph.agents.llm", art_director):
            result = run_validator(state)

        self.assertEqual(result["next_step"], "renderer")
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["validation_errors"], [])
        self.assertEqual(result["art_director_feedback"], "This is creatively weak; regenerate it.")


if __name__ == "__main__":
    unittest.main()
