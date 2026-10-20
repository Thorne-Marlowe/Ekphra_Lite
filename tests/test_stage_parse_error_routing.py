import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

from fastapi import HTTPException
from langchain_core.runnables import RunnableLambda

import api
from studio_graph.graph import create_storyboard_graph


def valid_plan() -> dict:
    return {
        "title": "The Archive",
        "language": "English",
        "tone": "reflective",
        "trope": "quiet discovery",
        "characters": [{"id": "elara", "name": "Elara"}],
        "beats": [{"panel": 1, "event": "Elara opens the archive ledger."}],
    }


def initial_state() -> dict:
    return {
        "generation_id": "parse-error-routing",
        "user_prompt": "An adult archivist opens a ledger.",
        "pageCount": 1,
        "requested_creative_intent": {},
        "inferred_creative_intent": {},
        "effective_creative_intent": {},
        "creative_intent_sources": {},
        "creative_intent_warnings": [],
        "current_schema": {},
        "story_plan": {},
        "dialogue_plan": {},
        "validation_errors": [],
        "normalization_notes": [],
        "storyboard_attempt_diagnostics": [],
        "retry_count": 0,
        "max_retries": 3,
        "generation_error": "",
    }


class StageParseErrorRoutingTests(unittest.TestCase):
    def test_malformed_planner_output_stops_before_dialogue_and_storyboard(self):
        dialogue_called = False
        artist_called = False

        def dialogue_should_not_run(_):
            nonlocal dialogue_called
            dialogue_called = True
            raise AssertionError("Dialogue Writer must not receive malformed Planner output")

        def artist_should_not_run(_):
            nonlocal artist_called
            artist_called = True
            raise AssertionError("Storyboard Artist must not receive malformed Planner output")

        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(
            lambda _: SimpleNamespace(content="not valid JSON")
        )), patch("studio_graph.agents.dialogue_writer_llm", RunnableLambda(dialogue_should_not_run)), patch(
            "studio_graph.agents.storyboarder_llm", RunnableLambda(artist_should_not_run)
        ):
            final = create_storyboard_graph().invoke(
                initial_state(), {"configurable": {"thread_id": "malformed-planner"}}
            )

        self.assertFalse(dialogue_called)
        self.assertFalse(artist_called)
        self.assertEqual(final["next_step"], "failed")
        self.assertEqual(final["generation_error"], final["validation_errors"][0])
        self.assertIn("Story plan JSON error:", final["generation_error"])

    def test_malformed_dialogue_output_stops_before_storyboard(self):
        artist_called = False

        def artist_should_not_run(_):
            nonlocal artist_called
            artist_called = True
            raise AssertionError("Storyboard Artist must not receive malformed Dialogue output")

        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(
            lambda _: SimpleNamespace(content=json.dumps(valid_plan()))
        )), patch("studio_graph.agents.dialogue_writer_llm", RunnableLambda(
            lambda _: SimpleNamespace(content="not valid JSON")
        )), patch("studio_graph.agents.storyboarder_llm", RunnableLambda(artist_should_not_run)):
            final = create_storyboard_graph().invoke(
                initial_state(), {"configurable": {"thread_id": "malformed-dialogue"}}
            )

        self.assertFalse(artist_called)
        self.assertEqual(final["next_step"], "failed")
        self.assertEqual(final["generation_error"], final["validation_errors"][0])
        self.assertIn("Dialogue plan JSON error:", final["generation_error"])

    def test_api_returns_422_for_terminal_stage_error_and_502_for_transport_failure(self):
        async def terminal_stage_error(_state, _request):
            return {"generation_error": "Story Planner JSON error: invalid JSON"}

        async def transport_failure(_state, _request):
            raise RuntimeError("local endpoint disconnected")

        request = api.StoryboardRequest(prompt="An adult archivist opens a ledger.")
        with patch.object(api, "invoke_storyboard_graph", terminal_stage_error):
            with self.assertRaises(HTTPException) as stage_error:
                asyncio.run(api.create_storyboard(request, object()))
        self.assertEqual(stage_error.exception.status_code, 422)
        self.assertIn("Story Planner JSON error", stage_error.exception.detail)

        with patch.object(api, "invoke_storyboard_graph", transport_failure):
            with self.assertRaises(HTTPException) as transport_error:
                asyncio.run(api.create_storyboard(request, object()))
        self.assertEqual(transport_error.exception.status_code, 502)
        self.assertIn("Storyboard LLM request failed", transport_error.exception.detail)


if __name__ == "__main__":
    unittest.main()
