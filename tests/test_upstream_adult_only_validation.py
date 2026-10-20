import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

from langchain_core.runnables import RunnableLambda

from studio_graph.graph import create_storyboard_graph


def planner_result(*, prohibited_event: str | None = None) -> dict:
    event = prohibited_event or "Adult Lena opens her late father's unfinished wooden box."
    required_props = ["wooden box"]
    if prohibited_event:
        required_props.append("framed photograph of Lena as a child")
    return {
        "title": "The Unfinished Box",
        "language": "English",
        "tone": "reflective",
        "trope": "quiet discovery",
        "creative_intent": {
            "genre": "drama",
            "tone": "reflective",
            "narrative_resolution": "emotional_beat",
        },
        "characters": [{"id": "1", "name": "Lena", "fixed_visual_identity": "adult woman in her thirties"}],
        "beats": [{
            "panel": 1,
            "event": event,
            "required_subjects": ["adult Lena", "wooden box"],
            "required_action": "Adult Lena opens the wooden box.",
            "required_props": required_props,
            "required_text": [],
            "dialogue_intent": "reflection",
            "payoff": "emotional_beat",
        }],
    }


def initial_state() -> dict:
    return {
        "generation_id": "adult_only_boundary",
        "user_prompt": "An adult woman returns to her late father's workshop.",
        "pageCount": 1,
        "requested_creative_intent": {
            "genre": "drama",
            "tone": "reflective",
            "narrative_resolution": "emotional_beat",
        },
        "inferred_creative_intent": {},
        "effective_creative_intent": {},
        "creative_intent_sources": {},
        "creative_intent_warnings": [],
        "retry_count": 0,
        "max_retries": 3,
        "generation_error": "",
        "current_schema": {},
        "story_plan": {},
        "dialogue_plan": {},
        "validation_errors": [],
        "normalization_notes": [],
    }


class UpstreamAdultOnlyValidationTests(unittest.TestCase):
    def test_childhood_photograph_fails_at_planner_before_downstream_agents(self):
        plan = planner_result(
            prohibited_event="Adult Lena studies a framed photograph of herself as a child."
        )
        dialogue_called = False
        artist_called = False

        def dialogue_should_not_run(_):
            nonlocal dialogue_called
            dialogue_called = True
            raise AssertionError("Dialogue Writer must not receive prohibited Planner content")

        def artist_should_not_run(_):
            nonlocal artist_called
            artist_called = True
            raise AssertionError("Storyboard Artist must not receive prohibited Planner content")

        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(plan)))), \
             patch("studio_graph.agents.dialogue_writer_llm", RunnableLambda(dialogue_should_not_run)), \
             patch("studio_graph.agents.storyboarder_llm", RunnableLambda(artist_should_not_run)):
            final = create_storyboard_graph().invoke(
                initial_state(), {"configurable": {"thread_id": "planner-adult-only"}}
            )

        self.assertFalse(dialogue_called)
        self.assertFalse(artist_called)
        self.assertEqual(final["retry_count"], 0)
        self.assertIn("Story Planner adult-only policy error", final["generation_error"])
        self.assertTrue(any("required_props[1]" in error and "child" in error for error in final["validation_errors"]))
        self.assertEqual(final["requested_creative_intent"]["genre"], "drama")
        self.assertEqual(final["requested_creative_intent"]["tone"], "reflective")
        self.assertEqual(final["requested_creative_intent"]["narrative_resolution"], "emotional_beat")

    def test_childhood_memory_in_dialogue_fails_before_storyboard_artist(self):
        dialogue = {
            "dialogues": [{
                "panel": 1,
                "character_id": "1",
                "text": "I remember begging him to build this when I was a child.",
                "intent": "reflection",
                "emotion": "reflective",
            }]
        }
        artist_called = False

        def artist_should_not_run(_):
            nonlocal artist_called
            artist_called = True
            raise AssertionError("Storyboard Artist must not receive prohibited Dialogue output")

        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(
            lambda _: SimpleNamespace(content=json.dumps(planner_result()))
        )), patch("studio_graph.agents.dialogue_writer_llm", RunnableLambda(
            lambda _: SimpleNamespace(content=json.dumps(dialogue))
        )), patch("studio_graph.agents.storyboarder_llm", RunnableLambda(artist_should_not_run)):
            final = create_storyboard_graph().invoke(
                initial_state(), {"configurable": {"thread_id": "dialogue-adult-only"}}
            )

        self.assertFalse(artist_called)
        self.assertEqual(final["retry_count"], 0)
        self.assertIn("Dialogue Writer adult-only policy error", final["generation_error"])
        self.assertTrue(any("dialogues[0].text" in error and "child" in error for error in final["validation_errors"]))
        self.assertEqual(final["effective_creative_intent"], {
            "genre": "drama",
            "tone": "reflective",
            "narrative_resolution": "emotional_beat",
        })

    def test_compliant_upstream_content_reaches_existing_storyboard_stage(self):
        dialogue = {"dialogues": []}
        artist_called = False

        def artist_returns_invalid_json(_):
            nonlocal artist_called
            artist_called = True
            return SimpleNamespace(content="not json")

        state = initial_state()
        state["max_retries"] = 0
        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(
            lambda _: SimpleNamespace(content=json.dumps(planner_result()))
        )), patch("studio_graph.agents.dialogue_writer_llm", RunnableLambda(
            lambda _: SimpleNamespace(content=json.dumps(dialogue))
        )), patch("studio_graph.agents.storyboarder_llm", RunnableLambda(artist_returns_invalid_json)):
            final = create_storyboard_graph().invoke(
                state, {"configurable": {"thread_id": "adult-only-compliant"}}
            )

        self.assertTrue(artist_called)
        self.assertNotIn("adult-only policy error", final["generation_error"])


if __name__ == "__main__":
    unittest.main()
