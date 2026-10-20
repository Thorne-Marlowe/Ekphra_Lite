import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from policy.adult_only import adult_only_errors
from studio_graph.agents import run_dialogue_writer, run_story_planner, run_storyboarder
from studio_graph.prompts import (
    DIALOGUE_WRITER_SYSTEM_PROMPT,
    STORY_PLANNER_SYSTEM_PROMPT,
    STORYBOARDER_SYSTEM_PROMPT,
    build_storyboarder_human_prompt,
)
from studio_graph.validation import validate_storyboard_structure


def planner_plan(intent):
    return {
        "title": "Test", "language": "English", "tone": intent.get("tone", "calm"), "trope": "test",
        "creative_intent": intent, "characters": [{"id": "ava", "name": "Ava"}],
        "beats": [{"panel": 1, "event": "Ava notices the change"}],
    }


def storyboard_schema(dialogue):
    return {
        "metadata": {"existing": "preserved"},
        "characters": {"ava": {"name": "Ava", "base_prompt_en": "adult woman in a coat"}},
        "pages": [{"page_number": 1, "panels": [
            {"id": "p1", "panel_prompt_en": "Ava studies a letter.", "active_char_ids": ["ava"], "dialogues": dialogue},
            {"id": "p2", "panel_prompt_en": "Ava looks through the window.", "active_char_ids": ["ava"], "dialogues": []},
        ]}],
    }


class GenreFlexibleStorytellingTests(unittest.TestCase):
    def _run_planner(self, requested, inferred):
        captured = {}

        def response(prompt_value):
            captured["prompt"] = "\n".join(str(message.content) for message in prompt_value.to_messages())
            return SimpleNamespace(content=json.dumps(planner_plan(inferred)))

        state = {"user_prompt": "A compact scene.", "pageCount": 1, "requested_creative_intent": requested}
        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(response)):
            result = run_story_planner(state)
        return result, captured["prompt"]

    def test_planner_receives_explicit_intent_and_preserves_precedence(self):
        requested = {"genre": "mystery", "tone": "quiet", "narrative_resolution": "revelation"}
        result, prompt = self._run_planner(
            requested,
            {"genre": "comedy", "tone": "broad", "narrative_resolution": "punchline"},
        )
        self.assertIn(json.dumps(requested), prompt)
        self.assertEqual(result["effective_creative_intent"], requested)
        self.assertEqual(set(result["creative_intent_sources"].values()), {"user_selection"})

    def test_genres_and_resolutions_can_be_inferred_without_a_comedy_default(self):
        scenarios = [
            ("mystery", "revelation"), ("comedy", "cliffhanger"), ("drama", "emotional_beat"),
            ("romance", "transition"), ("slice of life", "deliberate_nonresolution"),
        ]
        for genre, resolution in scenarios:
            with self.subTest(genre=genre):
                result, _ = self._run_planner({"narrative_resolution": "auto"}, {
                    "genre": genre, "narrative_resolution": resolution,
                })
                self.assertEqual(result["effective_creative_intent"]["genre"], genre)
                self.assertEqual(result["effective_creative_intent"]["narrative_resolution"], resolution)
        result, _ = self._run_planner({}, {})
        self.assertNotIn("narrative_resolution", result["effective_creative_intent"])

    def test_active_prompts_do_not_impose_a_universal_gag_or_dialogue(self):
        storyboard_human = build_storyboarder_human_prompt(1, False)
        for text in (STORY_PLANNER_SYSTEM_PROMPT, DIALOGUE_WRITER_SYSTEM_PROMPT, STORYBOARDER_SYSTEM_PROMPT, storyboard_human):
            self.assertNotIn("MUST contain at least one concise dialogue", text)
            self.assertNotIn("Include a clear visual punchline", text)
        self.assertIn("do not impose a joke", STORY_PLANNER_SYSTEM_PROMPT)
        self.assertIn("entire scene may be silent", DIALOGUE_WRITER_SYSTEM_PROMPT)
        self.assertIn("not necessarily the conclusion", storyboard_human)
        self.assertNotIn("You MUST NOT create perfect", STORYBOARDER_SYSTEM_PROMPT)
        self.assertNotIn("Webcomics must have rapid pacing", STORYBOARDER_SYSTEM_PROMPT)
        self.assertIn("Do not impose a setup-twist-punchline", STORYBOARDER_SYSTEM_PROMPT)

    def test_end_to_end_effective_intent_reaches_storyboard_and_dialogue_cannot_replace_it(self):
        requested = {"genre": "drama", "narrative_resolution": "auto"}
        state = {
            "user_prompt": "A quiet farewell.", "pageCount": 1, "current_schema": {}, "validation_errors": [],
            "requested_creative_intent": requested, "dialogue_plan": {},
        }
        planner = RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(
            planner_plan({"tone": "restrained", "narrative_resolution": "emotional_beat"})
        )))
        dialogue = RunnableLambda(lambda _: SimpleNamespace(content=json.dumps({"dialogues": [{
            "panel": 1, "character_id": "ava", "text": "Take care.", "intent": "farewell", "emotion": "sad",
        }]})))
        # The artist attempts to replace the approved line; merging must retain it.
        artist = RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(storyboard_schema([
            {"character_id": "ava", "text": "A twist!", "emotion": "excited"}
        ]))))
        with patch("studio_graph.agents.story_planner_llm", planner), \
             patch("studio_graph.agents.dialogue_writer_llm", dialogue), \
             patch("studio_graph.agents.storyboarder_llm", artist):
            run_story_planner(state)
            approved_intent = dict(state["effective_creative_intent"])
            run_dialogue_writer(state)
            run_storyboarder(state)
        self.assertEqual(state["effective_creative_intent"], approved_intent)
        self.assertEqual(state["current_schema"]["metadata"]["creative_intent"]["effective"], approved_intent)
        self.assertEqual(state["current_schema"]["metadata"]["existing"], "preserved")
        dialogue_text = state["current_schema"]["pages"][0]["panels"][0]["dialogues"][0]["text"]
        self.assertEqual(dialogue_text, "Take care.")

    def test_silent_panels_and_adult_only_validation_remain_valid(self):
        schema = storyboard_schema([])
        self.assertEqual(validate_storyboard_structure(schema, 1), [])
        self.assertEqual(adult_only_errors(schema), [])


if __name__ == "__main__":
    unittest.main()
