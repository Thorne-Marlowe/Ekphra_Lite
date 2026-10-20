import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import api
from langchain_core.runnables import RunnableLambda

from studio_graph.agents import run_dialogue_writer, run_story_planner, run_storyboarder
from studio_graph.narrative_contract import (
    NARRATIVE_RESOLUTIONS,
    NarrativeContractError,
    attach_creative_intent_metadata,
    conflict_warning,
    creative_intent_metadata,
    resolve_creative_intent,
)
from utils.parser import load_comic_from_json


class NarrativeContractTests(unittest.TestCase):
    def test_legacy_storyboard_request_remains_compatible(self):
        request = api.StoryboardRequest(prompt="A quiet conversation.")
        self.assertEqual(request.pageCount, 1)
        self.assertIsNone(request.creativeIntent)
        self.assertIsNone(creative_intent_metadata(request.creativeIntent))

    def test_optional_intent_fields_and_each_supported_resolution(self):
        self.assertEqual(resolve_creative_intent({})[2], {})
        for resolution in NARRATIVE_RESOLUTIONS:
            requested, inferred, effective, sources = resolve_creative_intent({"narrative_resolution": resolution})
            self.assertEqual(requested["narrative_resolution"], resolution)
            self.assertEqual(inferred, {})
            if resolution == "auto":
                self.assertNotIn("narrative_resolution", effective)
                self.assertNotIn("narrative_resolution", sources)
            else:
                self.assertEqual(effective["narrative_resolution"], resolution)
                self.assertEqual(sources["narrative_resolution"], "user_selection")

    def test_invalid_resolution_is_rejected(self):
        with self.assertRaisesRegex(NarrativeContractError, "narrative_resolution"):
            resolve_creative_intent({"narrative_resolution": "surprise-ending"})

    def test_auto_never_defaults_to_punchline_and_empty_inference_is_valid(self):
        requested, inferred, effective, sources = resolve_creative_intent(
            {"genre": "slice of life", "narrative_resolution": "auto"}, {}
        )
        self.assertEqual(requested["narrative_resolution"], "auto")
        self.assertEqual(inferred, {})
        self.assertEqual(effective, {"genre": "slice of life"})
        self.assertEqual(sources, {"genre": "user_selection"})
        self.assertNotIn("punchline", effective.values())

    def test_explicit_values_take_precedence_over_actual_inference(self):
        _, _, effective, sources = resolve_creative_intent(
            {"genre": "mystery", "tone": "quiet", "narrative_resolution": "revelation"},
            {"genre": "comedy", "tone": "broad", "narrative_resolution": "punchline"},
        )
        self.assertEqual(effective, {"genre": "mystery", "tone": "quiet", "narrative_resolution": "revelation"})
        self.assertEqual(set(sources.values()), {"user_selection"})

    def test_actual_planner_inference_has_planner_provenance_when_not_overridden(self):
        requested, inferred, effective, sources = resolve_creative_intent(
            {"narrative_resolution": "auto"},
            {"genre": "mystery", "narrative_resolution": "revelation"},
        )
        self.assertEqual(requested, {"narrative_resolution": "auto"})
        self.assertEqual(inferred["narrative_resolution"], "revelation")
        self.assertEqual(effective, {"genre": "mystery", "narrative_resolution": "revelation"})
        self.assertEqual(set(sources.values()), {"planner_inference"})

    def test_conflict_warning_has_a_deterministic_reviewable_shape(self):
        warning = conflict_warning("narrative_resolution", selection="revelation", prose_summary="The user asks for no resolution.")
        self.assertEqual(warning["code"], "creative_intent_conflict")
        self.assertEqual(warning["dimension"], "narrative_resolution")
        self.assertIn("review", warning["message"].lower())
        metadata = creative_intent_metadata({"narrative_resolution": "revelation"}, warnings=[warning])
        self.assertEqual(metadata["warnings"], [warning])

    def test_metadata_preserves_unrelated_values_and_parser_compatibility(self):
        schema = {
            "metadata": {"project_note": "keep this"},
            "characters": {},
            "pages": [{"page_number": 1, "panels": []}],
        }
        intent = creative_intent_metadata({"genre": "mystery", "narrative_resolution": "revelation"})
        attach_creative_intent_metadata(schema, intent)
        self.assertEqual(schema["metadata"]["project_note"], "keep this")
        self.assertEqual(schema["metadata"]["creative_intent"]["effective"]["genre"], "mystery")
        self.assertEqual(load_comic_from_json(schema).metadata, schema["metadata"])

    def test_existing_saved_storyboard_without_intent_stays_valid(self):
        legacy = {"characters": {}, "pages": [{"page_number": 1, "panels": []}]}
        self.assertEqual(load_comic_from_json(legacy).metadata, {})

    def test_storyboarder_persists_only_actual_contract_values(self):
        artist_schema = {
            "metadata": {"keep": "existing"},
            "characters": {"n": {"name": "N", "base_prompt_en": "adult"}},
            "pages": [{"page_number": 1, "panels": [{
                "id": "p1", "panel_prompt_en": "N waits quietly.", "active_char_ids": ["n"], "dialogues": [],
            }, {"id": "p2", "panel_prompt_en": "N leaves.", "active_char_ids": ["n"], "dialogues": []}]}],
        }
        state = {
            "user_prompt": "A quiet mystery.", "pageCount": 1, "current_schema": {}, "validation_errors": [],
            "story_plan": {}, "dialogue_plan": {"dialogues": []},
            "requested_creative_intent": {"genre": "mystery", "narrative_resolution": "auto"},
            "inferred_creative_intent": {"tone": "tense", "narrative_resolution": "revelation"},
            "effective_creative_intent": {"genre": "mystery", "tone": "tense", "narrative_resolution": "revelation"},
            "creative_intent_sources": {"genre": "user_selection", "tone": "planner_inference", "narrative_resolution": "planner_inference"},
            "creative_intent_warnings": [],
        }
        with patch("studio_graph.agents.storyboarder_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(artist_schema)))):
            result = run_storyboarder(state)
        intent = result["current_schema"]["metadata"]["creative_intent"]
        self.assertEqual(intent["requested"]["narrative_resolution"], "auto")
        self.assertEqual(intent["effective"], state["effective_creative_intent"])
        self.assertEqual(intent["source"], state["creative_intent_sources"])
        self.assertEqual(result["current_schema"]["metadata"]["keep"], "existing")

    def test_planner_captures_actual_inference_and_auto_resolves_from_it(self):
        plan = {
            "title": "Signal", "language": "English", "tone": "tense", "trope": "mystery",
            "creative_intent": {"genre": "mystery", "tone": "tense", "narrative_resolution": "revelation"},
            "characters": [], "beats": [],
        }
        state = {"user_prompt": "A signal in the dark.", "pageCount": 1,
                 "requested_creative_intent": {"narrative_resolution": "auto"}}
        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(plan)))):
            result = run_story_planner(state)
        self.assertEqual(result["inferred_creative_intent"], plan["creative_intent"])
        self.assertEqual(result["effective_creative_intent"]["narrative_resolution"], "revelation")
        self.assertEqual(result["creative_intent_sources"]["narrative_resolution"], "planner_inference")

    def test_explicit_request_wins_over_planner_inference(self):
        plan = {"title": "Signal", "language": "English", "tone": "broad", "trope": "comedy",
                "creative_intent": {"genre": "comedy", "tone": "broad", "narrative_resolution": "punchline"},
                "characters": [], "beats": []}
        state = {"user_prompt": "A mystery.", "pageCount": 1,
                 "requested_creative_intent": {"genre": "mystery", "tone": "quiet", "narrative_resolution": "revelation"}}
        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(plan)))):
            result = run_story_planner(state)
        self.assertEqual(result["effective_creative_intent"], {"genre": "mystery", "tone": "quiet", "narrative_resolution": "revelation"})
        self.assertEqual(set(result["creative_intent_sources"].values()), {"user_selection"})

    def test_missing_inference_is_valid_and_auto_remains_unresolved(self):
        plan = {"title": "Quiet", "language": "English", "tone": "calm", "trope": "slice of life", "characters": [], "beats": []}
        state = {"user_prompt": "Two people wait.", "pageCount": 1,
                 "requested_creative_intent": {"narrative_resolution": "auto"}}
        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(plan)))):
            result = run_story_planner(state)
        self.assertEqual(result["inferred_creative_intent"], {})
        self.assertEqual(result["effective_creative_intent"], {})

    def test_invalid_planner_inference_produces_no_fabricated_default(self):
        plan = {"title": "Quiet", "language": "English", "tone": "calm", "trope": "slice of life",
                "creative_intent": {"narrative_resolution": "not-a-resolution"}, "characters": [], "beats": []}
        state = {"user_prompt": "Two people wait.", "pageCount": 1,
                 "requested_creative_intent": {"narrative_resolution": "auto"},
                 "inferred_creative_intent": {}, "effective_creative_intent": {}, "creative_intent_sources": {}}
        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(plan)))):
            result = run_story_planner(state)
        self.assertEqual(result["next_step"], "failed")
        self.assertIn("Story Planner contract error", result["generation_error"])
        self.assertIn("creative_intent.narrative_resolution", result["validation_errors"][0])
        self.assertEqual(result["effective_creative_intent"], {})

    def test_downstream_stages_receive_but_cannot_replace_effective_intent(self):
        captured = {}
        state = {
            "story_plan": {"language": "English"}, "requested_language": "English", "dialogue_plan": {},
            "effective_creative_intent": {"genre": "mystery"},
        }
        def dialogue_response(prompt_value):
            captured["prompt"] = "\n".join(str(message.content) for message in prompt_value.to_messages())
            return SimpleNamespace(content=json.dumps({"dialogues": []}))

        with patch("studio_graph.agents.dialogue_writer_llm", RunnableLambda(dialogue_response)):
            run_dialogue_writer(state)
        self.assertIn('{"genre": "mystery"}', captured["prompt"])
        self.assertEqual(state["effective_creative_intent"], {"genre": "mystery"})


if __name__ == "__main__":
    unittest.main()
