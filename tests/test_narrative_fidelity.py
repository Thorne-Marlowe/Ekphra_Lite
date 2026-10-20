import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

from langchain_core.runnables import RunnableLambda

from studio_graph.agents import (
    _normalize_story_plan_action_metadata,
    _apply_story_constraints,
    merge_approved_dialogues,
    run_dialogue_writer,
    run_story_planner,
)
from studio_graph.graph import create_storyboard_graph
from studio_graph.prompts import (
    DIALOGUE_WRITER_SYSTEM_PROMPT,
    STORY_PLANNER_SYSTEM_PROMPT,
    build_storyboarder_human_prompt,
)
from studio_graph.structured_output import STORY_PLAN_SCHEMA


def schema() -> dict:
    return {
        "characters": {"elara": {"name": "Elara", "base_prompt_en": "adult archivist"}},
        "pages": [{"page_number": 1, "panels": [
            {"id": f"panel_{index}", "panel_prompt_en": f"Beat {index}", "active_char_ids": ["elara"], "dialogues": []}
            for index in range(1, 5)
        ]}],
    }


class NarrativeFidelityTests(unittest.TestCase):
    def test_active_prompts_distinguish_resolution_outcomes_without_universal_formula(self):
        self.assertIn("actually becomes known", STORY_PLANNER_SYSTEM_PROMPT)
        self.assertIn("unopened clue", STORY_PLANNER_SYSTEM_PROMPT)
        self.assertIn("consequential question, threat, or outcome pending", STORY_PLANNER_SYSTEM_PROMPT)
        self.assertIn("meaningful emotional realization or change", STORY_PLANNER_SYSTEM_PROMPT)
        self.assertIn("without requiring a dramatic payoff", STORY_PLANNER_SYSTEM_PROMPT)
        self.assertIn("only when appropriate", STORY_PLANNER_SYSTEM_PROMPT)
        self.assertIn("not universal formulas", STORY_PLANNER_SYSTEM_PROMPT)
        self.assertIn("final beat's `event` actually fulfill", STORY_PLANNER_SYSTEM_PROMPT)
        self.assertIn("state the meaningful information that becomes known", STORY_PLANNER_SYSTEM_PROMPT)
        self.assertIn("state that disclosure in the final event", STORY_PLANNER_SYSTEM_PROMPT)
        storyboard_prompt = build_storyboarder_human_prompt(1, False)
        self.assertIn("Preserve the Planner's actual final outcome", storyboard_prompt)
        self.assertIn("setup object, clue, or anticipation", storyboard_prompt)
        self.assertIn("supplemental staging panels", storyboard_prompt)
        self.assertIn("must not displace the Planner's final outcome", storyboard_prompt)

    def test_dialogue_prompt_supports_economy_without_a_silence_quota(self):
        self.assertIn("narrative contribution", DIALOGUE_WRITER_SYSTEM_PROMPT)
        self.assertIn("artwork does not already communicate", DIALOGUE_WRITER_SYSTEM_PROMPT)
        self.assertIn("may be silent", DIALOGUE_WRITER_SYSTEM_PROMPT)
        self.assertIn("Do not impose silence as a quota", DIALOGUE_WRITER_SYSTEM_PROMPT)
        self.assertIn("dialogue-heavy scenes remain valid", DIALOGUE_WRITER_SYSTEM_PROMPT)
        self.assertNotIn("MUST contain at least one concise dialogue", DIALOGUE_WRITER_SYSTEM_PROMPT)

    def test_planner_final_beat_requirements_are_attached_to_final_panel(self):
        draft = schema()
        final_requirements = {
            "panel": 4,
            "event": "Elara reads the revealed note.",
            "primary_subject": "Elara reading the revealed note",
            "secondary_subject": "open envelope with the surveyor's confession",
            "visual_hierarchy": "The note is legible beside Elara's realization.",
            "required_subjects": ["Elara", "open envelope", "confession note"],
            "required_action": "Opening the envelope and reading the confession.",
            "required_props": ["open envelope", "confession note"],
            "required_text": ["I corrected every survey."],
        }

        _apply_story_constraints(draft, {"beats": [final_requirements]})

        final_panel = draft["pages"][0]["panels"][3]
        self.assertEqual(final_panel["required_action"], final_requirements["required_action"])
        self.assertEqual(final_panel["required_props"], final_requirements["required_props"])
        self.assertEqual(final_panel["required_text"], final_requirements["required_text"])
        self.assertEqual(final_panel["primary_subject"], final_requirements["primary_subject"])
        self.assertNotIn("required_action", draft["pages"][0]["panels"][0])

    def test_planner_action_metadata_supports_singular_and_unambiguous_plural_compatibility(self):
        action_properties = STORY_PLAN_SCHEMA["properties"]["beats"]["items"]["properties"]
        self.assertIn("required_action", action_properties)
        self.assertNotIn("required_actions", action_properties)

        for beat, expected in (
            ({"panel": 1, "event": "Singular", "required_action": "Elara opens the envelope"}, "Elara opens the envelope"),
            ({"panel": 1, "event": "Plural alias", "required_actions": ["Elara opens the envelope"]}, "Elara opens the envelope"),
            ({
                "panel": 1,
                "event": "Both forms",
                "required_action": "Elara reads the note",
                "required_actions": ["Elara opens the envelope"],
            }, "Elara reads the note"),
        ):
            with self.subTest(beat=beat):
                plan = {"beats": [beat]}
                errors = _normalize_story_plan_action_metadata(plan)
                self.assertEqual(errors, [])
                self.assertNotIn("required_actions", plan["beats"][0])
                draft = schema()
                _apply_story_constraints(draft, plan, errors)
                self.assertEqual(errors, [])
                self.assertEqual(draft["pages"][0]["panels"][0]["required_action"], expected)

    def test_multiple_plural_actions_fail_explicitly_without_a_silent_loss(self):
        draft = schema()
        errors = []
        plan = {"beats": [{
            "panel": 1,
            "event": "Sequential action",
            "required_actions": ["Elara opens the envelope", "Elara reads the note"],
        }]}
        errors = _normalize_story_plan_action_metadata(plan)
        self.assertTrue(any(
            "required_actions: Multiple actions cannot be represented by the current singular required_action contract"
            in error for error in errors
        ), errors)
        self.assertNotIn("required_action", draft["pages"][0]["panels"][0])

    def test_invalid_planner_contract_stops_before_dialogue_or_artist_and_uses_no_artist_retry(self):
        invalid_plan = {
            "title": "Archive", "language": "English", "tone": "suspenseful", "trope": "mystery",
            "characters": [{"id": "elara", "name": "Elara"}],
            "beats": [{"panel": 1, "event": "Elara finds a ledger", "required_actions": ["opens it", "reads it"]}],
        }
        dialogue_called = False
        artist_called = False

        def dialogue_should_not_run(_):
            nonlocal dialogue_called
            dialogue_called = True
            raise AssertionError("Dialogue Writer must not receive an invalid Planner plan")

        def artist_should_not_run(_):
            nonlocal artist_called
            artist_called = True
            raise AssertionError("Storyboard Artist must not receive an invalid Planner plan")

        initial_state = {
            "generation_id": "planner_contract", "user_prompt": "An archivist searches.", "pageCount": 1,
            "requested_creative_intent": {}, "retry_count": 0, "max_retries": 3,
            "generation_error": "", "current_schema": {}, "story_plan": {}, "dialogue_plan": {},
            "validation_errors": [], "normalization_notes": [], "storyboard_attempt_diagnostics": [],
        }
        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(invalid_plan)))), \
             patch("studio_graph.agents.dialogue_writer_llm", RunnableLambda(dialogue_should_not_run)), \
             patch("studio_graph.agents.storyboarder_llm", RunnableLambda(artist_should_not_run)):
            final = create_storyboard_graph().invoke(initial_state, {"configurable": {"thread_id": "planner-contract"}})

        self.assertFalse(dialogue_called)
        self.assertFalse(artist_called)
        self.assertEqual(final["retry_count"], 0)
        self.assertIn("Story Planner contract error", final["generation_error"])
        self.assertEqual(final["storyboard_attempt_diagnostics"][-1]["stage"], "planner_contract")

    def test_approved_dialogue_overwrites_artist_dialogue_and_preserves_silence(self):
        draft = schema()
        draft["pages"][0]["panels"][0]["dialogues"] = [{"character_id": "elara", "text": "Artist replacement"}]
        draft["pages"][0]["panels"][1]["dialogues"] = [{"character_id": "elara", "text": "Invented speech"}]

        merge_approved_dialogues(draft, {"dialogues": [
            {"panel": 1, "character_id": "elara", "text": "Approved line", "emotion": "calm"},
        ]}, {"characters": [{"id": "elara", "name": "Elara"}]})

        self.assertEqual(draft["pages"][0]["panels"][0]["dialogues"][0]["text"], "Approved line")
        self.assertEqual(draft["pages"][0]["panels"][1]["dialogues"], [])

    def test_intermediate_plan_capture_is_disabled_by_default_and_opt_in_when_enabled(self):
        plan = {
            "title": "Archive", "language": "English", "tone": "suspenseful", "trope": "mystery",
            "characters": [{"id": "1", "name": "Elara"}], "beats": [{"panel": 1, "event": "Elara finds a ledger"}],
        }
        dialogue = {"dialogues": [{"panel": 1, "character_id": "1", "text": "This is older than it looks.", "intent": "inference", "emotion": "calm"}]}
        state = {"generation_id": "test_capture", "user_prompt": "Private premise.", "pageCount": 1, "requested_creative_intent": {}}

        with tempfile.TemporaryDirectory() as root, \
             patch("studio_graph.agents._DEBUG_STORYBOARD_CAPTURE_DIR", Path(root)), \
             patch("studio_graph.agents.story_planner_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(plan)))), \
             patch.dict(os.environ, {"EKPHRA_CAPTURE_STORYBOARD_PLANS": ""}, clear=False):
            run_story_planner(state)
            self.assertFalse((Path(root) / "test_capture").exists())

        with tempfile.TemporaryDirectory() as root, \
             patch("studio_graph.agents._DEBUG_STORYBOARD_CAPTURE_DIR", Path(root)), \
             patch("studio_graph.agents.story_planner_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(plan)))), \
             patch("studio_graph.agents.dialogue_writer_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(dialogue)))), \
             patch.dict(os.environ, {"EKPHRA_CAPTURE_STORYBOARD_PLANS": "true"}, clear=False):
            state = {"generation_id": "test_capture", "user_prompt": "Private premise.", "pageCount": 1, "requested_creative_intent": {}}
            run_story_planner(state)
            run_dialogue_writer(state)
            capture_dir = Path(root) / "test_capture"
            self.assertEqual(json.loads((capture_dir / "story_plan.json").read_text(encoding="utf-8")), plan)
            self.assertEqual(json.loads((capture_dir / "dialogue_plan.json").read_text(encoding="utf-8")), dialogue)


if __name__ == "__main__":
    unittest.main()
