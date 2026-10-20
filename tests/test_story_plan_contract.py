import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

from langchain_core.runnables import RunnableLambda

from studio_graph.agents import _normalize_and_validate_story_plan
from studio_graph.graph import create_storyboard_graph
from studio_graph.structured_output import STORY_PLAN_SCHEMA, validate_story_plan_contract


def valid_plan() -> dict:
    return {
        "title": "The Archive",
        "language": "English",
        "tone": "suspenseful",
        "trope": "historical mystery",
        "creative_intent": {
            "genre": "mystery",
            "tone": "suspenseful",
            "narrative_resolution": "revelation",
        },
        "characters": [{
            "id": "2",
            "name": "The Surveyor",
            "role": "off-panel witness",
            "goal": "correct the record",
            "personality_contrast": "reserved but determined",
            "fixed_visual_identity": "adult surveyor in a brown field jacket",
        }],
        "beats": [{
            "panel": 1,
            "role": "revelation",
            "event": "The confession identifies the falsified surveys.",
            "primary_subject": "the opened confession",
            "secondary_subject": "the corrected survey",
            "visual_hierarchy": ["confession", "survey"],
            "required_subjects": ["confession", "survey"],
            "required_action": "The archivist reads the opened confession.",
            "required_props": ["opened envelope", "survey"],
            "required_text": [],
            "dialogue_intent": "the surveyor admits the alteration",
            "payoff": "the historical injustice is documented",
        }],
    }


class StoryPlanContractTests(unittest.TestCase):
    def test_schema_closes_canonical_objects_and_advertises_only_singular_action(self):
        self.assertFalse(STORY_PLAN_SCHEMA["additionalProperties"])
        character_schema = STORY_PLAN_SCHEMA["properties"]["characters"]["items"]
        beat_schema = STORY_PLAN_SCHEMA["properties"]["beats"]["items"]
        self.assertFalse(character_schema["additionalProperties"])
        self.assertFalse(beat_schema["additionalProperties"])
        self.assertIn("required_action", beat_schema["properties"])
        self.assertNotIn("required_actions", beat_schema["properties"])

    def test_valid_canonical_plan_and_optional_metadata_are_preserved(self):
        plan = valid_plan()
        original = copy.deepcopy(plan)
        self.assertEqual(_normalize_and_validate_story_plan(plan), [])
        self.assertEqual(plan, original)

    def test_missing_required_top_level_fields_are_rejected(self):
        for field in STORY_PLAN_SCHEMA["required"]:
            with self.subTest(field=field):
                plan = valid_plan()
                plan.pop(field)
                errors = _normalize_and_validate_story_plan(plan)
                self.assertIn(f"{field}: Missing required field.", errors)

    def test_incorrect_top_level_types_are_rejected(self):
        plan = valid_plan()
        plan["title"] = 7
        plan["characters"] = {}
        plan["beats"] = "not an array"
        errors = _normalize_and_validate_story_plan(plan)
        self.assertIn("title: Expected string, received integer.", errors)
        self.assertIn("characters: Expected array, received object.", errors)
        self.assertIn("beats: Expected array, received string.", errors)

    def test_character_collection_and_ids_are_validated(self):
        for characters, expected in (
            ({"2": {"name": "Surveyor"}}, "characters: Expected array, received object."),
            ([{"name": "Surveyor"}], "characters[1].id: Missing required field."),
            ([{"id": 2, "name": "Surveyor"}], "characters[1].id: Expected string, received integer."),
            ([{"id": "   ", "name": "Surveyor"}], "characters[1].id: Must be a non-empty string."),
            ([{"id": " 2", "name": "Surveyor"}], "characters[1].id: Does not match the required format."),
        ):
            with self.subTest(characters=characters):
                plan = valid_plan()
                plan["characters"] = characters
                self.assertIn(expected, _normalize_and_validate_story_plan(plan))

        plan = valid_plan()
        plan["characters"].append({"id": "2", "name": "Duplicate"})
        self.assertIn(
            "characters[2].id: Character IDs must be unique.",
            _normalize_and_validate_story_plan(plan),
        )

    def test_missing_or_malformed_beats_are_rejected(self):
        plan = valid_plan()
        plan["beats"] = ["not an object", {"panel": 2}]
        errors = _normalize_and_validate_story_plan(plan)
        self.assertIn("beats[1]: Expected object, received string.", errors)
        self.assertIn("beats[2].event: Missing required field.", errors)

    def test_invalid_event_and_panel_values_are_rejected(self):
        for panel in (0, -1, True, "0", "panel_1_1", ""):
            with self.subTest(panel=panel):
                plan = valid_plan()
                plan["beats"][0]["panel"] = panel
                self.assertTrue(
                    any(error.startswith("beats[1].panel:") for error in _normalize_and_validate_story_plan(plan))
                )
        plan = valid_plan()
        plan["beats"][0]["event"] = "   "
        self.assertIn(
            "beats[1].event: Must be a non-empty string.",
            _normalize_and_validate_story_plan(plan),
        )

    def test_valid_singular_action_remains_unchanged(self):
        plan = valid_plan()
        self.assertEqual(_normalize_and_validate_story_plan(plan), [])
        self.assertEqual(plan["beats"][0]["required_action"], "The archivist reads the opened confession.")

    def test_one_item_legacy_action_alias_normalizes_before_strict_validation(self):
        plan = valid_plan()
        beat = plan["beats"][0]
        beat.pop("required_action")
        beat["required_actions"] = ["The archivist opens the envelope."]
        self.assertEqual(_normalize_and_validate_story_plan(plan), [])
        self.assertEqual(beat["required_action"], "The archivist opens the envelope.")
        self.assertNotIn("required_actions", beat)

    def test_singular_action_takes_precedence_over_legacy_alias(self):
        plan = valid_plan()
        plan["beats"][0]["required_actions"] = ["Ignored legacy action."]
        self.assertEqual(_normalize_and_validate_story_plan(plan), [])
        self.assertEqual(plan["beats"][0]["required_action"], "The archivist reads the opened confession.")
        self.assertNotIn("required_actions", plan["beats"][0])

    def test_multiple_or_invalid_legacy_actions_are_rejected_precisely(self):
        for value, expected in (
            (["opens envelope", "reads letter"], "Multiple actions cannot be represented"),
            ([], "Must contain exactly one non-empty string"),
            (["  "], "Must contain exactly one non-empty string"),
            ("opens envelope", "Must contain exactly one non-empty string"),
        ):
            with self.subTest(value=value):
                plan = valid_plan()
                plan["beats"][0].pop("required_action")
                plan["beats"][0]["required_actions"] = value
                errors = _normalize_and_validate_story_plan(plan)
                self.assertTrue(any(error.startswith("beats[1].required_actions:") and expected in error for error in errors))

    def test_malformed_near_canonical_fields_are_rejected_with_suggestions(self):
        for field in ("required_props ", "required_props': ["):
            with self.subTest(field=field):
                plan = valid_plan()
                plan["beats"][0][field] = ["envelope"]
                errors = _normalize_and_validate_story_plan(plan)
                self.assertTrue(any(field in error and "Did you mean required_props?" in error for error in errors), errors)

    def test_unknown_beat_and_character_fields_are_rejected(self):
        plan = valid_plan()
        plan["beats"][0]["camera_magic"] = "close"
        plan["characters"][0]["costume_hint"] = "brown coat"
        errors = _normalize_and_validate_story_plan(plan)
        self.assertTrue(any("beats[1].camera_magic: Unknown field." in error for error in errors))
        self.assertTrue(any("characters[1].costume_hint: Unknown field." in error for error in errors))

    def test_required_collection_field_types_are_rejected(self):
        for field, value in (
            ("required_subjects", "archivist"),
            ("required_props", {"item": "envelope"}),
            ("required_text", ["ARCHIVE", 7]),
        ):
            with self.subTest(field=field):
                plan = valid_plan()
                plan["beats"][0][field] = value
                errors = _normalize_and_validate_story_plan(plan)
                self.assertTrue(any(error.startswith(f"beats[1].{field}") for error in errors), errors)

    def test_contract_validator_does_not_normalize_noncanonical_input(self):
        plan = valid_plan()
        plan["beats"][0]["required_props "] = ["envelope"]
        before = copy.deepcopy(plan)
        self.assertTrue(validate_story_plan_contract(plan))
        self.assertEqual(plan, before)

    def test_invalid_plan_never_invokes_dialogue_or_storyboard_agents(self):
        plan = valid_plan()
        plan["beats"][0]["required_props "] = ["envelope"]
        dialogue_called = False
        artist_called = False

        def dialogue_response(_):
            nonlocal dialogue_called
            dialogue_called = True
            raise AssertionError("Dialogue Writer must not receive an invalid plan")

        def artist_response(_):
            nonlocal artist_called
            artist_called = True
            raise AssertionError("Storyboard Artist must not receive an invalid plan")

        state = {
            "generation_id": "contract_failure",
            "user_prompt": "An archivist finds a confession.",
            "pageCount": 1,
            "requested_creative_intent": {},
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
        with patch("studio_graph.agents.story_planner_llm", RunnableLambda(
            lambda _: SimpleNamespace(content=json.dumps(plan))
        )), patch("studio_graph.agents.dialogue_writer_llm", RunnableLambda(dialogue_response)), patch(
            "studio_graph.agents.storyboarder_llm", RunnableLambda(artist_response)
        ):
            result = create_storyboard_graph().invoke(
                state, {"configurable": {"thread_id": "contract-failure"}}
            )

        self.assertFalse(dialogue_called)
        self.assertFalse(artist_called)
        self.assertIn("Story Planner contract error", result["generation_error"])
        self.assertEqual(result["storyboard_attempt_diagnostics"][-1]["stage"], "planner_contract")


if __name__ == "__main__":
    unittest.main()
