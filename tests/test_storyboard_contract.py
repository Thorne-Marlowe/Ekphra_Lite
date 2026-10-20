import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

from langchain_core.runnables import RunnableLambda

from studio_graph.agents import run_storyboarder
from studio_graph.structured_output import STORYBOARD_SCHEMA, validate_storyboard_contract
from tests.test_storyboard_validation import VALID_FOUR_PANEL_STORYBOARD


def full_storyboard() -> dict:
    return {
        "title": "Existing-compatible storyboard",
        "metadata": {"creative_intent": {"effective": {"genre": "drama"}}},
        "background": {"prompt_en": "quiet studio", "negative_en": "", "seed": 2000},
        "characters": {
            "lena": {
                "name": "Lena",
                "description": "adult musician",
                "base_prompt_en": "adult woman in a blue coat",
                "base_negative_en": "",
                "camera_angle": "eye-level",
                "camera_distance": "medium shot",
                "seed": 3,
            }
        },
        "pages": [{
            "page_number": 1,
            "panels": [{
                "id": "panel_1_1",
                "panel_prompt_en": "Adult Lena studies a handwritten note in a quiet studio.",
                "panel_negative_en": "",
                "description_en": "Lena pauses at the worktable.",
                "background_prompt_en": "quiet studio",
                "active_char_ids": ["lena"],
                "dialogues": [{"character_id": "lena", "text": "I remember this.", "emotion": "calm", "intent": "reflection"}],
                "primary_subject": "Lena",
                "secondary_subject": "handwritten note",
                "visual_hierarchy": ["Lena", "note"],
                "required_subjects": {"primary": "Lena", "secondary": "handwritten note"},
                "required_action": "Lena studies the note.",
                "required_props": "handwritten note",
                "required_text": ["ARCHIVE"],
                "character_positions": {"lena": {"x": "left", "y": "middle", "anchor": "center"}},
                "character_actions": {"lena": {"action_en": "studying the note", "pose_en": "standing", "objects": ["handwritten note"], "interaction": "holding"}},
            }],
        }],
    }


class StoryboardContractTests(unittest.TestCase):
    def test_schema_closes_structural_objects_but_preserves_metadata(self):
        self.assertFalse(STORYBOARD_SCHEMA["additionalProperties"])
        page = STORYBOARD_SCHEMA["properties"]["pages"]["items"]
        panel = page["properties"]["panels"]["items"]
        self.assertFalse(page["additionalProperties"])
        self.assertFalse(panel["additionalProperties"])
        self.assertTrue(STORYBOARD_SCHEMA["properties"]["metadata"]["additionalProperties"])

    def test_existing_valid_storyboard_example_and_supported_metadata_are_accepted(self):
        self.assertEqual(validate_storyboard_contract(VALID_FOUR_PANEL_STORYBOARD), [])
        self.assertEqual(validate_storyboard_contract(full_storyboard()), [])

    def test_near_canonical_position_key_is_rejected(self):
        storyboard = full_storyboard()
        panel = storyboard["pages"][0]["panels"][0]
        panel.pop("character_positions")
        panel["character_positions': { "] = "lena"

        errors = validate_storyboard_contract(storyboard)

        self.assertTrue(any("character_positions': {" in error and "Unknown field" in error for error in errors), errors)

    def test_invalid_nested_position_and_action_objects_are_rejected(self):
        storyboard = full_storyboard()
        panel = storyboard["pages"][0]["panels"][0]
        panel["character_positions"]["lena"] = {"x": "left", "y": [], "anchor": "center", "unexpected": True}
        panel["character_actions"]["lena"] = {"action_en": "reading", "objects": "note"}

        errors = validate_storyboard_contract(storyboard)

        self.assertTrue(any("character_positions.lena.y" in error for error in errors), errors)
        self.assertTrue(any("character_positions.lena.unexpected" in error for error in errors), errors)
        self.assertTrue(any("character_actions.lena.objects" in error for error in errors), errors)

    def test_misplaced_page_object_inside_panels_is_rejected(self):
        storyboard = full_storyboard()
        storyboard["pages"][0]["panels"] = [{"page_number": 1, "panels": []}]

        errors = validate_storyboard_contract(storyboard)

        self.assertIn("pages[1].panels[1].id: Missing required field.", errors)
        self.assertIn("pages[1].panels[1].page_number: Unknown field.", errors)

    def test_artist_contract_error_precedes_normalization(self):
        malformed = full_storyboard()
        panel = malformed["pages"][0]["panels"][0]
        panel["character_positions': { "] = "lena"
        panel.pop("character_positions")
        state = {
            "user_prompt": "A quiet studio.", "pageCount": 1, "current_schema": {},
            "validation_errors": [], "story_plan": {}, "dialogue_plan": {}, "retry_count": 0,
        }

        with patch("studio_graph.agents.storyboarder_llm", RunnableLambda(
            lambda _: SimpleNamespace(content=json.dumps(malformed))
        )):
            result = run_storyboarder(state)

        self.assertEqual(result["current_schema"], {})
        self.assertTrue(any("character_positions': {" in error for error in result["validation_errors"]))
        self.assertEqual(result["storyboard_attempt_diagnostics"][-1]["stage"], "schema_validation")

    def test_extra_staging_panels_remain_contract_valid(self):
        storyboard = full_storyboard()
        panel = storyboard["pages"][0]["panels"][0]
        for number in range(2, 5):
            extra = json.loads(json.dumps(panel))
            extra["id"] = f"panel_1_{number}"
            extra["dialogues"] = []
            storyboard["pages"][0]["panels"].append(extra)

        self.assertEqual(validate_storyboard_contract(storyboard), [])


if __name__ == "__main__":
    unittest.main()
