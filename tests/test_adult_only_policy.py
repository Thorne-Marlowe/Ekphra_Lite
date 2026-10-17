import json
import tempfile
import unittest
from pathlib import Path

from panel_engine.comfy_renderer import render_comic_schema, render_panels_with_comfyui
from policy.adult_only import adult_only_errors, validate_metadata_payload
from studio_graph.prompts import STORY_PLANNER_SYSTEM_PROMPT, STORYBOARDER_SYSTEM_PROMPT
from studio_graph.validation import validate_storyboard_structure


def storyboard() -> dict:
    return {
        "characters": {
            "maya": {
                "name": "Maya",
                "description": "adult daughter",
                "base_prompt_en": "adult woman in her thirties",
            }
        },
        "pages": [{"page_number": 1, "panels": [
            {"id": "p1", "panel_prompt_en": "Maya studies a map", "active_char_ids": ["maya"], "dialogues": []},
            {"id": "p2", "panel_prompt_en": "Maya folds the map", "active_char_ids": ["maya"], "dialogues": []},
        ]}],
    }


class AdultOnlyPolicyTests(unittest.TestCase):
    def test_planner_and_storyboarder_explicitly_require_adults(self):
        for prompt in (STORY_PLANNER_SYSTEM_PROMPT, STORYBOARDER_SYSTEM_PROMPT):
            self.assertIn("adults aged 18 or older", prompt)
            self.assertIn("photographs", prompt)

    def test_singular_plural_case_and_word_boundaries(self):
        errors = adult_only_errors({"a": "CHILD", "b": "photograph of children", "c": "teenagers"})
        self.assertEqual(len(errors), 3)
        self.assertEqual(
            adult_only_errors({"text": "childcare is closed; minor conflict ends when an adult daughter visits her adult son, spouse, and grandparent."}),
            [],
        )

    def test_background_photograph_of_minors_is_rejected(self):
        data = storyboard()
        data["pages"][0]["panels"][0]["required_props"] = ["framed photograph of children at a beach"]
        errors = validate_storyboard_structure(data, 1)
        self.assertTrue(any("required_props[0]" in error and "children" in error for error in errors), errors)

    def test_all_nested_optional_fields_are_checked(self):
        data = storyboard()
        data["pages"][0]["panels"][1]["optional_reference"] = {"memory": "a 16-year-old figure"}
        errors = validate_storyboard_structure(data, 1)
        self.assertTrue(any("optional_reference.memory" in error and "16-year-old" in error for error in errors), errors)

    def test_numeric_age_in_optional_metadata_is_checked(self):
        data = storyboard()
        data["characters"]["maya"]["optional_reference"] = {"age": 16}
        errors = validate_storyboard_structure(data, 1)
        self.assertTrue(any("optional_reference.age" in error and "16" in error for error in errors), errors)

    def test_string_age_metadata_and_numeric_age_variants_are_checked(self):
        self.assertTrue(adult_only_errors({"age": "16"}))
        self.assertTrue(adult_only_errors({"text": "an aged 16 figure"}))
        self.assertTrue(adult_only_errors({"text": "16 y/o"}))

    def test_metadata_export_helper_rejects_exact_payload(self):
        with self.assertRaisesRegex(ValueError, "negative prompt.*underage"):
            validate_metadata_payload({"negative prompt": "underage"})

    def test_saved_storyboard_is_checked_before_rendering(self):
        data = storyboard()
        data["pages"][0]["panels"][0]["panel_prompt_en"] = "A minor studies a map"
        with self.assertRaisesRegex(ValueError, r"storyboard.pages\[0\].panels\[0\].panel_prompt_en.*minor"):
            render_comic_schema(data, "unused", None, None, {}, target_panel_ids=[])

    def test_assembled_character_description_is_checked_before_submit(self):
        data = storyboard()
        data["characters"]["maya"]["base_prompt_en"] = "teenager wearing a blue coat"
        self._assert_submit_blocked(data, {}, "positive prompt.*teenager")

    def test_request_negative_prompt_is_rejected_before_submit(self):
        self._assert_submit_blocked(storyboard(), {"negative_prompt": "minors"}, "negative prompt.*minors")

    def test_static_style_or_lora_prompt_input_is_checked_before_submit(self):
        self._assert_submit_blocked(storyboard(), {}, "workflow node '2' input 'style_prompt'.*teen", style_prompt="teen fashion")

    def test_nonstandard_custom_node_input_is_checked_before_submit(self):
        self._assert_submit_blocked(
            storyboard(), {}, "workflow node '2' input 'trigger'.*children",
            extra_inputs={"trigger": {"phrases": ["photograph of children"]}},
        )

    def _assert_submit_blocked(self, data, settings, expected_error, style_prompt=None, extra_inputs=None):
        class Profile:
            def load_workflow(self, _root):
                return {"1": {"inputs": {"text": "static style"}}}

            def apply_inputs(self, _workflow, values):
                workflow = {"1": {"inputs": {"text": values["positive_prompt"]}}}
                if style_prompt:
                    workflow["2"] = {"inputs": {"style_prompt": style_prompt}}
                elif extra_inputs:
                    workflow["2"] = {"inputs": extra_inputs}
                return workflow

        class Client:
            def submit(self, _workflow):
                raise AssertionError("ComfyUI submission must be blocked")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            story_path = root / "story.json"
            story_path.write_text(json.dumps(data), encoding="utf-8")
            (root / "panel_sizes.json").write_text(json.dumps({"p1": {"width": 800, "height": 800}, "p2": {"width": 800, "height": 800}}))
            with self.assertRaisesRegex(ValueError, expected_error):
                render_panels_with_comfyui(story_path, root / "panels", Profile(), Client(), settings)


if __name__ == "__main__":
    unittest.main()
