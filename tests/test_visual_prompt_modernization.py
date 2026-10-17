import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

from panel_engine.comfy_renderer import build_panel_prompt
from studio_graph.agents import merge_approved_dialogues
from studio_graph.prompts import STORYBOARDER_SYSTEM_PROMPT
from studio_graph.validation import normalize_single_frame_actions
from utils.parser import load_comic_from_json


class VisualPromptModernizationTests(unittest.TestCase):
    def test_storyboard_instructions_do_not_require_speaking_keywords_or_sd15_negatives(self):
        self.assertNotIn("STABLE DIFFUSION 1.5", STORYBOARDER_SYSTEM_PROMPT)
        self.assertNotIn("speaking, talking, open mouth", STORYBOARDER_SYSTEM_PROMPT)
        self.assertNotIn("blurry, distorted, deformed hands", STORYBOARDER_SYSTEM_PROMPT)

    def test_final_prompt_preserves_identity_props_camera_and_meaningful_hierarchy(self):
        comic, panel = self._comic({
            "primary_subject": "Nyx holding the map",
            "secondary_subject": "Elias at the doorway",
            "visual_hierarchy": "Nyx dominates the foreground; Elias remains small in the background",
            "required_action": "Nyx holds the map toward Elias",
            "required_props": ["weathered map"],
            "panel_prompt_en": "Rainy station platform at dusk",
            "background_prompt_en": "empty elevated train station",
        }, camera_angle="eye-level", camera_distance="medium shot")

        prompt = build_panel_prompt(comic, panel)

        self.assertIn("Nyx: silver android with cyan circuit markings", prompt)
        self.assertIn("required props: weathered map", prompt)
        self.assertIn("camera: medium shot, eye-level", prompt)
        self.assertIn("visual priority: Nyx dominates the foreground", prompt)
        self.assertIn("setting: empty elevated train station", prompt)
        self.assertNotIn("preserve the registered recurring cast", prompt)
        self.assertNotIn("speaking, open mouth", prompt)

    def test_exact_duplicate_structured_prose_is_included_once(self):
        comic, panel = self._comic({
            "required_action": "Nyx holds a weathered map",
            "required_props": ["weathered map"],
            "panel_prompt_en": "Nyx holds a weathered map",
        })

        prompt = build_panel_prompt(comic, panel)

        self.assertEqual(prompt.casefold().count("nyx holds a weathered map"), 1)

    def test_saved_storyboard_style_without_optional_camera_fields_remains_compatible(self):
        comic, panel = self._comic({
            "panel_prompt_en": "preserve the registered recurring cast and approved setting; Nyx studies a map",
            "panel_negative_en": "legacy negative text",
            "required_subjects": ["Nyx", "map"],
            "character_actions": {"nyx": {"action_en": "holds map"}},
        })

        prompt = build_panel_prompt(comic, panel)

        self.assertIn("Nyx: silver android with cyan circuit markings", prompt)
        self.assertIn("Nyx studies a map", prompt)
        self.assertNotIn("preserve the registered recurring cast", prompt)
        self.assertNotIn("camera:", prompt)
        self.assertNotIn("legacy negative text", prompt)

    def test_approved_dialogue_merge_does_not_change_panel_visual_prompt(self):
        schema = {
            "characters": {"nyx": {"name": "Nyx", "base_prompt_en": "android"}},
            "pages": [{"page_number": 1, "panels": [
                {"id": "p1", "panel_prompt_en": "Nyx studies a map", "active_char_ids": ["nyx"], "dialogues": []},
            ]}],
        }
        merge_approved_dialogues(
            schema,
            {"dialogues": [{"panel": 1, "character_id": "nyx", "text": "This route is wrong.", "emotion": "worried"}]},
            {"characters": [{"id": "nyx", "name": "Nyx"}]},
        )
        panel = schema["pages"][0]["panels"][0]
        self.assertEqual(panel["panel_prompt_en"], "Nyx studies a map")
        self.assertEqual(panel["dialogues"][0]["text"], "This route is wrong.")

    def test_safe_temporal_action_is_collapsed_and_ambiguous_sequence_is_reported(self):
        safe = {"pages": [{"panels": [{"required_action": "Elias pauses mid-grunt, then looks up at Nyx"}]}]}
        self.assertEqual(normalize_single_frame_actions(safe), [])
        self.assertEqual(safe["pages"][0]["panels"][0]["required_action"], "Elias pauses mid-grunt while looking up at Nyx")

        ambiguous = {"pages": [{"panels": [{
            "required_action": "Nyx blinks rapidly, then smiles faintly",
            "panel_prompt_en": "Nyx blinks rapidly, then smiles faintly in the station",
        }]}]}
        errors = normalize_single_frame_actions(ambiguous)
        self.assertEqual(len(errors), 2)
        self.assertIn("use one drawable instant", errors[0])
        self.assertEqual(ambiguous["pages"][0]["panels"][0]["required_action"], "Nyx blinks rapidly, then smiles faintly")

    @staticmethod
    def _comic(panel_fields, camera_angle="", camera_distance=""):
        data = {
            "characters": {
                "nyx": {
                    "name": "Nyx",
                    "base_prompt_en": "silver android with cyan circuit markings",
                    "camera_angle": camera_angle,
                    "camera_distance": camera_distance,
                },
            },
            "background": {"prompt_en": "fallback setting"},
            "pages": [{"page_number": 1, "panels": [{
                "id": "p1", "panel_prompt_en": "Nyx studies a map", "active_char_ids": ["nyx"],
                "dialogues": [{"character_id": "nyx", "text": "A spoken line", "emotion": "calm"}],
                **panel_fields,
            }]}],
        }
        comic = load_comic_from_json(data)
        return comic, comic.panels[0]


if __name__ == "__main__":
    unittest.main()
