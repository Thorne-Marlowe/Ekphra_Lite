import os
import json
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

from langchain_core.runnables import RunnableLambda
from panel_engine.comfy_renderer import build_panel_prompt
from studio_graph.agents import _normalize_panel_schema, merge_approved_dialogues, run_storyboarder
from studio_graph.prompts import STORYBOARDER_SYSTEM_PROMPT, build_storyboarder_human_prompt
from studio_graph.structured_output import STORYBOARD_SCHEMA
from studio_graph.validation import validate_storyboard_structure
from utils.parser import load_comic_from_json


class VisualPromptModernizationTests(unittest.TestCase):
    def test_storyboard_instructions_do_not_require_speaking_keywords_or_sd15_negatives(self):
        self.assertNotIn("STABLE DIFFUSION 1.5", STORYBOARDER_SYSTEM_PROMPT)
        self.assertNotIn("speaking, talking, open mouth", STORYBOARDER_SYSTEM_PROMPT)
        self.assertNotIn("blurry, distorted, deformed hands", STORYBOARDER_SYSTEM_PROMPT)

    def test_storyboard_instructions_allow_brief_temporal_context(self):
        self.assertIn("Brief temporal context is allowed", STORYBOARDER_SYSTEM_PROMPT)

    def test_storyboard_artist_has_visual_adapter_role_and_preserves_approved_authority(self):
        prompt = build_storyboarder_human_prompt(page_count=1, has_previous_draft=False)

        self.assertIn("sole responsibility is to translate the approved Story Planner beats", STORYBOARDER_SYSTEM_PROMPT)
        self.assertNotIn("Sole Creator", STORYBOARDER_SYSTEM_PROMPT)
        self.assertNotIn("must write the story and the dialogue yourself", STORYBOARDER_SYSTEM_PROMPT)
        self.assertNotIn("COMEDIC FORMULAS (YOU MUST STRICTLY USE", STORYBOARDER_SYSTEM_PROMPT)
        self.assertIn("approved story plan controls narrative events, cast, beat intent", STORYBOARDER_SYSTEM_PROMPT)
        self.assertIn("approved dialogue plan controls dialogue text, speaker IDs, and panel assignments", STORYBOARDER_SYSTEM_PROMPT)
        self.assertIn("positive requirements, not exhaustive lists", STORYBOARDER_SYSTEM_PROMPT)
        self.assertIn("Use the approved story plan and dialogue plan below", prompt)

    def test_storyboard_example_is_valid_json_with_compatible_metadata_shapes(self):
        example = re.search(r"```json\n(.*?)\n```", STORYBOARDER_SYSTEM_PROMPT, re.DOTALL)
        self.assertIsNotNone(example)
        # ChatPromptTemplate uses f-string syntax. Escaped source braces render
        # as ordinary JSON braces in the model-facing system prompt.
        schema = json.loads(example.group(1).replace("{{", "{").replace("}}", "}"))
        panel = schema["pages"][0]["panels"][0]

        self.assertEqual(validate_storyboard_structure(schema, 1), [])
        panel_properties = STORYBOARD_SCHEMA["properties"]["pages"]["items"]["properties"]["panels"]["items"]["properties"]
        self.assertTrue(all(field in panel for field in (
            "primary_subject", "secondary_subject", "visual_hierarchy", "required_subjects",
            "required_action", "required_props", "required_text", "character_actions",
        )))
        self.assertIn("character_positions", panel_properties)
        _normalize_panel_schema(schema)
        self.assertEqual(panel["required_props"], ["green slime", "crooked NO ENTRY sign"])
        self.assertEqual(panel["required_text"], ["NO ENTRY"])
        self.assertIn("NO ENTRY sign", panel["panel_prompt_en"])
        self.assertEqual(panel["character_actions"]["char_dad"]["objects"], ["green slime"])

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

    def test_character_actions_keep_each_action_with_its_actor(self):
        comic, panel = self._two_character_comic({
            "panel_prompt_en": "Lena and Jai sit together in the quiet music room.",
            "character_actions": {
                "lena": {"action_en": "plays the violin", "pose_en": "standing beside Jai"},
                "jai": {"action_en": "listens with his eyes closed", "pose_en": "seated in a chair"},
            },
        })

        prompt = build_panel_prompt(comic, panel)

        self.assertIn("Lena plays the violin", prompt)
        self.assertIn("Jai listens with his eyes closed", prompt)
        self.assertNotIn("Jai plays the violin", prompt)

    def test_action_already_expressed_in_panel_prose_is_not_repeated(self):
        comic, panel = self._two_character_comic({
            "panel_prompt_en": "Medium shot. Lena plays the violin while Jai listens with closed eyes.",
            "character_actions": {
                "lena": {"action_en": "playing violin"},
                "jai": {"action_en": "listening with closed eyes"},
            },
        })

        prompt = build_panel_prompt(comic, panel)

        self.assertEqual(prompt.casefold().count("lena plays the violin"), 1)
        self.assertNotIn("Lena playing violin", prompt)

    def test_structured_subject_ids_are_rendered_as_registered_names(self):
        comic, panel = self._two_character_comic({
            "primary_subject": "lena",
            "secondary_subject": "jai (background)",
            "visual_hierarchy": ["lena", "jai's hand"],
            "panel_prompt_en": "Lena reaches toward Jai in the music room.",
        })

        prompt = build_panel_prompt(comic, panel)

        self.assertIn("primary subject: Lena", prompt)
        self.assertIn("secondary subject: Jai (background)", prompt)
        self.assertNotIn("primary subject: lena", prompt)

    def test_panel_prose_wins_over_conflicting_lower_precedence_action(self):
        comic, panel = self._two_character_comic({
            "panel_prompt_en": "Medium close-up. Jai has stopped playing; the violin rests across his lap while Lena touches his hand.",
            "required_action": "Lena touches Jai's hand while he plays the violin",
            "character_actions": {"jai": {"action_en": "playing violin"}},
        })

        prompt = build_panel_prompt(comic, panel)

        self.assertIn("Jai has stopped playing", prompt)
        self.assertNotIn("action: Lena touches Jai's hand while he plays", prompt)
        self.assertNotIn("Jai playing violin", prompt)

    def test_jai_stopping_does_not_suppress_lenas_independent_playing_action(self):
        comic, panel = self._two_character_comic({
            "panel_prompt_en": "Jai has stopped playing and rests his violin in his lap while Lena waits beside him.",
            "character_actions": {
                "jai": {"action_en": "playing violin"},
                "lena": {"action_en": "playing violin"},
            },
        })

        prompt = build_panel_prompt(comic, panel)

        self.assertNotIn("Jai playing violin", prompt)
        self.assertIn("Lena playing violin", prompt)

    def test_explicit_panel_framing_overrides_character_card_camera(self):
        comic, panel = self._comic({"panel_prompt_en": "Close-up of Nyx studying a map."}, camera_distance="wide shot")

        prompt = build_panel_prompt(comic, panel)

        self.assertIn("Close-up of Nyx", prompt)
        self.assertNotIn("camera: wide shot", prompt)

    def test_temporal_context_survives_assembly(self):
        comic, panel = self._comic({
            "panel_prompt_en": "After the final note fades, Nyx holds the map against her chest in the rain.",
        })

        self.assertIn("After the final note fades", build_panel_prompt(comic, panel))

    def test_edited_panel_prompt_is_used_on_the_next_assembly(self):
        comic, panel = self._comic({"panel_prompt_en": "Nyx studies a map in the station."})
        original = build_panel_prompt(comic, panel)
        panel.panel_prompt_en = "Nyx studies a map beneath a flickering platform sign."

        edited = build_panel_prompt(comic, panel)

        self.assertNotEqual(original, edited)
        self.assertIn("flickering platform sign", edited)

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

    def test_mocked_storyboard_artist_cannot_replace_approved_dialogue(self):
        artist_schema = {
            "characters": {"nyx": {"name": "Nyx", "base_prompt_en": "android"}},
            "pages": [{"page_number": 1, "panels": [
                {"id": "p1", "panel_prompt_en": "Nyx studies a map", "active_char_ids": ["nyx"],
                 "dialogues": [{"character_id": "nyx", "text": "Artist replacement.", "emotion": "angry"}]},
                {"id": "p2", "panel_prompt_en": "Nyx leaves the station", "active_char_ids": ["nyx"], "dialogues": []},
            ]}],
        }
        state = {
            "user_prompt": "Nyx misses a train.", "pageCount": 1, "current_schema": {}, "validation_errors": [],
            "story_plan": {"characters": [{"id": "nyx", "name": "Nyx"}]},
            "dialogue_plan": {"dialogues": [{"panel": 1, "character_id": "nyx", "text": "This route is wrong.", "emotion": "worried"}]},
        }

        with patch("studio_graph.agents.storyboarder_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(artist_schema)))):
            result = run_storyboarder(state)

        dialogue = result["current_schema"]["pages"][0]["panels"][0]["dialogues"][0]
        self.assertEqual(dialogue["text"], "This route is wrong.")
        self.assertEqual(dialogue["character_id"], "nyx")
        self.assertEqual(dialogue["emotion"], "worried")

    def test_storyboarder_preserves_temporal_context_without_a_retry_error(self):
        schema = {
            "characters": {"nyx": {"name": "Nyx", "base_prompt_en": "silver android"}},
            "pages": [{"page_number": 1, "panels": [
                {"id": "p1", "panel_prompt_en": "After the train leaves, Nyx holds a map in the rain.", "active_char_ids": ["nyx"], "dialogues": []},
                {"id": "p2", "panel_prompt_en": "Nyx watches the empty platform.", "active_char_ids": ["nyx"], "dialogues": []},
            ]}],
        }
        state = {
            "user_prompt": "Nyx misses a train.", "pageCount": 1, "current_schema": {},
            "validation_errors": [], "story_plan": {}, "dialogue_plan": {"dialogues": []},
        }
        storyboarder = RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(schema)))

        with patch("studio_graph.agents.storyboarder_llm", storyboarder):
            result = run_storyboarder(state)

        self.assertEqual(result["validation_errors"], [])
        self.assertEqual(result["current_schema"]["pages"][0]["panels"][0]["panel_prompt_en"], schema["pages"][0]["panels"][0]["panel_prompt_en"])

    def test_last_melody_panel_four_remains_compatible(self):
        comic, panel = self._two_character_comic({
            "primary_subject": "Lena",
            "secondary_subject": "Jai listening",
            "visual_hierarchy": ["Lena", "violin"],
            "required_action": "Lena plays the melody, Jai listens with closed eyes",
            "required_props": ["violin now held by Lena"],
            "panel_prompt_en": "Full-body shot. Lena plays the violin with a clear, soothing melody. Jai smiles softly with his closed eyes.",
            "character_actions": {
                "jai": {"action_en": "listening with closed eyes", "pose_en": "sitting quietly"},
                "lena": {"action_en": "playing the violin", "pose_en": "holding it confidently"},
            },
        })

        prompt = build_panel_prompt(comic, panel)

        self.assertIn("Full-body shot", prompt)
        self.assertIn("Lena", prompt)
        self.assertIn("Jai", prompt)

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

    @staticmethod
    def _two_character_comic(panel_fields):
        data = {
            "characters": {
                "lena": {"name": "Lena", "base_prompt_en": "young violinist with a blue sweater"},
                "jai": {"name": "Jai", "base_prompt_en": "older musician with a black turtleneck"},
            },
            "background": {"prompt_en": "quiet music room at dusk"},
            "pages": [{"page_number": 1, "panels": [{
                "id": "panel_1_4", "panel_prompt_en": "Lena and Jai in a music room",
                "active_char_ids": ["lena", "jai"], "dialogues": [], **panel_fields,
            }]}],
        }
        comic = load_comic_from_json(data)
        return comic, comic.panels[0]


if __name__ == "__main__":
    unittest.main()
