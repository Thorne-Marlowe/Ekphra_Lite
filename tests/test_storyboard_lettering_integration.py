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
from PIL import Image

from bubbles.injector import inject_bubbles_to_page
from config import CONFIG
from panel_engine.comfy_renderer import render_comic_schema
from studio_graph.agents import _normalize_panel_schema, run_storyboarder
from studio_graph.prompts import STORYBOARDER_SYSTEM_PROMPT
from studio_graph.structured_output import STORYBOARD_SCHEMA
from studio_graph.validation import validate_storyboard_structure


class StoryboardLetteringIntegrationTests(unittest.TestCase):
    def test_artist_prompt_uses_viewer_facing_positions_not_anatomical_directions(self):
        self.assertIn("viewer's image-left and image-right", STORYBOARDER_SYSTEM_PROMPT)
        self.assertIn("never the character's anatomical left or right", STORYBOARDER_SYSTEM_PROMPT)
        self.assertIn("never claim positions were verified against a rendered image", STORYBOARDER_SYSTEM_PROMPT)

    def test_structured_schema_and_storyboarder_preserve_optional_last_melody_positions(self):
        panel_schema = STORYBOARD_SCHEMA["properties"]["pages"]["items"]["properties"]["panels"]["items"]["properties"]
        self.assertIn("character_positions", panel_schema)
        schema = self._last_melody_schema()
        state = {"user_prompt": "The Last Melody", "pageCount": 1, "current_schema": {}, "validation_errors": [], "story_plan": {}, "dialogue_plan": {"dialogues": []}}
        storyboarder = RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(schema)))
        with patch("studio_graph.agents.storyboarder_llm", storyboarder):
            result = run_storyboarder(state)

        panel = result["current_schema"]["pages"][0]["panels"][1]
        self.assertEqual(result["validation_errors"], [])
        self.assertEqual(panel["character_positions"]["char_lena"], {"x": "right", "y": "middle", "anchor": "center"})
        self.assertEqual(validate_storyboard_structure(result["current_schema"], 1), [])

    def test_normalization_preserves_positions_and_missing_field_remains_valid(self):
        schema = self._last_melody_schema()
        original = json.loads(json.dumps(schema["pages"][0]["panels"][1]["character_positions"]))
        _normalize_panel_schema(schema)
        self.assertEqual(schema["pages"][0]["panels"][1]["character_positions"], original)
        self.assertNotIn("character_positions", schema["pages"][0]["panels"][0])
        self.assertEqual(validate_storyboard_structure(schema, 1), [])

    def test_lenas_authored_position_reaches_page_lettering_without_segmentation(self):
        schema = self._last_melody_schema()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panels = root / "panels"
            panels.mkdir()
            Image.new("RGB", (500, 700), "white").save(panels / "p1.png")
            Image.new("RGB", (500, 700), "white").save(panels / "panel_1_4.png")
            story = root / "story.json"
            story.write_text(json.dumps(schema), encoding="utf-8")
            captured = []
            with patch("bubbles.injector.find_character_bounding_box", return_value=None), \
                    patch("bubbles.injector.draw_speech_bubble", side_effect=lambda **kwargs: captured.append(kwargs["speaker_target"])):
                inject_bubbles_to_page(Image.new("RGB", (500, 1400), "white"), str(story), str(panels), [(0, 0, 500, 700), (0, 700, 500, 700)], ["p1", "panel_1_4"])
        self.assertEqual(captured, [(400, 1050)])

    def test_dialogue_only_recomposition_persists_positions_without_diffusion_or_agents(self):
        schema = self._last_melody_schema()
        with tempfile.TemporaryDirectory() as directory:
            original_panel_dir = CONFIG.paths.panel_dir
            try:
                CONFIG.paths.panel_dir = str(Path(directory) / "panels")
                with patch("panel_engine.comfy_renderer.render_panels_with_comfyui") as render_panels, \
                        patch("panel_engine.comfy_renderer.build_comic_page"):
                    _pages, story_path = render_comic_schema(
                        schema, "dialogue_positions", None, None, {}, target_panel_ids=[], compose_page_numbers=[1]
                    )
            finally:
                CONFIG.paths.panel_dir = original_panel_dir
            persisted = json.loads(Path(story_path).read_text(encoding="utf-8"))
        render_panels.assert_not_called()
        self.assertEqual(persisted["pages"][0]["panels"][1]["character_positions"]["char_lena"]["x"], "right")

    @staticmethod
    def _last_melody_schema():
        return {
            "characters": {"char_jai": {"name": "Jai Thompson", "base_prompt_en": "older musician"}, "char_lena": {"name": "Lena", "base_prompt_en": "young violinist"}},
            "pages": [{"page_number": 1, "panels": [
                {"id": "p1", "panel_prompt_en": "Jai listens in the music room.", "active_char_ids": ["char_jai"], "dialogues": []},
                {"id": "panel_1_4", "panel_prompt_en": "Lena plays while Jai listens.", "active_char_ids": ["char_jai", "char_lena"],
                 "character_positions": {"char_jai": {"x": "left", "y": "middle", "anchor": "center"}, "char_lena": {"x": "right", "y": "middle", "anchor": "center"}},
                 "dialogues": [{"character_id": "char_lena", "text": "I think I know the next part.", "emotion": "hopeful"}]},
            ]}],
        }


if __name__ == "__main__":
    unittest.main()
