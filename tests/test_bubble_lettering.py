import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageFont

from bubbles.injector import (
    _place_bubble_without_overlap,
    _rects_overlap,
    caption_texts_for_panel,
    inject_bubbles_to_page,
    normalize_lettering_text,
    panel_font_size,
)


class BubbleLetteringTests(unittest.TestCase):
    def test_duplicate_caption_is_suppressed_after_case_whitespace_and_quote_normalization(self):
        panel = {
            "dialogues": [{"text": "Hello?   Who are you?"}],
            "required_text": ["  ‘hello? who are you?’ ", "EXIT →"],
        }
        self.assertEqual(normalize_lettering_text(" ‘Hello? Who are you?’ "), "hello? who are you?")
        self.assertEqual(caption_texts_for_panel(panel), ["EXIT →"])

    def test_distinct_environmental_text_remains_a_caption(self):
        panel = {
            "dialogues": [{"text": "Hello?"}],
            "required_text": ["SECTOR 7", "MAINTENANCE BAY"],
        }
        self.assertEqual(caption_texts_for_panel(panel), ["SECTOR 7", "MAINTENANCE BAY"])

    def test_captions_and_bubbles_share_panel_local_occupancy(self):
        with self._page_fixture({
            "dialogues": [{"character_id": "a", "text": "Hello", "emotion": ""}],
            "required_text": ["SECTOR 7"],
        }) as (page, schema_path, panels_dir):
            original_placer = _place_bubble_without_overlap
            observed_existing_counts = []

            def capture_placer(*args, **kwargs):
                observed_existing_counts.append(len(args[4]))
                return original_placer(*args, **kwargs)

            with patch("bubbles.injector._place_bubble_without_overlap", side_effect=capture_placer), \
                    patch("bubbles.injector.draw_speech_bubble"):
                inject_bubbles_to_page(page, str(schema_path), str(panels_dir), [(0, 0, 500, 700)], ["p1"])

        self.assertEqual(observed_existing_counts, [0, 1])

    def test_multiple_sentences_receive_non_overlapping_bubbles(self):
        with self._page_fixture({
            "dialogues": [{"character_id": "a", "text": "First sentence. Second sentence.", "emotion": ""}],
            "required_text": [],
        }) as (page, schema_path, panels_dir):
            drawn = []
            with patch("bubbles.injector.draw_speech_bubble", side_effect=lambda **kwargs: drawn.append(kwargs["bbox"])):
                inject_bubbles_to_page(page, str(schema_path), str(panels_dir), [(0, 0, 500, 700)], ["p1"])

        self.assertEqual(len(drawn), 2)
        self.assertFalse(_rects_overlap(drawn[0], drawn[1], margin=0))

    def test_insufficient_space_raises_instead_of_overlapping(self):
        with self.assertRaisesRegex(ValueError, "No non-overlapping lettering position"):
            _place_bubble_without_overlap(
                (20, 20), 100, 100, (0, 0, 160, 160), [(20, 20, 100, 100)], None,
            )

    def test_caption_and_dialogue_use_the_same_panel_relative_font_size(self):
        self.assertEqual(panel_font_size(500, 900), panel_font_size(500, 500))
        self.assertGreater(panel_font_size(900, 1200), panel_font_size(400, 800))

        with self._page_fixture({
            "dialogues": [{"character_id": "a", "text": "Hello", "emotion": ""}],
            "required_text": ["SECTOR 7"],
        }) as (page, schema_path, panels_dir):
            font_sizes = []

            def fake_font(_name, size):
                font_sizes.append(size)
                return ImageFont.load_default()

            with patch("bubbles.injector.load_font", side_effect=fake_font), \
                    patch("bubbles.injector.draw_speech_bubble"):
                inject_bubbles_to_page(page, str(schema_path), str(panels_dir), [(0, 0, 500, 700)], ["p1"])

        self.assertEqual(font_sizes, [panel_font_size(500, 700), panel_font_size(500, 700)])

    @staticmethod
    def _page_fixture(panel_fields):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        panels_dir = root / "panels"
        panels_dir.mkdir()
        Image.new("RGB", (500, 700), "white").save(panels_dir / "p1.png")
        schema = {
            "characters": {"a": {"name": "A", "base_prompt_en": "test"}},
            "pages": [{"page_number": 1, "panels": [{"id": "p1", "active_char_ids": ["a"], **panel_fields}]}],
        }
        schema_path = root / "story.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        page = Image.new("RGB", (500, 700), "white")

        class Fixture:
            def __enter__(self):
                return page, schema_path, panels_dir

            def __exit__(self, *args):
                directory.cleanup()

        return Fixture()


if __name__ == "__main__":
    unittest.main()
