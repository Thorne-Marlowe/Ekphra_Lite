import unittest

from studio_graph.validation import validate_storyboard_structure


VALID_FOUR_PANEL_STORYBOARD = {
    "characters": {"sam": {"name": "Sam", "base_prompt_en": "young person wearing a blue t-shirt"}},
    "pages": [{"page_number": 1, "panels": [
        {"id": "panel_1_1", "panel_prompt_en": "Sam reaches for a coffee mug", "active_char_ids": ["sam"], "dialogues": []},
        {"id": "panel_1_2", "panel_prompt_en": "The mug slides away", "active_char_ids": ["sam"], "dialogues": []},
        {"id": "panel_1_3", "panel_prompt_en": "Sam stares at the empty desk", "active_char_ids": ["sam"], "dialogues": []},
        {"id": "panel_1_4", "panel_prompt_en": "The mug sits in Sam's other hand", "active_char_ids": ["sam"], "dialogues": []},
    ]}],
}


class StoryboardValidationTests(unittest.TestCase):
    def test_valid_four_panel_storyboard_has_no_retry_errors(self):
        self.assertEqual(validate_storyboard_structure(VALID_FOUR_PANEL_STORYBOARD, requested_pages=1), [])

    def test_invalid_panel_count_remains_a_deterministic_error(self):
        malformed = {**VALID_FOUR_PANEL_STORYBOARD, "pages": [{**VALID_FOUR_PANEL_STORYBOARD["pages"][0], "panels": VALID_FOUR_PANEL_STORYBOARD["pages"][0]["panels"][:1]}]}
        self.assertEqual(validate_storyboard_structure(malformed, requested_pages=1), ["Page 1 must contain 2 to 6 panels, found 1."])

    def test_optional_character_positions_are_validated_only_when_supplied(self):
        malformed = {**VALID_FOUR_PANEL_STORYBOARD, "pages": [{**VALID_FOUR_PANEL_STORYBOARD["pages"][0], "panels": [
            {**VALID_FOUR_PANEL_STORYBOARD["pages"][0]["panels"][0], "character_positions": {"unknown": {"x": "sideways", "y": "up", "anchor": "left"}}},
            *VALID_FOUR_PANEL_STORYBOARD["pages"][0]["panels"][1:],
        ]}]}
        errors = validate_storyboard_structure(malformed, requested_pages=1)
        self.assertIn("Page 1, panel 1 character_positions references unregistered character 'unknown'.", errors)
        self.assertTrue(any(".x must be" in error for error in errors))
        self.assertTrue(any(".y must be" in error for error in errors))
        self.assertTrue(any(".anchor must be center" in error for error in errors))
