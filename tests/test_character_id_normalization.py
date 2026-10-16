import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

from studio_graph.agents import merge_approved_dialogues, run_validator
from studio_graph.validation import validate_storyboard_structure


def storyboard(characters: dict) -> dict:
    return {
        "characters": characters,
        "pages": [{"page_number": 1, "panels": [
            {"id": "panel_1_1", "panel_prompt_en": "Saiya opens a letter", "active_char_ids": [], "dialogues": []},
            {"id": "panel_1_2", "panel_prompt_en": "Saiya reads the letter", "active_char_ids": [], "dialogues": []},
        ]}],
    }


class CharacterIdNormalizationTests(unittest.TestCase):
    def test_planner_numeric_id_is_mapped_to_final_canonical_id_without_changing_content(self):
        schema = storyboard({"saiya_rivers": {"name": "Saiya Rivers", "base_prompt_en": "woman in a blue jacket"}})
        original_prompt = schema["pages"][0]["panels"][0]["panel_prompt_en"]
        dialogue_plan = {"dialogues": [{"panel": 1, "character_id": "1", "text": "I found it.", "emotion": "relieved"}]}
        story_plan = {"characters": [{"id": "1", "name": "Saiya Rivers"}]}

        notes = merge_approved_dialogues(schema, dialogue_plan, story_plan)

        dialogue = schema["pages"][0]["panels"][0]["dialogues"][0]
        self.assertEqual(dialogue["character_id"], "saiya_rivers")
        self.assertEqual(dialogue["text"], "I found it.")
        self.assertEqual(dialogue["emotion"], "relieved")
        self.assertEqual(schema["pages"][0]["panels"][0]["panel_prompt_en"], original_prompt)
        self.assertEqual(notes, ["Normalized dialogue character ID '1' to canonical ID 'saiya_rivers'."])
        self.assertEqual(validate_storyboard_structure(schema, 1), [])

    def test_existing_canonical_id_is_unchanged(self):
        schema = storyboard({"saiya_rivers": {"name": "Saiya Rivers", "base_prompt_en": "woman in a blue jacket"}})
        dialogue_plan = {"dialogues": [{"panel": 1, "character_id": "saiya_rivers", "text": "Already canonical.", "emotion": "neutral"}]}

        notes = merge_approved_dialogues(schema, dialogue_plan, {"characters": [{"id": "1", "name": "Saiya Rivers"}]})

        self.assertEqual(schema["pages"][0]["panels"][0]["dialogues"][0]["character_id"], "saiya_rivers")
        self.assertEqual(notes, [])

    def test_ambiguous_or_unknown_id_remains_a_validation_error(self):
        schema = storyboard({
            "saiya_rivers": {"name": "Saiya Rivers", "base_prompt_en": "woman in a blue jacket"},
            "saiya_clone": {"name": "Saiya Rivers", "base_prompt_en": "woman in a red jacket"},
        })
        dialogue_plan = {"dialogues": [{"panel": 1, "character_id": "1", "text": "Who am I?", "emotion": "confused"}]}

        merge_approved_dialogues(schema, dialogue_plan, {"characters": [{"id": "1", "name": "Saiya Rivers"}]})

        self.assertEqual(schema["pages"][0]["panels"][0]["dialogues"][0]["character_id"], "1")
        self.assertIn(
            "Page 1, panel 1 dialogue 1 references unregistered character '1'.",
            validate_storyboard_structure(schema, 1),
        )

    def test_structural_retry_logs_attempt_errors_and_next_generation(self):
        state = {
            "current_schema": storyboard({"saiya_rivers": {"name": "Saiya Rivers", "base_prompt_en": "woman in a blue jacket"}}),
            "validation_errors": ["Page 1, panel 1 dialogue 1 references unregistered character 'unknown'."],
            "pageCount": 1,
            "retry_count": 0,
            "max_retries": 3,
        }

        with self.assertLogs("studio_graph.agents", level="INFO") as logs:
            result = run_validator(state)

        self.assertEqual(result["next_step"], "storyboarder")
        self.assertEqual(result["retry_count"], 1)
        self.assertIn("attempt 1 (retry limit 3) rejected", "\n".join(logs.output))
        self.assertIn("Requesting storyboard regeneration after attempt 1 (retry 1 of 3)", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
