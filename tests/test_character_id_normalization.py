import os
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

from langchain_core.runnables import RunnableLambda

from studio_graph.agents import merge_approved_dialogues, run_storyboarder, run_validator
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
    def test_numeric_canonical_character_id_is_written_back_as_a_string(self):
        schema = storyboard({
            "1": {"name": "Elara", "base_prompt_en": "adult detective"},
            "2": {"name": "Surveyor", "base_prompt_en": "adult surveyor"},
        })
        dialogue_plan = {"dialogues": [{
            "panel": 1, "character_id": 2, "text": "I corrected the record.", "emotion": "solemn",
        }]}

        notes = merge_approved_dialogues(schema, dialogue_plan, {"characters": []})

        dialogue = schema["pages"][0]["panels"][0]["dialogues"][0]
        self.assertEqual(dialogue["character_id"], "2")
        self.assertIsInstance(dialogue["character_id"], str)
        self.assertEqual(notes, ["Normalized dialogue character ID '2' to canonical ID '2'."])

    def test_string_canonical_character_id_remains_a_string(self):
        schema = storyboard({"2": {"name": "Surveyor", "base_prompt_en": "adult surveyor"}})
        dialogue_plan = {"dialogues": [{
            "panel": 1, "character_id": "2", "text": "The record was altered.", "emotion": "solemn",
        }]}

        notes = merge_approved_dialogues(schema, dialogue_plan, {"characters": []})

        dialogue = schema["pages"][0]["panels"][0]["dialogues"][0]
        self.assertEqual(dialogue["character_id"], "2")
        self.assertIsInstance(dialogue["character_id"], str)
        self.assertEqual(notes, [])

    def test_original_numeric_surveyor_dialogue_merges_and_validates(self):
        schema = {
            "characters": {
                "1": {"name": "Dr. Elara Voss", "base_prompt_en": "adult archivist"},
                "2": {"name": "The Surveyor", "base_prompt_en": "adult surveyor"},
                "3": {"name": "The Curator", "base_prompt_en": "adult curator"},
            },
            "pages": [{"page_number": 1, "panels": [
                {"id": f"panel_1_{index}", "panel_prompt_en": "Adult archivist studies municipal records.",
                 "active_char_ids": ["1"], "dialogues": []}
                for index in range(1, 5)
            ]}],
        }
        dialogue_plan = {"dialogues": [{
            "panel": 4, "character_id": 2,
            "text": "I corrected the record to expose the stolen land.", "emotion": "solemn",
        }]}
        story_plan = {"characters": [
            {"id": "1", "name": "Dr. Elara Voss"},
            {"id": "2", "name": "The Surveyor"},
            {"id": "3", "name": "The Curator"},
        ]}

        merge_approved_dialogues(schema, dialogue_plan, story_plan)

        final_dialogue = schema["pages"][0]["panels"][3]["dialogues"][0]
        self.assertEqual(final_dialogue["character_id"], "2")
        self.assertEqual(schema["pages"][0]["panels"][3]["active_char_ids"], ["1"])
        self.assertEqual(validate_storyboard_structure(schema, 1), [])

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

    def test_registered_off_panel_dialogue_is_valid(self):
        schema = storyboard({
            "1": {"name": "Elara", "base_prompt_en": "adult detective"},
            "2": {"name": "Surveyor", "base_prompt_en": "adult surveyor"},
        })
        schema["pages"][0]["panels"][0].update({
            "active_char_ids": ["1"],
            "dialogues": [{"character_id": "2", "text": "The archive is sealed.", "emotion": "calm"}],
        })

        self.assertEqual(validate_storyboard_structure(schema, 1), [])

    def test_invalid_position_key_and_unregistered_dialogue_are_both_reported(self):
        schema = storyboard({"1": {"name": "Elara", "base_prompt_en": "adult detective"}})
        schema["pages"][0]["panels"][0].update({
            "character_positions": {"curator": {"x": "left", "y": "middle", "anchor": "center"}},
            "dialogues": [{"character_id": "2", "text": "Read this.", "emotion": "calm"}],
        })

        errors = validate_storyboard_structure(schema, 1)
        self.assertIn("Page 1, panel 1 character_positions references unregistered character 'curator'.", errors)
        self.assertIn("Page 1, panel 1 dialogue 1 references unregistered character '2'.", errors)

    def test_character_id_drift_after_failure_keeps_planner_ids_in_retry_feedback(self):
        drifted_schema = storyboard({
            "elara": {"name": "Elara", "base_prompt_en": "adult detective"},
            "mystery_guide": {"name": "The Surveyor", "base_prompt_en": "adult surveyor"},
            "curator": {"name": "Curator", "base_prompt_en": "adult curator"},
        })
        drifted_schema["pages"][0]["panels"][0]["active_char_ids"] = ["elara"]
        state = {
            "user_prompt": "An archive mystery.", "pageCount": 1, "current_schema": {}, "validation_errors": [],
            "story_plan": {"characters": [
                {"id": "1", "name": "Elara"}, {"id": "2", "name": "Surveyor"}, {"id": "3", "name": "Curator"},
            ]},
            "dialogue_plan": {"dialogues": [{"panel": 1, "character_id": "2", "text": "The archive is sealed.", "emotion": "calm"}]},
            "retry_count": 0, "max_retries": 3,
        }
        captured = {}

        def response(prompt_value):
            captured["prompt"] = "\n".join(str(message.content) for message in prompt_value.to_messages())
            return SimpleNamespace(content=json.dumps(drifted_schema))

        with patch("studio_graph.agents.storyboarder_llm", RunnableLambda(response)):
            run_storyboarder(state)
        run_validator(state)

        self.assertIn("dialogue 1 references unregistered character '2'", "\n".join(state["validation_errors"]).lower())
        self.assertEqual(state["storyboard_attempt_diagnostics"][-1]["stage"], "structural_validation")
        self.assertEqual(state["storyboard_attempt_diagnostics"][-1]["character_registry_ids"], ["elara", "mystery_guide", "curator"])

        with patch("studio_graph.agents.storyboarder_llm", RunnableLambda(response)):
            run_storyboarder(state)
        self.assertIn('["1", "2", "3"]', captured["prompt"])
        self.assertIn("A dialogue\nspeaker may be off-panel", captured["prompt"])

    def test_parse_failure_records_attempt_stage_without_story_content(self):
        state = {
            "user_prompt": "A private premise.", "pageCount": 1, "current_schema": {}, "validation_errors": [],
            "story_plan": {"characters": [{"id": "1", "name": "Elara"}]}, "dialogue_plan": {},
            "retry_count": 2,
        }
        with patch("studio_graph.agents.storyboarder_llm", RunnableLambda(lambda _: SimpleNamespace(content="not json"))):
            run_storyboarder(state)

        self.assertEqual(state["storyboard_attempt_diagnostics"], [{
            "attempt": 3, "stage": "parsing", "character_registry_ids": [],
            "errors": state["validation_errors"],
        }])
        self.assertNotIn("private premise", "\n".join(state["storyboard_attempt_diagnostics"][0]["errors"]))

    def test_normalization_failure_records_registry_ids(self):
        candidate = storyboard({"1": {"name": "Elara", "base_prompt_en": "adult detective"}})
        state = {
            "user_prompt": "A scene.", "pageCount": 1, "current_schema": {}, "validation_errors": [],
            "story_plan": {"characters": [{"id": "1", "name": "Elara"}], "beats": [{"panel": 9, "event": "Wrong panel"}]},
            "dialogue_plan": {}, "retry_count": 0,
        }
        with patch("studio_graph.agents.storyboarder_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(candidate)))):
            run_storyboarder(state)

        diagnostic = state["storyboard_attempt_diagnostics"][-1]
        self.assertEqual(diagnostic["stage"], "normalization")
        self.assertEqual(diagnostic["character_registry_ids"], ["1"])
        self.assertTrue(any("Story plan beat 1 references '9'" in error for error in diagnostic["errors"]))

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
