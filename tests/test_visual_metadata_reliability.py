import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from policy.adult_only import adult_only_errors
from studio_graph.agents import (
    _apply_story_constraints,
    _normalize_panel_schema,
    merge_approved_dialogues,
    run_storyboarder,
)
from studio_graph.validation import validate_storyboard_structure
from utils.parser import load_comic_from_json


def panel(panel_id):
    return {
        "id": panel_id,
        "panel_prompt_en": f"Adult subject in {panel_id}.",
        "active_char_ids": ["ava"],
        "dialogues": [],
    }


def schema(page_counts=(3, 3)):
    pages = []
    panel_number = 0
    for page_number, count in enumerate(page_counts, start=1):
        page_panels = []
        for local_number in range(1, count + 1):
            panel_number += 1
            page_panels.append(panel(f"panel_{page_number}_{local_number}"))
        pages.append({"page_number": page_number, "panels": page_panels})
    return {
        "metadata": {"creative_intent": {"effective": {"genre": "mystery"}}, "note": "preserve"},
        "characters": {"ava": {"name": "Ava", "base_prompt_en": "adult woman"}},
        "pages": pages,
    }


class VisualMetadataReliabilityTests(unittest.TestCase):
    def test_omitted_and_explicit_empty_requirements_have_different_pre_normalization_behavior(self):
        draft = schema((3,))
        target = draft["pages"][0]["panels"][0]
        target["required_props"] = ["artist prop"]
        target["required_text"] = ["ARTIST LABEL"]

        omitted = copy.deepcopy(draft)
        _apply_story_constraints(omitted, {"beats": [{"panel": 1, "event": "keep artist details"}]})
        self.assertEqual(omitted["pages"][0]["panels"][0]["required_props"], ["artist prop"])

        explicit_empty = copy.deepcopy(draft)
        _apply_story_constraints(explicit_empty, {"beats": [{
            "panel": 1, "event": "no mandatory objects", "required_props": [], "required_text": [],
        }]})
        self.assertEqual(explicit_empty["pages"][0]["panels"][0]["required_props"], [])
        self.assertEqual(explicit_empty["pages"][0]["panels"][0]["required_text"], [])

        final = schema((3,))
        _normalize_panel_schema(final)
        self.assertEqual(final["pages"][0]["panels"][0]["required_props"], [])
        self.assertEqual(final["pages"][0]["panels"][0]["required_text"], [])

    def test_planner_requirements_override_only_their_declared_fields_and_keep_artist_details(self):
        draft = schema((3,))
        target = draft["pages"][0]["panels"][0]
        target.update({
            "primary_subject": "artist subject", "required_props": ["artist prop"],
            "character_actions": {"ava": {"action_en": "leans toward the letter"}},
            "character_positions": {"ava": {"x": "left", "y": "middle", "anchor": "center"}},
        })
        errors = []
        _apply_story_constraints(draft, {"beats": [{
            "panel": 1, "event": "Ava finds the letter", "primary_subject": "Ava holding the letter",
            "secondary_subject": "the sealed envelope", "visual_hierarchy": "Ava first, envelope second",
            "required_subjects": ["Ava", "sealed envelope"], "required_action": "Ava opens the envelope",
            "required_props": ["sealed envelope"], "required_text": ["ARCHIVE"],
        }]}, errors)
        self.assertEqual(errors, [])
        self.assertEqual(target["primary_subject"], "Ava holding the letter")
        self.assertEqual(target["required_action"], "Ava opens the envelope")
        self.assertEqual(target["required_props"], ["sealed envelope"])
        self.assertEqual(target["required_text"], ["ARCHIVE"])
        self.assertEqual(target["character_actions"]["ava"]["action_en"], "leans toward the letter")
        self.assertEqual(target["character_positions"]["ava"]["x"], "left")

    def test_invalid_metadata_types_are_reported_without_inventing_requirements(self):
        draft = schema((3,))
        draft["pages"][0]["panels"][0]["required_props"] = {"not": "a list"}
        draft["pages"][0]["panels"][0]["primary_subject"] = {"not": "a string"}
        errors = []
        _apply_story_constraints(draft, {"beats": [{
            "panel": 1, "event": "bad data", "required_action": ["not a string"], "required_text": {"bad": "shape"},
        }]}, errors)
        _normalize_panel_schema(draft, errors)
        self.assertTrue(any("required_action must be a string" in error for error in errors), errors)
        self.assertTrue(any("required_text must be an array" in error for error in errors), errors)
        self.assertTrue(any("required_props must be an array" in error for error in errors), errors)
        self.assertTrue(any("primary_subject must be a string" in error for error in errors), errors)

    def test_required_metadata_survives_repeated_normalization_for_a_retry(self):
        draft = schema((3,))
        plan = {"beats": [{
            "panel": 1, "event": "Ava finds evidence", "required_subjects": ["Ava", "photograph"],
            "required_action": "Ava lifts the photograph", "required_props": ["photograph"], "required_text": ["ARCHIVE"],
        }]}
        first_errors, second_errors = [], []
        _apply_story_constraints(draft, plan, first_errors)
        _normalize_panel_schema(draft, first_errors)
        _apply_story_constraints(draft, plan, second_errors)
        _normalize_panel_schema(draft, second_errors)
        target = draft["pages"][0]["panels"][0]
        self.assertEqual(first_errors + second_errors, [])
        self.assertEqual(target["required_subjects"], ["Ava", "photograph"])
        self.assertEqual(target["required_props"], ["photograph"])
        self.assertEqual(target["required_text"], ["ARCHIVE"])

    def test_three_panels_per_page_use_actual_global_order_for_constraints_and_dialogue(self):
        draft = schema((3, 3))
        errors = []
        _apply_story_constraints(draft, {"beats": [
            {"panel": 3, "event": "last on first page", "required_props": ["page one prop"]},
            {"panel": 4, "event": "first on second page", "required_props": ["page two prop"]},
        ]}, errors)
        merge_approved_dialogues(draft, {"dialogues": [
            {"panel": 3, "character_id": "ava", "text": "Third.", "emotion": "calm"},
            {"panel": 4, "character_id": "ava", "text": "Fourth.", "emotion": "calm"},
        ]}, {"characters": [{"id": "ava", "name": "Ava"}]}, errors)
        self.assertEqual(errors, [])
        self.assertEqual(draft["pages"][0]["panels"][2]["required_props"], ["page one prop"])
        self.assertEqual(draft["pages"][1]["panels"][0]["required_props"], ["page two prop"])
        self.assertEqual(draft["pages"][0]["panels"][2]["dialogues"][0]["text"], "Third.")
        self.assertEqual(draft["pages"][1]["panels"][0]["dialogues"][0]["text"], "Fourth.")

    def test_unequal_page_counts_and_exact_panel_ids_map_without_six_panel_stride(self):
        draft = schema((2, 5))
        errors = []
        _apply_story_constraints(draft, {"beats": [
            {"panel": 3, "event": "first on page two", "required_action": "Ava enters"},
            {"panel": "panel_2_5", "event": "last panel", "required_action": "Ava leaves"},
        ]}, errors)
        merge_approved_dialogues(draft, {"dialogues": [
            {"panel": 3, "character_id": "ava", "text": "Entering.", "emotion": "calm"},
            {"panel": 7, "character_id": "ava", "text": "Leaving.", "emotion": "calm"},
        ]}, {"characters": [{"id": "ava", "name": "Ava"}]}, errors)
        self.assertEqual(errors, [])
        self.assertEqual(draft["pages"][1]["panels"][0]["required_action"], "Ava enters")
        self.assertEqual(draft["pages"][1]["panels"][4]["required_action"], "Ava leaves")
        self.assertEqual(draft["pages"][1]["panels"][0]["dialogues"][0]["text"], "Entering.")
        self.assertEqual(draft["pages"][1]["panels"][4]["dialogues"][0]["text"], "Leaving.")

    def test_invalid_or_legacy_stride_reference_is_a_deterministic_error_not_a_wrong_assignment(self):
        draft = schema((3, 3))
        errors = []
        _apply_story_constraints(draft, {"beats": [{"panel": 7, "event": "legacy stride"}]}, errors)
        merge_approved_dialogues(draft, {"dialogues": [
            {"panel": 7, "character_id": "ava", "text": "Wrong reference.", "emotion": "calm"},
        ]}, {"characters": [{"id": "ava", "name": "Ava"}]}, errors)
        self.assertTrue(any("global panel index from 1 to 6" in error for error in errors), errors)
        self.assertTrue(all(not panel["dialogues"] for page in draft["pages"] for panel in page["panels"]))

    def test_exact_ids_and_numeric_global_references_are_the_only_supported_dialogue_references(self):
        draft = schema((3, 3))
        errors = []
        merge_approved_dialogues(draft, {"dialogues": [
            {"panel": "panel_2_1", "character_id": "ava", "text": "Exact.", "emotion": "calm"},
            {"panel": "6", "character_id": "ava", "text": "Global.", "emotion": "calm"},
        ]}, {"characters": [{"id": "ava", "name": "Ava"}]}, errors)
        self.assertEqual(errors, [])
        self.assertEqual(draft["pages"][1]["panels"][0]["dialogues"][0]["text"], "Exact.")
        self.assertEqual(draft["pages"][1]["panels"][2]["dialogues"][0]["text"], "Global.")

    def test_unknown_textual_and_malformed_references_are_rejected_without_trailing_digit_guessing(self):
        for reference in ("panel_9_3", "panel three", True, 0, -1, "-1"):
            with self.subTest(reference=reference):
                draft = schema((3, 3))
                errors = []
                merge_approved_dialogues(draft, {"dialogues": [{
                    "panel": reference, "character_id": "ava", "text": "Rejected.", "emotion": "calm",
                }]}, {"characters": [{"id": "ava", "name": "Ava"}]}, errors)
                self.assertTrue(errors)
                self.assertTrue(all(not panel_data["dialogues"] for page in draft["pages"] for panel_data in page["panels"]))

    def test_invalid_dialogue_mapping_stays_in_storyboarder_validation_errors(self):
        artist = schema((3,))
        state = {
            "user_prompt": "A quiet scene.", "pageCount": 1, "current_schema": {"previous": "approved"},
            "validation_errors": [], "story_plan": {"characters": [{"id": "ava", "name": "Ava"}]},
            "dialogue_plan": {"dialogues": [{
                "panel": "panel_9_3", "character_id": "ava", "text": "Wrong panel.", "emotion": "calm",
            }]},
        }
        with patch("studio_graph.agents.storyboarder_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(artist)))):
            result = run_storyboarder(state)
        self.assertTrue(any("Dialogue 1 references 'panel_9_3'" in error for error in result["validation_errors"]))
        self.assertEqual(result["current_schema"], {"previous": "approved"})
        self.assertNotIn("normalization_notes", result)

    def test_creative_intent_legacy_storyboards_silence_and_adult_policy_remain_compatible(self):
        draft = schema((3,))
        _normalize_panel_schema(draft)
        self.assertEqual(draft["metadata"]["creative_intent"]["effective"]["genre"], "mystery")
        self.assertEqual(validate_storyboard_structure(draft, 1), [])
        self.assertEqual(adult_only_errors(draft), [])
        self.assertEqual(load_comic_from_json(draft).metadata["note"], "preserve")
        legacy = {"characters": {}, "pages": [{"page_number": 1, "panels": []}]}
        self.assertEqual(load_comic_from_json(legacy).metadata, {})


if __name__ == "__main__":
    unittest.main()
