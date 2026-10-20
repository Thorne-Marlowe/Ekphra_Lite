import unittest

from studio_graph.structured_output import STORY_PLAN_SCHEMA, STORYBOARD_SCHEMA, create_structured_llm, json_schema_response_format


class StructuredOutputTests(unittest.TestCase):
    def test_lm_studio_request_uses_json_schema_not_json_object(self):
        captured = {}

        def fake_chat_openai(**kwargs):
            captured.update(kwargs)
            return object()

        create_structured_llm("ministral-3-3b-reasoning-2512", "http://127.0.0.1:1234/v1", "comic_storyboard", STORYBOARD_SCHEMA, fake_chat_openai)
        response_format = captured["model_kwargs"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertNotIn("json_object", str(response_format))
        self.assertEqual(response_format["json_schema"]["name"], "comic_storyboard")
        self.assertEqual(response_format["json_schema"]["schema"]["required"], ["characters", "pages"])

    def test_schema_envelope_is_valid_for_all_structured_calls(self):
        response_format = json_schema_response_format("story_plan", {"type": "object", "properties": {}})
        self.assertEqual(response_format, {"type": "json_schema", "json_schema": {"name": "story_plan", "schema": {"type": "object", "properties": {}}}})

    def test_story_plan_allows_optional_inferred_creative_intent_not_auto(self):
        intent = STORY_PLAN_SCHEMA["properties"]["creative_intent"]
        self.assertNotIn("creative_intent", STORY_PLAN_SCHEMA["required"])
        self.assertEqual(intent["additionalProperties"], False)
        self.assertNotIn("auto", intent["properties"]["narrative_resolution"]["enum"])


if __name__ == "__main__":
    unittest.main()
