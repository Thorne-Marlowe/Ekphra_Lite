import asyncio
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

import api
from config import CONFIG
from panel_engine.comfy_renderer import build_panel_prompt, render_comic_schema
from style.presets import (
    ArtDirectionError,
    MAX_RESOLVED_ART_DIRECTION_LENGTH,
    STYLE_PRESETS,
    resolve_art_direction,
)
from studio_graph.agents import run_storyboarder
from utils.parser import load_comic_from_json


def storyboard(metadata=None):
    return {
        "title": "Style test", "metadata": metadata or {},
        "characters": {"ava": {"name": "Ava", "base_prompt_en": "adult investigator in a wool coat"}},
        "pages": [{"page_number": 1, "panels": [{
            "id": "panel_1_1", "panel_prompt_en": "Ava studies an archive ledger.",
            "active_char_ids": ["ava"], "dialogues": [],
        }]}],
    }


class ArtDirectionTests(unittest.TestCase):
    def test_all_stable_presets_resolve_and_none_adds_nothing(self):
        self.assertIsNone(resolve_art_direction(None))
        self.assertIsNone(resolve_art_direction({}))
        for preset_id, preset in STYLE_PRESETS.items():
            self.assertEqual(resolve_art_direction({"preset_id": preset_id}), {
                "preset_id": preset_id, "resolved_prompt": preset.prompt_fragment,
            })

    def test_custom_combined_and_invalid_direction_inputs(self):
        self.assertEqual(resolve_art_direction({"custom_prompt": "dry brush texture"}), {
            "preset_id": None, "resolved_prompt": "dry brush texture",
        })
        resolved = resolve_art_direction({
            "preset_id": "cinematic_graphic_novel", "custom_prompt": "restrained watercolor texture",
        })
        self.assertEqual(resolved["resolved_prompt"],
                         f"{STYLE_PRESETS['cinematic_graphic_novel'].prompt_fragment}, restrained watercolor texture")
        with self.assertRaisesRegex(ArtDirectionError, "Unknown art-direction"):
            resolve_art_direction({"preset_id": "not-real"})
        with self.assertRaisesRegex(ArtDirectionError, "nonempty"):
            resolve_art_direction({"custom_prompt": "  "})
        with self.assertRaisesRegex(ArtDirectionError, "object"):
            resolve_art_direction("manga")
        with self.assertRaisesRegex(ArtDirectionError, "at most"):
            resolve_art_direction({"custom_prompt": "x" * 1_001})

    def test_approved_direction_replaces_llm_value_and_preserves_creative_metadata(self):
        generated = storyboard({"art_direction": {"preset_id": "wrong", "resolved_prompt": "LLM choice"}, "note": "keep"})
        state = {
            "user_prompt": "An archive scene.", "pageCount": 1, "current_schema": {},
            "story_plan": {"characters": [{"id": "ava", "name": "Ava"}], "beats": []},
            "dialogue_plan": {"dialogues": []}, "validation_errors": [],
            "requested_creative_intent": {"genre": "mystery"}, "inferred_creative_intent": {},
            "effective_creative_intent": {"genre": "mystery"}, "creative_intent_sources": {"genre": "user_selection"},
            "creative_intent_warnings": [],
            "approved_art_direction": {"preset_id": "flat_color_webcomic", "resolved_prompt": "approved flat color"},
        }
        with patch("studio_graph.agents.storyboarder_llm", RunnableLambda(lambda _: SimpleNamespace(content=json.dumps(generated)))):
            result = run_storyboarder(state)
        metadata = result["current_schema"]["metadata"]
        self.assertEqual(metadata["art_direction"], state["approved_art_direction"])
        self.assertEqual(metadata["creative_intent"]["effective"], {"genre": "mystery"})
        self.assertEqual(metadata["note"], "keep")

    def test_human_persisted_edit_is_used_once_and_legacy_prompt_is_unchanged(self):
        legacy_comic = load_comic_from_json(storyboard())
        legacy_prompt = build_panel_prompt(legacy_comic, legacy_comic.panels[0])
        edited = storyboard({"art_direction": {"preset_id": "cinematic_graphic_novel", "resolved_prompt": "human edited ink wash"}})
        comic = load_comic_from_json(edited)
        prompt = build_panel_prompt(comic, comic.panels[0])
        self.assertEqual(prompt.count("human edited ink wash"), 1)
        self.assertNotIn("human edited ink wash", legacy_prompt)
        self.assertEqual(build_panel_prompt(legacy_comic, legacy_comic.panels[0]), legacy_prompt)
        malformed = load_comic_from_json(storyboard({"art_direction": {"resolved_prompt": "missing provenance"}}))
        with self.assertRaisesRegex(ArtDirectionError, "preset_id"):
            build_panel_prompt(malformed, malformed.panels[0])

    def test_persisted_resolved_prompt_has_a_non_destructive_length_bound(self):
        prompt = "x" * MAX_RESOLVED_ART_DIRECTION_LENGTH
        comic = load_comic_from_json(storyboard({"art_direction": {"preset_id": None, "resolved_prompt": prompt}}))
        self.assertIn(prompt, build_panel_prompt(comic, comic.panels[0]))
        too_long = load_comic_from_json(storyboard({"art_direction": {
            "preset_id": None, "resolved_prompt": "x" * (MAX_RESOLVED_ART_DIRECTION_LENGTH + 1),
        }}))
        with self.assertRaisesRegex(ArtDirectionError, "at most"):
            build_panel_prompt(too_long, too_long.panels[0])

    def test_render_api_rejects_malformed_art_direction_before_renderer_execution(self):
        invalid = storyboard({"art_direction": {"preset_id": None, "resolved_prompt": "x" * (MAX_RESOLVED_ART_DIRECTION_LENGTH + 1)}})
        request = api.RenderRequest(storyboard=invalid, rendererProfile="test")
        with patch.object(api, "profile_or_422") as profile, patch.object(api, "render_comic_schema") as render:
            with self.assertRaises(api.HTTPException) as raised:
                asyncio.run(api.render_storyboard(request))
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("resolved_prompt", raised.exception.detail)
        profile.assert_not_called()
        render.assert_not_called()

    def test_dialogue_only_composition_persists_art_direction_and_adult_policy_blocks_before_submit(self):
        schema = storyboard({"creative_intent": {"effective": {"genre": "mystery"}}, "art_direction": {
            "preset_id": "flat_color_webcomic", "resolved_prompt": "persisted flat treatment",
        }})
        with tempfile.TemporaryDirectory() as directory, patch("panel_engine.comfy_renderer.build_comic_page"):
            original = CONFIG.paths.panel_dir
            try:
                CONFIG.paths.panel_dir = str(Path(directory) / "panels")
                _, schema_path = render_comic_schema(schema, "style_job", None, None, {}, target_panel_ids=[], compose_page_numbers=[1])
            finally:
                CONFIG.paths.panel_dir = original
            stored = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["metadata"]["art_direction"], schema["metadata"]["art_direction"])
        self.assertEqual(stored["metadata"]["creative_intent"], schema["metadata"]["creative_intent"])

        unsafe = storyboard({"art_direction": {"preset_id": None, "resolved_prompt": "underage figure illustration"}})
        with tempfile.TemporaryDirectory() as directory:
            original = CONFIG.paths.panel_dir
            try:
                CONFIG.paths.panel_dir = str(Path(directory) / "panels")
                with self.assertRaisesRegex(ValueError, "Adult-only policy violation"):
                    render_comic_schema(unsafe, "unsafe_style", None, None, {}, target_panel_ids=[])
            finally:
                CONFIG.paths.panel_dir = original

    def test_targeted_regeneration_uses_persisted_art_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            job_dir = output / "jobs" / "job_style"
            panels_dir = job_dir / "panels"
            panels_dir.mkdir(parents=True)
            schema = storyboard({"art_direction": {"preset_id": None, "resolved_prompt": "human-approved wash"}})
            (job_dir / "story.json").write_text(json.dumps(schema), encoding="utf-8")
            (job_dir / "panel_sizes.json").write_text(json.dumps({"panel_1_1": {"width": 100, "height": 100}}))
            (job_dir / "layout_config.json").write_text(json.dumps({"page_layouts": {"1": "paged_splash"}, "image_fit": "cover"}))
            (job_dir / "page_layout_1.json").write_text("{}")
            (output / "comic_page_job_style_1.png").parent.mkdir(parents=True, exist_ok=True)
            (output / "comic_page_job_style_1.png").write_bytes(b"old-page")
            captured = {}

            def staged_render(updated, stage_id, *_args, **_kwargs):
                captured["direction"] = updated["metadata"]["art_direction"]
                stage = output / "jobs" / stage_id
                (stage / "story.json").write_text(json.dumps(updated), encoding="utf-8")
                (stage / "panel_sizes.json").write_text(json.dumps({"panel_1_1": {"width": 100, "height": 100}}))
                (stage / "layout_config.json").write_text(json.dumps({
                    "page_layouts": {"1": "paged_splash"}, "page_image_fits": {"1": "cover"},
                }))
                (stage / "page_layout_1.json").write_text(json.dumps({}))
                page = output / f"comic_page_{stage_id}_1.png"
                page.write_bytes(b"new-page")
                return [str(page)], stage / "story.json"

            request = api.RegeneratePanelRequest(
                pageUrl="/outputs/comic_page_job_style_1.png", panelId="panel_1_1", panelPrompt="edited prompt",
                dialogues=[], mode="dialogue", rendererProfile="test",
            )
            with patch.object(api, "output_dir", output), patch.object(api, "profile_or_422", return_value=object()), \
                    patch.object(api, "comfy_client", return_value=object()), patch.object(api, "render_comic_schema", side_effect=staged_render):
                asyncio.run(api.regenerate_panel(request))
            self.assertEqual(captured["direction"], schema["metadata"]["art_direction"])
            stored = json.loads((job_dir / "story.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["metadata"]["art_direction"], schema["metadata"]["art_direction"])


if __name__ == "__main__":
    unittest.main()
