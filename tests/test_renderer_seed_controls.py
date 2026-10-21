import json
import asyncio
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import api
from config import CONFIG
from comfyui.client import ComfyImage
from comfyui.profiles import RendererProfileStore
from panel_engine.comfy_renderer import (
    MAX_RANDOM_NOISE_SEED,
    effective_panel_seed,
    render_comic_schema,
    render_panels_with_comfyui,
)


def schema(panel_ids: list[str]) -> dict:
    return {
        "characters": {},
        "pages": [{"page_number": 1, "panels": [
            {"id": panel_id, "panel_prompt_en": f"scene {panel_id}", "active_char_ids": [], "dialogues": []}
            for panel_id in panel_ids
        ]}],
    }


class FakeComfyClient:
    def __init__(self):
        self.submissions: list[dict] = []
        image = BytesIO()
        Image.new("RGB", (32, 32), "navy").save(image, format="PNG")
        self.image_bytes = image.getvalue()

    def submit(self, workflow):
        self.submissions.append(workflow)
        return f"job-{len(self.submissions)}"

    def wait_for_images(self, _prompt_id, _output_node_id):
        return [ComfyImage(filename="panel.png")]

    def download_image(self, _image):
        return self.image_bytes


class RendererSeedControlTests(unittest.TestCase):
    @staticmethod
    def profile():
        root = Path(__file__).resolve().parents[1]
        return RendererProfileStore(root / "renderer_profiles.example.json", root).get("flux2-klein-base-9b")

    def test_random_mode_allocates_a_distinct_seed_for_each_panel_and_records_them(self):
        client = FakeComfyClient()
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "story.json").write_text(json.dumps(schema(["panel_1_1", "panel_1_2"])), encoding="utf-8")
            (job / "panel_sizes.json").write_text(json.dumps({}), encoding="utf-8")
            with patch("panel_engine.comfy_renderer.generate_random_seed", side_effect=[41, 42]):
                render_panels_with_comfyui(job / "story.json", job / "panels", self.profile(), client, {"seed_mode": "random"})
            recorded = json.loads((job / "panel_seeds.json").read_text(encoding="utf-8"))

        self.assertEqual([item["114"]["inputs"]["noise_seed"] for item in client.submissions], [41, 42])
        self.assertEqual(recorded, {"panel_1_1": 41, "panel_1_2": 42})

    def test_fixed_mode_submits_exact_unsigned_64_bit_seed_without_changing_workflow_controls(self):
        client = FakeComfyClient()
        seed = 18_446_744_073_709_551_615
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "story.json").write_text(json.dumps(schema(["panel_1_1"])), encoding="utf-8")
            render_panels_with_comfyui(
                job / "story.json", job / "panels", self.profile(), client,
                {"seed_mode": "fixed", "seed": seed},
            )
            recorded = json.loads((job / "panel_seeds.json").read_text(encoding="utf-8"))

        submitted = client.submissions[0]
        self.assertEqual(submitted["114"]["inputs"]["noise_seed"], seed)
        self.assertEqual(submitted["155"]["inputs"]["steps"], 4)
        self.assertEqual(submitted["152"]["inputs"]["cfg"], 1)
        self.assertEqual(recorded, {"panel_1_1": seed})

    def test_invalid_fixed_seed_is_rejected_and_legacy_requests_remain_compatible(self):
        with self.assertRaisesRegex(ValueError, "Fixed seed"):
            effective_panel_seed({"seed_mode": "fixed", "seed": MAX_RANDOM_NOISE_SEED + 1})
        with self.assertRaisesRegex(ValueError, "Fixed seed"):
            effective_panel_seed({"seed_mode": "fixed", "seed": True})

        legacy_fixed = api.RenderRequest(rendererProfile="flux2-klein-base-9b", seed=9)
        legacy_random = api.RenderRequest(rendererProfile="flux2-klein-base-9b")
        self.assertEqual(api.rendering_settings(legacy_fixed)["seed_mode"], "fixed")
        self.assertEqual(api.rendering_settings(legacy_random)["seed_mode"], "random")
        ui_fixed = api.RenderRequest(
            rendererProfile="flux2-klein-base-9b", seedMode="fixed", seed="299955909870984"
        )
        self.assertEqual(api.rendering_settings(ui_fixed), {
            "steps": None, "guidance": None, "negative_prompt": "",
            "seed": 299955909870984, "seed_mode": "fixed",
        })

    def test_dialogue_only_composition_does_not_allocate_an_image_seed(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("panel_engine.comfy_renderer.build_comic_page") as build_page, \
                patch("panel_engine.comfy_renderer.render_panels_with_comfyui") as render_panels:
            original = CONFIG.paths.panel_dir
            try:
                CONFIG.paths.panel_dir = str(Path(directory) / "panels")
                _pages, story_path = render_comic_schema(
                    schema(["panel_1_1"]), "dialogue_only", None, None,
                    {"seed_mode": "random"}, target_panel_ids=[], compose_page_numbers=[1],
                )
            finally:
                CONFIG.paths.panel_dir = original

            render_panels.assert_not_called()
            self.assertFalse((story_path.parent / "panel_seeds.json").exists())
            build_page.assert_called_once()

    def test_api_rejects_unavailable_krea_profile(self):
        root = Path(__file__).resolve().parents[1]
        store = RendererProfileStore(root / "renderer_profiles.example.json", root)
        with patch.object(api, "profile_store", return_value=store), self.assertRaises(api.HTTPException) as raised:
            api.profile_or_422("krea-2")
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("not available", raised.exception.detail)

    def test_targeted_image_regeneration_merges_only_its_staged_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            job = output / "jobs" / "seed_job"
            panels = job / "panels"
            panels.mkdir(parents=True)
            (panels / "panel_1_1.png").write_bytes(b"old")
            (job / "story.json").write_text(json.dumps(schema(["panel_1_1"])), encoding="utf-8")
            (job / "panel_seeds.json").write_text(json.dumps({"other_panel": 3}), encoding="utf-8")
            (job / "panel_sizes.json").write_text(json.dumps({"panel_1_1": {"width": 100, "height": 100}}), encoding="utf-8")
            (job / "layout_config.json").write_text(json.dumps({"page_layouts": {"1": "paged_splash"}}), encoding="utf-8")
            (job / "page_layout_1.json").write_text("{}", encoding="utf-8")
            page = output / "comic_page_seed_job_1.png"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_bytes(b"old-page")

            def staged_render(updated, staging_id, *_args, **_kwargs):
                stage = output / "jobs" / staging_id
                (stage / "story.json").write_text(json.dumps(updated), encoding="utf-8")
                (stage / "panels" / "panel_1_1.png").parent.mkdir(parents=True, exist_ok=True)
                (stage / "panels" / "panel_1_1.png").write_bytes(b"new")
                (stage / "panel_seeds.json").write_text(json.dumps({"panel_1_1": 77}), encoding="utf-8")
                (stage / "panel_sizes.json").write_text(json.dumps({"panel_1_1": {"width": 100, "height": 100}}), encoding="utf-8")
                (stage / "layout_config.json").write_text(json.dumps({"page_layouts": {"1": "paged_splash"}}), encoding="utf-8")
                (stage / "page_layout_1.json").write_text("{}", encoding="utf-8")
                staged_page = output / f"comic_page_{staging_id}_1.png"
                staged_page.write_bytes(b"new-page")
                return [str(staged_page)], stage / "story.json"

            request = api.RegeneratePanelRequest(
                pageUrl="/outputs/comic_page_seed_job_1.png", panelId="panel_1_1", panelPrompt="updated scene",
                dialogues=[], mode="panel", rendererProfile="flux2-klein-base-9b", seedMode="random",
            )
            with patch.object(api, "output_dir", output), \
                    patch.object(api, "profile_or_422", return_value=object()), \
                    patch.object(api, "comfy_client", return_value=object()), \
                    patch.object(api, "render_comic_schema", side_effect=staged_render):
                asyncio.run(api.regenerate_panel(request))

            self.assertEqual(json.loads((job / "panel_seeds.json").read_text(encoding="utf-8")), {
                "other_panel": 3, "panel_1_1": 77,
            })


if __name__ == "__main__":
    unittest.main()
