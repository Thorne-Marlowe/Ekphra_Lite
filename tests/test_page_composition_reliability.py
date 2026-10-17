import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

import api
from page.builder import build_comic_page
from panel_engine.comfy_renderer import render_comic_schema


def panel(panel_id: str) -> dict:
    return {"id": panel_id, "panel_prompt_en": "original", "active_char_ids": [], "dialogues": []}


class ExpectedPanelImageTests(unittest.TestCase):
    def test_expected_panels_must_all_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            panels_dir = Path(directory) / "panels"
            panels_dir.mkdir()
            Image.new("RGB", (20, 20), "red").save(panels_dir / "p1.png")

            with self.assertRaisesRegex(ValueError, r"missing expected panel image\(s\): p2"):
                build_comic_page(
                    panels_dir=str(panels_dir), output_path=str(Path(directory) / "page.png"),
                    expected_panel_ids=["p1", "p2"], inject_bubbles=False,
                )

    def test_expected_panels_must_be_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            panels_dir = Path(directory) / "panels"
            panels_dir.mkdir()
            (panels_dir / "p1.png").write_bytes(b"not a PNG")

            with self.assertRaisesRegex(ValueError, r"unreadable expected panel image\(s\): p1"):
                build_comic_page(
                    panels_dir=str(panels_dir), output_path=str(Path(directory) / "page.png"),
                    expected_panel_ids=["p1"], inject_bubbles=False,
                )


class RegenerationIsolationTests(unittest.TestCase):
    def test_renderer_can_recompose_only_the_requested_page(self):
        schema = {
            "characters": {},
            "pages": [
                {"page_number": 1, "panels": [panel("p1")]},
                {"page_number": 2, "panels": [panel("p2")]},
            ],
        }
        captured = []
        with tempfile.TemporaryDirectory() as directory, \
                patch("panel_engine.comfy_renderer.build_comic_page", side_effect=lambda **kwargs: captured.append(kwargs)):
            old_panel_dir = api.CONFIG.paths.panel_dir
            try:
                api.CONFIG.paths.panel_dir = str(Path(directory) / "panels")
                pages, _ = render_comic_schema(
                    schema, "targeted", None, None, {}, target_panel_ids=[], compose_page_numbers=[2],
                )
            finally:
                api.CONFIG.paths.panel_dir = old_panel_dir
        self.assertEqual([Path(path).name for path in pages], ["comic_page_targeted_2.png"])
        self.assertEqual([item["expected_panel_ids"] for item in captured], [["p2"]])

    def test_failed_regeneration_does_not_change_approved_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            job_dir = self._approved_job(output)
            before = {path.name: path.read_bytes() for path in job_dir.iterdir() if path.is_file()}
            before_page_two = (output / "comic_page_job_1_2.png").read_bytes()
            request = self._request()

            with patch.object(api, "output_dir", output), \
                    patch.object(api, "profile_or_422", return_value=object()), \
                    patch.object(api, "comfy_client", return_value=object()), \
                    patch.object(api, "render_comic_schema", side_effect=RuntimeError("mock render failure")):
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(api.regenerate_panel(request))

            self.assertEqual(raised.exception.status_code, 502)
            self.assertEqual(before, {path.name: path.read_bytes() for path in job_dir.iterdir() if path.is_file()})
            self.assertEqual(before_page_two, (output / "comic_page_job_1_2.png").read_bytes())
            self.assertFalse(any("_staging_" in path.name for path in (output / "jobs").iterdir()))

    def test_successful_regeneration_keeps_unrelated_page_files_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            job_dir = self._approved_job(output)
            old_page_two = (output / "comic_page_job_1_2.png").read_bytes()
            old_page_two_metadata = (job_dir / "page_layout_2.json").read_bytes()
            old_panel_two = (job_dir / "panels" / "p2.png").read_bytes()

            def staged_render(schema, stage_job_id, *_args, **_kwargs):
                stage = output / "jobs" / stage_job_id
                (stage / "story.json").write_text(json.dumps(schema), encoding="utf-8")
                (stage / "panel_sizes.json").write_text(json.dumps({"p1": {"width": 100, "height": 200}, "p2": {"width": 200, "height": 100}}))
                (stage / "layout_config.json").write_text(json.dumps({
                    "publication_format": "paged_comic", "layout_mode": "manual", "image_fit": "cover",
                    "page_layouts": {"1": "paged_splash", "2": "paged_splash"},
                    "page_image_fits": {"1": "cover", "2": "cover"},
                }))
                (stage / "page_layout_1.json").write_text(json.dumps({"page": 1, "changed": True}))
                (stage / "panels" / "p1.png").write_bytes(b"new-panel-one")
                staged_page = output / f"comic_page_{stage_job_id}_1.png"
                staged_page.write_bytes(b"new-page-one")
                return [str(staged_page)], stage / "story.json"

            with patch.object(api, "output_dir", output), \
                    patch.object(api, "profile_or_422", return_value=object()), \
                    patch.object(api, "comfy_client", return_value=object()), \
                    patch.object(api, "render_comic_schema", side_effect=staged_render):
                result = asyncio.run(api.regenerate_panel(self._request()))

            self.assertEqual(result["url"], "/outputs/comic_page_job_1_1.png")
            self.assertEqual((output / "comic_page_job_1_1.png").read_bytes(), b"new-page-one")
            self.assertEqual((job_dir / "panels" / "p1.png").read_bytes(), b"new-panel-one")
            self.assertEqual((output / "comic_page_job_1_2.png").read_bytes(), old_page_two)
            self.assertEqual((job_dir / "page_layout_2.json").read_bytes(), old_page_two_metadata)
            self.assertEqual((job_dir / "panels" / "p2.png").read_bytes(), old_panel_two)
            stored = json.loads((job_dir / "story.json").read_text())
            self.assertEqual(stored["pages"][0]["panels"][0]["panel_prompt_en"], "replacement")
            self.assertEqual(stored["pages"][1]["panels"][0]["panel_prompt_en"], "original")
            self.assertFalse(any("_staging_" in path.name for path in (output / "jobs").iterdir()))

    @staticmethod
    def _request():
        return api.RegeneratePanelRequest(
            pageUrl="/outputs/comic_page_job_1_1.png", panelId="p1", panelPrompt="replacement",
            dialogues=[], mode="panel", rendererProfile="test", layoutMode="auto",
        )

    @staticmethod
    def _approved_job(output: Path) -> Path:
        job_dir = output / "jobs" / "job_1"
        panels = job_dir / "panels"
        panels.mkdir(parents=True)
        schema = {"characters": {}, "pages": [
            {"page_number": 1, "panels": [panel("p1")]},
            {"page_number": 2, "panels": [panel("p2")]},
        ]}
        (job_dir / "story.json").write_text(json.dumps(schema))
        (job_dir / "panel_sizes.json").write_text(json.dumps({"p1": {"width": 1, "height": 1}, "p2": {"width": 2, "height": 2}}))
        (job_dir / "layout_config.json").write_text(json.dumps({
            "page_layouts": {"1": "paged_splash", "2": "paged_splash"}, "image_fit": "cover",
        }))
        (job_dir / "page_layout_1.json").write_text(json.dumps({"page": 1}))
        (job_dir / "page_layout_2.json").write_text(json.dumps({"page": 2, "preserve": True}))
        (panels / "p1.png").write_bytes(b"old-panel-one")
        (panels / "p2.png").write_bytes(b"old-panel-two")
        (output / "comic_page_job_1_1.png").write_bytes(b"old-page-one")
        (output / "comic_page_job_1_2.png").write_bytes(b"old-page-two")
        return job_dir


if __name__ == "__main__":
    unittest.main()
