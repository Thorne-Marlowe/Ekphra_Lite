import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

import api
from panel_engine.comfy_renderer import render_comic_schema
from panel_engine.job_status import completed_job_pages, load_manifest, start_render


def schema() -> dict:
    return {"characters": {}, "pages": [
        {"page_number": 1, "panels": [{"id": "p1", "panel_prompt_en": "one", "active_char_ids": [], "dialogues": []}]},
        {"page_number": 2, "panels": [{"id": "p2", "panel_prompt_en": "two", "active_char_ids": [], "dialogues": []}]},
    ]}


def save_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), "navy").save(path)


class JobCompletionIntegrityTests(unittest.TestCase):
    def test_successful_multi_page_render_writes_completed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            old_panel_dir = api.CONFIG.paths.panel_dir
            try:
                api.CONFIG.paths.panel_dir = str(output / "panels")
                with patch("panel_engine.comfy_renderer.render_panels_with_comfyui", side_effect=self._render_panels), \
                        patch("panel_engine.comfy_renderer.build_comic_page", side_effect=self._compose_page):
                    pages, _ = render_comic_schema(schema(), "complete", object(), object(), {})
            finally:
                api.CONFIG.paths.panel_dir = old_panel_dir
            manifest = load_manifest(output / "jobs" / "complete")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["expected_panel_ids"], ["p1", "p2"])
            self.assertEqual([Path(path).name for path in pages], ["comic_page_complete_1.png", "comic_page_complete_2.png"])

    def test_later_page_failure_marks_job_failed_and_history_excludes_partial_page(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            old_panel_dir = api.CONFIG.paths.panel_dir
            calls = 0

            def fail_second_page(**kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("page 2 composition failed")
                self._compose_page(**kwargs)

            try:
                api.CONFIG.paths.panel_dir = str(output / "panels")
                with patch("panel_engine.comfy_renderer.render_panels_with_comfyui", side_effect=self._render_panels), \
                        patch("panel_engine.comfy_renderer.build_comic_page", side_effect=fail_second_page):
                    with self.assertRaisesRegex(RuntimeError, "page 2 composition failed"):
                        render_comic_schema(schema(), "failed", object(), object(), {})
            finally:
                api.CONFIG.paths.panel_dir = old_panel_dir
            self.assertEqual(load_manifest(output / "jobs" / "failed")["status"], "failed")
            self.assertTrue((output / "comic_page_failed_1.png").exists())
            self.assertEqual(completed_job_pages(output), [])

    def test_rendering_manifest_is_not_completed_after_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            job = output / "jobs" / "interrupted"
            (job / "panels").mkdir(parents=True)
            save_image(job / "panels" / "p1.png")
            save_image(output / "comic_page_interrupted_1.png")
            start_render(job, "interrupted", ["p1"], [1])
            self.assertEqual(completed_job_pages(output), [])

    def test_history_includes_completed_manifest_and_verified_legacy_not_failed_or_ambiguous_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            self._complete_legacy_job(output, "legacy")
            self._complete_legacy_job(output, "manifested")
            start_render(output / "jobs" / "manifested", "manifested", ["p1", "p2"], [1, 2])
            from panel_engine.job_status import mark_completed
            mark_completed(output / "jobs" / "manifested", [output / "comic_page_manifested_1.png", output / "comic_page_manifested_2.png"])
            partial = output / "jobs" / "partial"
            (partial / "panels").mkdir(parents=True)
            save_image(output / "comic_page_partial_1.png")
            with patch.object(api, "output_dir", output):
                history = asyncio.run(api.get_history())["history"]
            self.assertIn("/outputs/comic_page_legacy_1.png", history)
            self.assertIn("/outputs/comic_page_manifested_1.png", history)
            self.assertNotIn("/outputs/comic_page_partial_1.png", history)

    def test_corrupt_manifest_does_not_fall_back_to_legacy_history(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            self._complete_legacy_job(output, "corrupt")
            (output / "jobs" / "corrupt" / "job_status.json").write_text("not json", encoding="utf-8")
            self.assertEqual(completed_job_pages(output), [])

    def test_layout_preparation_failure_marks_identified_job_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            old_panel_dir = api.CONFIG.paths.panel_dir
            try:
                api.CONFIG.paths.panel_dir = str(output / "panels")
                with self.assertRaisesRegex(ValueError, "needs a manually selected layout"):
                    render_comic_schema(schema(), "layout_failure", object(), object(), {}, layout_mode="manual")
            finally:
                api.CONFIG.paths.panel_dir = old_panel_dir
            manifest = load_manifest(output / "jobs" / "layout_failure")
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("manually selected layout", manifest["error"])

    def test_completed_job_id_collision_does_not_modify_existing_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            self._complete_legacy_job(output, "collision")
            job = output / "jobs" / "collision"
            start_render(job, "collision", ["p1", "p2"], [1, 2])
            from panel_engine.job_status import mark_completed
            mark_completed(job, [output / "comic_page_collision_1.png", output / "comic_page_collision_2.png"])
            before = {path.name: path.read_bytes() for path in job.rglob("*") if path.is_file()}
            old_panel_dir = api.CONFIG.paths.panel_dir
            try:
                api.CONFIG.paths.panel_dir = str(output / "panels")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    render_comic_schema(schema(), "collision", object(), object(), {})
            finally:
                api.CONFIG.paths.panel_dir = old_panel_dir
            self.assertEqual(before, {path.name: path.read_bytes() for path in job.rglob("*") if path.is_file()})

    def test_failed_regeneration_leaves_completed_manifest_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            self._complete_legacy_job(output, "job_1")
            job = output / "jobs" / "job_1"
            start_render(job, "job_1", ["p1", "p2"], [1, 2])
            from panel_engine.job_status import mark_completed
            mark_completed(job, [output / "comic_page_job_1_1.png", output / "comic_page_job_1_2.png"])
            before = (job / "job_status.json").read_bytes()
            request = api.RegeneratePanelRequest(pageUrl="/outputs/comic_page_job_1_1.png", panelId="p1", panelPrompt="replacement", dialogues=[], mode="panel", rendererProfile="test")
            with patch.object(api, "output_dir", output), \
                    patch.object(api, "profile_or_422", return_value=object()), \
                    patch.object(api, "comfy_client", return_value=object()), \
                    patch.object(api, "render_comic_schema", side_effect=RuntimeError("staging failed")):
                with self.assertRaises(HTTPException):
                    asyncio.run(api.regenerate_panel(request))
            self.assertEqual((job / "job_status.json").read_bytes(), before)
            self.assertEqual(load_manifest(job)["status"], "completed")

    @staticmethod
    def _render_panels(schema_path, panel_dir, *_args):
        data = json.loads(Path(schema_path).read_text())
        for page in data["pages"]:
            for panel in page["panels"]:
                save_image(Path(panel_dir) / f"{panel['id']}.png")

    @staticmethod
    def _compose_page(**kwargs):
        save_image(Path(kwargs["output_path"]))

    @staticmethod
    def _complete_legacy_job(output: Path, job_id: str):
        job = output / "jobs" / job_id
        (job / "panels").mkdir(parents=True)
        (job / "story.json").write_text(json.dumps(schema()), encoding="utf-8")
        for page_number in (1, 2):
            (job / f"page_layout_{page_number}.json").write_text("{}", encoding="utf-8")
            save_image(output / f"comic_page_{job_id}_{page_number}.png")
        save_image(job / "panels" / "p1.png")
        save_image(job / "panels" / "p2.png")


if __name__ == "__main__":
    unittest.main()
