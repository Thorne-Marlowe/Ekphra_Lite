import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException
from PIL import Image

import api
from panel_engine.job_status import mark_completed, start_render


def save_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), "navy").save(path)


def story() -> dict:
    return {"characters": {}, "pages": [{"page_number": 1, "panels": [{
        "id": "panel_1", "panel_prompt_en": "A clear adult portrait.", "active_char_ids": [], "dialogues": [],
    }]}]}


def report(job_id: str, run_id: str, *, status: str = "completed", completed_at: str = "2026-01-01T00:00:00+00:00") -> dict:
    raw = {"panel_id": "panel_1", "page_number": 1, "status": "assessed", "observations": []}
    page = {"page_number": 1, "panel_ids": ["panel_1"], "status": "assessed", "observations": []}
    if status == "partial":
        page = {**page, "status": "error", "error": "Page assessment unavailable."}
    elif status == "unavailable":
        raw = {**raw, "status": "error", "error": "Vision model unavailable."}
        page = {**page, "status": "error", "error": "Vision model unavailable."}
    return {
        "version": 1, "job_id": job_id, "run_id": run_id, "model": "local-vision",
        "started_at": "2026-01-01T00:00:00+00:00", "completed_at": completed_at, "status": status,
        "raw_panels": [raw], "pages": [page],
        "errors": [],
    }


class VisionQAApiTests(unittest.TestCase):
    def _completed_job(self, output: Path, job_id: str = "job") -> Path:
        job = output / "jobs" / job_id
        (job / "panels").mkdir(parents=True)
        (job / "story.json").write_text(json.dumps(story()), encoding="utf-8")
        save_image(job / "panels" / "panel_1.png")
        page = output / f"comic_page_{job_id}_1.png"
        save_image(page)
        start_render(job, job_id, ["panel_1"], [1])
        mark_completed(job, [page])
        return job

    @staticmethod
    def _write_report(job: Path, data: dict) -> Path:
        path = job / "vision_qa" / f"{data['run_id']}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_manual_qa_runs_engine_in_worker_and_returns_report_without_changing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            job = self._completed_job(output)
            manifest_before = (job / "job_status.json").read_bytes()

            class FakeEngine:
                def __init__(self, root):
                    self.root = Path(root)

                def run(self, job_id):
                    return VisionQAApiTests._write_report(self.root / "jobs" / job_id, report(job_id, "run_1"))

            with patch.object(api, "output_dir", output), patch.object(api, "VisionQAEngine", FakeEngine):
                result = asyncio.run(api.run_manual_vision_qa("job"))
            self.assertEqual(result["report"]["run_id"], "run_1")
            self.assertEqual((job / "job_status.json").read_bytes(), manifest_before)

    def test_get_and_list_reports_newest_first_and_ignore_malformed_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            job = self._completed_job(output)
            first = self._write_report(job, report("job", "first", completed_at="2026-01-01T00:00:00+00:00"))
            second = self._write_report(job, report("job", "second", completed_at="2026-01-02T00:00:00+00:00"))
            # Completion time, rather than contradictory filesystem mtime,
            # determines the public list order.
            os.utime(first, (20, 20)); os.utime(second, (10, 10))
            (job / "vision_qa" / "broken.json").write_text("not json", encoding="utf-8")
            (job / "vision_qa" / ".ignored.pending").write_text("", encoding="utf-8")
            with patch.object(api, "output_dir", output):
                listing = asyncio.run(api.list_vision_qa_reports("job"))
                retrieved = asyncio.run(api.get_vision_qa_report("job", "first"))
            self.assertEqual([item["runId"] for item in listing["reports"]], ["second", "first"])
            self.assertEqual(retrieved["report"]["run_id"], "first")

    def test_missing_incomplete_and_invalid_identifiers_return_useful_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            (output / "jobs" / "rendering").mkdir(parents=True)
            with patch.object(api, "output_dir", output):
                for call, detail in [
                    (lambda: api.list_vision_qa_reports("missing"), "Render job not found"),
                    (lambda: api.list_vision_qa_reports("rendering"), "verified completed"),
                    (lambda: api.list_vision_qa_reports("../escape"), "Invalid job ID"),
                ]:
                    with self.assertRaises(HTTPException) as raised:
                        asyncio.run(call())
                    self.assertIn(detail, raised.exception.detail)
                job = self._completed_job(output)
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(api.get_vision_qa_report("job", "../escape"))
                self.assertIn("Invalid Vision QA run ID", raised.exception.detail)
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(api.get_vision_qa_report("job", "missing"))
                self.assertIn("report not found", raised.exception.detail)

    def test_malformed_report_is_excluded_and_unavailable_report_remains_retrievable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            job = self._completed_job(output)
            self._write_report(job, {"version": 1, "job_id": "job", "run_id": "bad"})
            self._write_report(job, report("job", "unavailable", status="unavailable"))
            with patch.object(api, "output_dir", output):
                listing = asyncio.run(api.list_vision_qa_reports("job"))
                available = asyncio.run(api.get_vision_qa_report("job", "unavailable"))
                with self.assertRaises(HTTPException):
                    asyncio.run(api.get_vision_qa_report("job", "bad"))
            self.assertEqual([item["runId"] for item in listing["reports"]], ["unavailable"])
            self.assertEqual(available["report"]["status"], "unavailable")

    def test_partial_and_unavailable_reports_remain_accessible(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            job = self._completed_job(output)
            self._write_report(job, report("job", "partial", status="partial"))
            self._write_report(job, report("job", "unavailable", status="unavailable"))
            with patch.object(api, "output_dir", output):
                partial = asyncio.run(api.get_vision_qa_report("job", "partial"))
                unavailable = asyncio.run(api.get_vision_qa_report("job", "unavailable"))
            self.assertEqual(partial["report"]["status"], "partial")
            self.assertEqual(unavailable["report"]["status"], "unavailable")

    def test_invalid_observation_values_and_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            job = self._completed_job(output)
            invalid_cases = {
                "category": ("raw_panels", "category", "not-a-category"),
                "severity": ("raw_panels", "severity", "fatal"),
                "confidence": ("pages", "confidence", "certain"),
            }
            for run_id, (section, key, value) in invalid_cases.items():
                data = report("job", run_id)
                data[section][0]["observations"] = [{
                    "category": "identity" if section == "raw_panels" else "layout",
                    "severity": "minor", "confidence": "high", "explanation": "Visible evidence.",
                }]
                data[section][0]["observations"][0][key] = value
                self._write_report(job, data)
            malformed = report("job", "bad-time")
            malformed["started_at"] = "2026-01-01 00:00:00"
            self._write_report(job, malformed)
            backwards = report("job", "backwards", completed_at="2025-12-31T00:00:00+00:00")
            self._write_report(job, backwards)
            assessed_with_error = report("job", "assessed-error")
            assessed_with_error["raw_panels"][0]["error"] = "Should not accompany assessed status."
            self._write_report(job, assessed_with_error)
            error_without_message = report("job", "error-without-message", status="partial")
            error_without_message["pages"][0].pop("error")
            self._write_report(job, error_without_message)
            with patch.object(api, "output_dir", output):
                for run_id in (*invalid_cases, "bad-time", "backwards", "assessed-error", "error-without-message"):
                    with self.assertRaises(HTTPException) as raised:
                        asyncio.run(api.get_vision_qa_report("job", run_id))
                    self.assertEqual(raised.exception.status_code, 404)

    def test_unavailable_model_returns_informational_report_and_execution_exception_is_502(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_VISION_MODEL": ""}):
            output = Path(directory) / "outputs"
            job = self._completed_job(output)
            manifest_before = (job / "job_status.json").read_bytes()
            with patch.object(api, "output_dir", output):
                unavailable = asyncio.run(api.run_manual_vision_qa("job"))
            self.assertEqual(unavailable["report"]["status"], "unavailable")
            self.assertEqual((job / "job_status.json").read_bytes(), manifest_before)

            class BrokenEngine:
                def __init__(self, *_args):
                    pass

                def run(self, *_args):
                    raise RuntimeError("vision endpoint exploded")

            with patch.object(api, "output_dir", output), patch.object(api, "VisionQAEngine", BrokenEngine):
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(api.run_manual_vision_qa("job"))
            self.assertEqual(raised.exception.status_code, 502)
            self.assertIn("Vision QA execution failed", raised.exception.detail)

    def test_rendering_does_not_automatically_invoke_vision_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            page = output / "comic_page_render_1.png"
            save_image(page)
            request = api.RenderRequest(storyboard=story(), rendererProfile="test")
            engine = Mock()
            with patch.object(api, "output_dir", output), \
                    patch.object(api, "profile_or_422", return_value=object()), \
                    patch.object(api, "comfy_client", return_value=object()), \
                    patch.object(api, "render_comic_schema", return_value=([str(page)], output / "jobs" / "render" / "story.json")), \
                    patch.object(api, "VisionQAEngine", engine):
                asyncio.run(api.render_storyboard(request))
            engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
