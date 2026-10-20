import concurrent.futures
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from panel_engine.job_status import mark_completed, start_render
from vision_qa.engine import PAGE_CATEGORIES, RAW_CATEGORIES, VisionQAEngine, VisionQAError


def save_image(path: Path, color: str = "navy") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color).save(path)


def storyboard(page_count: int = 1) -> dict:
    pages = []
    for number in range(1, page_count + 1):
        panel_id = f"panel_{number}_1"
        pages.append({"page_number": number, "panels": [{
            "id": panel_id,
            "panel_prompt_en": f"Panel {number} composition",
            "active_char_ids": ["lena"],
            "required_action": "Lena holds a violin.",
            "required_props": ["violin"],
            "primary_subject": "Lena",
            "secondary_subject": "violin",
            "visual_hierarchy": "Lena first, violin second",
            "dialogues": [{"character_id": "lena", "text": "Listen."}],
        }]})
    return {"characters": {"lena": {"name": "Lena", "description": "adult violinist", "base_prompt_en": "adult Lena"}}, "pages": pages}


class Response:
    def __init__(self, content):
        self.content = content


class FakeVisionClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return Response(response)


class VisionQAEngineTests(unittest.TestCase):
    def _completed_job(self, root: Path, job_id: str = "job", page_count: int = 1) -> tuple[Path, dict]:
        data = storyboard(page_count)
        job = root / "jobs" / job_id
        panel_ids = [f"panel_{number}_1" for number in range(1, page_count + 1)]
        page_numbers = list(range(1, page_count + 1))
        (job / "panels").mkdir(parents=True)
        (job / "story.json").write_text(json.dumps(data), encoding="utf-8")
        pages = []
        for number, panel_id in zip(page_numbers, panel_ids):
            save_image(job / "panels" / f"{panel_id}.png")
            (job / f"page_layout_{number}.json").write_text(json.dumps({"panels": [[0, 0, 80, 60]]}), encoding="utf-8")
            page = root / f"comic_page_{job_id}_{number}.png"
            save_image(page)
            pages.append(page)
        start_render(job, job_id, panel_ids, page_numbers)
        mark_completed(job, pages)
        return job, data

    @staticmethod
    def _factory(client):
        def factory(**kwargs):
            factory.calls.append(kwargs)
            return client
        factory.calls = []
        return factory

    @staticmethod
    def _valid(category):
        return json.dumps({"observations": [{"category": category, "severity": "minor", "confidence": "high", "explanation": "Visible evidence."}]})

    def test_one_panel_successful_qa_writes_isolated_valid_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            job, story = self._completed_job(root)
            manifest_before = (job / "job_status.json").read_bytes()
            story_before = (job / "story.json").read_bytes()
            raw_before = (job / "panels" / "panel_1_1.png").read_bytes()
            page_before = (root / "comic_page_job_1.png").read_bytes()
            client = FakeVisionClient([self._valid("identity"), self._valid("dialogue_readability")])
            factory = self._factory(client)
            report_path = VisionQAEngine(root, model="local-vision", client_factory=factory).run("job")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["version"], 1)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["job_id"], "job")
            self.assertEqual(report["raw_panels"][0]["status"], "assessed")
            self.assertEqual(report["pages"][0]["status"], "assessed")
            self.assertEqual((job / "job_status.json").read_bytes(), manifest_before)
            self.assertEqual((job / "story.json").read_bytes(), story_before)
            self.assertEqual((job / "panels" / "panel_1_1.png").read_bytes(), raw_before)
            self.assertEqual((root / "comic_page_job_1.png").read_bytes(), page_before)
            self.assertEqual(len(factory.calls), 2)
            self.assertTrue(all(call["model"] == "local-vision" for call in factory.calls))
            self.assertTrue(all(call["max_retries"] == 0 for call in factory.calls))
            raw_schema = factory.calls[0]["model_kwargs"]["response_format"]["json_schema"]["schema"]
            page_schema = factory.calls[1]["model_kwargs"]["response_format"]["json_schema"]["schema"]
            raw_categories = raw_schema["properties"]["observations"]["items"]["properties"]["category"]["enum"]
            page_categories = page_schema["properties"]["observations"]["items"]["properties"]["category"]["enum"]
            self.assertEqual(raw_categories, sorted(RAW_CATEGORIES))
            self.assertEqual(page_categories, sorted(PAGE_CATEGORIES))
            raw_instruction = client.calls[0][1]["content"][0]["text"]
            page_instruction = client.calls[1][1]["content"][0]["text"]
            for category in RAW_CATEGORIES:
                self.assertIn(f"`{category}`", raw_instruction)
            for category in PAGE_CATEGORIES:
                self.assertIn(f"`{category}`", page_instruction)

    def test_multi_page_qa_assesses_every_raw_panel_and_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            self._completed_job(root, page_count=2)
            client = FakeVisionClient([
                self._valid("identity"), self._valid("identity"),
                self._valid("layout"), self._valid("speaker_attribution"),
            ])
            report = json.loads(VisionQAEngine(root, model="local-vision", client_factory=self._factory(client)).run("job").read_text())
            self.assertEqual([item["panel_id"] for item in report["raw_panels"]], ["panel_1_1", "panel_2_1"])
            self.assertEqual([item["page_number"] for item in report["pages"]], [1, 2])
            self.assertEqual(report["pages"][1]["observations"][0]["category"], "speaker_attribution")

    def test_missing_failed_or_corrupt_manifest_is_rejected_without_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            job, _ = self._completed_job(root)
            (job / "job_status.json").unlink()
            engine = VisionQAEngine(root, model="local-vision", client_factory=self._factory(FakeVisionClient([])))
            with self.assertRaisesRegex(VisionQAError, "not a verified completed"):
                engine.run("job")
            self._completed_job(root, "failed")
            (root / "jobs" / "failed" / "job_status.json").write_text("{bad json", encoding="utf-8")
            with self.assertRaisesRegex(VisionQAError, "not a verified completed"):
                engine.run("failed")
            self._completed_job(root, "interrupted")
            status_path = root / "jobs" / "interrupted" / "job_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["status"] = "failed"
            status_path.write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(VisionQAError, "not a verified completed"):
                engine.run("interrupted")

    def test_duplicate_storyboard_panel_id_is_rejected_before_qa_directory_or_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            job, data = self._completed_job(root)
            duplicate = dict(data["pages"][0]["panels"][0])
            data["pages"][0]["panels"].append(duplicate)
            (job / "story.json").write_text(json.dumps(data), encoding="utf-8")
            factory = self._factory(FakeVisionClient([]))
            with self.assertRaisesRegex(VisionQAError, "duplicate panel ID"):
                VisionQAEngine(root, model="local-vision", client_factory=factory).run("job")
            self.assertFalse((job / "vision_qa").exists())
            self.assertFalse(hasattr(factory, "kwargs"))

    def test_mismatched_page_association_is_rejected_before_qa_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            job, _ = self._completed_job(root)
            manifest_path = job / "job_status.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["expected_page_numbers"] = [2]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(VisionQAError, "page numbers do not match"):
                VisionQAEngine(root, model="local-vision", client_factory=self._factory(FakeVisionClient([]))).run("job")
            self.assertFalse((job / "vision_qa").exists())

    def test_unavailable_model_is_recorded_without_touching_completed_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            job, _ = self._completed_job(root)
            before = (job / "job_status.json").read_bytes()
            report = json.loads(VisionQAEngine(root, model="local-vision", client_factory=lambda **_: (_ for _ in ()).throw(ConnectionError("offline"))).run("job").read_text())
            self.assertEqual(report["status"], "unavailable")
            self.assertIn("Vision model unavailable", report["errors"][0]["error"])
            self.assertEqual((job / "job_status.json").read_bytes(), before)

    def test_missing_vision_model_writes_unavailable_report_after_artifact_validation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_VISION_MODEL": ""}):
            root = Path(directory) / "outputs"
            job, _ = self._completed_job(root)
            factory = self._factory(FakeVisionClient([]))
            report = json.loads(VisionQAEngine(root, client_factory=factory).run("job").read_text())
            self.assertEqual(report["status"], "unavailable")
            self.assertIn("OPENAI_VISION_MODEL", report["errors"][0]["error"])
            self.assertTrue((job / "vision_qa").is_dir())
            self.assertFalse(hasattr(factory, "kwargs"))

    def test_malformed_model_response_is_an_assessment_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            self._completed_job(root)
            client = FakeVisionClient(["not json", self._valid("layout")])
            report = json.loads(VisionQAEngine(root, model="local-vision", client_factory=self._factory(client)).run("job").read_text())
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["raw_panels"][0]["status"], "error")
            self.assertIn("invalid JSON", report["raw_panels"][0]["error"])
            self.assertEqual(report["pages"][0]["status"], "assessed")

    def test_invalid_assessment_schema_is_an_assessment_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            self._completed_job(root)
            invalid_schema = json.dumps({"observations": [{"category": "unrecognized", "severity": "minor", "confidence": "high", "explanation": "No."}]})
            client = FakeVisionClient([invalid_schema, self._valid("layout")])
            report = json.loads(VisionQAEngine(root, model="local-vision", client_factory=self._factory(client)).run("job").read_text())
            self.assertEqual(report["status"], "partial")
            self.assertIn("unsupported category", report["raw_panels"][0]["error"])

    def test_supported_categories_are_accepted_and_cross_context_categories_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            self._completed_job(root)
            client = FakeVisionClient([
                self._valid("obvious_visual_defect"),
                self._valid("identity"),
            ])
            report = json.loads(VisionQAEngine(root, model="local-vision", client_factory=self._factory(client)).run("job").read_text())
            self.assertEqual(report["raw_panels"][0]["status"], "assessed")
            self.assertEqual(report["raw_panels"][0]["observations"][0]["category"], "obvious_visual_defect")
            self.assertEqual(report["pages"][0]["status"], "error")
            self.assertIn("unsupported category 'identity'", report["pages"][0]["error"])

    def test_partial_image_failure_preserves_other_assessments_and_existing_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            job, _ = self._completed_job(root, page_count=2)
            report_dir = job / "vision_qa"
            report_dir.mkdir()
            prior = report_dir / "prior.json"
            prior.write_text('{"preserve": true}', encoding="utf-8")
            client = FakeVisionClient([
                self._valid("identity"), RuntimeError("panel unavailable"),
                self._valid("layout"), self._valid("speaker_attribution"),
            ])
            report_path = VisionQAEngine(root, model="local-vision", client_factory=self._factory(client)).run("job")
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["raw_panels"][0]["status"], "assessed")
            self.assertEqual(report["raw_panels"][1]["status"], "error")
            self.assertEqual(prior.read_text(encoding="utf-8"), '{"preserve": true}')
            self.assertNotEqual(report_path.name, prior.name)

    def test_not_assessable_speaker_attribution_is_informational(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            self._completed_job(root)
            response = json.dumps({"observations": [{"category": "speaker_attribution", "severity": "major", "confidence": "not_assessable", "explanation": "Speaker cannot be visually established."}]})
            client = FakeVisionClient([self._valid("identity"), response])
            report = json.loads(VisionQAEngine(root, model="local-vision", client_factory=self._factory(client)).run("job").read_text())
            observation = report["pages"][0]["observations"][0]
            self.assertEqual(observation["severity"], "info")
            self.assertEqual(observation["confidence"], "not_assessable")

    def test_concurrent_runs_reserve_distinct_report_names_without_pending_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            job, _ = self._completed_job(root)

            def run_once():
                client = FakeVisionClient([self._valid("identity"), self._valid("layout")])
                return VisionQAEngine(root, model="local-vision", client_factory=self._factory(client)).run("job")

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                reports = list(executor.map(lambda _: run_once(), range(2)))
            self.assertEqual(len({path.name for path in reports}), 2)
            self.assertTrue(all(path.is_file() for path in reports))
            self.assertEqual(list((job / "vision_qa").glob("*.pending")), [])
            self.assertEqual(list((job / "vision_qa").glob("*.tmp")), [])

    def test_existing_report_filename_collision_is_skipped_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "vision_qa"
            report_dir.mkdir()
            existing = report_dir / "collision.json"
            existing.write_text('{"preserve": true}', encoding="utf-8")

            class UUID:
                def __init__(self, value):
                    self.hex = value

            with patch("vision_qa.engine.uuid.uuid4", side_effect=[UUID("collision"), UUID("fresh")]):
                run_id, report_path, reservation = VisionQAEngine._reserve_report_path(report_dir)
            self.assertEqual(run_id, "fresh")
            self.assertEqual(report_path.name, "fresh.json")
            self.assertTrue(reservation.exists())
            self.assertEqual(existing.read_text(encoding="utf-8"), '{"preserve": true}')
            reservation.unlink()

    def test_unexpected_raw_processing_exception_cleans_reservation_and_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            job, _ = self._completed_job(root)
            engine = VisionQAEngine(root, model="local-vision", client_factory=self._factory(FakeVisionClient([])))
            with patch.object(engine, "_assess", side_effect=KeyboardInterrupt("unexpected raw interruption")):
                with self.assertRaisesRegex(KeyboardInterrupt, "unexpected raw interruption"):
                    engine.run("job")
            report_dir = job / "vision_qa"
            self.assertEqual(list(report_dir.glob("*.json")), [])
            self.assertEqual(list(report_dir.glob("*.pending")), [])
            self.assertEqual(list(report_dir.glob("*.tmp")), [])

    def test_destination_appearing_after_reservation_is_preserved_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            job, _ = self._completed_job(root)
            engine = VisionQAEngine(root, model="local-vision", client_factory=self._factory(FakeVisionClient([
                self._valid("identity"), self._valid("layout"),
            ])))
            publish = VisionQAEngine._publish_report

            def destination_appears(temporary, destination):
                destination.write_text('{"preexisting": true}', encoding="utf-8")
                publish(temporary, destination)

            with patch.object(VisionQAEngine, "_publish_report", side_effect=destination_appears):
                with self.assertRaisesRegex(VisionQAError, "appeared before publication"):
                    engine.run("job")
            report_dir = job / "vision_qa"
            reports = list(report_dir.glob("*.json"))
            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0].read_text(encoding="utf-8"), '{"preexisting": true}')
            self.assertEqual(list(report_dir.glob("*.pending")), [])
            self.assertEqual(list(report_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
