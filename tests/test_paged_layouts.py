import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from config import CONFIG
from comfyui.client import ComfyImage
from comfyui.profiles import RendererProfileStore
from page.builder import preserve_aspect_resize
from page.layouts import compatible_layout_names, compute_layout_placements, select_layout_name
from panel_engine.comfy_renderer import RESOLUTION_PRESETS, render_comic_schema, render_panels_with_comfyui, select_panel_resolution


class PagedLayoutTests(unittest.TestCase):
    def test_auto_templates_cover_one_through_six_panels(self):
        expected = {
            1: "paged_splash", 2: "paged_top_bottom_2", 3: "paged_rows_3",
            4: "paged_grid_4", 5: "paged_full_plus_four_5", 6: "paged_grid_6",
        }
        for count, layout in expected.items():
            self.assertEqual(select_layout_name(count), layout)
            self.assertIn(layout, compatible_layout_names(count))

    def test_manual_layout_and_reading_order_are_preserved(self):
        layout = select_layout_name(4, preferred="paged_split_wide_wide_4", mode="manual")
        placements = compute_layout_placements(layout, 4, CONFIG.page.width, CONFIG.page.height, CONFIG.page.margin, CONFIG.page.gutter)
        self.assertEqual([placement["idx"] for placement in placements], [0, 1, 2, 3])
        self.assertLess(placements[0]["x"], placements[1]["x"])
        self.assertLess(placements[0]["y"], placements[2]["y"])
        self.assertEqual(placements[2]["x"], placements[3]["x"])

    def test_resolution_presets_stay_near_the_workflow_pixel_budget(self):
        for width, height in [(2360, 1100), (1170, 1100), (760, 1100), (1100, 2360), (1100, 1100)]:
            generated_width, generated_height = select_panel_resolution(width, height)
            self.assertIn((generated_width, generated_height), RESOLUTION_PRESETS.values())
            self.assertLessEqual(generated_width * generated_height, 1_048_576)

    def test_cover_crops_and_contain_preserves_the_source_image(self):
        source = Image.new("RGB", (400, 100), "red")
        cover = preserve_aspect_resize(source, 100, 100, fill_mode="crop")
        contain = preserve_aspect_resize(source, 100, 100, fill_mode="contain")
        self.assertEqual(cover.size, (100, 100))
        self.assertEqual(contain.size, (100, 100))
        self.assertEqual(contain.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(contain.getpixel((50, 50)), (255, 0, 0))

    def test_multi_page_layouts_are_independent_and_share_render_geometry(self):
        schema = {
            "characters": {},
            "pages": [
                {"page_number": 1, "panels": [self._panel("p1"), self._panel("p2")]},
                {"page_number": 2, "panels": [self._panel("p3"), self._panel("p4"), self._panel("p5"), self._panel("p6")]},
            ],
        }
        captured = []
        with tempfile.TemporaryDirectory() as directory, patch("panel_engine.comfy_renderer.build_comic_page", side_effect=lambda **kwargs: captured.append(kwargs)):
            old_panel_dir = CONFIG.paths.panel_dir
            try:
                CONFIG.paths.panel_dir = str(Path(directory) / "panels")
                pages, schema_path = render_comic_schema(
                    schema, "layout_test", None, None, {}, target_panel_ids=[], layout_mode="manual",
                    page_layouts={"1": "paged_left_right_2", "2": "paged_wide_split_wide_4"}, image_fit="contain",
                )
                sizes = json.loads((schema_path.parent / "panel_sizes.json").read_text())
                layout_config = json.loads((schema_path.parent / "layout_config.json").read_text())
            finally:
                CONFIG.paths.panel_dir = old_panel_dir
        self.assertEqual([Path(path).name for path in pages], ["comic_page_layout_test_1.png", "comic_page_layout_test_2.png"])
        self.assertEqual([item["layout_name"] for item in captured], ["paged_left_right_2", "paged_wide_split_wide_4"])
        self.assertTrue(all(item["image_fit"] == "contain" for item in captured))
        self.assertEqual(layout_config["page_layouts"], {"1": "paged_left_right_2", "2": "paged_wide_split_wide_4"})
        first = compute_layout_placements("paged_left_right_2", 2, CONFIG.page.width, CONFIG.page.height, CONFIG.page.margin, CONFIG.page.gutter)
        self.assertEqual(sizes["p1"], {"width": first[0]["width"], "height": first[0]["height"]})

    def test_existing_storyboard_json_without_layout_fields_uses_auto_layout(self):
        schema = {"characters": {}, "pages": [{"page_number": 1, "panels": [self._panel("p1"), self._panel("p2")]}]}
        captured = []
        with tempfile.TemporaryDirectory() as directory, patch("panel_engine.comfy_renderer.build_comic_page", side_effect=lambda **kwargs: captured.append(kwargs)):
            old_panel_dir = CONFIG.paths.panel_dir
            try:
                CONFIG.paths.panel_dir = str(Path(directory) / "panels")
                render_comic_schema(schema, "compat_test", None, None, {}, target_panel_ids=[])
            finally:
                CONFIG.paths.panel_dir = old_panel_dir
        self.assertEqual(captured[0]["layout_name"], "paged_top_bottom_2")

    def test_mocked_comfy_submission_binds_layout_aware_klein_dimensions(self):
        root = Path(__file__).resolve().parents[1]
        profile = RendererProfileStore(root / "renderer_profiles.json", root).get("flux2-klein-base-9b")
        submitted = []
        image_buffer = BytesIO()
        Image.new("RGB", (64, 64), "blue").save(image_buffer, format="PNG")

        class FakeComfyClient:
            def submit(self, workflow):
                submitted.append(workflow)
                return "mock-job"

            def wait_for_images(self, prompt_id, output_node_id):
                self_outer.assertEqual((prompt_id, output_node_id), ("mock-job", "9"))
                return [ComfyImage(filename="mock.png")]

            def download_image(self, image):
                self_outer.assertEqual(image.filename, "mock.png")
                return image_buffer.getvalue()

        self_outer = self
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            schema_path = job_dir / "story.json"
            schema_path.write_text(json.dumps({"characters": {}, "pages": [{"page_number": 1, "panels": [self._panel("wide")]}]}))
            (job_dir / "panel_sizes.json").write_text(json.dumps({"wide": {"width": 2300, "height": 1100}}))
            output_dir = job_dir / "panels"
            render_panels_with_comfyui(schema_path, output_dir, profile, FakeComfyClient(), {})

            self.assertTrue((output_dir / "wide.png").exists())
        self.assertEqual(submitted[0]["148"]["inputs"]["text"], "test")
        self.assertEqual((submitted[0]["150"]["inputs"]["value"], submitted[0]["151"]["inputs"]["value"]), RESOLUTION_PRESETS["wide"])

    @staticmethod
    def _panel(panel_id: str) -> dict:
        return {"id": panel_id, "panel_prompt_en": "test", "active_char_ids": [], "dialogues": []}


if __name__ == "__main__":
    unittest.main()
