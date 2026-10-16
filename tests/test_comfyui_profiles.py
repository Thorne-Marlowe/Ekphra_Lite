import json
import tempfile
import unittest
from pathlib import Path

from comfyui.client import ComfyUIClient
from comfyui.profiles import RendererProfileStore


class RendererProfileTests(unittest.TestCase):
    def test_profile_binds_declared_inputs_without_guessing_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workflow.json").write_text(json.dumps({"7": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}}}))
            (root / "profiles.json").write_text(json.dumps({"profiles": {"test": {
                "workflow_path": "workflow.json",
                "input_map": {"positive_prompt": {"node_id": "7", "input_name": "text"}},
                "output_node_id": "9",
            }}}))
            profile = RendererProfileStore(root / "profiles.json", root).get("test")
            self.assertEqual(profile.apply_inputs(profile.load_workflow(root), {"positive_prompt": "panel text"})["7"]["inputs"]["text"], "panel text")

    def test_client_extracts_only_declared_output_images(self):
        images = ComfyUIClient._images_from_outputs({"8": {"images": [{"filename": "other.png"}]}, "9": {"images": [{"filename": "panel.png", "subfolder": "comic"}]}}, "9")
        self.assertEqual([(item.filename, item.subfolder) for item in images], [("panel.png", "comic")])


if __name__ == "__main__":
    unittest.main()
