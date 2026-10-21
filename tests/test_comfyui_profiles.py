import json
import tempfile
import unittest
from pathlib import Path

from comfyui.client import ComfyUIClient
from comfyui.profiles import ProfileError, RendererProfileStore


class RendererProfileTests(unittest.TestCase):
    def test_klein_profile_binds_the_checked_in_api_workflow_prompt_and_output(self):
        root = Path(__file__).resolve().parents[1]
        profile = RendererProfileStore(root / "renderer_profiles.example.json", root).get("flux2-klein-base-9b")

        workflow = profile.load_workflow(root)
        original_prompt = workflow["148"]["inputs"]["text"]
        bound = profile.apply_inputs(workflow, {
            "positive_prompt": "a comic panel test prompt", "width": 832, "height": 1216,
            "seed": 299955909870984,
        })

        self.assertEqual(profile.output_node_id, "9")
        self.assertEqual(bound["148"]["inputs"]["text"], "a comic panel test prompt")
        self.assertEqual(bound["150"]["inputs"]["value"], 832)
        self.assertEqual(bound["151"]["inputs"]["value"], 1216)
        self.assertEqual(bound["114"]["inputs"]["noise_seed"], 299955909870984)
        self.assertEqual(workflow["148"]["inputs"]["text"], original_prompt)
        self.assertEqual(bound["9"]["class_type"], "SaveImage")
        self.assertEqual(bound["144"]["inputs"]["unet_name"], workflow["144"]["inputs"]["unet_name"])
        self.assertEqual(bound["155"]["inputs"]["steps"], workflow["155"]["inputs"]["steps"])
        self.assertEqual(bound["152"]["inputs"]["cfg"], workflow["152"]["inputs"]["cfg"])

    def test_placeholder_profile_is_not_operational(self):
        root = Path(__file__).resolve().parents[1]
        profile = RendererProfileStore(root / "renderer_profiles.example.json", root).get("krea-2")
        self.assertFalse(profile.available)
        with self.assertRaisesRegex(ProfileError, "not available"):
            profile.load_workflow(root)

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
