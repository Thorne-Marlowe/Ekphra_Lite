"""Validated, explicit mappings between comic render inputs and Comfy workflows.

Profiles deliberately contain no model-specific node guesses.  Export an API
workflow from the target ComfyUI installation, then bind its real node IDs and
input names here (or in the file specified by RENDERER_PROFILES_FILE).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProfileError(ValueError):
    """A profile cannot safely be used to submit a ComfyUI job."""


@dataclass(frozen=True)
class WorkflowInput:
    node_id: str
    input_name: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkflowInput":
        node_id = str(value.get("node_id", "")).strip()
        input_name = str(value.get("input_name", "")).strip()
        if not node_id or not input_name:
            raise ProfileError("Every input_map entry needs node_id and input_name.")
        return cls(node_id=node_id, input_name=input_name)


@dataclass(frozen=True)
class RendererProfile:
    id: str
    label: str
    workflow_path: str
    input_map: dict[str, WorkflowInput] = field(default_factory=dict)
    output_node_id: str | None = None
    description: str = ""

    @classmethod
    def from_dict(cls, profile_id: str, value: dict[str, Any]) -> "RendererProfile":
        raw_map = value.get("input_map", {})
        if not isinstance(raw_map, dict):
            raise ProfileError(f"Profile '{profile_id}' input_map must be an object.")
        return cls(
            id=profile_id,
            label=str(value.get("label") or profile_id),
            workflow_path=str(value.get("workflow_path", "")).strip(),
            input_map={key: WorkflowInput.from_dict(item) for key, item in raw_map.items()},
            output_node_id=str(value["output_node_id"]).strip() if value.get("output_node_id") else None,
            description=str(value.get("description", "")),
        )

    def load_workflow(self, root: Path) -> dict[str, Any]:
        if not self.workflow_path:
            raise ProfileError(
                f"Renderer profile '{self.id}' has no workflow_path. Export its ComfyUI API workflow first."
            )
        path = Path(self.workflow_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise ProfileError(
                f"Renderer profile '{self.id}' needs workflow file '{path}'. "
                "Export the workflow with ComfyUI's Save (API Format) control."
            )
        try:
            workflow = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(f"Cannot read ComfyUI workflow '{path}': {exc}") from exc
        if not isinstance(workflow, dict) or not workflow:
            raise ProfileError(f"ComfyUI workflow '{path}' must be a non-empty API-format JSON object.")
        return workflow

    def apply_inputs(self, workflow: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
        # JSON round-trip makes a safe deep copy for ComfyUI's JSON graph.
        bound = json.loads(json.dumps(workflow))
        for app_name, binding in self.input_map.items():
            if app_name not in values or values[app_name] is None:
                continue
            node = bound.get(binding.node_id)
            if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
                raise ProfileError(
                    f"Profile '{self.id}' maps '{app_name}' to node '{binding.node_id}', "
                    "which is absent or has no inputs in its workflow export."
                )
            node["inputs"][binding.input_name] = values[app_name]
        if "positive_prompt" not in self.input_map:
            raise ProfileError(
                f"Profile '{self.id}' must map 'positive_prompt' before it can render comic panels."
            )
        return bound


class RendererProfileStore:
    def __init__(self, path: str | Path, root: str | Path = "."):
        self.path = Path(path)
        self.root = Path(root)

    def load(self) -> dict[str, RendererProfile]:
        if not self.path.is_file():
            raise ProfileError(
                f"Renderer profile file '{self.path}' is missing. Copy renderer_profiles.example.json "
                "to renderer_profiles.json and bind a ComfyUI API workflow."
            )
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(f"Cannot read renderer profile file '{self.path}': {exc}") from exc
        items = raw.get("profiles", raw) if isinstance(raw, dict) else None
        if not isinstance(items, dict):
            raise ProfileError("Renderer profile file must contain a 'profiles' object.")
        return {profile_id: RendererProfile.from_dict(profile_id, item) for profile_id, item in items.items()}

    def get(self, profile_id: str) -> RendererProfile:
        profiles = self.load()
        try:
            return profiles[profile_id]
        except KeyError as exc:
            available = ", ".join(profiles) or "none"
            raise ProfileError(f"Unknown renderer profile '{profile_id}'. Available profiles: {available}.") from exc
