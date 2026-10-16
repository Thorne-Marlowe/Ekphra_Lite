"""Small dependency-free client for ComfyUI's local HTTP API."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ComfyUIError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComfyImage:
    filename: str
    subfolder: str = ""
    image_type: str = "output"


class ComfyUIClient:
    def __init__(self, base_url: str, timeout_seconds: int = 300, poll_seconds: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds

    def _request(self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self.base_url}{path}", body, method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urlopen(request, timeout=30) as response:
                data = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ComfyUIError(f"ComfyUI {method} {path} returned HTTP {exc.code}: {detail[:1000]}") from exc
        except URLError as exc:
            raise ComfyUIError(
                f"Cannot reach ComfyUI at {self.base_url}. Start ComfyUI with --listen 127.0.0.1: {exc.reason}"
            ) from exc
        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ComfyUIError(f"ComfyUI {method} {path} returned invalid JSON: {data[:300]!r}") from exc

    def submit(self, workflow: dict[str, Any]) -> str:
        response = self._request("/prompt", method="POST", payload={"prompt": workflow, "client_id": uuid.uuid4().hex})
        prompt_id = response.get("prompt_id") if isinstance(response, dict) else None
        if not prompt_id:
            node_errors = response.get("node_errors") if isinstance(response, dict) else None
            raise ComfyUIError(f"ComfyUI rejected the workflow: {node_errors or response}")
        return str(prompt_id)

    def wait_for_images(self, prompt_id: str, output_node_id: str | None = None) -> list[ComfyImage]:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            history = self._request(f"/history/{prompt_id}")
            job = history.get(prompt_id) if isinstance(history, dict) else None
            if job:
                status = job.get("status", {})
                if status.get("status_str") == "error":
                    messages = status.get("messages") or job.get("outputs") or job
                    raise ComfyUIError(f"ComfyUI job {prompt_id} failed: {messages}")
                outputs = job.get("outputs", {})
                images = self._images_from_outputs(outputs, output_node_id)
                if images:
                    return images
                if status.get("completed"):
                    raise ComfyUIError(
                        f"ComfyUI job {prompt_id} completed without image output. "
                        "Set output_node_id to the SaveImage node that produces the panel."
                    )
            time.sleep(self.poll_seconds)
        raise ComfyUIError(f"ComfyUI job {prompt_id} timed out after {self.timeout_seconds} seconds.")

    @staticmethod
    def _images_from_outputs(outputs: Any, output_node_id: str | None) -> list[ComfyImage]:
        if not isinstance(outputs, dict):
            return []
        nodes = {output_node_id: outputs.get(output_node_id)} if output_node_id else outputs
        images: list[ComfyImage] = []
        for output in nodes.values():
            if not isinstance(output, dict):
                continue
            for item in output.get("images", []):
                if isinstance(item, dict) and item.get("filename"):
                    images.append(ComfyImage(
                        filename=str(item["filename"]), subfolder=str(item.get("subfolder", "")),
                        image_type=str(item.get("type", "output")),
                    ))
        return images

    def download_image(self, image: ComfyImage) -> bytes:
        query = urlencode({"filename": image.filename, "subfolder": image.subfolder, "type": image.image_type})
        request = Request(f"{self.base_url}/view?{query}")
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except (HTTPError, URLError) as exc:
            raise ComfyUIError(f"Could not download ComfyUI image '{image.filename}': {exc}") from exc
