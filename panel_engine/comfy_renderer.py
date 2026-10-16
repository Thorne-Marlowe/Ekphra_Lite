"""Application-level comic rendering backed by a configured ComfyUI profile."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image

from comfyui.client import ComfyUIClient
from comfyui.profiles import RendererProfile
from config import CONFIG
from page.builder import build_comic_page
from page.layouts import compute_layout_placements, select_layout_name
from utils.parser import Comic, Panel, load_comic_from_json


def build_panel_prompt(comic: Comic, panel: Panel) -> str:
    """Keep storyboard intent intact without model-family prompt syntax."""
    parts: list[str] = []
    for character_id in panel.active_char_ids:
        character = comic.characters.get(character_id)
        if character and character.base_prompt_en:
            parts.append(f"{character.name or character_id}: {character.base_prompt_en}")
    if panel.primary_subject:
        parts.append(f"primary subject: {panel.primary_subject}")
    if panel.secondary_subject:
        parts.append(f"secondary subject: {panel.secondary_subject}")
    if panel.required_action:
        parts.append(f"action: {panel.required_action}")
    if panel.required_props:
        parts.append("required props: " + ", ".join(panel.required_props))
    if panel.panel_prompt_en:
        parts.append(panel.panel_prompt_en)
    if panel.background_prompt_en:
        parts.append(f"setting: {panel.background_prompt_en}")
    elif comic.background and comic.background.prompt_en:
        parts.append(f"setting: {comic.background.prompt_en}")
    return ". ".join(part.strip().rstrip(".") for part in parts if part and part.strip())


def _profile_values(panel: Panel, prompt: str, width: int, height: int, settings: dict) -> dict:
    values = {
        "positive_prompt": prompt,
        "negative_prompt": settings.get("negative_prompt", ""),
        "width": width,
        "height": height,
        "steps": settings.get("steps"),
        "cfg": settings.get("guidance"),
        "seed": settings.get("seed"),
        "filename_prefix": f"comic/{panel.id}",
    }
    return values


def render_panels_with_comfyui(
    schema_path: str | Path,
    output_panel_dir: str | Path,
    profile: RendererProfile,
    client: ComfyUIClient,
    settings: dict,
    target_panel_ids: Iterable[str] | None = None,
) -> None:
    schema_path = Path(schema_path)
    comic = load_comic_from_json(json.loads(schema_path.read_text(encoding="utf-8")))
    workflow = profile.load_workflow(Path.cwd())
    # None means all panels; an explicit empty iterable means no ComfyUI work
    # (used for dialogue-only recomposition).
    target_ids = None if target_panel_ids is None else set(target_panel_ids)
    output_dir = Path(output_panel_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_sizes = json.loads((schema_path.parent / "panel_sizes.json").read_text(encoding="utf-8")) \
        if (schema_path.parent / "panel_sizes.json").exists() else {}

    for panel in comic.panels:
        if target_ids is not None and panel.id not in target_ids:
            continue
        size = panel_sizes.get(panel.id, {})
        width = int(size.get("width", CONFIG.panel.width))
        height = int(size.get("height", CONFIG.panel.height))
        prompt = build_panel_prompt(comic, panel)
        bound_workflow = profile.apply_inputs(workflow, _profile_values(panel, prompt, width, height, settings))
        prompt_id = client.submit(bound_workflow)
        images = client.wait_for_images(prompt_id, profile.output_node_id)
        image_bytes = client.download_image(images[0])
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"ComfyUI returned an unreadable image for panel {panel.id}: {exc}") from exc
        image.save(output_dir / f"{panel.id}.png", format="PNG")


def render_comic_schema(
    schema: dict,
    job_id: str,
    profile: RendererProfile,
    client: ComfyUIClient,
    settings: dict,
    layout_style: str = "auto",
    manga_layout: str = "style1",
    target_panel_ids: Iterable[str] | None = None,
) -> tuple[list[str], Path]:
    out_dir = Path(CONFIG.paths.panel_dir).parent
    job_dir = out_dir / "jobs" / job_id
    panel_dir = job_dir / "panels"
    job_dir.mkdir(parents=True, exist_ok=True)
    schema_path = job_dir / "story.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    comic = load_comic_from_json(schema)
    pages: dict[int, list[Panel]] = {}
    for panel in comic.panels:
        pages.setdefault(panel.page_number, []).append(panel)
    if not pages:
        raise ValueError("Storyboard has no panels to render.")

    preferred = manga_layout.replace("style", "Layout") if layout_style == "manga" else None
    sizes: dict[str, dict[str, int]] = {}
    for page_number, panels in pages.items():
        layout = select_layout_name(len(panels), preferred=preferred, seed_text=f"{job_id}_{page_number}")
        for placement, panel in zip(compute_layout_placements(layout, len(panels)), panels):
            sizes[panel.id] = {"width": placement["width"], "height": placement["height"]}
    (job_dir / "panel_sizes.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")

    # Dialogue-only edits keep existing panel PNGs and only recompose bubbles.
    targets = None if target_panel_ids is None else list(target_panel_ids)
    if targets is None or targets:
        render_panels_with_comfyui(schema_path, panel_dir, profile, client, settings, targets)
    output_paths: list[str] = []
    for page_number, panels in sorted(pages.items()):
        expected_ids = [panel.id for panel in panels]
        layout = select_layout_name(len(panels), preferred=preferred, seed_text=f"{job_id}_{page_number}")
        page_path = out_dir / f"comic_page_{job_id}_{page_number}.png"
        build_comic_page(
            panels_dir=str(panel_dir), output_path=str(page_path), expected_panel_ids=expected_ids,
            schema_path=str(schema_path), inject_bubbles=True, use_adaptive_layout=True, layout_name=layout,
        )
        output_paths.append(str(page_path))
    return output_paths, schema_path
