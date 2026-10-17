"""Application-level comic rendering backed by a configured ComfyUI profile."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from comfyui.client import ComfyUIClient
from comfyui.profiles import RendererProfile
from config import CONFIG
from page.builder import build_comic_page
from page.layouts import compute_layout_placements, select_layout_name
from utils.parser import Comic, Panel, load_comic_from_json


_LEGACY_CONTINUITY_PREFIX = "preserve the registered recurring cast and approved setting;"


def _field_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(text for item in value if (text := _field_text(item)))
    return str(value or "").strip()


def _normalized_field_text(value: Any) -> str:
    return " ".join(_field_text(value).casefold().split())


def _without_legacy_continuity_prefix(value: Any) -> str:
    text = _field_text(value)
    if text.casefold().startswith(_LEGACY_CONTINUITY_PREFIX):
        return text[len(_LEGACY_CONTINUITY_PREFIX):].lstrip()
    return text


def _camera_description(comic: Comic, panel: Panel) -> str:
    """Use an explicit, unambiguous character-card framing instruction only."""
    descriptions: list[str] = []
    for character_id in panel.active_char_ids:
        character = comic.characters.get(character_id)
        if not character:
            continue
        values = [_field_text(character.camera_distance), _field_text(character.camera_angle)]
        description = ", ".join(value for value in values if value)
        if description and _normalized_field_text(description) not in {
            _normalized_field_text(item) for item in descriptions
        }:
            descriptions.append(description)
    return descriptions[0] if len(descriptions) == 1 else ""


def _hierarchy_adds_information(hierarchy: Any, primary: Any, secondary: Any) -> bool:
    normalized = _normalized_field_text(hierarchy)
    if not normalized:
        return False
    known = {
        _normalized_field_text(primary),
        _normalized_field_text(secondary),
        _normalized_field_text([primary, secondary]),
    }
    return normalized not in known


def build_panel_prompt(comic: Comic, panel: Panel) -> str:
    """Assemble one concrete prompt without model-family-specific syntax.

    Character cards establish identity; primary/secondary subjects establish
    composition; action and props establish the drawable beat; panel prose
    supplies any remaining visual detail; and the panel/global background
    supplies setting. `required_subjects`, character actions/positions, and
    required text remain validation, layout, or lettering data rather than
    being blindly concatenated into image conditioning.
    """
    parts: list[str] = []
    seen: set[str] = set()

    def add(label: str, value: Any) -> None:
        text = _field_text(value)
        normalized = _normalized_field_text(text)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        parts.append(f"{label}: {text}" if label else text)

    for character_id in panel.active_char_ids:
        character = comic.characters.get(character_id)
        if character and character.base_prompt_en:
            add(character.name or character_id, character.base_prompt_en)
    add("primary subject", panel.primary_subject)
    add("secondary subject", panel.secondary_subject)
    if _hierarchy_adds_information(panel.visual_hierarchy, panel.primary_subject, panel.secondary_subject):
        add("visual priority", panel.visual_hierarchy)
    add("action", panel.required_action)
    add("required props", panel.required_props)
    add("camera", _camera_description(comic, panel))
    add("", _without_legacy_continuity_prefix(panel.panel_prompt_en))
    if panel.background_prompt_en:
        add("setting", panel.background_prompt_en)
    elif comic.background and comic.background.prompt_en:
        add("setting", comic.background.prompt_en)
    return ". ".join(part.strip().rstrip(".") for part in parts if part and part.strip())


RESOLUTION_PRESETS = {
    "square": (1024, 1024),
    "landscape": (1216, 832),
    "wide": (1344, 640),
    "portrait": (832, 1216),
    "tall": (640, 1344),
}


def select_panel_resolution(layout_width: int, layout_height: int) -> tuple[int, int]:
    """Choose a conservative Klein-compatible preset from panel geometry."""
    aspect = layout_width / max(1, layout_height)
    if aspect >= 1.7:
        return RESOLUTION_PRESETS["wide"]
    if aspect >= 1.15:
        return RESOLUTION_PRESETS["landscape"]
    if aspect <= 0.59:
        return RESOLUTION_PRESETS["tall"]
    if aspect <= 0.87:
        return RESOLUTION_PRESETS["portrait"]
    return RESOLUTION_PRESETS["square"]


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
        layout_width = int(size.get("width", CONFIG.panel.width))
        layout_height = int(size.get("height", CONFIG.panel.height))
        width, height = select_panel_resolution(layout_width, layout_height)
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
    publication_format: str = "paged_comic",
    layout_mode: str = "auto",
    page_layouts: dict[str, str] | None = None,
    image_fit: str = "cover",
    compose_page_numbers: Iterable[int] | None = None,
) -> tuple[list[str], Path]:
    if publication_format != "paged_comic":
        raise ValueError(f"Unsupported publication format '{publication_format}'.")
    if layout_mode not in {"auto", "simple", "manual"}:
        raise ValueError("layout_mode must be auto, simple, or manual.")
    if image_fit not in {"cover", "contain"}:
        raise ValueError("image_fit must be cover or contain.")
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

    requested_layouts = page_layouts or {}
    resolved_layouts: dict[int, str] = {}
    sizes: dict[str, dict[str, int]] = {}
    for page_number, panels in pages.items():
        requested = requested_layouts.get(str(page_number)) or requested_layouts.get(page_number)  # type: ignore[arg-type]
        # Preserve the old request fields for API callers that previously used
        # them, but new callers use page_layouts and layout_mode.
        legacy = manga_layout.replace("style", "Layout") if layout_style == "manga" else None
        preferred = requested or legacy
        if layout_mode == "manual" and not preferred:
            raise ValueError(f"Page {page_number} needs a manually selected layout.")
        layout = select_layout_name(len(panels), preferred=preferred, mode=layout_mode)
        placements = compute_layout_placements(
            layout, len(panels), CONFIG.page.width, CONFIG.page.height,
            CONFIG.page.margin, CONFIG.page.gutter,
        )
        resolved_layouts[page_number] = layout
        for placement, panel in zip(placements, panels):
            sizes[panel.id] = {"width": placement["width"], "height": placement["height"]}
    (job_dir / "panel_sizes.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")
    (job_dir / "layout_config.json").write_text(json.dumps({
        "publication_format": publication_format,
        "layout_mode": layout_mode,
        "page_layouts": {str(page_number): layout for page_number, layout in resolved_layouts.items()},
        "image_fit": image_fit,
        "page_image_fits": {str(page_number): image_fit for page_number in resolved_layouts},
    }, indent=2), encoding="utf-8")

    # Dialogue-only edits keep existing panel PNGs and only recompose bubbles.
    targets = None if target_panel_ids is None else list(target_panel_ids)
    if targets is None or targets:
        render_panels_with_comfyui(schema_path, panel_dir, profile, client, settings, targets)
    pages_to_compose = None if compose_page_numbers is None else set(compose_page_numbers)
    output_paths: list[str] = []
    for page_number, panels in sorted(pages.items()):
        if pages_to_compose is not None and page_number not in pages_to_compose:
            continue
        expected_ids = [panel.id for panel in panels]
        layout = resolved_layouts[page_number]
        page_path = out_dir / f"comic_page_{job_id}_{page_number}.png"
        build_comic_page(
            panels_dir=str(panel_dir), output_path=str(page_path), expected_panel_ids=expected_ids,
            schema_path=str(schema_path), inject_bubbles=True, use_adaptive_layout=True, layout_name=layout,
            image_fit=image_fit, layout_metadata_path=str(job_dir / f"page_layout_{page_number}.json"),
        )
        output_paths.append(str(page_path))
    return output_paths, schema_path
