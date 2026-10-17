"""Application-level comic rendering backed by a configured ComfyUI profile."""
from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from comfyui.client import ComfyUIClient
from comfyui.profiles import RendererProfile
from config import CONFIG
from page.builder import build_comic_page
from page.layouts import compute_layout_placements, select_layout_name
from policy.adult_only import validate_diffusion_prompts, validate_metadata_payload
from utils.parser import Comic, Panel, load_comic_from_json


_LEGACY_CONTINUITY_PREFIX = "preserve the registered recurring cast and approved setting;"
_PANEL_FRAMING = re.compile(
    r"\b(?:close[- ]up|medium(?:[- ](?:close|wide))? shot|wide shot|full[- ]body|"
    r"eye[- ]level|low[- ]angle|high[- ]angle|over[- ]the[- ]shoulder)\b",
    re.IGNORECASE,
)
_STOPPED_ACTION = re.compile(
    r"\b(?:stopped|stops|no longer|not)\s+(?:\w+\s+){0,2}?(?P<verb>\w+ing)\b",
    re.IGNORECASE,
)


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


def _panel_has_explicit_framing(panel: Panel) -> bool:
    return bool(_PANEL_FRAMING.search(_without_legacy_continuity_prefix(panel.panel_prompt_en)))


def _phrase_is_already_present(value: Any, scene_text: str) -> bool:
    """Return true only for a useful exact-ish authored duplication."""
    phrase = _normalized_field_text(value)
    scene = _normalized_field_text(scene_text)
    return bool(phrase and (phrase in scene or scene in phrase))


def _word_stem(word: str) -> str:
    """Small deterministic inflection normalizer used only for de-duplication."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _action_is_already_present(value: Any, scene_text: str, actor_name: str = "") -> bool:
    if _phrase_is_already_present(value, scene_text):
        return True
    scene_for_terms = _normalized_field_text(scene_text)
    if actor_name:
        # A different character's "stopped playing" must not make this
        # character's independent "playing" action appear duplicated.
        for match in reversed(list(_STOPPED_ACTION.finditer(scene_for_terms))):
            if not _actor_is_named_before(scene_for_terms, match.start(), actor_name):
                scene_for_terms = scene_for_terms[:match.start("verb")] + scene_for_terms[match.end("verb"):]
    ignored = {"a", "an", "the", "with", "and", "at", "in", "of", "to", "is", "are"}
    terms = {
        _word_stem(word)
        for word in re.findall(r"[a-z0-9]+", _normalized_field_text(value))
        if word not in ignored
    }
    scene_terms = {
        _word_stem(word)
        for word in re.findall(r"[a-z0-9]+", scene_for_terms)
        if word not in ignored
    }
    return len(terms) >= 2 and len(terms & scene_terms) / len(terms) >= 0.8


def _actor_is_named_before(text: str, position: int, actor_name: str) -> bool:
    """Whether an explicitly stopped action belongs to this assembled actor."""
    actor_terms = [term for term in re.findall(r"[a-z0-9]+", actor_name.casefold()) if len(term) > 1]
    if not actor_terms:
        return True
    clause = text[:position].rsplit(".", 1)[-1]
    return any(re.search(rf"\b{re.escape(term)}\b", clause) for term in actor_terms)


def _conflicting_action(left: Any, right: Any, actor_name: str = "") -> bool:
    """Detect a narrow, explicit stopped-versus-continuing action conflict.

    Prompt composition must not guess at nuanced prose.  This deliberately
    handles only an objectively incompatible state such as "stopped playing"
    alongside "still playing"; other wording is retained as approved context.
    """
    left_text = _normalized_field_text(left)
    right_text = _normalized_field_text(right)
    if not left_text or not right_text:
        return False
    for stopped, other in ((left_text, right_text), (right_text, left_text)):
        match = _STOPPED_ACTION.search(stopped)
        if match and actor_name and not _actor_is_named_before(stopped, match.start(), actor_name):
            continue
        other_terms = {_word_stem(word) for word in re.findall(r"[a-z0-9]+", other)}
        if match and _word_stem(match.group("verb").casefold()) in other_terms:
            return True
    return False


def _action_is_compatible_with_panel_prose(panel: Panel, action: Any, actor_name: str = "") -> bool:
    """Panel prose wins over a lower-precedence incompatible action field."""
    scene = _without_legacy_continuity_prefix(panel.panel_prompt_en)
    return not _conflicting_action(scene, action, actor_name)


def _character_action_descriptions(character_name: str, action: Any) -> list[str]:
    """Keep each structured action detail attached to its actor."""
    if not action:
        return []
    pieces: list[str] = []
    action_text = _field_text(getattr(action, "action_en", ""))
    pose = _field_text(getattr(action, "pose_en", ""))
    objects = _field_text(getattr(action, "objects", []))
    interaction = _field_text(getattr(action, "interaction", ""))
    if action_text:
        pieces.append(action_text if _normalized_field_text(action_text).startswith(_normalized_field_text(character_name))
                      else f"{character_name} {action_text}")
    if pose:
        pieces.append(f"{character_name} {pose}")
    if objects and not _phrase_is_already_present(objects, ", ".join(pieces)):
        pieces.append(f"with {objects}")
    if interaction:
        pieces.append(interaction)
    return [piece for piece in pieces if piece]


def _subject_text(comic: Comic, value: Any) -> str:
    """Use readable registered names without modifying the source field."""
    text = _field_text(value)
    for character_id, character in comic.characters.items():
        if character.name:
            text = re.sub(rf"\b{re.escape(character_id)}\b", character.name, text)
    return text


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
    """Assemble the approved panel fields into one renderer-ready scene.

    Precedence is deterministic: panel_prompt_en is the approved scene prose
    (including any explicit framing); character cards supply identity; primary
    and secondary subjects plus visual_hierarchy supply composition; required
    action and per-character actions supply missing, non-conflicting physical
    detail; required props supply missing objects; and panel background wins
    over the comic background.  When panel prose explicitly says an action has
    stopped while a lower-precedence action says it continues, the panel prose
    wins and the conflicting lower-precedence field is omitted, never combined
    or written back to the saved storyboard.

    required_subjects, character_positions, required_text, dialogues, and
    negative prompt fields remain validation/layout/lettering or renderer-
    profile data and are intentionally not positive-prompt prose.
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
    scene_prose = _without_legacy_continuity_prefix(panel.panel_prompt_en)
    add("primary subject", _subject_text(comic, panel.primary_subject))
    add("secondary subject", _subject_text(comic, panel.secondary_subject))
    if _hierarchy_adds_information(panel.visual_hierarchy, panel.primary_subject, panel.secondary_subject):
        add("visual priority", _subject_text(comic, panel.visual_hierarchy))
    if (_action_is_compatible_with_panel_prose(panel, panel.required_action)
            and not _action_is_already_present(panel.required_action, scene_prose)):
        add("action", panel.required_action)
    for character_id in panel.active_char_ids:
        character = comic.characters.get(character_id)
        action = panel.character_actions.get(character_id)
        descriptions = _character_action_descriptions((character.name if character else character_id) or character_id, action)
        for description in descriptions:
            if (_action_is_compatible_with_panel_prose(panel, description, character.name if character else character_id)
                    and not _action_is_already_present(description, scene_prose, character.name if character else character_id)):
                add("", description)
    if not _phrase_is_already_present(panel.required_props, scene_prose):
        add("required props", panel.required_props)
    if not _panel_has_explicit_framing(panel):
        add("camera", _camera_description(comic, panel))
    add("", scene_prose)
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
        values = _profile_values(panel, prompt, width, height, settings)
        bound_workflow = profile.apply_inputs(workflow, values)
        validate_diffusion_prompts(
            panel_id=panel.id,
            positive_prompt=prompt,
            negative_prompt=str(values["negative_prompt"] or ""),
            bound_workflow=bound_workflow,
        )
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
    # Direct rendering and older saved storyboards bypass generation-time
    # validation, so enforce the same policy before any panel is submitted.
    validate_metadata_payload(schema, path="storyboard")
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
