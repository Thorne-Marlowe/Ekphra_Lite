from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class StylePreset:
    id: str
    name: str
    prompt_fragment: str


STYLE_PRESETS: Dict[str, StylePreset] = {
    "contemporary_western_commercial_comic": StylePreset("contemporary_western_commercial_comic", "Contemporary Western Commercial Comic", "contemporary Western commercial comic illustration, clean confident linework, polished digital shading, saturated controlled color"),
    "cinematic_graphic_novel": StylePreset("cinematic_graphic_novel", "Cinematic Graphic Novel", "cinematic graphic novel illustration, expressive ink linework, rich tonal shading, textured painted color"),
    "black_and_white_manga": StylePreset("black_and_white_manga", "Black-and-White Manga", "black-and-white manga illustration, precise ink linework, screentone shading, high-contrast monochrome rendering"),
    "european_ligne_claire": StylePreset("european_ligne_claire", "European Ligne Claire", "European ligne claire comic illustration, crisp uniform linework, flat clear color, minimal clean shading"),
    "flat_color_webcomic": StylePreset("flat_color_webcomic", "Flat-Color Webcomic", "flat-color webcomic illustration, bold clean outlines, simplified cel shading, bright flat color treatment"),
    "painterly_graphic_novel": StylePreset("painterly_graphic_novel", "Painterly Graphic Novel", "painterly graphic novel illustration, visible brush texture, layered painterly shading, nuanced color rendering"),
}

MAX_CUSTOM_ART_DIRECTION_LENGTH = 1_000
# A persisted prompt may combine the longest current preset, the separator,
# and the full allowed custom direction. This is a validation bound only: it
# never truncates an approved human edit.
MAX_RESOLVED_ART_DIRECTION_LENGTH = (
    MAX_CUSTOM_ART_DIRECTION_LENGTH
    + len(", ")
    + max(len(preset.prompt_fragment) for preset in STYLE_PRESETS.values())
)


class ArtDirectionError(ValueError):
    """An invalid user-selected or persisted art direction."""


def get_style_preset(preset_id: str | None) -> StylePreset | None:
    """Look up a selected preset without silently substituting another."""
    if preset_id is None:
        return None
    try:
        return STYLE_PRESETS[preset_id]
    except KeyError as exc:
        raise ArtDirectionError(f"Unknown art-direction preset_id '{preset_id}'.") from exc


def resolve_art_direction(value: Any) -> dict[str, str | None] | None:
    """Validate a request and return durable renderer-ready metadata."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ArtDirectionError("artDirection must be an object.")
    unknown = set(value) - {"preset_id", "custom_prompt"}
    if unknown:
        raise ArtDirectionError(f"artDirection contains unsupported field(s): {', '.join(sorted(unknown))}.")
    preset_value, custom_value = value.get("preset_id"), value.get("custom_prompt")
    if preset_value is not None and (not isinstance(preset_value, str) or not preset_value.strip()):
        raise ArtDirectionError("artDirection.preset_id must be a nonempty string or null.")
    if custom_value is not None and (not isinstance(custom_value, str) or not custom_value.strip()):
        raise ArtDirectionError("artDirection.custom_prompt must be a nonempty string when supplied.")
    preset_id = preset_value.strip() if isinstance(preset_value, str) else None
    custom_prompt = custom_value.strip() if isinstance(custom_value, str) else None
    if custom_prompt and len(custom_prompt) > MAX_CUSTOM_ART_DIRECTION_LENGTH:
        raise ArtDirectionError(f"artDirection.custom_prompt must be at most {MAX_CUSTOM_ART_DIRECTION_LENGTH} characters.")
    preset = get_style_preset(preset_id)
    fragments = [fragment for fragment in (preset.prompt_fragment if preset else "", custom_prompt or "") if fragment]
    if not fragments:
        return None
    return {"preset_id": preset_id, "resolved_prompt": ", ".join(fragments)}


def validated_storyboard_art_direction(metadata: Any) -> str:
    """Return persisted prompt or fail before a ComfyUI submission."""
    if not isinstance(metadata, dict) or "art_direction" not in metadata:
        return ""
    direction = metadata["art_direction"]
    if not isinstance(direction, dict):
        raise ArtDirectionError("metadata.art_direction must be an object.")
    if set(direction) != {"preset_id", "resolved_prompt"}:
        raise ArtDirectionError("metadata.art_direction must contain only preset_id and resolved_prompt.")
    preset_id, prompt = direction.get("preset_id"), direction.get("resolved_prompt")
    if preset_id is not None and (not isinstance(preset_id, str) or not preset_id.strip()):
        raise ArtDirectionError("metadata.art_direction.preset_id must be a nonempty string or null.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ArtDirectionError("metadata.art_direction.resolved_prompt must be a nonempty string.")
    if len(prompt) > MAX_RESOLVED_ART_DIRECTION_LENGTH:
        raise ArtDirectionError(
            "metadata.art_direction.resolved_prompt must be at most "
            f"{MAX_RESOLVED_ART_DIRECTION_LENGTH} characters."
        )
    return prompt.strip()

