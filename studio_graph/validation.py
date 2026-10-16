"""Deterministic storyboard checks used before a draft reaches the user."""
from __future__ import annotations

from typing import Any


def validate_storyboard_structure(schema: Any, requested_pages: int) -> list[str]:
    """Return data-integrity errors, never subjective creative criticism."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return ["Storyboard must be a JSON object."]
    characters = schema.get("characters")
    if not isinstance(characters, dict):
        errors.append("Storyboard must contain a characters object.")
        characters = {}
    else:
        for character_id, character in characters.items():
            if not isinstance(character, dict):
                errors.append(f"Character '{character_id}' must be an object.")
            else:
                if not str(character.get("name", "")).strip():
                    errors.append(f"Character '{character_id}' needs a non-empty name.")
                if not str(character.get("base_prompt_en", "")).strip():
                    errors.append(f"Character '{character_id}' needs a non-empty base_prompt_en.")

    pages = schema.get("pages")
    if not isinstance(pages, list):
        return errors + ["Storyboard must contain a pages array."]
    if len(pages) != requested_pages:
        errors.append(f"Storyboard must contain exactly {requested_pages} page(s), found {len(pages)}.")

    panel_ids: set[str] = set()
    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            errors.append(f"Page {page_index} must be an object.")
            continue
        if page.get("page_number") != page_index:
            errors.append(f"Page {page_index} must have page_number {page_index}.")
        panels = page.get("panels")
        if not isinstance(panels, list):
            errors.append(f"Page {page_index} must contain a panels array.")
            continue
        if not 2 <= len(panels) <= 6:
            errors.append(f"Page {page_index} must contain 2 to 6 panels, found {len(panels)}.")
        for panel_index, panel in enumerate(panels, start=1):
            label = f"Page {page_index}, panel {panel_index}"
            if not isinstance(panel, dict):
                errors.append(f"{label} must be an object.")
                continue
            panel_id = str(panel.get("id", "")).strip()
            if not panel_id:
                errors.append(f"{label} needs an id.")
            elif panel_id in panel_ids:
                errors.append(f"Panel id '{panel_id}' must be unique.")
            else:
                panel_ids.add(panel_id)
            if not str(panel.get("panel_prompt_en", "")).strip():
                errors.append(f"{label} needs a non-empty panel_prompt_en.")
            if "active_char_ids" not in panel:
                errors.append(f"{label} needs active_char_ids.")
            active_ids = panel.get("active_char_ids", [])
            if not isinstance(active_ids, list):
                errors.append(f"{label} active_char_ids must be an array.")
                active_ids = []
            for character_id in active_ids:
                if character_id not in characters:
                    errors.append(f"{label} references unregistered character '{character_id}'.")
            if "dialogues" not in panel:
                errors.append(f"{label} needs a dialogues array.")
            dialogues = panel.get("dialogues", [])
            if not isinstance(dialogues, list):
                errors.append(f"{label} dialogues must be an array.")
                continue
            for dialogue_index, dialogue in enumerate(dialogues, start=1):
                if not isinstance(dialogue, dict):
                    errors.append(f"{label} dialogue {dialogue_index} must be an object.")
                    continue
                speaker = dialogue.get("character_id")
                if speaker not in characters:
                    errors.append(f"{label} dialogue {dialogue_index} references unregistered character '{speaker}'.")
                if not str(dialogue.get("text", "")).strip():
                    errors.append(f"{label} dialogue {dialogue_index} needs text.")
    return errors
