"""Deterministic storyboard checks used before a draft reaches the user."""
from __future__ import annotations

import re
from typing import Any


_PAUSE_THEN_ACTION = re.compile(
    r"^(?P<subject>.+?)\s+(?P<hold>pauses|holds|remains|stands|stays)(?P<detail>[^,;]*),\s*(?:and\s+)?then\s+(?P<next>.+)$",
    re.IGNORECASE,
)
_SEQUENCE_CONNECTOR = re.compile(
    r"(?:,\s*(?:and\s+)?then\b|\bafter\s+(?:he|she|they|[A-Za-z]+ing)\b)",
    re.IGNORECASE,
)
_GERUNDS = {
    "looks": "looking", "turns": "turning", "reaches": "reaching", "leans": "leaning",
    "stares": "staring", "glances": "glancing", "raises": "raising", "lowers": "lowering",
}


def normalize_single_frame_actions(schema: Any) -> list[str]:
    """Safely collapse only an unambiguous hold-then-action into one still.

    Other sequential wording is retained and returned as a deterministic
    validation issue so the storyboard artist, rather than code, can choose
    the intended single visual moment.
    """
    if not isinstance(schema, dict):
        return []
    errors: list[str] = []
    for page_index, page in enumerate(schema.get("pages", []), start=1):
        if not isinstance(page, dict):
            continue
        for panel_index, panel in enumerate(page.get("panels", []), start=1):
            if not isinstance(panel, dict):
                continue
            action = str(panel.get("required_action", "")).strip()
            if _SEQUENCE_CONNECTOR.search(action):
                match = _PAUSE_THEN_ACTION.match(action)
                if match:
                    next_action = match.group("next").strip()
                    words = next_action.split(maxsplit=1)
                    if words and words[0].casefold() in _GERUNDS:
                        next_action = _GERUNDS[words[0].casefold()] + (f" {words[1]}" if len(words) > 1 else "")
                        panel["required_action"] = (
                            f"{match.group('subject')} {match.group('hold')}{match.group('detail')} while {next_action}"
                        )
                    else:
                        errors.append(
                            f"Page {page_index}, panel {panel_index} required_action describes a sequence; use one drawable instant."
                        )
                else:
                    errors.append(
                        f"Page {page_index}, panel {panel_index} required_action describes a sequence; use one drawable instant."
                    )
            panel_prompt = str(panel.get("panel_prompt_en", "")).strip()
            if _SEQUENCE_CONNECTOR.search(panel_prompt):
                errors.append(
                    f"Page {page_index}, panel {panel_index} panel_prompt_en describes a sequence; use one drawable instant."
                )
    return errors


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
            if "character_positions" in panel:
                positions = panel.get("character_positions")
                if not isinstance(positions, dict):
                    errors.append(f"{label} character_positions must be an object.")
                else:
                    valid_x, valid_y = {"left", "center", "right"}, {"top", "middle", "bottom"}
                    for character_id, position in positions.items():
                        if character_id not in characters:
                            errors.append(f"{label} character_positions references unregistered character '{character_id}'.")
                        if not isinstance(position, dict):
                            errors.append(f"{label} character_positions.{character_id} must be an object.")
                            continue
                        x, y, anchor = position.get("x", "center"), position.get("y", "middle"), position.get("anchor", "center")
                        valid_number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1
                        if not (valid_number(x) or (isinstance(x, str) and x.casefold() in valid_x)):
                            errors.append(f"{label} character_positions.{character_id}.x must be left, center, right, or a number from 0 to 1.")
                        if not (valid_number(y) or (isinstance(y, str) and y.casefold() in valid_y)):
                            errors.append(f"{label} character_positions.{character_id}.y must be top, middle, bottom, or a number from 0 to 1.")
                        if anchor != "center":
                            errors.append(f"{label} character_positions.{character_id}.anchor must be center.")
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
