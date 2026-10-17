"""Curated paged-comic layout templates and shared placement calculations."""
from __future__ import annotations

from typing import Dict, List, Optional

from config import CONFIG


# Entries are read in comic reading order. ``col_span``/``row_span`` define
# actual geometry; nominal image dimensions are deliberately not stored here.
LAYOUT_SETTINGS: Dict[str, List[Dict[str, int]]] = {
    "paged_splash": [{"col_span": 1, "row_span": 1}],
    "paged_top_bottom_2": [{"col_span": 1, "row_span": 1}, {"col_span": 1, "row_span": 1}],
    "paged_left_right_2": [{"col_span": 1, "row_span": 1}, {"col_span": 1, "row_span": 1}],
    "paged_rows_3": [{"col_span": 1, "row_span": 1}] * 3,
    "paged_grid_4": [{"col_span": 1, "row_span": 1}] * 4,
    "paged_wide_split_wide_4": [
        {"col_span": 2, "row_span": 1}, {"col_span": 1, "row_span": 1},
        {"col_span": 1, "row_span": 1}, {"col_span": 2, "row_span": 1},
    ],
    "paged_wide_wide_split_4": [
        {"col_span": 2, "row_span": 1}, {"col_span": 2, "row_span": 1},
        {"col_span": 1, "row_span": 1}, {"col_span": 1, "row_span": 1},
    ],
    "paged_split_wide_wide_4": [
        {"col_span": 1, "row_span": 1}, {"col_span": 1, "row_span": 1},
        {"col_span": 2, "row_span": 1}, {"col_span": 2, "row_span": 1},
    ],
    "paged_full_plus_four_5": [
        {"col_span": 2, "row_span": 1}, {"col_span": 1, "row_span": 1},
        {"col_span": 1, "row_span": 1}, {"col_span": 1, "row_span": 1},
        {"col_span": 1, "row_span": 1},
    ],
    "paged_grid_6": [{"col_span": 1, "row_span": 1}] * 6,
    # Names retained for callers that already used the earlier layout module.
    "Layout1Panel": [{"col_span": 1, "row_span": 1}],
    "Layout2Panels": [{"col_span": 1, "row_span": 1}, {"col_span": 1, "row_span": 1}],
    "Layout3Panels": [{"col_span": 1, "row_span": 1}] * 3,
    "Layout4Panels_Dynamic": [
        {"col_span": 2, "row_span": 1}, {"col_span": 1, "row_span": 1},
        {"col_span": 1, "row_span": 1}, {"col_span": 2, "row_span": 1},
    ],
    "Layout5Panels_Action": [
        {"col_span": 2, "row_span": 1}, {"col_span": 1, "row_span": 1},
        {"col_span": 1, "row_span": 1}, {"col_span": 1, "row_span": 1},
        {"col_span": 1, "row_span": 1},
    ],
    "Layout6Panels_Reveal": [{"col_span": 1, "row_span": 1}] * 6,
}

GRID_CONFIGS: Dict[str, Dict[str, int]] = {
    "paged_splash": {"cols": 1, "rows": 1},
    "paged_top_bottom_2": {"cols": 1, "rows": 2},
    "paged_left_right_2": {"cols": 2, "rows": 1},
    "paged_rows_3": {"cols": 1, "rows": 3},
    "paged_grid_4": {"cols": 2, "rows": 2},
    "paged_wide_split_wide_4": {"cols": 2, "rows": 3},
    "paged_wide_wide_split_4": {"cols": 2, "rows": 3},
    "paged_split_wide_wide_4": {"cols": 2, "rows": 3},
    "paged_full_plus_four_5": {"cols": 2, "rows": 3},
    "paged_grid_6": {"cols": 2, "rows": 3},
    "Layout1Panel": {"cols": 1, "rows": 1},
    "Layout2Panels": {"cols": 1, "rows": 2},
    "Layout3Panels": {"cols": 1, "rows": 3},
    "Layout4Panels_Dynamic": {"cols": 2, "rows": 3},
    "Layout5Panels_Action": {"cols": 2, "rows": 3},
    "Layout6Panels_Reveal": {"cols": 2, "rows": 3},
}

DEFAULT_LAYOUTS = {1: "paged_splash", 2: "paged_top_bottom_2", 3: "paged_rows_3", 4: "paged_grid_4", 5: "paged_full_plus_four_5", 6: "paged_grid_6"}
SIMPLE_LAYOUTS = DEFAULT_LAYOUTS.copy()


def compatible_layout_names(num_panels: int) -> List[str]:
    """Return curated paged-comic templates compatible with ``num_panels``."""
    return [name for name, entries in LAYOUT_SETTINGS.items() if name.startswith("paged_") and len(entries) == num_panels]


def select_layout_name(num_panels: int, preferred: Optional[str] = None, mode: str = "auto", seed_text: str = "") -> str:
    """Resolve a deterministic layout, accepting a validated manual template."""
    if preferred:
        if preferred not in LAYOUT_SETTINGS or len(LAYOUT_SETTINGS[preferred]) != num_panels:
            raise ValueError(f"Layout '{preferred}' is not compatible with {num_panels} panels.")
        return preferred
    defaults = SIMPLE_LAYOUTS if mode == "simple" else DEFAULT_LAYOUTS
    try:
        return defaults[num_panels]
    except KeyError as exc:
        raise ValueError(f"No paged-comic layout is defined for {num_panels} panels.") from exc


def compute_layout_placements(
    layout_name: str,
    panels_count: int,
    page_width: int | None = None,
    page_height: int | None = None,
    margin: int | None = None,
    gutter: int | None = None,
) -> List[Dict[str, int]]:
    if layout_name not in LAYOUT_SETTINGS:
        raise ValueError(f"Unknown layout '{layout_name}'.")
    layout_settings = LAYOUT_SETTINGS[layout_name]
    if panels_count != len(layout_settings):
        raise ValueError(f"Layout '{layout_name}' requires {len(layout_settings)} panels, got {panels_count}.")
    page_width = CONFIG.page.width if page_width is None else page_width
    page_height = CONFIG.page.height if page_height is None else page_height
    margin = CONFIG.page.margin if margin is None else margin
    gutter = CONFIG.page.gutter if gutter is None else gutter
    grid_config = GRID_CONFIGS[layout_name]
    num_cols, num_rows = grid_config["cols"], grid_config["rows"]
    content_width, content_height = page_width - margin * 2, page_height - margin * 2
    cell_width = (content_width - gutter * (num_cols - 1)) // num_cols
    cell_height = (content_height - gutter * (num_rows - 1)) // num_rows
    width_adjustment = (content_width - (cell_width * num_cols + gutter * (num_cols - 1))) // 2
    height_adjustment = (content_height - (cell_height * num_rows + gutter * (num_rows - 1))) // 2
    occupied = [[False for _ in range(num_cols)] for _ in range(num_rows)]
    placements: List[Dict[str, int]] = []
    for idx, entry in enumerate(layout_settings):
        col_span, row_span = entry["col_span"], entry["row_span"]
        position: tuple[int, int] | None = None
        for row in range(num_rows):
            for col in range(num_cols):
                if row + row_span <= num_rows and col + col_span <= num_cols and all(not occupied[row + dr][col + dc] for dr in range(row_span) for dc in range(col_span)):
                    position = (row, col)
                    break
            if position:
                break
        if position is None:
            raise ValueError(f"Layout '{layout_name}' cannot place panel {idx + 1}.")
        row, col = position
        for dr in range(row_span):
            for dc in range(col_span):
                occupied[row + dr][col + dc] = True
        placements.append({
            "idx": idx, "width": cell_width * col_span + gutter * (col_span - 1),
            "height": cell_height * row_span + gutter * (row_span - 1),
            "x": margin + width_adjustment + col * (cell_width + gutter),
            "y": margin + height_adjustment + row * (cell_height + gutter),
            "col": col, "row": row, "col_span": col_span, "row_span": row_span,
        })
    return placements


__all__ = ["LAYOUT_SETTINGS", "GRID_CONFIGS", "compatible_layout_names", "select_layout_name", "compute_layout_placements"]
