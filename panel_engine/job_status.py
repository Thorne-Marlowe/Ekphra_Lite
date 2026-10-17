"""Persistent completion records for rendered comic jobs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


MANIFEST_NAME = "job_status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_path(job_dir: str | Path) -> Path:
    return Path(job_dir) / MANIFEST_NAME


def _write(job_dir: Path, manifest: dict[str, Any]) -> None:
    path = manifest_path(job_dir)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_manifest(job_dir: str | Path) -> dict[str, Any] | None:
    try:
        data = json.loads(manifest_path(job_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def manifest_exists(job_dir: str | Path) -> bool:
    """Whether a status record exists, even if it cannot be parsed."""
    return manifest_path(job_dir).exists()


def reserve_new_render_job(job_dir: str | Path, job_id: str) -> Path:
    """Create a new full-render directory without overwriting an existing job."""
    directory = Path(job_dir)
    if directory.exists():
        raise ValueError(f"Render job ID '{job_id}' already exists; choose a new job ID.")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def start_render(job_dir: str | Path, job_id: str, panel_ids: list[str], page_numbers: list[int]) -> None:
    directory = Path(job_dir)
    _write(directory, {
        "version": 1,
        "job_id": job_id,
        "status": "rendering",
        "started_at": _now(),
        "expected_panel_ids": panel_ids,
        "expected_page_numbers": page_numbers,
        "page_files": [],
    })


def set_render_expectations(job_dir: str | Path, panel_ids: list[str], page_numbers: list[int]) -> None:
    directory = Path(job_dir)
    manifest = load_manifest(directory)
    if not manifest or manifest.get("status") != "rendering":
        raise RuntimeError(f"Cannot set render expectations for job '{directory.name}'.")
    manifest.update({"expected_panel_ids": panel_ids, "expected_page_numbers": page_numbers})
    _write(directory, manifest)


def mark_failed(job_dir: str | Path, error: Exception | str) -> None:
    directory = Path(job_dir)
    manifest = load_manifest(directory) or {"version": 1, "job_id": directory.name}
    manifest.update({"status": "failed", "failed_at": _now(), "error": str(error)[:2000]})
    _write(directory, manifest)


def mark_completed(job_dir: str | Path, page_files: list[Path]) -> None:
    directory = Path(job_dir)
    manifest = load_manifest(directory)
    if not manifest:
        raise RuntimeError(f"Cannot complete job '{directory.name}' without a rendering manifest.")
    completed_manifest = {
        **manifest,
        "status": "completed",
        "completed_at": _now(),
        "page_files": [path.name for path in page_files],
    }
    completed_manifest.pop("error", None)
    if _expected_job_files(directory, directory.parent.parent, completed_manifest) is None:
        raise RuntimeError(f"Render job '{directory.name}' is missing or has unreadable expected output.")
    _write(directory, completed_manifest)


def _readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _expected_job_files(job_dir: Path, output_dir: Path, manifest: dict[str, Any]) -> list[Path] | None:
    panel_ids = manifest.get("expected_panel_ids")
    page_numbers = manifest.get("expected_page_numbers")
    page_files = manifest.get("page_files")
    if not isinstance(panel_ids, list) or not isinstance(page_numbers, list) or not isinstance(page_files, list):
        return None
    if len(page_files) != len(page_numbers):
        return None
    for panel_id in panel_ids:
        if not isinstance(panel_id, str) or not _readable_image(job_dir / "panels" / f"{panel_id}.png"):
            return None
    pages: list[Path] = []
    for page_number, filename in zip(page_numbers, page_files):
        if not isinstance(page_number, int) or not isinstance(filename, str) or Path(filename).name != filename:
            return None
        expected_name = f"comic_page_{job_dir.name}_{page_number}.png"
        if filename != expected_name:
            return None
        page = output_dir / filename
        if not _readable_image(page):
            return None
        pages.append(page)
    return pages


def completed_job_pages(output_dir: str | Path) -> list[Path]:
    """Return only manifest-complete or conservatively verified legacy pages."""
    root = Path(output_dir)
    jobs_dir = root / "jobs"
    if not jobs_dir.is_dir():
        return []
    pages: list[Path] = []
    for job_dir in jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue
        manifest = load_manifest(job_dir)
        if manifest_exists(job_dir):
            if manifest is not None and manifest.get("status") == "completed":
                pages.extend(_expected_job_files(job_dir, root, manifest) or [])
            # A failed, interrupted, or malformed manifest is never inferred
            # from leftover page files.
            continue
        pages.extend(_legacy_completed_pages(job_dir, root))
    return sorted(pages, key=lambda path: path.stat().st_mtime, reverse=True)


def _legacy_completed_pages(job_dir: Path, output_dir: Path) -> list[Path]:
    """Read-only compatibility check for pre-manifest jobs; ambiguity is excluded."""
    try:
        schema = json.loads((job_dir / "story.json").read_text(encoding="utf-8"))
        page_entries = schema.get("pages")
        if not isinstance(page_entries, list) or not page_entries:
            return []
    except (OSError, json.JSONDecodeError, AttributeError):
        return []
    pages: list[Path] = []
    for index, entry in enumerate(page_entries, start=1):
        if not isinstance(entry, dict) or not isinstance(entry.get("panels"), list):
            return []
        page_number = entry.get("page_number", index)
        if not isinstance(page_number, int):
            return []
        layout = job_dir / f"page_layout_{page_number}.json"
        if not layout.exists() and not (job_dir / "page_layout.json").exists():
            return []
        for panel in entry["panels"]:
            panel_id = panel.get("id") if isinstance(panel, dict) else None
            if not isinstance(panel_id, str) or not _readable_image(job_dir / "panels" / f"{panel_id}.png"):
                return []
        page = output_dir / f"comic_page_{job_dir.name}_{page_number}.png"
        if not _readable_image(page):
            return []
        pages.append(page)
    return pages
