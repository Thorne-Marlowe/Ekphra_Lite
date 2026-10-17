"""Best-effort visual QA for already-completed comic render jobs.

This module deliberately has no dependency on the creative LangGraph or the
renderer.  It only reads verified job artifacts and writes a new QA report.
"""
from __future__ import annotations

import base64
import json
import os
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from panel_engine.job_status import load_manifest
from studio_graph.structured_output import json_schema_response_format


RAW_CATEGORIES = {
    "identity", "action", "object", "composition", "obvious_visual_defect",
}
PAGE_CATEGORIES = {
    "dialogue_readability", "bubble_placement", "lettering_overlap",
    "lettering_clipping", "layout", "speaker_attribution",
}
SEVERITIES = {"info", "minor", "major"}
CONFIDENCES = {"high", "medium", "low", "not_assessable"}

ASSESSMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["observations"],
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "severity", "confidence", "explanation"],
                "properties": {
                    "category": {"type": "string"},
                    "severity": {"enum": ["info", "minor", "major"]},
                    "confidence": {"enum": ["high", "medium", "low", "not_assessable"]},
                    "explanation": {"type": "string"},
                },
            },
        },
    },
}


class VisionQAError(RuntimeError):
    """A job is ineligible for QA or its saved artifacts are invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _parse_model_json(content: Any) -> dict[str, Any]:
    raw = content if isinstance(content, str) else json.dumps(content)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Vision model returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Vision model response must be a JSON object.")
    return data


def _validate_observations(data: dict[str, Any], allowed_categories: set[str]) -> list[dict[str, str]]:
    observations = data.get("observations")
    if not isinstance(observations, list):
        raise ValueError("Vision assessment must contain an observations array.")
    validated: list[dict[str, str]] = []
    for index, item in enumerate(observations):
        if not isinstance(item, dict):
            raise ValueError(f"Observation {index} must be an object.")
        category = item.get("category")
        severity = item.get("severity")
        confidence = item.get("confidence")
        explanation = item.get("explanation")
        if category not in allowed_categories:
            raise ValueError(f"Observation {index} has unsupported category {category!r}.")
        if severity not in SEVERITIES:
            raise ValueError(f"Observation {index} has unsupported severity {severity!r}.")
        if confidence not in CONFIDENCES:
            raise ValueError(f"Observation {index} has unsupported confidence {confidence!r}.")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"Observation {index} needs a nonempty explanation.")
        # Uncertain speaker association is explicitly informational, never a
        # fabricated major failure.
        if category == "speaker_attribution" and confidence == "not_assessable":
            severity = "info"
        validated.append({
            "category": category,
            "severity": severity,
            "confidence": confidence,
            "explanation": explanation.strip(),
        })
    return validated


def _data_url(path: Path, maximum_size: tuple[int, int]) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(maximum_size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _response_content(response: Any) -> Any:
    return getattr(response, "content", response)


class VisionQAEngine:
    """Assess all fixed artifacts of one completed render job.

    ``client_factory`` is injectable for tests.  The production default uses
    the existing local OpenAI-compatible configuration and never falls back to
    a text-only model.
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs",
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 120,
        client_factory: Callable[..., Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.model = model or os.getenv("OPENAI_VISION_MODEL")
        self.base_url = base_url if base_url is not None else os.getenv("OPENAI_BASE_URL")
        self.timeout_seconds = timeout_seconds
        self.client_factory = client_factory or self._default_client_factory

    @staticmethod
    def _default_client_factory(**kwargs: Any) -> Any:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**kwargs)

    def _client(self) -> Any:
        if not self.model:
            raise VisionQAError("OPENAI_VISION_MODEL must name a vision-capable local model; no text-model fallback is used.")
        return self.client_factory(
            model=self.model,
            temperature=0,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
            model_kwargs={"response_format": json_schema_response_format("vision_qa_assessment", ASSESSMENT_SCHEMA)},
        )

    @staticmethod
    def _validated_panels(story: dict[str, Any], job_id: str) -> tuple[list[int], list[tuple[int, dict[str, Any]]]]:
        pages = story.get("pages")
        if not isinstance(pages, list) or not pages:
            raise VisionQAError(f"Job '{job_id}' storyboard has invalid pages data.")
        page_numbers: list[int] = []
        panels: list[tuple[int, dict[str, Any]]] = []
        panel_ids: set[str] = set()
        for index, page in enumerate(pages, start=1):
            if not isinstance(page, dict) or not isinstance(page.get("page_number"), int):
                raise VisionQAError(f"Job '{job_id}' storyboard page {index} has no valid page_number.")
            page_number = page["page_number"]
            if page_number in page_numbers:
                raise VisionQAError(f"Job '{job_id}' storyboard has duplicate page number {page_number}.")
            if not isinstance(page.get("panels"), list) or not page["panels"]:
                raise VisionQAError(f"Job '{job_id}' storyboard page {page_number} has no panels.")
            page_numbers.append(page_number)
            for panel in page["panels"]:
                panel_id = panel.get("id") if isinstance(panel, dict) else None
                if (not isinstance(panel_id, str) or not panel_id or Path(panel_id).name != panel_id):
                    raise VisionQAError(f"Job '{job_id}' storyboard page {page_number} has an invalid panel ID.")
                if panel_id in panel_ids:
                    raise VisionQAError(f"Job '{job_id}' storyboard has duplicate panel ID '{panel_id}'.")
                panel_ids.add(panel_id)
                panels.append((page_number, panel))
        return page_numbers, panels

    def _artifacts(self, job_id: str) -> tuple[Path, dict[str, Any], dict[str, Any], list[Path], list[tuple[int, dict[str, Any]]]]:
        if not isinstance(job_id, str) or not job_id or Path(job_id).name != job_id:
            raise VisionQAError("Job ID must name one existing job directory.")
        job_dir = self.output_dir / "jobs" / job_id
        manifest = load_manifest(job_dir)
        if not isinstance(manifest, dict) or manifest.get("status") != "completed":
            raise VisionQAError(f"Job '{job_id}' is not a verified completed render job.")
        try:
            story = json.loads((job_dir / "story.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VisionQAError(f"Job '{job_id}' has no readable storyboard artifact: {exc}") from exc
        if not isinstance(story, dict):
            raise VisionQAError(f"Job '{job_id}' storyboard has invalid data.")
        storyboard_page_numbers, panels = self._validated_panels(story, job_id)
        page_names = manifest.get("page_files")
        expected_ids = manifest.get("expected_panel_ids")
        expected_numbers = manifest.get("expected_page_numbers")
        if (not isinstance(page_names, list) or not isinstance(expected_ids, list)
                or not isinstance(expected_numbers, list) or len(page_names) != len(expected_numbers)):
            raise VisionQAError(f"Job '{job_id}' completion manifest is malformed.")
        if (any(not isinstance(item, str) or not item or Path(item).name != item for item in expected_ids)
                or len(set(expected_ids)) != len(expected_ids)
                or any(not isinstance(item, int) for item in expected_numbers)
                or len(set(expected_numbers)) != len(expected_numbers)):
            raise VisionQAError(f"Job '{job_id}' completion manifest has invalid expected IDs or page numbers.")
        if expected_numbers != storyboard_page_numbers:
            raise VisionQAError(f"Job '{job_id}' manifest page numbers do not match storyboard page associations.")
        if expected_ids != [panel["id"] for _, panel in panels]:
            raise VisionQAError(f"Job '{job_id}' manifest panel IDs do not match storyboard page associations.")
        pages = [self.output_dir / name for name in page_names if isinstance(name, str) and Path(name).name == name]
        if len(pages) != len(page_names) or not pages or any(not _readable_image(page) for page in pages):
            raise VisionQAError(f"Job '{job_id}' is missing a readable completed page artifact.")
        for page_number, page in zip(expected_numbers, pages):
            if not isinstance(page_number, int) or page.name != f"comic_page_{job_id}_{page_number}.png":
                raise VisionQAError(f"Job '{job_id}' completion manifest has invalid page metadata.")
        panel_dir = job_dir / "panels"
        if any(not isinstance(panel_id, str) or not _readable_image(panel_dir / f"{panel_id}.png") for panel_id in expected_ids):
            raise VisionQAError(f"Job '{job_id}' is missing a readable raw panel artifact.")
        return job_dir, manifest, story, pages, panels

    @staticmethod
    def _reserve_report_path(report_dir: Path) -> tuple[str, Path, Path]:
        """Reserve a filename exclusively so concurrent runs cannot collide."""
        for _ in range(100):
            run_id = uuid.uuid4().hex
            report_path = report_dir / f"{run_id}.json"
            reservation = report_dir / f".{run_id}.pending"
            if report_path.exists():
                continue
            try:
                with reservation.open("x", encoding="utf-8"):
                    pass
            except FileExistsError:
                continue
            if report_path.exists():
                reservation.unlink(missing_ok=True)
                continue
            return run_id, report_path, reservation
        raise VisionQAError("Could not reserve a unique Vision QA report filename.")

    @staticmethod
    def _publish_report(temporary: Path, report_path: Path) -> None:
        """Publish without replacing a report that appeared after reservation.

        ``os.link`` is an exclusive, atomic name creation on the local NTFS
        filesystem used by this application.  Unlike ``Path.replace``, it
        refuses to clobber a newly-created destination.  On a filesystem that
        does not support hard links, fail safely instead of overwriting an
        existing audit.
        """
        try:
            os.link(temporary, report_path)
        except FileExistsError as exc:
            raise VisionQAError(f"Vision QA report '{report_path.name}' appeared before publication.") from exc
        except OSError as exc:
            raise VisionQAError(f"Could not atomically publish Vision QA report '{report_path.name}': {exc}") from exc

    @staticmethod
    def _raw_context(story: dict[str, Any], panel: dict[str, Any]) -> dict[str, Any]:
        characters = story.get("characters", {}) if isinstance(story.get("characters"), dict) else {}
        active_ids = panel.get("active_char_ids", [])
        active_ids = active_ids if isinstance(active_ids, list) else []
        active_characters = {
            character_id: characters[character_id]
            for character_id in active_ids
            if isinstance(character_id, str) and isinstance(characters.get(character_id), dict)
        }
        return {
            "panel_id": panel["id"],
            "characters": active_characters,
            "required_action": panel.get("required_action", ""),
            "required_props": panel.get("required_props", []),
            "primary_subject": panel.get("primary_subject", ""),
            "secondary_subject": panel.get("secondary_subject", ""),
            "visual_hierarchy": panel.get("visual_hierarchy", ""),
            "composition": panel.get("panel_prompt_en", ""),
        }

    def _assess(self, client: Any, *, image_path: Path, instruction: str, context: dict[str, Any], categories: set[str]) -> list[dict[str, str]]:
        content = [
            {"type": "text", "text": instruction + "\nContext:\n" + json.dumps(context, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": _data_url(image_path, (1400, 1400))}},
        ]
        response = client.invoke([
            {"role": "system", "content": "Return only the requested JSON. Report only visible evidence. Do not infer unseen details."},
            {"role": "user", "content": content},
        ])
        return _validate_observations(_parse_model_json(_response_content(response)), categories)

    def run(self, job_id: str) -> Path:
        """Write and return a new QA report for a verified completed job.

        Individual image failures remain in the report; they never affect the
        completed job's manifest or its render artifacts.
        """
        # All artifact/association checks occur before any QA directory is
        # created or any vision client is initialized.
        job_dir, manifest, story, pages, panels = self._artifacts(job_id)
        report_dir = job_dir / "vision_qa"
        report_dir.mkdir(parents=True, exist_ok=True)
        run_id, report_path, reservation = self._reserve_report_path(report_dir)
        temporary: Path | None = None
        try:
            client_error: str | None = None
            try:
                client = self._client()
            except Exception as exc:
                client = None
                client_error = str(exc)
            report: dict[str, Any] = {
                "version": 1,
                "job_id": job_id,
                "run_id": run_id,
                "model": self.model,
                "started_at": _now(),
                "completed_at": None,
                "status": "running",
                "raw_panels": [],
                "pages": [],
                "errors": [],
            }
            if client_error:
                report["errors"].append({"scope": "run", "error": f"Vision model unavailable: {client_error}"})
            for page_number, panel in panels:
                entry: dict[str, Any] = {"panel_id": panel["id"], "page_number": page_number, "status": "error", "observations": []}
                if client is None:
                    entry["error"] = "Vision model unavailable."
                else:
                    try:
                        entry["observations"] = self._assess(
                            client, image_path=job_dir / "panels" / f"{panel['id']}.png",
                            instruction="Assess this raw comic panel for visible character consistency, required actions and objects, composition, and obvious visual defects. Do not require perfection or infer unseen story details.",
                            context=self._raw_context(story, panel), categories=RAW_CATEGORIES,
                        )
                        entry["status"] = "assessed"
                    except Exception as exc:
                        entry["error"] = str(exc)
                report["raw_panels"].append(entry)
            page_entries = {number: [] for number, _ in panels}
            for page_number, panel in panels:
                page_entries[page_number].append(panel)
            for page_path, page_number in zip(pages, manifest["expected_page_numbers"]):
                try:
                    layout = json.loads((job_dir / f"page_layout_{page_number}.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    layout = {}
                panels_on_page = page_entries[page_number]
                context = {
                    "page_number": page_number,
                    "panel_ids": [panel["id"] for panel in panels_on_page],
                    "dialogues": [{"panel_id": panel["id"], "dialogues": panel.get("dialogues", [])} for panel in panels_on_page],
                    "layout": layout,
                }
                entry = {"page_number": page_number, "panel_ids": context["panel_ids"], "status": "error", "observations": []}
                if client is None:
                    entry["error"] = "Vision model unavailable."
                else:
                    try:
                        entry["observations"] = self._assess(
                            client, image_path=page_path,
                            instruction="Assess this composed comic page for dialogue readability, bubble placement, obvious lettering clipping or overlap, and layout problems. Assess speaker attribution only where the image itself supports it; otherwise use confidence not_assessable and severity info.",
                            context=context, categories=PAGE_CATEGORIES,
                        )
                        entry["status"] = "assessed"
                    except Exception as exc:
                        entry["error"] = str(exc)
                report["pages"].append(entry)
            successful = sum(item["status"] == "assessed" for item in report["raw_panels"] + report["pages"])
            total = len(report["raw_panels"]) + len(report["pages"])
            report["status"] = "completed" if successful == total else ("partial" if successful else "unavailable")
            report["completed_at"] = _now()
            temporary = report_dir / f".{run_id}.{uuid.uuid4().hex}.tmp"
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            self._publish_report(temporary, report_path)
            return report_path
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            reservation.unlink(missing_ok=True)
