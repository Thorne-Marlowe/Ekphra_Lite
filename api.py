import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from comfyui.client import ComfyUIClient, ComfyUIError
from comfyui.profiles import ProfileError, RendererProfileStore
from config import CONFIG
from panel_engine.comfy_renderer import render_comic_schema
from panel_engine.job_status import completed_job_pages
from page.layouts import compatible_layout_names
from studio_graph.graph import create_storyboard_graph
from utils.parser import load_comic_from_json

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="Ekphra FLUX.2 Comic Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "panels").mkdir(parents=True, exist_ok=True)
CONFIG.paths.panel_dir = str(output_dir / "panels")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
storyboard_graph = create_storyboard_graph()


class StoryboardRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    pageCount: int = Field(default=1, ge=1, le=8)


class RenderRequest(BaseModel):
    storyboard: dict | None = None
    rendererProfile: str
    layoutStyle: str = "auto"
    mangaLayout: str = "style1"
    steps: int | None = Field(default=None, ge=1, le=500)
    guidance: float | None = Field(default=None, ge=0, le=100)
    negativePrompt: str = ""
    seed: int | None = Field(default=None, ge=0)
    jobId: str | None = Field(default=None, max_length=100)
    publicationFormat: Literal["paged_comic"] = "paged_comic"
    layoutMode: Literal["auto", "simple", "manual"] = "auto"
    pageLayouts: dict[str, str] = Field(default_factory=dict)
    imageFit: Literal["cover", "contain"] = "cover"


class RegeneratePanelRequest(RenderRequest):
    pageUrl: str = Field(min_length=1)
    panelId: str = Field(min_length=1, max_length=100)
    panelPrompt: str = Field(min_length=1, max_length=4000)
    dialogues: list[dict] = Field(default_factory=list)
    mode: str = Field(default="panel", pattern="^(panel|dialogue)$")


class SavePageRequest(BaseModel):
    pageUrl: str = Field(min_length=1)
    filename: str = Field(default="comic-page.png", min_length=1, max_length=120)


def profile_store() -> RendererProfileStore:
    return RendererProfileStore(os.getenv("RENDERER_PROFILES_FILE", CONFIG.comfyui.profiles_file))


def comfy_client() -> ComfyUIClient:
    return ComfyUIClient(
        os.getenv("COMFYUI_BASE_URL", CONFIG.comfyui.base_url),
        timeout_seconds=int(os.getenv("COMFYUI_TIMEOUT_SECONDS", CONFIG.comfyui.timeout_seconds)),
        poll_seconds=float(os.getenv("COMFYUI_POLL_SECONDS", CONFIG.comfyui.poll_seconds)),
    )


def profile_or_422(profile_id: str):
    try:
        return profile_store().get(profile_id)
    except ProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def rendering_settings(req: RenderRequest) -> dict:
    return {"steps": req.steps, "guidance": req.guidance, "negative_prompt": req.negativePrompt, "seed": req.seed}


def page_job_info(page_url: str) -> tuple[Path, str, int]:
    relative = page_url.removeprefix("/outputs/").replace("/", os.sep)
    page_path = (output_dir / relative).resolve()
    if output_dir.resolve() not in page_path.parents or not page_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    parts = page_path.stem.split("_")
    if len(parts) < 4 or parts[:2] != ["comic", "page"]:
        raise HTTPException(status_code=400, detail="Unsupported page URL")
    return page_path, "_".join(parts[2:-1]), int(parts[-1])


@app.get("/api/capabilities")
async def get_capabilities():
    try:
        profiles = profile_store().load()
        return {
            "renderer": "comfyui",
            "profiles": [{"id": item.id, "label": item.label, "description": item.description} for item in profiles.values()],
            "visionQaEnabled": CONFIG.vision.enabled,
            "publicationFormats": ["paged_comic"],
            "layouts": {str(count): compatible_layout_names(count) for count in range(1, 7)},
        }
    except ProfileError as exc:
        return {
            "renderer": "comfyui", "profiles": [], "profileError": str(exc), "visionQaEnabled": CONFIG.vision.enabled,
            "publicationFormats": ["paged_comic"],
            "layouts": {str(count): compatible_layout_names(count) for count in range(1, 7)},
        }


async def invoke_storyboard_graph(state: dict, request: Request) -> dict:
    """Release a disconnected HTTP request without claiming to stop LM Studio.

    LangGraph and ChatOpenAI run synchronously. The worker thread lets the
    async endpoint observe a browser disconnect, but an already-issued local
    LLM HTTP request cannot be cancelled reliably from that thread.
    """
    task = asyncio.create_task(asyncio.to_thread(
        storyboard_graph.invoke,
        state,
        {"configurable": {"thread_id": f"storyboard_{uuid.uuid4().hex}"}},
    ))
    while not task.done():
        if await request.is_disconnected():
            logger.info("Storyboard request cancelled by client; LM Studio inference may continue.")
            # Retrieve a later exception, if any, without awaiting the work.
            task.add_done_callback(lambda completed: None if completed.cancelled() else completed.exception())
            raise HTTPException(status_code=499, detail="Storyboard request cancelled by client.")
        await asyncio.sleep(0.1)
    return task.result()


@app.post("/api/storyboards")
async def create_storyboard(req: StoryboardRequest, request: Request):
    state = {
        "generation_id": uuid.uuid4().hex, "user_prompt": req.prompt, "requested_language": "",
        "current_schema": {}, "story_plan": {}, "dialogue_plan": {}, "normalization_notes": [], "vision_qa": {},
        "vision_retry_count": 0, "max_vision_retries": 0, "validation_errors": [],
        "next_step": "", "pageCount": req.pageCount, "retry_count": 0,
        "max_retries": CONFIG.story.max_retries, "generation_error": "",
    }
    try:
        final_state = await invoke_storyboard_graph(state, request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Storyboard LLM request failed before structural validation.")
        raise HTTPException(status_code=502, detail=f"Storyboard LLM request failed: {exc}") from exc
    if final_state.get("generation_error"):
        logger.error("Storyboard request rejected: %s", final_state["generation_error"])
        raise HTTPException(status_code=422, detail=final_state["generation_error"])
    schema = final_state.get("current_schema")
    if not isinstance(schema, dict) or not schema.get("pages"):
        logger.error("Storyboard request ended without an approved schema: %s", final_state.get("validation_errors", []))
        raise HTTPException(status_code=422, detail=f"Storyboard was not approved: {final_state.get('validation_errors', [])}")
    return {"storyboard": schema, "storyPlan": final_state.get("story_plan", {}), "dialoguePlan": final_state.get("dialogue_plan", {})}


@app.post("/api/render-storyboard")
async def render_storyboard(req: RenderRequest):
    if not isinstance(req.storyboard, dict):
        raise HTTPException(status_code=422, detail="A storyboard JSON object is required before rendering.")
    try:
        load_comic_from_json(req.storyboard)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid storyboard JSON: {exc}") from exc
    profile = profile_or_422(req.rendererProfile)
    job_id = req.jobId or uuid.uuid4().hex
    try:
        pages, _ = render_comic_schema(
            req.storyboard, job_id, profile, comfy_client(), rendering_settings(req), req.layoutStyle, req.mangaLayout,
            publication_format=req.publicationFormat, layout_mode=req.layoutMode,
            page_layouts=req.pageLayouts, image_fit=req.imageFit,
        )
    except (ComfyUIError, ProfileError, ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"jobId": job_id, "pages": ["/outputs/" + Path(path).relative_to(output_dir).as_posix() for path in pages]}


@app.get("/api/history")
async def get_history():
    return {"history": ["/outputs/" + page.relative_to(output_dir).as_posix() for page in completed_job_pages(output_dir)]}


@app.get("/api/page-meta")
async def get_page_meta(url: str = Query(..., min_length=1)):
    page_path, job_id, page_number = page_job_info(url)
    job_dir = output_dir / "jobs" / job_id
    try:
        schema = json.loads((job_dir / "story.json").read_text(encoding="utf-8"))
        layout_path = job_dir / f"page_layout_{page_number}.json"
        if not layout_path.exists():
            layout_path = job_dir / "page_layout.json"  # Existing rendered jobs.
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail=f"Page metadata not found: {exc}") from exc
    panels = next((page.get("panels", []) for page in schema.get("pages", []) if int(page.get("page_number", 1)) == page_number), [])
    return {"url": "/outputs/" + page_path.relative_to(output_dir.resolve()).as_posix(), "pageWidth": layout.get("page_width", CONFIG.page.width), "pageHeight": layout.get("page_height", CONFIG.page.height), "panels": [
        {"x": pos[0], "y": pos[1], "width": pos[2], "height": pos[3], "id": panel.get("id"), "prompt": panel.get("panel_prompt_en", ""), "dialogues": panel.get("dialogues", [])}
        for pos, panel in zip(layout.get("panels", []), panels) if len(pos) == 4
    ]}


@app.post("/api/regenerate-panel")
async def regenerate_panel(req: RegeneratePanelRequest):
    page_path, job_id, requested_page_number = page_job_info(req.pageUrl)
    job_dir = output_dir / "jobs" / job_id
    schema_path = job_dir / "story.json"
    if not schema_path.exists():
        raise HTTPException(status_code=404, detail="Generation job not found")
    # Keep the approved job untouched until the renderer has produced a valid
    # replacement page in an isolated staging job.
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    updated_schema = json.loads(json.dumps(schema))
    found = False
    affected_page_number = requested_page_number
    for page in updated_schema.get("pages", []):
        for panel in page.get("panels", []):
            if panel.get("id") == req.panelId:
                panel["panel_prompt_en"] = req.panelPrompt
                panel["dialogues"] = req.dialogues
                found = True
                affected_page_number = int(page.get("page_number", requested_page_number))
    if not found:
        raise HTTPException(status_code=404, detail="Panel not found")
    profile = profile_or_422(req.rendererProfile)
    layout_mode = req.layoutMode
    page_layouts = req.pageLayouts
    image_fit = req.imageFit
    # Reuse the original page geometry for legacy editor requests that do not
    # explicitly make a new layout choice.
    stored_layout_path = output_dir / "jobs" / job_id / "layout_config.json"
    if layout_mode == "auto" and not page_layouts and stored_layout_path.exists():
        try:
            stored_layout = json.loads(stored_layout_path.read_text(encoding="utf-8"))
            page_layouts = stored_layout.get("page_layouts", {})
            layout_mode = "manual"
            page_image_fits = stored_layout.get("page_image_fits", {})
            image_fit = page_image_fits.get(str(affected_page_number), stored_layout.get("image_fit", image_fit))
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    staging_job_id = f"{job_id}_staging_{uuid.uuid4().hex}"
    staging_dir = output_dir / "jobs" / staging_job_id
    staged_pages: list[Path] = []
    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
        source_panels = job_dir / "panels"
        if source_panels.exists():
            shutil.copytree(source_panels, staging_dir / "panels")
        else:
            (staging_dir / "panels").mkdir()
        # An empty target set means compositor-only dialogue editing.
        targets = [] if req.mode == "dialogue" else [req.panelId]
        pages, _ = render_comic_schema(
            updated_schema, staging_job_id, profile, comfy_client(), rendering_settings(req), req.layoutStyle, req.mangaLayout, targets,
            publication_format=req.publicationFormat, layout_mode=layout_mode,
            page_layouts=page_layouts, image_fit=image_fit, compose_page_numbers=[affected_page_number],
        )
        staged_pages = [Path(item) for item in pages]
        if len(staged_pages) != 1 or not staged_pages[0].is_file():
            raise RuntimeError(
                f"Expected one completed recomposed page, received {len(staged_pages)}."
            )
        staged_story_path = staging_dir / "story.json"
        staged_sizes_path = staging_dir / "panel_sizes.json"
        staged_page_layout_path = staging_dir / f"page_layout_{affected_page_number}.json"
        staged_layout_path = staging_dir / "layout_config.json"
        if req.mode != "dialogue" and not (staging_dir / "panels" / f"{req.panelId}.png").is_file():
            raise RuntimeError(f"Regenerated panel image was not produced for {req.panelId}.")
        # Read all staged metadata before changing the approved job.  This
        # catches an incomplete renderer result as a clean rollback.
        json.loads(staged_story_path.read_text(encoding="utf-8"))
        staged_sizes = json.loads(staged_sizes_path.read_text(encoding="utf-8"))
        json.loads(staged_page_layout_path.read_text(encoding="utf-8"))
        staged_layout = json.loads(staged_layout_path.read_text(encoding="utf-8"))

        # Commit only the changed page and its panel/story metadata after the
        # staged composition has completed successfully.
        if req.mode != "dialogue":
            source = staging_dir / "panels" / f"{req.panelId}.png"
            destination = job_dir / "panels" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copy2(staged_story_path, schema_path)

        affected_ids = {
            panel.get("id") for page in updated_schema.get("pages", [])
            if int(page.get("page_number", 1)) == affected_page_number
            for panel in page.get("panels", [])
        }
        existing_sizes_path = job_dir / "panel_sizes.json"
        existing_sizes = json.loads(existing_sizes_path.read_text(encoding="utf-8")) if existing_sizes_path.exists() else {}
        existing_sizes.update({panel_id: staged_sizes[panel_id] for panel_id in affected_ids if panel_id in staged_sizes})
        existing_sizes_path.write_text(json.dumps(existing_sizes, indent=2), encoding="utf-8")

        shutil.copy2(staged_page_layout_path, job_dir / f"page_layout_{affected_page_number}.json")
        existing_layout_path = job_dir / "layout_config.json"
        existing_layout = json.loads(existing_layout_path.read_text(encoding="utf-8")) if existing_layout_path.exists() else {}
        existing_layout.setdefault("page_layouts", {}).update({
            str(affected_page_number): staged_layout["page_layouts"][str(affected_page_number)]
        })
        existing_layout.setdefault("page_image_fits", {}).update({
            str(affected_page_number): staged_layout.get("page_image_fits", {}).get(str(affected_page_number), image_fit)
        })
        existing_layout.setdefault("publication_format", staged_layout.get("publication_format", req.publicationFormat))
        existing_layout.setdefault("layout_mode", staged_layout.get("layout_mode", layout_mode))
        existing_layout.setdefault("image_fit", staged_layout.get("image_fit", image_fit))
        existing_layout_path.write_text(json.dumps(existing_layout, indent=2), encoding="utf-8")

        shutil.copy2(staged_pages[0], page_path)
    except (ComfyUIError, ProfileError, ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        for staged_page in staged_pages:
            if staged_page.parent.resolve() == output_dir.resolve() and staged_page.name.startswith(f"comic_page_{staging_job_id}_"):
                staged_page.unlink(missing_ok=True)
    return {"url": "/outputs/" + page_path.relative_to(output_dir.resolve()).as_posix(), "panelId": req.panelId, "mode": req.mode}


@app.post("/api/save-page")
async def save_page(req: SavePageRequest):
    page_path, _, _ = page_job_info(req.pageUrl)
    saved_dir = output_dir / "saved"
    saved_dir.mkdir(parents=True, exist_ok=True)
    name = Path(req.filename).name
    if Path(name).suffix.lower() != ".png":
        name += ".png"
    destination = saved_dir / name
    destination.write_bytes(page_path.read_bytes())
    return {"url": "/outputs/saved/" + destination.name}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
