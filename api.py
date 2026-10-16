import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

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
        }
    except ProfileError as exc:
        return {"renderer": "comfyui", "profiles": [], "profileError": str(exc), "visionQaEnabled": CONFIG.vision.enabled}


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
        pages, _ = render_comic_schema(req.storyboard, job_id, profile, comfy_client(), rendering_settings(req), req.layoutStyle, req.mangaLayout)
    except (ComfyUIError, ProfileError, ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"jobId": job_id, "pages": ["/outputs/" + Path(path).relative_to(output_dir).as_posix() for path in pages]}


@app.get("/api/history")
async def get_history():
    files = sorted(((f.stat().st_mtime, f.name) for f in output_dir.glob("comic_page_*.png")), reverse=True)
    return {"history": [f"/outputs/{name}" for _, name in files]}


@app.get("/api/page-meta")
async def get_page_meta(url: str = Query(..., min_length=1)):
    page_path, job_id, page_number = page_job_info(url)
    job_dir = output_dir / "jobs" / job_id
    try:
        schema = json.loads((job_dir / "story.json").read_text(encoding="utf-8"))
        layout = json.loads((job_dir / "page_layout.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail=f"Page metadata not found: {exc}") from exc
    panels = next((page.get("panels", []) for page in schema.get("pages", []) if int(page.get("page_number", 1)) == page_number), [])
    return {"url": "/outputs/" + page_path.relative_to(output_dir.resolve()).as_posix(), "pageWidth": layout.get("page_width", CONFIG.page.width), "pageHeight": layout.get("page_height", CONFIG.page.height), "panels": [
        {"x": pos[0], "y": pos[1], "width": pos[2], "height": pos[3], "id": panel.get("id"), "prompt": panel.get("panel_prompt_en", ""), "dialogues": panel.get("dialogues", [])}
        for pos, panel in zip(layout.get("panels", []), panels) if len(pos) == 4
    ]}


@app.post("/api/regenerate-panel")
async def regenerate_panel(req: RegeneratePanelRequest):
    page_path, job_id, _ = page_job_info(req.pageUrl)
    schema_path = output_dir / "jobs" / job_id / "story.json"
    if not schema_path.exists():
        raise HTTPException(status_code=404, detail="Generation job not found")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    found = False
    for page in schema.get("pages", []):
        for panel in page.get("panels", []):
            if panel.get("id") == req.panelId:
                panel["panel_prompt_en"] = req.panelPrompt
                panel["dialogues"] = req.dialogues
                found = True
    if not found:
        raise HTTPException(status_code=404, detail="Panel not found")
    profile = profile_or_422(req.rendererProfile)
    try:
        # An empty target set means compositor-only dialogue editing.
        targets = [] if req.mode == "dialogue" else [req.panelId]
        render_comic_schema(schema, job_id, profile, comfy_client(), rendering_settings(req), req.layoutStyle, req.mangaLayout, targets)
    except (ComfyUIError, ProfileError, ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
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
