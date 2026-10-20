from typing import TypedDict, List, Any

class StudioState(TypedDict):
    # Stable identifier for one user request. Vision-QA retries reuse the same
    # output directory/page instead of creating another History entry.
    generation_id: str
    user_prompt: str
    requested_language: str
    current_schema: dict
    validation_errors: List[str]
    next_step: str
    output_page: str
    output_pages: List[str]
    layoutStyle: str
    mangaLayout: str
    steps: int
    guidance: float
    lora: str
    negativePrompt: str
    seed: str
    pageCount: int
    retry_count: int
    max_retries: int
    generation_error: str
    story_plan: dict
    dialogue_plan: dict
    # Optional explicit selections and only actual Planner-derived values.
    # Prompt integration is intentionally deferred; these fields establish
    # deterministic provenance without changing creative behavior.
    requested_creative_intent: dict
    inferred_creative_intent: dict
    effective_creative_intent: dict
    creative_intent_sources: dict
    creative_intent_warnings: List[dict]
    # Resolved before graph execution; only deterministic postprocessing may
    # attach it to the final approved storyboard.
    approved_art_direction: dict
    # Safe, deterministic cross-stage ID repairs applied to dialogue only.
    normalization_notes: List[str]
    # Development diagnostics only; never returned as part of the public API.
    storyboard_attempt_diagnostics: List[dict]
    vision_qa: dict
    # Optional creative-review text. It is informational and never controls
    # storyboard retry routing.
    art_director_feedback: str
    vision_retry_count: int
    max_vision_retries: int
