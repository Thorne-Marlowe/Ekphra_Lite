"""Storyboard-only LangGraph flow.

Rendering is intentionally outside this graph: a user must approve the JSON
storyboard before a ComfyUI job can be submitted.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .agents import run_dialogue_writer, run_story_planner, run_storyboarder, run_validator
from .state import StudioState


def create_storyboard_graph() -> StateGraph:
    workflow = StateGraph(StudioState)
    workflow.add_node("story_planner", run_story_planner)
    workflow.add_node("dialogue_writer", run_dialogue_writer)
    workflow.add_node("storyboarder", run_storyboarder)
    workflow.add_node("validator", run_validator)
    workflow.set_entry_point("story_planner")
    workflow.add_conditional_edges(
        "story_planner",
        lambda state: "failed" if state.get("generation_error") else "dialogue_writer",
        {"dialogue_writer": "dialogue_writer", "failed": END},
    )
    workflow.add_conditional_edges(
        "dialogue_writer",
        lambda state: "failed" if state.get("generation_error") else "storyboarder",
        {"storyboarder": "storyboarder", "failed": END},
    )
    workflow.add_edge("storyboarder", "validator")
    workflow.add_conditional_edges(
        "validator",
        lambda state: state.get("next_step", "failed"),
        {
            "story_planner": "story_planner",
            "dialogue_writer": "dialogue_writer",
            "storyboarder": "storyboarder",
            # The old graph rendered at this point. Approval now ends here.
            "renderer": END,
            "failed": END,
        },
    )
    return workflow.compile(checkpointer=MemorySaver())


# Name-only compatibility alias. It contains neither an image renderer nor
# Vision-QA retry routing.
create_studio_graph = create_storyboard_graph
