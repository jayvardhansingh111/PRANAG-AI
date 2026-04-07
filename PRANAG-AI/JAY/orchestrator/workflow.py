# orchestrator/workflow.py
# LangGraph state machine that orchestrates the full pipeline with retry logic.
#
# Graph:
#   parse_input → fetch_traits → fetch_research → build_spec
#       → validate_spec ──FAIL──→ fix_spec → validate_spec (retry)
#                      ──PASS──→ END

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
from typing import TypedDict, Optional, List, Dict, Any

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from JAY.shared.models import ParsedPrompt, CropTraitVector, ResearchInsight
from JAY.shared.config import MAX_RETRIES


# ── Pipeline State ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    original_prompt: str
    pipeline_id:     str
    parsed_prompt:   Optional[ParsedPrompt]
    traits:          List[CropTraitVector]
    research:        List[ResearchInsight]
    spec:            Optional[Dict[str, Any]]
    retry_count:     int
    errors:          List[str]
    status:          str


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_parse_input(state: PipelineState) -> dict:
    from JAY.orchestrator.prompt_parser import parse_prompt
    try:
        parsed = parse_prompt(state["original_prompt"])
        return {"parsed_prompt": parsed, "status": "parsed"}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Parse error: {e}"],
                "status": "parse_failed"}


def node_fetch_traits(state: PipelineState) -> dict:
    from JAY.search_engine.similarity_search import search_traits
    parsed = state.get("parsed_prompt")
    if not parsed:
        return {"errors": ["No parsed prompt for trait search"], "status": "error"}
    try:
        query  = f"{parsed.crop_type} {parsed.location_raw} {' '.join(parsed.stress_hints)}"
        if parsed.temp_celsius:
            query += f" {parsed.temp_celsius}°C heat tolerance"
        traits = search_traits(query, top_k=10)
        return {"traits": traits, "status": "traits_fetched"}
    except Exception as e:
        print(f"[WORKFLOW] Trait search failed (non-fatal): {e}")
        return {"traits": [], "errors": state.get("errors", []) + [str(e)]}


def node_fetch_research(state: PipelineState) -> dict:
    from JAY.orchestrator.research_fetcher import fetch_research
    parsed = state.get("parsed_prompt")
    if not parsed:
        return {"research": []}
    try:
        stress = " ".join(parsed.stress_hints) or "general"
        query  = f"{parsed.crop_type} {stress} stress tolerance"
        return {"research": fetch_research(query, max_results=5)}
    except Exception as e:
        print(f"[WORKFLOW] Research fetch failed (non-fatal): {e}")
        return {"research": [], "errors": state.get("errors", []) + [str(e)]}


def node_build_spec(state: PipelineState) -> dict:
    from JAY.orchestrator.spec_builder import build_spec
    try:
        spec = build_spec(
            parsed_prompt   = state["parsed_prompt"],
            traits          = state.get("traits", []),
            research        = state.get("research", []),
            original_prompt = state["original_prompt"],
            pipeline_id     = state["pipeline_id"]
        )
        return {"spec": spec.model_dump(), "status": "spec_built"}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Build error: {e}"],
                "status": "build_failed"}


def node_validate_spec(state: PipelineState) -> dict:
    from JAY.orchestrator.output_validator import validate_spec
    if not state.get("spec"):
        return {"status": "validation_failed"}
    try:
        validated = validate_spec(state["spec"])
        return {"spec": validated, "status": "validated"}
    except Exception as e:
        return {
            "errors": state.get("errors", []) + [f"Validation: {e}"],
            "status": "validation_failed",
            "retry_count": state.get("retry_count", 0) + 1
        }


def node_fix_spec(state: PipelineState) -> dict:
    """Apply safe defaults to repair a spec that failed validation."""
    from datetime import datetime, timezone
    spec = state.get("spec", {})
    if spec:
        if "confidence_score"  not in spec: spec["confidence_score"]  = 0.5
        if "validation_passed" not in spec: spec["validation_passed"] = False
        if "generated_at"      not in spec: spec["generated_at"]      = datetime.now(timezone.utc).isoformat()
    return {"spec": spec, "status": "fix_attempted"}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_validation(state: PipelineState) -> str:
    if state.get("status") == "validated":
        return "success"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "fail"
    return "fix"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("parse_input",    node_parse_input)
    graph.add_node("fetch_traits",   node_fetch_traits)
    graph.add_node("fetch_research", node_fetch_research)
    graph.add_node("build_spec",     node_build_spec)
    graph.add_node("validate_spec",  node_validate_spec)
    graph.add_node("fix_spec",       node_fix_spec)

    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input",    "fetch_traits")
    graph.add_edge("fetch_traits",   "fetch_research")
    graph.add_edge("fetch_research", "build_spec")
    graph.add_edge("build_spec",     "validate_spec")
    graph.add_edge("fix_spec",       "validate_spec")

    graph.add_conditional_edges(
        "validate_spec",
        route_after_validation,
        {"fix": "fix_spec", "success": END, "fail": END}
    )
    return graph.compile()


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_pipeline(prompt: str) -> Dict[str, Any]:
    """Run the full pipeline for a user prompt. Returns final state with spec."""
    initial: PipelineState = {
        "original_prompt": prompt,
        "pipeline_id":     str(uuid.uuid4()),
        "parsed_prompt":   None,
        "traits":          [],
        "research":        [],
        "spec":            None,
        "retry_count":     0,
        "errors":          [],
        "status":          "running"
    }

    if LANGGRAPH_AVAILABLE:
        return build_pipeline().invoke(initial)

    # Fallback: sequential run without retry
    state = dict(initial)
    for node_fn in [node_parse_input, node_fetch_traits, node_fetch_research,
                    node_build_spec, node_validate_spec]:
        state.update(node_fn(state))
    return state


if __name__ == "__main__":
    result = run_pipeline("wheat for Jodhpur at 48°C")
    print(f"Status: {result.get('status')}")
    print(f"Errors: {result.get('errors')}")
