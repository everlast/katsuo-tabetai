from __future__ import annotations

import json
from datetime import datetime, timezone

from agents import RunContextWrapper, function_tool

from .context import KatsuoContext
from .models import (
    CandidateStore,
    RestaurantCandidateInput,
    TopFiveStore,
)
from .report import render_top_five_html
from .scoring import apply_range_rule, rank_top_five


def _deduplicate(
    candidates: list[RestaurantCandidateInput],
) -> list[RestaurantCandidateInput]:
    unique: dict[str, RestaurantCandidateInput] = {}
    for candidate in candidates:
        key = "".join(candidate.name.casefold().split())
        unique.setdefault(key, candidate)
    return list(unique.values())


def persist_restaurant_candidates(
    context: KatsuoContext,
    candidates: list[RestaurantCandidateInput],
) -> dict[str, object]:
    deduplicated = _deduplicate(candidates)
    ranged = [
        apply_range_rule(candidate, context.hotel, context.max_distance_km)
        for candidate in deduplicated
    ]
    eligible_count = sum(candidate.within_range for candidate in ranged)
    if len(ranged) < 5 or eligible_count < 5:
        raise ValueError(
            "Save rejected: provide at least 5 unique restaurants inside the "
            f"{context.max_distance_km:.2f} km range. "
            f"Received {len(ranged)} unique and {eligible_count} in range."
        )

    store = CandidateStore(
        generated_at=datetime.now(timezone.utc),
        hotel=context.hotel,
        max_distance_km=context.max_distance_km,
        candidates=ranged,
    )
    context.output_dir.mkdir(parents=True, exist_ok=True)
    context.candidates_path.write_text(
        store.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return {
        "status": "saved",
        "path": str(context.candidates_path),
        "unique_candidates": len(ranged),
        "within_range": eligible_count,
        "outside_range": len(ranged) - eligible_count,
        "next_action": "handoff to the evaluation agent",
    }


@function_tool
def save_restaurant_candidates(
    wrapper: RunContextWrapper[KatsuoContext],
    candidates: list[RestaurantCandidateInput],
) -> str:
    """Validate and save researched restaurant candidates as structured JSON.

    Args:
        candidates: Restaurants found by web search. Every entry must have a page
            that explicitly names its katsuo dish and map coordinates.
    """
    context = wrapper.context
    context.candidate_save_calls += 1
    return json.dumps(
        persist_restaurant_candidates(context, candidates),
        ensure_ascii=False,
    )


def create_top_five_report(context: KatsuoContext) -> dict[str, object]:
    if not context.candidates_path.exists():
        raise FileNotFoundError(
            "Candidates have not been saved. The research agent must call "
            "save_restaurant_candidates before handoff."
        )

    candidates = CandidateStore.model_validate_json(
        context.candidates_path.read_text(encoding="utf-8")
    )
    ranked = rank_top_five(candidates.candidates, candidates.max_distance_km)
    report = TopFiveStore(
        generated_at=datetime.now(timezone.utc),
        hotel=candidates.hotel,
        max_distance_km=candidates.max_distance_km,
        restaurants=ranked,
    )
    context.top_five_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    render_top_five_html(report, context.html_path)
    return {
        "status": "completed",
        "top_five_json": str(context.top_five_path),
        "html": str(context.html_path),
        "restaurant_names": [item.name for item in ranked],
    }


@function_tool
def evaluate_and_render_top_five(
    wrapper: RunContextWrapper[KatsuoContext],
) -> str:
    """Load saved candidates, score them deterministically, and write TOP 5 HTML."""
    context = wrapper.context
    context.evaluation_tool_calls += 1
    return json.dumps(create_top_five_report(context), ensure_ascii=False)
