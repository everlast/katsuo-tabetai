from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from agents import AgentBase, RunContextWrapper, function_tool

from .context import KatsuoContext
from .models import (
    CandidateStore,
    RestaurantCandidateInput,
    TopFiveStore,
)
from .report import render_top_five_html
from .scoring import apply_range_rule, rank_top_five

RECENT_REVIEW_MAX_AGE_DAYS = 365


def _deduplicate(
    candidates: list[RestaurantCandidateInput],
) -> list[RestaurantCandidateInput]:
    unique: dict[str, RestaurantCandidateInput] = {}
    for candidate in candidates:
        key = "".join(candidate.name.casefold().split())
        unique.setdefault(key, candidate)
    return list(unique.values())


def _validate_recent_reviews(
    candidates: list[RestaurantCandidateInput],
    as_of: date,
) -> None:
    oldest_allowed = as_of - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS)
    for candidate in candidates:
        fingerprints: set[tuple[date, str]] = set()
        for review in candidate.recent_reviews:
            if review.published_at > as_of:
                raise ValueError(
                    f"Save rejected: {candidate.name} has a future-dated review "
                    f"({review.published_at.isoformat()})."
                )
            if review.published_at < oldest_allowed:
                raise ValueError(
                    f"Save rejected: {candidate.name} has a review older than "
                    f"{RECENT_REVIEW_MAX_AGE_DAYS} days "
                    f"({review.published_at.isoformat()})."
                )
            fingerprint = (
                review.published_at,
                "".join(review.summary.casefold().split()),
            )
            if fingerprint in fingerprints:
                raise ValueError(
                    f"Save rejected: {candidate.name} contains a duplicate review "
                    f"dated {review.published_at.isoformat()}."
                )
            fingerprints.add(fingerprint)


def persist_restaurant_candidates(
    context: KatsuoContext,
    candidates: list[RestaurantCandidateInput],
) -> dict[str, object]:
    deduplicated = _deduplicate(candidates)
    generated_at = datetime.now(timezone.utc)
    _validate_recent_reviews(deduplicated, generated_at.date())
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
        generated_at=generated_at,
        hotel=context.hotel,
        max_distance_km=context.max_distance_km,
        candidates=ranged,
    )
    context.output_dir.mkdir(parents=True, exist_ok=True)
    context.candidates_path.write_text(
        store.model_dump_json(indent=2),
        encoding="utf-8",
    )
    context.candidates_saved = True
    return {
        "status": "saved",
        "path": str(context.candidates_path),
        "unique_candidates": len(ranged),
        "within_range": eligible_count,
        "outside_range": len(ranged) - eligible_count,
        "next_action": "handoff to the evaluation agent",
    }


def candidate_save_is_enabled(
    wrapper: RunContextWrapper[KatsuoContext],
    _: AgentBase,
) -> bool:
    context = wrapper.context
    return bool(context.pending_candidates) and not context.candidates_saved


@function_tool(
    is_enabled=candidate_save_is_enabled,
    failure_error_function=None,
)
def save_restaurant_candidates(
    wrapper: RunContextWrapper[KatsuoContext],
) -> str:
    """Validate and save the structured restaurant candidates from web research."""
    context = wrapper.context
    result = persist_restaurant_candidates(context, context.pending_candidates)
    context.candidate_save_calls += 1
    return json.dumps(result, ensure_ascii=False)


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


@function_tool(failure_error_function=None)
def evaluate_and_render_top_five(
    wrapper: RunContextWrapper[KatsuoContext],
) -> str:
    """Load saved candidates, score them deterministically, and write TOP 5 HTML."""
    context = wrapper.context
    result = create_top_five_report(context)
    context.evaluation_tool_calls += 1
    return json.dumps(result, ensure_ascii=False)
