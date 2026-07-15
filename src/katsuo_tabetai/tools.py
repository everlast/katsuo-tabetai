from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from agents import AgentBase, RunContextWrapper, function_tool

from .context import KatsuoContext
from .models import (
    CandidateStore,
    RestaurantCandidate,
    RestaurantCandidateInput,
    TopFiveStore,
)
from .report import render_top_five_html
from .scoring import apply_range_rule, haversine_km, normalized_url_host, rank_top_five

RECENT_REVIEW_MAX_AGE_DAYS = 365
MIN_REVIEW_SOURCE_SITES = 2
MIN_IN_RANGE_CANDIDATES = 5
DUPLICATE_LOCATION_THRESHOLD_KM = 0.05


@dataclass(frozen=True)
class CandidatePoolSummary:
    candidates: list[RestaurantCandidate]
    unique_candidates: int
    within_range: int
    outside_range: int

    @property
    def is_ready(self) -> bool:
        return (
            self.unique_candidates >= MIN_IN_RANGE_CANDIDATES
            and self.within_range >= MIN_IN_RANGE_CANDIDATES
        )


def _normalize_identity_text(value: str) -> str:
    return "".join(value.casefold().split())


def _is_same_restaurant_location(
    existing: RestaurantCandidateInput,
    candidate: RestaurantCandidateInput,
) -> bool:
    if _normalize_identity_text(existing.name) != _normalize_identity_text(
        candidate.name
    ):
        return False
    if _normalize_identity_text(existing.address) == _normalize_identity_text(
        candidate.address
    ):
        return True
    distance = haversine_km(
        existing.latitude,
        existing.longitude,
        candidate.latitude,
        candidate.longitude,
    )
    return distance <= DUPLICATE_LOCATION_THRESHOLD_KM


def deduplicate_restaurant_candidates(
    candidates: list[RestaurantCandidateInput],
) -> list[RestaurantCandidateInput]:
    unique: list[RestaurantCandidateInput] = []
    for candidate in candidates:
        if any(
            _is_same_restaurant_location(existing, candidate) for existing in unique
        ):
            continue
        unique.append(candidate)
    return unique


def _summarize_deduplicated_candidate_pool(
    context: KatsuoContext,
    candidates: list[RestaurantCandidateInput],
) -> CandidatePoolSummary:
    ranged = [
        apply_range_rule(candidate, context.hotel, context.max_distance_km)
        for candidate in candidates
    ]
    within_range = sum(candidate.within_range for candidate in ranged)
    return CandidatePoolSummary(
        candidates=ranged,
        unique_candidates=len(ranged),
        within_range=within_range,
        outside_range=len(ranged) - within_range,
    )


def summarize_candidate_pool(
    context: KatsuoContext,
    candidates: list[RestaurantCandidateInput],
) -> CandidatePoolSummary:
    return _summarize_deduplicated_candidate_pool(
        context,
        deduplicate_restaurant_candidates(candidates),
    )


def insufficient_candidate_pool_message(
    summary: CandidatePoolSummary,
    max_distance_km: float,
) -> str:
    return (
        "Save rejected: provide at least "
        f"{MIN_IN_RANGE_CANDIDATES} unique restaurants inside the "
        f"{max_distance_km:.2f} km range. "
        f"Received {summary.unique_candidates} unique and "
        f"{summary.within_range} in range."
    )


def _validate_recent_reviews(
    candidates: list[RestaurantCandidateInput],
    as_of: date,
) -> None:
    oldest_allowed = as_of - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS)
    for candidate in candidates:
        fingerprints: set[tuple[date, str]] = set()
        review_source_sites = {
            normalized_url_host(review.review_url)
            for review in candidate.recent_reviews
        }
        if len(review_source_sites) < MIN_REVIEW_SOURCE_SITES:
            raise ValueError(
                f"Save rejected: {candidate.name} has reviews from fewer than "
                f"{MIN_REVIEW_SOURCE_SITES} source sites (URL domains)."
            )
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


def partition_candidates_by_review_validity(
    candidates: list[RestaurantCandidateInput],
    as_of: date,
) -> tuple[list[RestaurantCandidateInput], list[str]]:
    """Separate candidates that can be persisted from invalid research output."""
    accepted: list[RestaurantCandidateInput] = []
    rejections: list[str] = []
    for candidate in candidates:
        try:
            _validate_recent_reviews([candidate], as_of)
        except ValueError as exc:
            rejections.append(str(exc))
        else:
            accepted.append(candidate)
    return accepted, rejections


def persist_restaurant_candidates(
    context: KatsuoContext,
    candidates: list[RestaurantCandidateInput],
) -> dict[str, object]:
    deduplicated = deduplicate_restaurant_candidates(candidates)
    generated_at = datetime.now(timezone.utc)
    _validate_recent_reviews(deduplicated, generated_at.date())
    summary = _summarize_deduplicated_candidate_pool(context, deduplicated)
    if not summary.is_ready:
        raise ValueError(
            insufficient_candidate_pool_message(summary, context.max_distance_km)
        )

    store = CandidateStore(
        generated_at=generated_at,
        hotel=context.hotel,
        max_distance_km=context.max_distance_km,
        candidates=summary.candidates,
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
        "unique_candidates": summary.unique_candidates,
        "within_range": summary.within_range,
        "outside_range": summary.outside_range,
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
