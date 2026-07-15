from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from agents import AgentBase, RunContextWrapper, function_tool

from .context import KatsuoContext
from .models import (
    CandidateStore,
    RecentReview,
    RestaurantCandidate,
    RestaurantCacheEntry,
    RestaurantCandidateInput,
    TopFiveStore,
)
from .report import render_top_five_html
from .scoring import apply_range_rule, haversine_km, normalized_url_host, rank_top_five

RECENT_REVIEW_MAX_AGE_DAYS = 365
MIN_RECENT_REVIEW_COUNT = 5
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


def merge_restaurant_candidates(
    existing_candidates: list[RestaurantCandidateInput],
    incoming_candidates: list[RestaurantCandidateInput],
) -> list[RestaurantCandidateInput]:
    """Merge candidates while letting newer research replace the same location."""
    merged = list(existing_candidates)
    for candidate in incoming_candidates:
        for index, existing in enumerate(merged):
            if _is_same_restaurant_location(existing, candidate):
                merged[index] = candidate
                break
        else:
            merged.append(candidate)
    return merged


def _candidate_cache_filename(candidate: RestaurantCandidateInput) -> str:
    identity = "\0".join(
        (
            _normalize_identity_text(candidate.name),
            _normalize_identity_text(candidate.address),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    normalized_name = unicodedata.normalize("NFKC", candidate.name)
    readable_name = re.sub(r"[^\w-]+", "-", normalized_name).strip("-_")[:48]
    return f"{readable_name or 'restaurant'}-{digest}.json"


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


def _filter_valid_reviews(
    candidate: RestaurantCandidateInput,
    as_of: date,
) -> tuple[list[RecentReview], list[str]]:
    oldest_allowed = as_of - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS)
    fingerprints: set[tuple[date, str]] = set()
    valid_reviews: list[RecentReview] = []
    issues: list[str] = []
    for review in candidate.recent_reviews:
        if review.published_at > as_of:
            issues.append(
                f"a future-dated review ({review.published_at.isoformat()})"
            )
            continue
        if review.published_at < oldest_allowed:
            issues.append(
                f"a review older than {RECENT_REVIEW_MAX_AGE_DAYS} days "
                f"({review.published_at.isoformat()})"
            )
            continue
        fingerprint = (
            review.published_at,
            "".join(review.summary.casefold().split()),
        )
        if fingerprint in fingerprints:
            issues.append(
                f"a duplicate review dated {review.published_at.isoformat()}"
            )
            continue
        fingerprints.add(fingerprint)
        valid_reviews.append(review)
    return valid_reviews, issues


def _validate_recent_reviews(
    candidates: list[RestaurantCandidateInput],
    as_of: date,
) -> None:
    for candidate in candidates:
        valid_reviews, issues = _filter_valid_reviews(candidate, as_of)
        if issues:
            raise ValueError(f"Save rejected: {candidate.name} has {issues[0]}.")
        review_source_sites = {
            normalized_url_host(review.review_url) for review in valid_reviews
        }
        if len(review_source_sites) < MIN_REVIEW_SOURCE_SITES:
            raise ValueError(
                f"Save rejected: {candidate.name} has reviews from fewer than "
                f"{MIN_REVIEW_SOURCE_SITES} source sites (URL domains)."
            )


def partition_candidates_by_review_validity(
    candidates: list[RestaurantCandidateInput],
    as_of: date,
) -> tuple[list[RestaurantCandidateInput], list[str]]:
    """Separate candidates that can be persisted from invalid research output."""
    accepted: list[RestaurantCandidateInput] = []
    rejections: list[str] = []
    for candidate in candidates:
        valid_reviews, issues = _filter_valid_reviews(candidate, as_of)
        if len(valid_reviews) < MIN_RECENT_REVIEW_COUNT:
            issue_detail = "; ".join(issues[:3]) or "too few review entries"
            if len(issues) > 3:
                issue_detail += f"; and {len(issues) - 3} more issue(s)"
            rejections.append(
                f"Save rejected: {candidate.name} has only {len(valid_reviews)} "
                "valid recent reviews after filtering; at least "
                f"{MIN_RECENT_REVIEW_COUNT} required. Removed: {issue_detail}."
            )
            continue
        review_source_sites = {
            normalized_url_host(review.review_url) for review in valid_reviews
        }
        if len(review_source_sites) < MIN_REVIEW_SOURCE_SITES:
            rejections.append(
                f"Save rejected: {candidate.name} has reviews from fewer than "
                f"{MIN_REVIEW_SOURCE_SITES} source sites (URL domains) after filtering."
            )
            continue
        accepted.append(candidate.model_copy(update={"recent_reviews": valid_reviews}))
    return accepted, rejections


def cache_restaurant_candidates(
    context: KatsuoContext,
    candidates: list[RestaurantCandidateInput],
    updated_at: datetime | None = None,
) -> int:
    """Persist one independently reusable JSON file per restaurant."""
    unique_candidates = deduplicate_restaurant_candidates(candidates)
    if not unique_candidates:
        return 0
    context.restaurant_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_timestamp = updated_at or datetime.now(timezone.utc)
    for candidate in unique_candidates:
        entry = RestaurantCacheEntry(
            updated_at=cache_timestamp,
            candidate=candidate,
        )
        cache_path = context.restaurant_cache_dir / _candidate_cache_filename(candidate)
        temporary_path = cache_path.with_suffix(".json.tmp")
        temporary_path.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        temporary_path.replace(cache_path)
    return len(unique_candidates)


def _bootstrap_restaurant_cache(context: KatsuoContext) -> str | None:
    if context.restaurant_cache_dir.exists() and any(
        context.restaurant_cache_dir.glob("*.json")
    ):
        return None
    if not context.candidates_path.exists():
        return None
    try:
        store = CandidateStore.model_validate_json(
            context.candidates_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        return f"Existing aggregate candidate file could not seed cache: {exc}"
    candidates = [
        RestaurantCandidateInput.model_validate(
            candidate.model_dump(exclude={"distance_km", "within_range"})
        )
        for candidate in store.candidates
    ]
    cache_restaurant_candidates(context, candidates, updated_at=store.generated_at)
    return None


def load_cached_restaurant_candidates(
    context: KatsuoContext,
    as_of: date,
) -> tuple[list[RestaurantCandidateInput], list[str]]:
    """Load cached restaurants that remain review-valid and in the current range."""
    rejections: list[str] = []
    bootstrap_error = _bootstrap_restaurant_cache(context)
    if bootstrap_error:
        rejections.append(bootstrap_error)
    if not context.restaurant_cache_dir.exists():
        return [], rejections

    entries: list[RestaurantCacheEntry] = []
    for cache_path in sorted(context.restaurant_cache_dir.glob("*.json")):
        try:
            entries.append(
                RestaurantCacheEntry.model_validate_json(
                    cache_path.read_text(encoding="utf-8")
                )
            )
        except ValueError as exc:
            rejections.append(f"Cache ignored: {cache_path.name} is invalid: {exc}")
    entries.sort(key=lambda entry: entry.updated_at.isoformat(), reverse=True)
    cached_candidates = deduplicate_restaurant_candidates(
        [entry.candidate for entry in entries]
    )
    review_valid, review_rejections = partition_candidates_by_review_validity(
        cached_candidates,
        as_of,
    )
    rejections.extend(review_rejections)
    in_range = [
        candidate
        for candidate in review_valid
        if apply_range_rule(candidate, context.hotel, context.max_distance_km).within_range
    ]
    return in_range, rejections


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
