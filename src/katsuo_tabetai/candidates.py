"""Deterministic domain logic for the restaurant candidate pool."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta

from .config import (
    DUPLICATE_LOCATION_THRESHOLD_KM,
    MIN_IN_RANGE_CANDIDATES,
    MIN_RECENT_REVIEW_COUNT,
    MIN_REVIEW_SOURCE_SITES,
    RECENT_REVIEW_MAX_AGE_DAYS,
)
from .context import KatsuoContext
from .evidence import (
    partition_reviews_by_reference_validity,
    sanitize_candidate_claims,
    validate_candidate_references,
)
from .models import (
    EVIDENCE_SOURCE_PRIORITY,
    RecentReview,
    RestaurantCandidate,
    RestaurantCandidateInput,
    ScrapedPage,
)
from .scoring import apply_range_rule, haversine_km, normalized_url_host
from .scraping import canonical_url


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


def normalize_identity_text(value: str) -> str:
    """Normalize a name or address into a whitespace-free identity key."""
    return "".join(value.casefold().split())


def summarize_issue_list(issues: list[str], noun: str) -> str:
    """Join the first three issues and note how many more were omitted."""
    summary = "; ".join(issues[:3])
    if len(issues) > 3:
        summary += f"; and {len(issues) - 3} more {noun}"
    return summary


def candidate_within_range(
    context: KatsuoContext,
    candidate: RestaurantCandidateInput,
) -> bool:
    return apply_range_rule(
        candidate,
        context.hotel,
        context.max_distance_km,
    ).within_range


def _is_same_restaurant_location(
    existing: RestaurantCandidateInput,
    candidate: RestaurantCandidateInput,
) -> bool:
    existing_name = normalize_identity_text(existing.name)
    candidate_name = normalize_identity_text(candidate.name)
    names_match = existing_name == candidate_name
    addresses_match = normalize_identity_text(
        existing.address
    ) == normalize_identity_text(candidate.address)
    name_is_qualified_alias = (
        min(len(existing_name), len(candidate_name)) >= 4
        and (existing_name in candidate_name or candidate_name in existing_name)
    )
    if not names_match and not (addresses_match and name_is_qualified_alias):
        return False
    if addresses_match:
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


def _review_fingerprint(review: RecentReview) -> tuple[str, str, date, float]:
    return (
        canonical_url(review.review_url),
        normalize_identity_text(review.reviewer_name),
        review.published_at,
        review.rating,
    )


def merge_candidate_observations(
    existing: RestaurantCandidateInput,
    incoming: RestaurantCandidateInput,
) -> RestaurantCandidateInput:
    reviews: dict[tuple[str, str, date, float], RecentReview] = {}
    for review in [*incoming.recent_reviews, *existing.recent_reviews]:
        reviews.setdefault(_review_fingerprint(review), review)
    recent_reviews = sorted(
        reviews.values(),
        key=lambda review: review.published_at,
        reverse=True,
    )[:10]

    source_urls: dict[str, object] = {}
    for source_url in [*incoming.source_urls, *existing.source_urls]:
        source_urls.setdefault(canonical_url(source_url), source_url)

    def observation_quality(candidate: RestaurantCandidateInput) -> tuple[int, int, int]:
        return (
            len(candidate.recent_reviews),
            len(candidate.source_urls),
            EVIDENCE_SOURCE_PRIORITY[candidate.evidence_source_type],
        )

    # Broad discovery often returns only a location and no reviews. Keep the
    # richer primary observation in that case; let equally rich fresh data win.
    base = (
        incoming
        if observation_quality(incoming) >= observation_quality(existing)
        else existing
    )
    return base.model_copy(
        update={
            "source_urls": list(source_urls.values()),
            "recent_reviews": recent_reviews,
            "has_warayaki": existing.has_warayaki or incoming.has_warayaki,
            "has_shio_tataki": existing.has_shio_tataki
            or incoming.has_shio_tataki,
            "has_seasonal_katsuo": existing.has_seasonal_katsuo
            or incoming.has_seasonal_katsuo,
        }
    )


def accumulate_restaurant_candidates(
    existing_candidates: list[RestaurantCandidateInput],
    incoming_candidates: list[RestaurantCandidateInput],
) -> list[RestaurantCandidateInput]:
    """Accumulate discovery observations without losing earlier review data."""
    accumulated = list(existing_candidates)
    for candidate in incoming_candidates:
        for index, existing in enumerate(accumulated):
            if _is_same_restaurant_location(existing, candidate):
                accumulated[index] = merge_candidate_observations(existing, candidate)
                break
        else:
            accumulated.append(candidate)
    return accumulated


def summarize_deduplicated_candidate_pool(
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
    return summarize_deduplicated_candidate_pool(
        context,
        deduplicate_restaurant_candidates(candidates),
    )


def insufficient_candidate_pool_message(
    summary: CandidatePoolSummary,
    max_distance_km: float,
) -> str:
    return (
        "Evaluation stopped: provide at least "
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
    fingerprints: set[tuple[str, str, date, float]] = set()
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
        fingerprint = _review_fingerprint(review)
        if fingerprint in fingerprints:
            issues.append(
                f"a duplicate review dated {review.published_at.isoformat()}"
            )
            continue
        fingerprints.add(fingerprint)
        valid_reviews.append(review)
    return valid_reviews, issues


def prepare_candidate_for_evaluation(
    candidate: RestaurantCandidateInput,
    as_of: date,
    scraped_pages: Mapping[str, ScrapedPage] | None = None,
) -> tuple[RestaurantCandidateInput, list[str]]:
    """Remove unusable reviews and optional claims before eligibility checks."""
    valid_reviews, issues = _filter_valid_reviews(candidate, as_of)
    prepared = candidate.model_copy(update={"recent_reviews": valid_reviews})
    if scraped_pages is None:
        return prepared, issues

    prepared = sanitize_candidate_claims(prepared, scraped_pages)
    verified_reviews, reference_issues = partition_reviews_by_reference_validity(
        prepared,
        scraped_pages,
    )
    issues.extend(reference_issues)
    return prepared.model_copy(update={"recent_reviews": verified_reviews}), issues


def validate_recent_reviews(
    candidates: list[RestaurantCandidateInput],
    as_of: date,
    context: KatsuoContext,
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
        reference_issues = validate_candidate_references(candidate, context.scraped_pages)
        if reference_issues:
            raise ValueError(f"Save rejected: {candidate.name} has {reference_issues[0]}.")


def partition_candidates_by_review_validity(
    candidates: list[RestaurantCandidateInput],
    as_of: date,
    scraped_pages: Mapping[str, ScrapedPage] | None = None,
) -> tuple[list[RestaurantCandidateInput], list[str]]:
    """Separate candidates that can be persisted from invalid research output."""
    accepted: list[RestaurantCandidateInput] = []
    rejections: list[str] = []
    for candidate in candidates:
        candidate, issues = prepare_candidate_for_evaluation(
            candidate,
            as_of,
            scraped_pages,
        )
        if len(candidate.recent_reviews) < MIN_RECENT_REVIEW_COUNT:
            issue_detail = (
                summarize_issue_list(issues, "issue(s)") or "too few review entries"
            )
            rejections.append(
                f"Evaluation excluded: {candidate.name} has only "
                f"{len(candidate.recent_reviews)} "
                "valid recent reviews after filtering; at least "
                f"{MIN_RECENT_REVIEW_COUNT} required. Removed: {issue_detail}."
            )
            continue
        review_source_sites = {
            normalized_url_host(review.review_url) for review in candidate.recent_reviews
        }
        if len(review_source_sites) < MIN_REVIEW_SOURCE_SITES:
            rejections.append(
                f"Evaluation excluded: {candidate.name} has reviews from fewer than "
                f"{MIN_REVIEW_SOURCE_SITES} source sites (URL domains) after filtering."
            )
            continue
        if scraped_pages is not None:
            reference_issues = validate_candidate_references(candidate, scraped_pages)
            if reference_issues:
                issue_detail = summarize_issue_list(reference_issues, "issue(s)")
                rejections.append(
                    f"Evaluation excluded: {candidate.name} has unverified references: "
                    f"{issue_detail}."
                )
                continue
        accepted.append(candidate)
    return accepted, rejections
