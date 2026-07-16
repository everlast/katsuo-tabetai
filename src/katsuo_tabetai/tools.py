from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from agents import AgentBase, RunContextWrapper, function_tool

from .context import KatsuoContext
from .evidence import (
    sanitize_candidate_claims,
    scraped_pages_for_candidate,
    validate_candidate_references,
)
from .models import (
    EVIDENCE_SOURCE_PRIORITY,
    CandidateStore,
    RecentReview,
    RestaurantCandidate,
    RestaurantCacheEntry,
    RestaurantCandidateInput,
    ScrapedPage,
    TopFiveStore,
)
from .report import render_top_five_html
from .scoring import apply_range_rule, haversine_km, normalized_url_host, rank_top_five
from .scraping import canonical_url

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


def _merge_candidate_observations(
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
                accumulated[index] = _merge_candidate_observations(existing, candidate)
                break
        else:
            accumulated.append(candidate)
    return accumulated


def _candidate_cache_filename(candidate: RestaurantCandidateInput) -> str:
    identity = "\0".join(
        (
            normalize_identity_text(candidate.name),
            normalize_identity_text(candidate.address),
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


def _validate_recent_reviews(
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
        valid_reviews, issues = _filter_valid_reviews(candidate, as_of)
        if len(valid_reviews) < MIN_RECENT_REVIEW_COUNT:
            issue_detail = (
                summarize_issue_list(issues, "issue(s)") or "too few review entries"
            )
            rejections.append(
                f"Evaluation excluded: {candidate.name} has only {len(valid_reviews)} "
                "valid recent reviews after filtering; at least "
                f"{MIN_RECENT_REVIEW_COUNT} required. Removed: {issue_detail}."
            )
            continue
        review_source_sites = {
            normalized_url_host(review.review_url) for review in valid_reviews
        }
        if len(review_source_sites) < MIN_REVIEW_SOURCE_SITES:
            rejections.append(
                f"Evaluation excluded: {candidate.name} has reviews from fewer than "
                f"{MIN_REVIEW_SOURCE_SITES} source sites (URL domains) after filtering."
            )
            continue
        if scraped_pages is not None:
            candidate = sanitize_candidate_claims(candidate, scraped_pages)
            reference_issues = validate_candidate_references(candidate, scraped_pages)
            if reference_issues:
                issue_detail = summarize_issue_list(reference_issues, "issue(s)")
                rejections.append(
                    f"Evaluation excluded: {candidate.name} has unverified references: "
                    f"{issue_detail}."
                )
                continue
        accepted.append(candidate.model_copy(update={"recent_reviews": valid_reviews}))
    return accepted, rejections


def cache_restaurant_candidates(
    context: KatsuoContext,
    candidates: list[RestaurantCandidateInput],
    updated_at: datetime | None = None,
) -> int:
    """Persist every discovered restaurant inside the configured radius."""
    discovered_candidates = [
        candidate
        for candidate in deduplicate_restaurant_candidates(candidates)
        if candidate_within_range(context, candidate)
    ]
    if not discovered_candidates:
        return 0
    context.restaurant_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_timestamp = updated_at or datetime.now(timezone.utc)
    for candidate in discovered_candidates:
        cache_path = context.restaurant_cache_dir / _candidate_cache_filename(candidate)
        existing_pages: list[ScrapedPage] = []
        if cache_path.exists():
            try:
                existing_entry = RestaurantCacheEntry.model_validate_json(
                    cache_path.read_text(encoding="utf-8")
                )
            except ValueError:
                pass
            else:
                candidate = _merge_candidate_observations(
                    existing_entry.candidate,
                    candidate,
                )
                existing_pages = existing_entry.scraped_pages

        pages_by_url = {
            canonical_url(page.requested_url): page for page in existing_pages
        }
        for page in scraped_pages_for_candidate(candidate, context.scraped_pages):
            pages_by_url[canonical_url(page.requested_url)] = page
        entry = RestaurantCacheEntry(
            updated_at=cache_timestamp,
            candidate=candidate,
            scraped_pages=list(pages_by_url.values()),
        )
        temporary_path = cache_path.with_suffix(".json.tmp")
        temporary_path.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        temporary_path.replace(cache_path)
    return len(discovered_candidates)


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
    for page in store.scraped_pages:
        context.scraped_pages[canonical_url(page.requested_url)] = page
    cache_restaurant_candidates(context, candidates, updated_at=store.generated_at)
    return None


def load_cached_restaurant_candidates(
    context: KatsuoContext,
    as_of: date,
) -> tuple[list[RestaurantCandidateInput], list[str]]:
    """Load every cached discovery that remains inside the current range."""
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
    entries.sort(key=lambda entry: entry.updated_at.isoformat())
    for entry in entries:
        for page in entry.scraped_pages:
            context.scraped_pages[canonical_url(page.requested_url)] = page
    cached_candidates = accumulate_restaurant_candidates(
        [],
        [entry.candidate for entry in entries],
    )
    in_range = [
        candidate
        for candidate in cached_candidates
        if candidate_within_range(context, candidate)
    ]
    return in_range, rejections


def persist_discovered_restaurants(
    context: KatsuoContext,
    as_of: date,
) -> dict[str, int]:
    """Write the full discovery pool and its separate evaluation status."""
    records: list[dict[str, object]] = []
    evaluation_eligible = 0
    for candidate in deduplicate_restaurant_candidates(context.collected_candidates):
        ranged = apply_range_rule(candidate, context.hotel, context.max_distance_km)
        if not ranged.within_range:
            continue
        accepted, issues = partition_candidates_by_review_validity(
            [candidate],
            as_of,
            context.scraped_pages,
        )
        is_eligible = bool(accepted)
        evaluation_eligible += is_eligible
        records.append(
            {
                "candidate": ranged.model_dump(mode="json"),
                "evaluation_eligible": is_eligible,
                "evaluation_issues": issues,
            }
        )

    context.output_dir.mkdir(parents=True, exist_ok=True)
    context.discovered_candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": context.model,
                "trace_id": context.trace_id,
                "hotel": context.hotel.model_dump(mode="json"),
                "max_distance_km": context.max_distance_km,
                "collected_count": len(records),
                "evaluation_eligible_count": evaluation_eligible,
                "restaurants": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "collected": len(records),
        "evaluation_eligible": evaluation_eligible,
    }


def persist_restaurant_candidates(
    context: KatsuoContext,
    candidates: list[RestaurantCandidateInput],
) -> dict[str, object]:
    deduplicated = deduplicate_restaurant_candidates(
        [
            sanitize_candidate_claims(candidate, context.scraped_pages)
            for candidate in candidates
        ]
    )
    generated_at = datetime.now(timezone.utc)
    _validate_recent_reviews(deduplicated, generated_at.date(), context)
    summary = _summarize_deduplicated_candidate_pool(context, deduplicated)
    if not summary.is_ready:
        raise ValueError(
            insufficient_candidate_pool_message(summary, context.max_distance_km)
        )

    stored_pages: dict[str, ScrapedPage] = {}
    for candidate in deduplicated:
        for page in scraped_pages_for_candidate(candidate, context.scraped_pages):
            stored_pages[canonical_url(page.requested_url)] = page
    store = CandidateStore(
        generated_at=generated_at,
        model=context.model,
        trace_id=context.trace_id,
        hotel=context.hotel,
        max_distance_km=context.max_distance_km,
        candidates=summary.candidates,
        scraped_pages=list(stored_pages.values()),
    )
    context.output_dir.mkdir(parents=True, exist_ok=True)
    context.candidates_path.write_text(
        store.model_dump_json(indent=2),
        encoding="utf-8",
    )
    context.context_markdown_path.write_text(
        render_context_markdown(store),
        encoding="utf-8",
    )
    context.scrape_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": generated_at.isoformat(),
                "model": context.model,
                "trace_id": context.trace_id,
                "pages": [
                    page.model_dump(mode="json", exclude={"content"})
                    for page in store.scraped_pages
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    context.candidates_saved = True
    return {
        "status": "saved",
        "path": str(context.candidates_path),
        "context_markdown": str(context.context_markdown_path),
        "scrape_manifest": str(context.scrape_manifest_path),
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
        model=candidates.model,
        trace_id=candidates.trace_id,
        context_markdown=candidates.context_markdown,
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


def render_context_markdown(store: CandidateStore) -> str:
    lines = [
        "# Katsuo Restaurant Context",
        "",
        f"- Generated at: `{store.generated_at.isoformat()}`",
        f"- Model: `{store.model}`",
        f"- Trace ID: `{store.trace_id}`",
        f"- Hotel: {store.hotel.name}",
        f"- Maximum straight-line distance: {store.max_distance_km:.2f} km",
        "",
        "## Verified Restaurant Candidates",
        "",
    ]
    for index, candidate in enumerate(store.candidates, start=1):
        feature_labels = ["katsuo dish"]
        if candidate.has_warayaki:
            feature_labels.append("warayaki")
        if candidate.has_shio_tataki:
            feature_labels.append("shio tataki")
        if candidate.has_seasonal_katsuo:
            feature_labels.append("seasonal katsuo")
        lines.extend(
            [
                f"{index}. **{candidate.name}**",
                f"   - Address: {candidate.address}",
                f"   - Coordinates: {candidate.latitude}, {candidate.longitude}",
                f"   - Distance: {candidate.distance_km:.2f} km",
                f"   - Katsuo dish: {candidate.katsuo_dish}",
                f"   - Verified features: {', '.join(feature_labels)}",
                f"   - Evidence: [{candidate.evidence_url}]({candidate.evidence_url})",
                "   - Additional verified sources:",
                *(
                    f"     - [{source_url}]({source_url})"
                    for source_url in candidate.source_urls
                ),
                "   - Verified reviews:",
            ]
        )
        for review in candidate.recent_reviews:
            lines.append(
                "     - "
                f"{review.published_at.isoformat()} | {review.rating:g}/5 | "
                f"{review.reviewer_name} | {review.source_name} | "
                f"[{review.review_url}]({review.review_url}) | {review.summary}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
