"""Persistence of discovered restaurants, candidate stores, and manifests."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path

from .candidates import (
    accumulate_restaurant_candidates,
    candidate_within_range,
    deduplicate_restaurant_candidates,
    insufficient_candidate_pool_message,
    merge_candidate_observations,
    normalize_identity_text,
    partition_candidates_by_review_validity,
    summarize_deduplicated_candidate_pool,
    validate_recent_reviews,
)
from .context import KatsuoContext
from .evidence import sanitize_candidate_claims, scraped_pages_for_candidate
from .models import (
    CandidateStore,
    RestaurantCacheEntry,
    RestaurantCandidateInput,
    ScrapedPage,
    TopFiveStore,
)
from .report import render_context_markdown, render_top_five_html
from .scoring import apply_range_rule, rank_top_five
from .scraping import canonical_url


def write_json_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Serialize an output artifact with the shared JSON conventions."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
                candidate = merge_candidate_observations(
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
    write_json_artifact(
        context.discovered_candidates_path,
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
    )
    return {
        "collected": len(records),
        "evaluation_eligible": evaluation_eligible,
    }


def persist_run_manifest(
    context: KatsuoContext,
    trace_id: str,
    audit: dict[str, object],
) -> None:
    """Write the run manifest with the model, trace, audit, and artifact paths."""
    write_json_artifact(
        context.run_manifest_path,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": context.model,
            "trace_id": trace_id,
            "trace_dashboard": "https://platform.openai.com/traces",
            "audit": audit,
            "artifacts": {
                "discovered_restaurants": str(context.discovered_candidates_path),
                "context_markdown": str(context.context_markdown_path),
                "scrape_manifest": str(context.scrape_manifest_path),
                "candidates_json": str(context.candidates_path),
                "top_five_json": str(context.top_five_path),
                "html": str(context.html_path),
            },
            "collection": {
                "collected": len(context.collected_candidates),
                "evaluation_eligible": len(context.pending_candidates),
            },
        },
    )


def _collect_candidate_pages(
    candidates: list[RestaurantCandidateInput],
    pages: Mapping[str, ScrapedPage],
) -> list[ScrapedPage]:
    """Collect each candidate's scraped pages once, keyed by requested URL."""
    stored_pages: dict[str, ScrapedPage] = {}
    for candidate in candidates:
        for page in scraped_pages_for_candidate(candidate, pages):
            stored_pages[canonical_url(page.requested_url)] = page
    return list(stored_pages.values())


def _write_candidate_artifacts(
    context: KatsuoContext,
    store: CandidateStore,
) -> None:
    """Write the candidate store JSON, Markdown context, and scrape manifest."""
    context.output_dir.mkdir(parents=True, exist_ok=True)
    context.candidates_path.write_text(
        store.model_dump_json(indent=2),
        encoding="utf-8",
    )
    context.context_markdown_path.write_text(
        render_context_markdown(store),
        encoding="utf-8",
    )
    write_json_artifact(
        context.scrape_manifest_path,
        {
            "schema_version": 1,
            "generated_at": store.generated_at.isoformat(),
            "model": context.model,
            "trace_id": context.trace_id,
            "pages": [
                page.model_dump(mode="json", exclude={"content"})
                for page in store.scraped_pages
            ],
        },
    )


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
    validate_recent_reviews(deduplicated, generated_at.date(), context)
    summary = summarize_deduplicated_candidate_pool(context, deduplicated)
    if not summary.is_ready:
        raise ValueError(
            insufficient_candidate_pool_message(summary, context.max_distance_km)
        )

    store = CandidateStore(
        generated_at=generated_at,
        model=context.model,
        trace_id=context.trace_id,
        hotel=context.hotel,
        max_distance_km=context.max_distance_km,
        candidates=summary.candidates,
        scraped_pages=_collect_candidate_pages(deduplicated, context.scraped_pages),
    )
    _write_candidate_artifacts(context, store)
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
