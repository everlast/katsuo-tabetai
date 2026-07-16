"""Function tools exposed to the agents.

Deterministic domain logic lives in ``candidates``; artifact persistence
lives in ``persistence``. Their public names are re-exported here so that
existing imports keep working.
"""

from __future__ import annotations

import json

from agents import AgentBase, RunContextWrapper, function_tool

from .candidates import (
    CandidatePoolSummary,
    accumulate_restaurant_candidates,
    candidate_within_range,
    deduplicate_restaurant_candidates,
    insufficient_candidate_pool_message,
    merge_restaurant_candidates,
    normalize_identity_text,
    partition_candidates_by_review_validity,
    summarize_candidate_pool,
    summarize_issue_list,
)
from .config import (
    DUPLICATE_LOCATION_THRESHOLD_KM,
    MIN_IN_RANGE_CANDIDATES,
    MIN_RECENT_REVIEW_COUNT,
    MIN_REVIEW_SOURCE_SITES,
    RECENT_REVIEW_MAX_AGE_DAYS,
)
from .context import KatsuoContext
from .persistence import (
    cache_restaurant_candidates,
    create_top_five_report,
    load_cached_restaurant_candidates,
    persist_discovered_restaurants,
    persist_restaurant_candidates,
    persist_run_manifest,
    write_json_artifact,
)
from .report import render_context_markdown

__all__ = [
    "CandidatePoolSummary",
    "DUPLICATE_LOCATION_THRESHOLD_KM",
    "MIN_IN_RANGE_CANDIDATES",
    "MIN_RECENT_REVIEW_COUNT",
    "MIN_REVIEW_SOURCE_SITES",
    "RECENT_REVIEW_MAX_AGE_DAYS",
    "accumulate_restaurant_candidates",
    "cache_restaurant_candidates",
    "candidate_save_is_enabled",
    "candidate_within_range",
    "create_top_five_report",
    "deduplicate_restaurant_candidates",
    "evaluate_and_render_top_five",
    "insufficient_candidate_pool_message",
    "load_cached_restaurant_candidates",
    "merge_restaurant_candidates",
    "normalize_identity_text",
    "partition_candidates_by_review_validity",
    "persist_discovered_restaurants",
    "persist_restaurant_candidates",
    "persist_run_manifest",
    "render_context_markdown",
    "save_restaurant_candidates",
    "summarize_candidate_pool",
    "summarize_issue_list",
    "write_json_artifact",
]


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


@function_tool(failure_error_function=None)
def evaluate_and_render_top_five(
    wrapper: RunContextWrapper[KatsuoContext],
) -> str:
    """Load saved candidates, score them deterministically, and write TOP 5 HTML."""
    context = wrapper.context
    result = create_top_five_report(context)
    context.evaluation_tool_calls += 1
    return json.dumps(result, ensure_ascii=False)
