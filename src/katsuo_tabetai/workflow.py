from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from agents import (
    Agent,
    ModelSettings,
    ModelBehaviorError,
    RunContextWrapper,
    Runner,
    WebSearchTool,
    gen_trace_id,
    handoff,
    trace,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from pydantic import BaseModel, Field

from .candidates import (
    CandidatePoolSummary,
    accumulate_restaurant_candidates,
    candidate_within_range,
    deduplicate_restaurant_candidates,
    insufficient_candidate_pool_message,
    merge_restaurant_candidates,
    normalize_identity_text,
    partition_candidates_by_review_validity,
    prepare_candidate_for_evaluation,
    summarize_candidate_pool,
    summarize_issue_list,
)
from .config import (
    DEFAULT_DISCOVERY_ATTEMPTS,
    DEFAULT_MODEL,
    DEFAULT_REVIEW_ENRICHMENT_ATTEMPTS,
    DISCOVERY_TARGET_IN_RANGE_CANDIDATES,
    MIN_IN_RANGE_CANDIDATES,
    MIN_RECENT_REVIEW_COUNT,
    MIN_REVIEW_SOURCE_SITES,
    RECENT_REVIEW_MAX_AGE_DAYS,
    TARGET_KATSUO_EVIDENCE_DOMAINS,
)
from .context import KatsuoContext
from .models import (
    EVIDENCE_SOURCE_PRIORITY,
    ResearchBatch,
    RestaurantCandidateInput,
)
from .persistence import (
    cache_restaurant_candidates,
    load_cached_restaurant_candidates,
    persist_discovered_restaurants,
    persist_run_manifest,
)
from .scoring import INDEPENDENT_SOURCE_MAX_DOMAINS, normalized_url_host
from .tools import evaluate_and_render_top_five, save_restaurant_candidates
from .scraping import scrape_reference_page

PROGRESS_HEARTBEAT_SECONDS = 15.0


class EvaluationHandoffInput(BaseModel):
    research_summary: str = Field(
        min_length=1,
        description="A short summary of the completed search and saved candidate count.",
    )


class NoValidResearchCandidatesError(RuntimeError):
    """Raised when research leaves no enabled action for the storage phase."""


class InsufficientResearchCandidatesError(RuntimeError):
    """Raised when research did not find enough unique in-range restaurants."""


class InvalidResearchOutputError(RuntimeError):
    """Raised when web research repeatedly returns malformed structured output."""


class RunAudit(BaseModel):
    runner_calls: int
    web_search_calls: int
    rejected_research_candidates: int
    function_tool_calls: int
    handoff_items: int
    candidate_save_calls: int
    evaluation_tool_calls: int
    handoff_callbacks: int
    cached_candidates_loaded: int
    cached_candidates_written: int
    scrape_tool_calls: int
    last_agent: str


@dataclass
class WorkflowOutcome:
    final_output: Any
    last_agent: str
    model: str
    trace_id: str
    audit: RunAudit


@dataclass(frozen=True)
class WorkflowAgents:
    web_researcher: Agent[KatsuoContext]
    researcher: Agent[KatsuoContext]
    evaluator: Agent[KatsuoContext]


def _emit_progress(context: KatsuoContext, message: str) -> None:
    if context.progress_callback is not None:
        context.progress_callback(message)


async def _await_with_progress(
    awaitable: Awaitable[Any],
    context: KatsuoContext,
    label: str,
) -> Any:
    task = asyncio.ensure_future(awaitable)
    started_at = time.monotonic()
    try:
        while not task.done():
            done, _ = await asyncio.wait(
                {task},
                timeout=PROGRESS_HEARTBEAT_SECONDS,
            )
            if task in done:
                break
            elapsed = round(time.monotonic() - started_at)
            _emit_progress(context, f"{label} 実行中（{elapsed}秒経過）")
        return await task
    except BaseException:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise


def _raw_field(raw_item: Any, field: str) -> Any:
    if isinstance(raw_item, dict):
        return raw_item.get(field)
    return getattr(raw_item, field, None)


@dataclass(frozen=True)
class _RunItemCounts:
    web_search_calls: int
    function_tool_calls: int
    scrape_tool_calls: int
    handoff_items: int


def _count_run_items(run_results: tuple[Any, ...]) -> _RunItemCounts:
    """Count SDK run items relevant to the acceptance checks."""
    web_search_calls = 0
    function_tool_calls = 0
    scrape_tool_calls = 0
    handoff_items = 0

    for run_result in run_results:
        for item in run_result.new_items:
            item_type = getattr(item, "type", "")
            raw_item = getattr(item, "raw_item", None)
            raw_type = _raw_field(raw_item, "type")
            raw_name = _raw_field(raw_item, "name")

            if raw_type == "web_search_call":
                web_search_calls += 1
            if raw_type in {"function_call", "function_call_output"} and raw_name in {
                "save_restaurant_candidates",
                "evaluate_and_render_top_five",
            }:
                function_tool_calls += 1
            if raw_type in {"function_call", "function_call_output"} and raw_name == (
                "scrape_reference_page"
            ):
                scrape_tool_calls += 1
            if item_type in {"handoff_call_item", "handoff_output_item"}:
                handoff_items += 1

    return _RunItemCounts(
        web_search_calls=web_search_calls,
        function_tool_calls=function_tool_calls,
        scrape_tool_calls=scrape_tool_calls,
        handoff_items=handoff_items,
    )


def _verify_acceptance_checks(audit: RunAudit, context: KatsuoContext) -> None:
    if audit.web_search_calls < 1:
        raise RuntimeError(
            "Acceptance check failed: no WebSearchTool call was recorded."
        )
    if audit.scrape_tool_calls < 1 or context.scrape_calls < 1:
        raise RuntimeError(
            "Acceptance check failed: no successful scrape_reference_page call was recorded."
        )
    if (
        audit.function_tool_calls < 2
        or audit.candidate_save_calls < 1
        or audit.evaluation_tool_calls < 1
    ):
        raise RuntimeError(
            "Acceptance check failed: custom function tools did not run."
        )
    if audit.handoff_callbacks < 1 or audit.handoff_items < 1:
        raise RuntimeError("Acceptance check failed: no actual handoff was recorded.")


def audit_run_items(
    result: Any,
    context: KatsuoContext,
    prior_results: tuple[Any, ...] = (),
) -> RunAudit:
    run_results = (*prior_results, result)
    counts = _count_run_items(run_results)
    audit = RunAudit(
        runner_calls=len(run_results),
        web_search_calls=counts.web_search_calls,
        rejected_research_candidates=len(context.candidate_rejections),
        function_tool_calls=counts.function_tool_calls,
        handoff_items=counts.handoff_items,
        candidate_save_calls=context.candidate_save_calls,
        evaluation_tool_calls=context.evaluation_tool_calls,
        handoff_callbacks=context.handoff_calls,
        cached_candidates_loaded=context.cached_candidates_loaded,
        cached_candidates_written=context.cached_candidates_written,
        scrape_tool_calls=counts.scrape_tool_calls,
        last_agent=result.last_agent.name,
    )
    _verify_acceptance_checks(audit, context)
    return audit


def candidates_are_saved(context: KatsuoContext) -> bool:
    return context.candidates_saved and context.candidates_path.exists()


def evaluation_handoff_is_enabled(
    wrapper: RunContextWrapper[KatsuoContext],
    _: Agent[KatsuoContext],
) -> bool:
    return candidates_are_saved(wrapper.context)


async def on_evaluation_handoff(
    wrapper: RunContextWrapper[KatsuoContext],
    input_data: EvaluationHandoffInput,
) -> None:
    if not candidates_are_saved(wrapper.context):
        raise RuntimeError(
            "Handoff rejected: save_restaurant_candidates must run first."
        )
    wrapper.context.handoff_calls += 1


def _model_override(model: str) -> dict[str, str]:
    return {"model": model}


def _format_rejection_detail(rejections: list[str]) -> str:
    rejection_summary = summarize_issue_list(rejections, "rejection(s)")
    return f" Rejections: {rejection_summary}" if rejection_summary else ""


def _format_candidate_detail(summary: CandidatePoolSummary) -> str:
    detail = "\n".join(
        (
            f"- {candidate.name} / {candidate.address} / "
            f"{candidate.latitude:.7f}, {candidate.longitude:.7f} / "
            f"{candidate.distance_km:.2f} km / "
            f"{'IN RANGE' if candidate.within_range else 'OUTSIDE RANGE'}"
        )
        for candidate in summary.candidates[:50]
    )
    return detail or "- none"


def _build_discovery_prompt(
    base_prompt: str,
    context: KatsuoContext,
) -> str:
    collected_summary = summarize_candidate_pool(
        context,
        context.collected_candidates,
    )
    return f"""
{base_prompt}

RESEARCH MODE: DISCOVERY
範囲内でカツオ料理を提供する店舗を網羅的に発見する回です。
現在 {collected_summary.within_range} 店を収集済みです。食べログ、ホットペッパーグルメ、
Google Maps、Yahoo!マップ、Rettyなどの口コミサイトにある個別店舗ページを起点に、
新規候補と実在する個別口コミを探してください。口コミ取得の見込みがある未収集店舗を
優先し、その後に店舗公式・観光公式ページで店名、住所、カツオ料理、料理特徴を補完して
ください。公式ページだけで見つけた店より、検証可能な口コミを持つ店を優先します。
新規候補が5店未満なら、出力スキーマを満たすため収集済み候補も再提出してください。
口コミが5件揃わない店舗も省略せず、確認できた分または空配列で返してください。
各候補について、主根拠を含めて最低{TARGET_KATSUO_EVIDENCE_DOMAINS}つの独立ドメインで
カツオ料理を確認してください。主根拠と異なるドメインの店舗公式メニュー、観光公式、
予約サイトの料理ページをsource_urlsへ追加し、最大{INDEPENDENT_SOURCE_MAX_DOMAINS}ドメイン
まで見つかる範囲で収集してください。
範囲内{DISCOVERY_TARGET_IN_RANGE_CANDIDATES}店は最低収集目標です。すでに到達済みでも、
この店舗発見回では前回と異なる検索語・口コミサイト・エリア名を使って未収集店舗を
探し続けてください。

Collected candidates:
{_format_candidate_detail(collected_summary)}
""".strip()


def _enrichment_target_key(
    candidate: RestaurantCandidateInput,
) -> tuple[str, str]:
    return (
        normalize_identity_text(candidate.name),
        normalize_identity_text(candidate.address),
    )


def _katsuo_evidence_domains(candidate: RestaurantCandidateInput) -> set[str]:
    return {
        normalized_url_host(candidate.evidence_url),
        *(normalized_url_host(url) for url in candidate.source_urls),
    }


def _select_enrichment_targets(
    context: KatsuoContext,
    as_of: date,
    limit: int = MIN_IN_RANGE_CANDIDATES,
    target_attempts: dict[tuple[str, str], int] | None = None,
) -> list[RestaurantCandidateInput]:
    target_attempts = target_attempts or {}
    oldest_allowed = as_of - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS)
    eligible_keys = {
        _enrichment_target_key(candidate) for candidate in context.pending_candidates
    }
    def priority(
        candidate: RestaurantCandidateInput,
    ) -> tuple[int, int, int, int, int, str]:
        recent_reviews = {
            (
                str(review.review_url),
                review.reviewer_name.casefold(),
                review.published_at,
                review.rating,
            )
            for review in candidate.recent_reviews
            if oldest_allowed <= review.published_at <= as_of
        }
        feature_count = sum(
            (
                candidate.has_warayaki,
                candidate.has_shio_tataki,
                candidate.has_seasonal_katsuo,
            )
        )
        return (
            target_attempts.get(_enrichment_target_key(candidate), 0),
            -len(recent_reviews),
            -len(candidate.source_urls),
            -EVIDENCE_SOURCE_PRIORITY[candidate.evidence_source_type],
            -feature_count,
            candidate.name,
        )

    prepared_candidates = [
        prepare_candidate_for_evaluation(
            candidate,
            as_of,
            context.scraped_pages,
        )[0]
        for candidate in context.collected_candidates
    ]
    ineligible = [
        candidate
        for candidate in prepared_candidates
        if _enrichment_target_key(candidate) not in eligible_keys
    ]
    eligible = [
        candidate
        for candidate in prepared_candidates
        if _enrichment_target_key(candidate) in eligible_keys
    ]
    ineligible.sort(key=priority)
    eligible.sort(key=priority)
    evidence_incomplete = [
        candidate
        for candidate in eligible
        if len(_katsuo_evidence_domains(candidate))
        < TARGET_KATSUO_EVIDENCE_DOMAINS
    ]
    evidence_complete = [
        candidate for candidate in eligible if candidate not in evidence_incomplete
    ]
    if len(context.pending_candidates) >= MIN_IN_RANGE_CANDIDATES:
        return [*evidence_incomplete, *ineligible, *evidence_complete][:limit]
    return [*ineligible, *evidence_incomplete, *evidence_complete][:limit]


def _format_enrichment_targets(
    targets: list[RestaurantCandidateInput],
) -> str:
    target_details: list[str] = []
    for candidate in targets:
        review_domains = {
            normalized_url_host(review.review_url)
            for review in candidate.recent_reviews
        }
        evidence_domains = _katsuo_evidence_domains(candidate)
        target_details.append(
            (
                f"- {candidate.name} / {candidate.address} / "
                f"{candidate.latitude:.7f}, {candidate.longitude:.7f} / "
                f"dish={candidate.katsuo_dish} / evidence={candidate.evidence_url} / "
                f"current_reviews={len(candidate.recent_reviews)} / "
                "current_review_domains="
                f"{','.join(sorted(review_domains)) or 'none'} / "
                "missing_reviews="
                f"{max(0, MIN_RECENT_REVIEW_COUNT - len(candidate.recent_reviews))} / "
                "missing_review_domains="
                f"{max(0, MIN_REVIEW_SOURCE_SITES - len(review_domains))} / "
                "current_evidence_domains="
                f"{','.join(sorted(evidence_domains))} / "
                "missing_evidence_domains="
                f"{max(0, TARGET_KATSUO_EVIDENCE_DOMAINS - len(evidence_domains))}"
            )
        )
        target_details.extend(
            f"  existing-evidence: {source_url}"
            for source_url in [candidate.evidence_url, *candidate.source_urls]
        )
        target_details.extend(
            (
                "  existing-review: "
                f"{review.review_url} | {review.reviewer_name} | "
                f"{review.published_at.isoformat()} | {review.rating:.1f}"
            )
            for review in candidate.recent_reviews
        )
    return "\n".join(target_details)


def _build_enrichment_prompt(
    base_prompt: str,
    context: KatsuoContext,
    summary: CandidatePoolSummary,
    as_of: date,
    targets: list[RestaurantCandidateInput] | None = None,
) -> str:
    missing_in_range = max(0, MIN_IN_RANGE_CANDIDATES - summary.within_range)
    collected_summary = summarize_candidate_pool(
        context,
        context.collected_candidates,
    )
    targets = targets or _select_enrichment_targets(context, as_of)
    rejection_detail = "\n".join(
        f"- {rejection}" for rejection in context.candidate_rejections[:10]
    )
    if not rejection_detail:
        rejection_detail = "- none"
    return f"""
{base_prompt}

RESEARCH MODE: ENRICHMENT
前回までの調査では評価条件を満たす店舗が不足しています。
範囲内の収集済み候補は {collected_summary.within_range} 店、
評価可能候補は {summary.within_range} 店で、あと {missing_in_range} 店必要です。
この回は下記5店舗だけを調査してください。existing-reviewはすでに保存済みなので、
同じ口コミを再提出せず、missing_reviewsとmissing_review_domainsを満たす不足分だけを
recent_reviewsへ入れてください。特にmissing_review_domainsが1なら、
current_review_domainsにない
口コミサイトを最初に検索してください。食べログ、ホットペッパーグルメ、Google Maps、
Yahoo!マップ、Rettyなどの個別店舗・個別口コミページを起点にし、店舗公式・観光公式は
店名、住所、カツオ料理、料理特徴の補完に使ってください。新しい店舗や対象外の店舗は
返さないでください。existing-reviewには検証済み口コミだけを載せているため、キャッシュ
から除外された無効口コミは再利用せず、missing_reviews分の別口コミを探してください。
さらに、missing_evidence_domainsが1以上なら、current_evidence_domainsにないドメインから、
店名、支店または住所、カツオ料理名を同じページで確認できる店舗公式メニュー、観光公式、
予約サイトの料理ページを探してsource_urlsへ入れてください。既存根拠を再提出しても
ドメイン数は増えません。最低{TARGET_KATSUO_EVIDENCE_DOMAINS}ドメインを満たした後も、
見つかる場合は最大{INDEPENDENT_SOURCE_MAX_DOMAINS}ドメインまで収集してください。
各口コミページを必ずscrape_reference_pageで取得し、投稿者名、公開日または訪問月、
5点評価を本文上で照合してください。見つからない口コミを推測して補わず、対象店舗
自体はrecent_reviewsを空配列にしてでも必ず返してください。既存分と新規分はコードで
重複排除して累積します。

Enrichment targets (return each exactly once):
{_format_enrichment_targets(targets)}

Previously evaluation-eligible candidates:
{_format_candidate_detail(summary)}

All collected candidates:
{_format_candidate_detail(collected_summary)}

Rejected candidates:
{rejection_detail}
""".strip()


def _build_invalid_output_retry_prompt(base_prompt: str, attempt: int) -> str:
    return f"""
{base_prompt}

前回のWeb調査出力はResearchBatchとして解釈できない壊れたJSONでした。
再調査し、必ずスキーマに合う構造化出力だけを返してください。
特に文字列の引用符、配列、オブジェクトの閉じ括弧を壊さないでください。
これは構造化出力の再試行 {attempt} 回目です。
""".strip()


def _format_invalid_research_output_error(
    attempts: int,
    error: ModelBehaviorError,
) -> str:
    message = str(error).splitlines()[0]
    if len(message) > 240:
        message = f"{message[:237]}..."
    return (
        "Web research returned malformed structured JSON "
        f"after {attempts} attempt(s). Last error: {message}"
    )


def build_evaluator(model: str = DEFAULT_MODEL) -> Agent[KatsuoContext]:
    return Agent[KatsuoContext](
        name="Katsuo Evaluation Agent",
        handoff_description="Deterministically scores saved candidates and renders TOP 5.",
        instructions=(
            "You are the final evaluation agent. Immediately call "
            "evaluate_and_render_top_five exactly once. Do not recalculate scores, "
            "change rankings, or write prose before the tool call."
        ),
        tools=[evaluate_and_render_top_five],
        model_settings=ModelSettings(tool_choice="required"),
        tool_use_behavior="stop_on_first_tool",
        **_model_override(model),
    )


def build_researcher(
    evaluator: Agent[KatsuoContext],
    model: str = DEFAULT_MODEL,
) -> Agent[KatsuoContext]:
    evaluation_handoff = handoff(
        agent=evaluator,
        on_handoff=on_evaluation_handoff,
        input_type=EvaluationHandoffInput,
        tool_name_override="transfer_to_katsuo_evaluation",
        tool_description_override=(
            "Transfer control after web research and structured candidate storage are complete."
        ),
        is_enabled=evaluation_handoff_is_enabled,
    )
    researcher = Agent[KatsuoContext](
        name="Katsuo Research Agent",
        instructions=prompt_with_handoff_instructions(
            """
Required workflow, in this exact order:
1. The completed structured web research is already stored in the run context.
2. Immediately call save_restaurant_candidates exactly once.
3. After the save tool succeeds, call transfer_to_katsuo_evaluation. Do not
   produce a final answer yourself and do not use an agent-as-tool pattern.
""".strip()
        ),
        tools=[save_restaurant_candidates],
        handoffs=[evaluation_handoff],
        model_settings=ModelSettings(tool_choice="required"),
        reset_tool_choice=False,
        **_model_override(model),
    )
    return researcher


def build_web_researcher(model: str = DEFAULT_MODEL) -> Agent[KatsuoContext]:
    return Agent[KatsuoContext](
        name="Katsuo Web Research Agent",
        instructions=f"""
You research restaurants serving excellent katsuo near the configured hotel in Kochi, Japan.

Required workflow:
1. Read RESEARCH MODE from the user input. DISCOVERY finds restaurants broadly;
   ENRICHMENT researches only the five explicitly listed targets. Never mix modes.
2. Use WebSearchTool to start from current individual restaurant and review pages
   on review platforms such as Tabelog, Hot Pepper Gourmet, Google Maps, Yahoo!
   Maps, and Retty. Use these pages to identify candidates with verifiable review
   coverage. Then use official restaurant and official tourism pages to complement
   the candidate's name, address, katsuo dish, and dish features. Review platforms
   drive candidate discovery; official sources strengthen dish evidence and never
   count as reviews. Use more searches if evidence is weak.
3. In DISCOVERY mode, discover every distinct restaurant serving katsuo inside the
   configured hotel radius. Aim for at least 20 candidates and return up to 30;
   prioritize candidates with verifiable review coverage instead of collecting
   official-page-only candidates. In ENRICHMENT mode, return only the five listed
   targets and focus on obtaining exactly five verifiable reviews per target from
   at least two domains. In both modes, check coordinates against
   a map or official location page and never invent a URL, dish, address, coordinate,
   or review. Do not omit a target merely because its reviews are incomplete.
4. Call scrape_reference_page for every evidence_url and for each source_urls or
   review_url entry you include. Every evidence or additional source page must name
   the restaurant, its address, and its katsuo dish. Set a katsuo feature flag to
   true only when one of those pages explicitly supports that feature. Prefer
   official restaurant pages, then official tourism pages, reservation sites,
   and lastly review sites. source_urls are only for additional katsuo dish
   evidence; never put review-list, review-detail, or map pages in source_urls.
   For each candidate, collect katsuo dish evidence from at least
   {TARGET_KATSUO_EVIDENCE_DOMAINS} independent domains including evidence_url,
   and up to {INDEPENDENT_SOURCE_MAX_DOMAINS} domains when readily available.
5. Try to collect distinct reviews from at least two independent review platforms
   for every candidate: exactly five in ENRICHMENT mode and up to ten when readily
   available in DISCOVERY mode. When that is not possible, return the restaurant
   anyway with the smaller verified review set or an empty recent_reviews list.
   A ranking-eligible candidate needs reviews from at least two distinct domains.
   Never invent reviews to reach five. Never use a platform root such as
   google.com/maps, tabelog.com, or retty.me. A review
   page must name the restaurant and display the reviewer's name, date or visit
   month, and exact rating near each other. Store the displayed reviewer in
   reviewer_name. When only YYYY-MM is displayed, store YYYY-MM-01 in published_at
   for recency calculations. Never infer a reviewer, year, month, or rating. Paraphrase each
   review in under 500 characters and record 1 to 3 praised aspects as natural
   Japanese phrases of 2 to 30 characters. Include cautions the reviewer actually
   mentioned. Point arrays must contain only the point text, such as
   "藁焼きの香りが良い". Never include field names, JSON syntax, character-count
   notes, or instruction text in a point. Do not copy review text.
6. Return every candidate required by the active mode as structured output. Review
   and evidence completeness is evaluated later in code. Do not rank candidates.
""".strip(),
        tools=[
            WebSearchTool(
                user_location={
                    "type": "approximate",
                    "country": "JP",
                    "city": "Kochi",
                    "region": "Kochi",
                },
                search_context_size="medium",
            ),
            scrape_reference_page,
        ],
        output_type=ResearchBatch,
        model_settings=ModelSettings(tool_choice="required"),
        reset_tool_choice=True,
        **_model_override(model),
    )


def build_agents(model: str = DEFAULT_MODEL) -> WorkflowAgents:
    evaluator = build_evaluator(model=model)
    return WorkflowAgents(
        web_researcher=build_web_researcher(model=model),
        researcher=build_researcher(evaluator=evaluator, model=model),
        evaluator=evaluator,
    )


def _ingest_research_batch(
    context: KatsuoContext,
    research_batch: ResearchBatch,
    as_of: date,
) -> None:
    """Accumulate, cache, and re-validate the candidates from one research run."""
    discovered = [
        candidate
        for candidate in deduplicate_restaurant_candidates(
            list(research_batch.candidates)
        )
        if candidate_within_range(context, candidate)
    ]
    context.collected_candidates = accumulate_restaurant_candidates(
        context.collected_candidates,
        discovered,
    )
    context.cached_candidates_written += cache_restaurant_candidates(
        context,
        discovered,
    )
    accepted, rejections = partition_candidates_by_review_validity(
        context.collected_candidates,
        as_of,
        context.scraped_pages,
    )
    context.pending_candidates = deduplicate_restaurant_candidates(accepted)
    context.candidate_rejections = rejections
    persist_discovered_restaurants(context, as_of)


async def run_web_research_phase(
    agent: Agent[KatsuoContext],
    prompt: str,
    context: KatsuoContext,
    max_turns: int,
    progress_label: str = "Web調査",
) -> Any:
    as_of = datetime.now(timezone.utc).date()
    oldest_allowed = as_of - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS)
    _emit_progress(context, f"{progress_label}を開始")
    result = await _await_with_progress(
        Runner.run(
            starting_agent=agent,
            input=(
                f"{prompt}\n口コミの公開日または訪問月は {oldest_allowed.isoformat()} から "
                f"{as_of.isoformat()} まで（両端を含む）のものだけを採用してください。"
                "日まで表示されない場合は、その年月の1日（YYYY-MM-01）として保存してください。"
            ),
            context=context,
            max_turns=max_turns,
        ),
        context,
        progress_label,
    )
    _emit_progress(context, f"{progress_label}のAPI応答を受信")
    research_batch = ResearchBatch.model_validate(result.final_output)
    _ingest_research_batch(context, research_batch, as_of)
    return result


async def run_storage_and_evaluation_phase(
    agent: Agent[KatsuoContext],
    context: KatsuoContext,
    max_turns: int,
) -> Any:
    if not context.pending_candidates:
        raise NoValidResearchCandidatesError(
            f"Collected {len(context.collected_candidates)} in-range restaurant(s), "
            "but none passed evaluation validation, so the storage and evaluation "
            "phase cannot start."
            f"{_format_rejection_detail(context.candidate_rejections)}"
        )
    summary = summarize_candidate_pool(context, context.pending_candidates)
    if not summary.is_ready:
        raise InsufficientResearchCandidatesError(
            insufficient_candidate_pool_message(summary, context.max_distance_km)
            + f" Collected {len(context.collected_candidates)} in-range restaurant(s)."
            + _format_rejection_detail(context.candidate_rejections)
        )
    label = "候補保存・評価"
    _emit_progress(context, f"{label}を開始")
    result = await _await_with_progress(
        Runner.run(
            starting_agent=agent,
            input=(
                "Web research is complete and its structured candidates are in context. "
                "Save them, then hand off to the evaluation agent."
            ),
            context=context,
            max_turns=max_turns,
        ),
        context,
        label,
    )
    _emit_progress(context, f"{label}を完了")
    return result


def _prime_context_from_cache(context: KatsuoContext, as_of: date) -> None:
    """Load cached discoveries and rebuild the evaluation pool before research."""
    cached_candidates, cache_rejections = load_cached_restaurant_candidates(
        context,
        as_of,
    )
    context.collected_candidates = accumulate_restaurant_candidates(
        context.collected_candidates,
        cached_candidates,
    )
    cached_evaluation_candidates, cached_evaluation_rejections = (
        partition_candidates_by_review_validity(
            cached_candidates,
            as_of,
            context.scraped_pages,
        )
    )
    context.pending_candidates = merge_restaurant_candidates(
        context.pending_candidates,
        cached_evaluation_candidates,
    )
    context.candidate_rejections.extend(cache_rejections)
    context.candidate_rejections.extend(cached_evaluation_rejections)
    context.cached_candidates_loaded = len(cached_candidates)
    persist_discovered_restaurants(context, as_of)
    _emit_progress(
        context,
        "キャッシュ読込完了: "
        f"収集済み{len(cached_candidates)}店 / "
        f"評価可能{len(cached_evaluation_candidates)}店",
    )


async def _run_research_attempts(
    web_researcher: Agent[KatsuoContext],
    context: KatsuoContext,
    base_prompt: str,
    max_turns: int,
    discovery_attempts: int,
    review_enrichment_attempts: int,
    as_of: date,
) -> list[Any]:
    """Run every configured discovery and review-enrichment attempt."""
    research_results: list[Any] = []
    enrichment_target_attempts: dict[tuple[str, str], int] = {}
    schedule = (
        [
            ("店舗発見", attempt, discovery_attempts)
            for attempt in range(1, discovery_attempts + 1)
        ]
        + [
            ("口コミ補完", attempt, review_enrichment_attempts)
            for attempt in range(1, review_enrichment_attempts + 1)
        ]
    )
    retry_invalid_output = False
    for overall_attempt, (research_mode, phase_attempt, phase_total) in enumerate(
        schedule,
        start=1,
    ):
        current_enrichment_targets: list[RestaurantCandidateInput] = []
        if research_mode == "店舗発見":
            current_prompt = _build_discovery_prompt(base_prompt, context)
        else:
            summary = summarize_candidate_pool(context, context.pending_candidates)
            current_enrichment_targets = _select_enrichment_targets(
                context,
                as_of,
                target_attempts=enrichment_target_attempts,
            )
            current_prompt = _build_enrichment_prompt(
                base_prompt,
                context,
                summary,
                as_of,
                targets=current_enrichment_targets,
            )
        if retry_invalid_output:
            current_prompt = _build_invalid_output_retry_prompt(
                current_prompt,
                overall_attempt,
            )
        try:
            research_result = await run_web_research_phase(
                agent=web_researcher,
                prompt=current_prompt,
                context=context,
                max_turns=max_turns,
                progress_label=f"{research_mode} {phase_attempt}/{phase_total}",
            )
        except ModelBehaviorError as exc:
            if overall_attempt >= len(schedule):
                raise InvalidResearchOutputError(
                    _format_invalid_research_output_error(overall_attempt, exc)
                ) from exc
            retry_invalid_output = True
            continue

        retry_invalid_output = False
        research_results.append(research_result)
        if research_mode == "口コミ補完":
            for candidate in current_enrichment_targets:
                key = _enrichment_target_key(candidate)
                enrichment_target_attempts[key] = (
                    enrichment_target_attempts.get(key, 0) + 1
                )
        summary = summarize_candidate_pool(context, context.pending_candidates)
        _emit_progress(
            context,
            "候補収集・検証完了: "
            f"収集済み{len(context.collected_candidates)}店 / "
            f"評価可能{summary.unique_candidates}店",
        )
    return research_results


async def run_katsuo_workflow(
    context: KatsuoContext,
    model: str = DEFAULT_MODEL,
    max_turns: int = 24,
    discovery_attempts: int = DEFAULT_DISCOVERY_ATTEMPTS,
    review_enrichment_attempts: int = DEFAULT_REVIEW_ENRICHMENT_ATTEMPTS,
) -> WorkflowOutcome:
    if discovery_attempts < 0:
        raise ValueError("discovery_attempts must be zero or greater.")
    if review_enrichment_attempts < 0:
        raise ValueError("review_enrichment_attempts must be zero or greater.")
    if discovery_attempts + review_enrichment_attempts < 1:
        raise ValueError("At least one research attempt is required.")
    context.model = model
    context.trace_id = gen_trace_id()
    agents = build_agents(model=model)
    as_of = datetime.now(timezone.utc).date()
    _prime_context_from_cache(context, as_of)
    trace_id = context.trace_id
    prompt = (
        "高知駅周辺でカツオ料理がおいしい店を調査し、TOP 5を作成してください。"
        f"基準ホテルは{context.hotel.name} "
        f"({context.hotel.latitude}, {context.hotel.longitude})、"
        f"許容する直線距離は{context.max_distance_km:.2f} kmです。"
    )
    with trace(
        workflow_name="Kochi Katsuo TOP 5",
        trace_id=trace_id,
        metadata={
            "hotel": context.hotel.name,
            "max_distance_km": str(context.max_distance_km),
        },
    ):
        research_results = await _run_research_attempts(
            web_researcher=agents.web_researcher,
            context=context,
            base_prompt=prompt,
            max_turns=max_turns,
            discovery_attempts=discovery_attempts,
            review_enrichment_attempts=review_enrichment_attempts,
            as_of=as_of,
        )
        result = await run_storage_and_evaluation_phase(
            agent=agents.researcher,
            context=context,
            max_turns=max_turns,
        )

    if result.last_agent is not agents.evaluator:
        raise RuntimeError(
            "Acceptance check failed: result.last_agent is "
            f"{result.last_agent.name!r}, expected {agents.evaluator.name!r}."
        )
    audit = audit_run_items(result, context, prior_results=tuple(research_results))
    _emit_progress(context, "実行監査と成果物保存を開始")
    persist_run_manifest(context, trace_id, audit.model_dump())
    return WorkflowOutcome(
        final_output=result.final_output,
        last_agent=result.last_agent.name,
        model=context.model,
        trace_id=trace_id,
        audit=audit,
    )
