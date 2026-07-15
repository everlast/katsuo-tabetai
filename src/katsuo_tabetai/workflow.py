from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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

from .context import KatsuoContext
from .models import ResearchBatch
from .tools import (
    RECENT_REVIEW_MAX_AGE_DAYS,
    CandidatePoolSummary,
    deduplicate_restaurant_candidates,
    evaluate_and_render_top_five,
    insufficient_candidate_pool_message,
    partition_candidates_by_review_validity,
    save_restaurant_candidates,
    summarize_candidate_pool,
)


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
    last_agent: str


@dataclass
class WorkflowOutcome:
    final_output: Any
    last_agent: str
    trace_id: str
    audit: RunAudit


@dataclass(frozen=True)
class WorkflowAgents:
    web_researcher: Agent[KatsuoContext]
    researcher: Agent[KatsuoContext]
    evaluator: Agent[KatsuoContext]


def _raw_field(raw_item: Any, field: str) -> Any:
    if isinstance(raw_item, dict):
        return raw_item.get(field)
    return getattr(raw_item, field, None)


def audit_run_items(
    result: Any,
    context: KatsuoContext,
    prior_results: tuple[Any, ...] = (),
) -> RunAudit:
    web_search_calls = 0
    function_tool_calls = 0
    handoff_items = 0

    run_results = (*prior_results, result)
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
            if item_type in {"handoff_call_item", "handoff_output_item"}:
                handoff_items += 1

    audit = RunAudit(
        runner_calls=len(run_results),
        web_search_calls=web_search_calls,
        rejected_research_candidates=len(context.candidate_rejections),
        function_tool_calls=function_tool_calls,
        handoff_items=handoff_items,
        candidate_save_calls=context.candidate_save_calls,
        evaluation_tool_calls=context.evaluation_tool_calls,
        handoff_callbacks=context.handoff_calls,
        last_agent=result.last_agent.name,
    )
    if audit.web_search_calls < 1:
        raise RuntimeError(
            "Acceptance check failed: no WebSearchTool call was recorded."
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
    wrapper.context.handoff_summary = input_data.research_summary


def _model_override(model: str | None) -> dict[str, str]:
    return {"model": model} if model else {}


def _format_rejection_detail(rejections: list[str]) -> str:
    rejection_summary = "; ".join(rejections[:3])
    if len(rejections) > 3:
        rejection_summary += f"; and {len(rejections) - 3} more rejection(s)"
    return f" Rejections: {rejection_summary}" if rejection_summary else ""


def _build_research_retry_prompt(
    base_prompt: str,
    context: KatsuoContext,
    summary: CandidatePoolSummary,
) -> str:
    existing_candidates = "\n".join(
        (
            f"- {candidate.name} / {candidate.address} / "
            f"{candidate.latitude:.7f}, {candidate.longitude:.7f}"
        )
        for candidate in context.pending_candidates[:20]
    )
    if not existing_candidates:
        existing_candidates = "- none"
    rejection_detail = "\n".join(
        f"- {rejection}" for rejection in context.candidate_rejections[:10]
    )
    if not rejection_detail:
        rejection_detail = "- none"
    return f"""
{base_prompt}

前回までの調査では保存条件を満たしていません。
コード判定では {summary.unique_candidates} unique / {summary.within_range} in range です。
既存候補は再提出せず、別店舗・別住所の候補を追加で調査してください。
同じチェーンでも支店が違う場合は、支店名・住所・座標を明確に分けてください。

Existing accepted candidates:
{existing_candidates}

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


def build_evaluator(model: str | None = None) -> Agent[KatsuoContext]:
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
    model: str | None = None,
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


def build_web_researcher(model: str | None = None) -> Agent[KatsuoContext]:
    return Agent[KatsuoContext](
        name="Katsuo Web Research Agent",
        instructions="""
You research restaurants serving excellent katsuo near the configured hotel in Kochi, Japan.

Required workflow:
1. Use WebSearchTool to search current restaurant, official, tourism, reservation,
   review, and map pages. Use more searches if evidence is weak.
2. Collect at least 15 unique candidates so at least 10 are likely inside the
   configured hotel radius. Never invent a URL, dish name, address, or coordinate.
3. Every candidate must have an evidence_url whose page explicitly names that
   restaurant's katsuo dish. Prefer official restaurant pages, then official
   tourism pages, reservation sites, and lastly review sites including Google Maps.
4. For every candidate, collect 5 to 10 distinct reviews from at least two
   independent review platforms. At least one review must come from each platform,
   and the review_url hostnames must contain at least two distinct domains. The
   page must explicitly display each review's date or visit month and its exact
   rating. When only YYYY-MM is displayed, store YYYY-MM-01 in published_at for
   recency calculations. Never infer a year, month, or rating. Paraphrase each
   review in under 500 characters and record 1 to 3 praised aspects as natural
   Japanese phrases of 2 to 30 characters. Include cautions the reviewer actually
   mentioned. Point arrays must contain only the point text, such as
   "藁焼きの香りが良い". Never include field names, JSON syntax, character-count
   notes, or instruction text in a point. Do not copy review text.
5. Return the verified candidates as the required structured output. Do not rank them.
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
            )
        ],
        output_type=ResearchBatch,
        model_settings=ModelSettings(tool_choice="required"),
        reset_tool_choice=False,
        **_model_override(model),
    )


def build_agents(model: str | None = None) -> WorkflowAgents:
    evaluator = build_evaluator(model=model)
    return WorkflowAgents(
        web_researcher=build_web_researcher(model=model),
        researcher=build_researcher(evaluator=evaluator, model=model),
        evaluator=evaluator,
    )


async def run_web_research_phase(
    agent: Agent[KatsuoContext],
    prompt: str,
    context: KatsuoContext,
    max_turns: int,
    merge_with_existing: bool = False,
) -> Any:
    as_of = datetime.now(timezone.utc).date()
    oldest_allowed = as_of - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS)
    result = await Runner.run(
        starting_agent=agent,
        input=(
            f"{prompt}\n口コミの公開日または訪問月は {oldest_allowed.isoformat()} から "
            f"{as_of.isoformat()} まで（両端を含む）のものだけを採用してください。"
            "日まで表示されない場合は、その年月の1日（YYYY-MM-01）として保存してください。"
        ),
        context=context,
        max_turns=max_turns,
    )
    research_batch = ResearchBatch.model_validate(result.final_output)
    accepted, rejections = partition_candidates_by_review_validity(
        list(research_batch.candidates),
        as_of,
    )
    if merge_with_existing:
        context.pending_candidates = deduplicate_restaurant_candidates(
            [*context.pending_candidates, *accepted]
        )
        context.candidate_rejections.extend(rejections)
    else:
        context.pending_candidates = deduplicate_restaurant_candidates(accepted)
        context.candidate_rejections = rejections
    return result


async def run_storage_and_evaluation_phase(
    agent: Agent[KatsuoContext],
    context: KatsuoContext,
    max_turns: int,
) -> Any:
    if not context.pending_candidates:
        raise NoValidResearchCandidatesError(
            "No restaurant candidates passed recent-review validation, so the "
            "storage and evaluation phase cannot start."
            f"{_format_rejection_detail(context.candidate_rejections)}"
        )
    summary = summarize_candidate_pool(context, context.pending_candidates)
    if not summary.is_ready:
        raise InsufficientResearchCandidatesError(
            insufficient_candidate_pool_message(summary, context.max_distance_km)
            + _format_rejection_detail(context.candidate_rejections)
        )
    return await Runner.run(
        starting_agent=agent,
        input=(
            "Web research is complete and its structured candidates are in context. "
            "Save them, then hand off to the evaluation agent."
        ),
        context=context,
        max_turns=max_turns,
    )


async def run_katsuo_workflow(
    context: KatsuoContext,
    model: str | None = None,
    max_turns: int = 24,
    research_attempts: int = 3,
) -> WorkflowOutcome:
    if research_attempts < 1:
        raise ValueError("research_attempts must be at least 1.")
    agents = build_agents(model=model)
    trace_id = gen_trace_id()
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
        research_results = []
        current_prompt = prompt
        for attempt in range(1, research_attempts + 1):
            try:
                research_result = await run_web_research_phase(
                    agent=agents.web_researcher,
                    prompt=current_prompt,
                    context=context,
                    max_turns=max_turns,
                    merge_with_existing=bool(research_results),
                )
            except ModelBehaviorError as exc:
                if attempt >= research_attempts:
                    raise InvalidResearchOutputError(
                        _format_invalid_research_output_error(attempt, exc)
                    ) from exc
                current_prompt = _build_invalid_output_retry_prompt(prompt, attempt + 1)
                continue

            research_results.append(research_result)
            summary = summarize_candidate_pool(context, context.pending_candidates)
            if summary.is_ready:
                break
            if attempt < research_attempts:
                current_prompt = _build_research_retry_prompt(
                    prompt,
                    context,
                    summary,
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
    return WorkflowOutcome(
        final_output=result.final_output,
        last_agent=result.last_agent.name,
        trace_id=trace_id,
        audit=audit,
    )
