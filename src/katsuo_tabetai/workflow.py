from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents import (
    Agent,
    ModelSettings,
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
from .tools import evaluate_and_render_top_five, save_restaurant_candidates


class EvaluationHandoffInput(BaseModel):
    research_summary: str = Field(
        min_length=1,
        description="A short summary of the completed search and saved candidate count.",
    )


class RunAudit(BaseModel):
    runner_calls: int
    web_search_calls: int
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
4. For every candidate, collect 5 to 10 distinct reviews published within the
   last 12 months. The page must explicitly display each review's date and rating.
   Prefer multiple review platforms when available. Never infer a date or rating.
   Paraphrase each review in under 500 characters, record 1 to 3 praised aspects
   in 10 to 30 characters, and include cautions the reviewer actually mentioned.
   Do not copy review text.
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
) -> Any:
    result = await Runner.run(
        starting_agent=agent,
        input=prompt,
        context=context,
        max_turns=max_turns,
    )
    research_batch = ResearchBatch.model_validate(result.final_output)
    context.pending_candidates = list(research_batch.candidates)
    return result


async def run_storage_and_evaluation_phase(
    agent: Agent[KatsuoContext],
    context: KatsuoContext,
    max_turns: int,
) -> Any:
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
) -> WorkflowOutcome:
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
        research_result = await run_web_research_phase(
            agent=agents.web_researcher,
            prompt=prompt,
            context=context,
            max_turns=max_turns,
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
    audit = audit_run_items(result, context, prior_results=(research_result,))
    return WorkflowOutcome(
        final_output=result.final_output,
        last_agent=result.last_agent.name,
        trace_id=trace_id,
        audit=audit,
    )
