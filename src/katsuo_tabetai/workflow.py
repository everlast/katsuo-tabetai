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
from .tools import evaluate_and_render_top_five, save_restaurant_candidates


class EvaluationHandoffInput(BaseModel):
    research_summary: str = Field(
        min_length=1,
        description="A short summary of the completed search and saved candidate count.",
    )


class RunAudit(BaseModel):
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


def _raw_field(raw_item: Any, field: str) -> Any:
    if isinstance(raw_item, dict):
        return raw_item.get(field)
    return getattr(raw_item, field, None)


def audit_run_items(result: Any, context: KatsuoContext) -> RunAudit:
    web_search_calls = 0
    function_tool_calls = 0
    handoff_items = 0

    for item in result.new_items:
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


def build_agents(
    model: str | None = None,
) -> tuple[Agent[KatsuoContext], Agent[KatsuoContext]]:
    def evaluation_handoff_is_enabled(
        wrapper: RunContextWrapper[KatsuoContext],
        _: Agent[KatsuoContext],
    ) -> bool:
        return (
            wrapper.context.candidates_saved
            and wrapper.context.candidates_path.exists()
        )

    async def on_evaluation_handoff(
        wrapper: RunContextWrapper[KatsuoContext],
        input_data: EvaluationHandoffInput,
    ) -> None:
        if not evaluation_handoff_is_enabled(wrapper, evaluator):
            raise RuntimeError(
                "Handoff rejected: save_restaurant_candidates must run first."
            )
        wrapper.context.handoff_calls += 1
        wrapper.context.handoff_summary = input_data.research_summary

    shared_model = {"model": model} if model else {}
    evaluator = Agent[KatsuoContext](
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
        **shared_model,
    )

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
You research restaurants serving excellent katsuo near the configured hotel in Kochi, Japan.

Required workflow, in this exact order:
1. Your first action must be WebSearchTool. Search current restaurant, official,
   tourism, reservation, and map pages. Use more searches if evidence is weak.
2. Collect at least 15 unique candidates so at least 10 are likely inside the
   configured hotel radius. Never invent a URL, dish name, address, or coordinate.
3. Every candidate must have an evidence_url whose page explicitly names that
   restaurant's katsuo dish. Prefer official restaurant pages, then official
   tourism pages, reservation sites, and lastly review sites including Google Maps.
4. For every candidate, collect 5 to 10 distinct reviews published within the
   last 12 months. The page must explicitly display each review's date and rating.
   Prefer multiple review platforms when available. Never infer a date or rating.
   Paraphrase each review in under 500 characters, record 1 to 3 praised aspects in 10 to 30 characters,
   and include any cautions the reviewer actually mentioned. Do not copy review text.
5. Call save_restaurant_candidates exactly once with all verified candidates.
   Boolean dish features must be supported by the evidence page.
6. After the save tool succeeds, call transfer_to_katsuo_evaluation. Do not
   produce a final answer yourself and do not use an agent-as-tool pattern.
""".strip()
        ),
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
            save_restaurant_candidates,
        ],
        handoffs=[evaluation_handoff],
        model_settings=ModelSettings(tool_choice="required"),
        reset_tool_choice=False,
        **shared_model,
    )
    return researcher, evaluator


async def run_katsuo_workflow(
    context: KatsuoContext,
    model: str | None = None,
    max_turns: int = 24,
) -> WorkflowOutcome:
    researcher, evaluator = build_agents(model=model)
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
        result = await Runner.run(
            starting_agent=researcher,
            input=prompt,
            context=context,
            max_turns=max_turns,
        )

    if result.last_agent is not evaluator:
        raise RuntimeError(
            "Acceptance check failed: result.last_agent is "
            f"{result.last_agent.name!r}, expected {evaluator.name!r}."
        )
    audit = audit_run_items(result, context)
    return WorkflowOutcome(
        final_output=result.final_output,
        last_agent=result.last_agent.name,
        trace_id=trace_id,
        audit=audit,
    )
