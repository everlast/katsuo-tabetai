from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from agents import ModelBehaviorError, RunContextWrapper, WebSearchTool
from agents.tool_context import ToolContext

from katsuo_tabetai.context import KatsuoContext
from katsuo_tabetai.models import HotelLocation, ResearchBatch, RestaurantCandidateInput
from katsuo_tabetai.tools import (
    candidate_save_is_enabled,
    evaluate_and_render_top_five,
    save_restaurant_candidates,
)
from katsuo_tabetai.workflow import (
    InsufficientResearchCandidatesError,
    InvalidResearchOutputError,
    NoValidResearchCandidatesError,
    audit_run_items,
    build_agents,
    run_storage_and_evaluation_phase,
    run_web_research_phase,
    run_katsuo_workflow,
)
from test_scoring import HOTEL, make_candidate


def test_workflow_has_real_handoff_and_required_tools() -> None:
    agents = build_agents()
    web_researcher = agents.web_researcher
    researcher = agents.researcher
    evaluator = agents.evaluator

    assert web_researcher.name == "Katsuo Web Research Agent"
    assert researcher.name == "Katsuo Research Agent"
    assert evaluator.name == "Katsuo Evaluation Agent"
    assert researcher.handoffs
    assert any(isinstance(tool, WebSearchTool) for tool in web_researcher.tools)
    assert not any(isinstance(tool, WebSearchTool) for tool in researcher.tools)
    assert save_restaurant_candidates in researcher.tools
    assert evaluate_and_render_top_five in evaluator.tools
    assert web_researcher.output_type is ResearchBatch
    assert web_researcher.model_settings.tool_choice == "required"
    assert researcher.model_settings.tool_choice == "required"
    assert researcher.reset_tool_choice is False
    assert evaluator.tool_use_behavior == "stop_on_first_tool"


def test_evaluation_handoff_is_hidden_until_candidates_are_saved(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    researcher = build_agents().researcher
    evaluation_handoff = researcher.handoffs[0]
    assert callable(evaluation_handoff.is_enabled)

    async def is_enabled() -> bool:
        return await evaluation_handoff.is_enabled(
            RunContextWrapper(context=context), researcher
        )

    assert asyncio.run(is_enabled()) is False

    context.candidates_path.write_text("{}", encoding="utf-8")
    assert asyncio.run(is_enabled()) is False

    context.candidates_saved = True
    assert asyncio.run(is_enabled()) is True


def test_save_tool_is_enabled_only_for_unsaved_pending_candidates(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    researcher = build_agents().researcher
    wrapper = RunContextWrapper(context=context)

    assert candidate_save_is_enabled(wrapper, researcher) is False

    candidate = make_candidate(1)
    context.pending_candidates = [
        RestaurantCandidateInput.model_validate(
            candidate.model_dump(exclude={"distance_km", "within_range"})
        )
    ]
    assert candidate_save_is_enabled(wrapper, researcher) is True

    context.candidates_saved = True
    assert candidate_save_is_enabled(wrapper, researcher) is False


def test_storage_phase_rejects_empty_valid_candidate_set_before_runner(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
        candidate_rejections=[
            "Save rejected: Restaurant 1 has reviews from fewer than 2 source sites.",
            "Save rejected: Restaurant 2 has a review older than 365 days.",
        ],
    )
    runner_called = False

    async def fake_run(**kwargs):
        nonlocal runner_called
        runner_called = True

    monkeypatch.setattr("katsuo_tabetai.workflow.Runner.run", fake_run)

    with pytest.raises(
        NoValidResearchCandidatesError,
        match="No restaurant candidates passed recent-review validation",
    ) as exc_info:
        asyncio.run(
            run_storage_and_evaluation_phase(
                build_agents().researcher,
                context,
                max_turns=24,
            )
        )

    assert "Restaurant 1" in str(exc_info.value)
    assert "Restaurant 2" in str(exc_info.value)
    assert runner_called is False


def test_tool_failures_propagate_without_incrementing_success_counters(
    tmp_path,
) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    save_context = ToolContext(
        context=context,
        tool_name="save_restaurant_candidates",
        tool_call_id="save-call",
        tool_arguments="{}",
    )
    evaluation_context = ToolContext(
        context=context,
        tool_name="evaluate_and_render_top_five",
        tool_call_id="evaluation-call",
        tool_arguments="{}",
    )

    with pytest.raises(ValueError, match="provide at least 5"):
        asyncio.run(save_restaurant_candidates.on_invoke_tool(save_context, "{}"))
    with pytest.raises(FileNotFoundError, match="Candidates have not been saved"):
        asyncio.run(
            evaluate_and_render_top_five.on_invoke_tool(evaluation_context, "{}")
        )

    assert context.candidate_save_calls == 0
    assert context.evaluation_tool_calls == 0


def test_research_output_schema_uses_supported_url_strings() -> None:
    tool_schema = ResearchBatch.model_json_schema(mode="validation")
    candidate_schema = tool_schema["$defs"]["RestaurantCandidateInput"]
    properties = candidate_schema["properties"]
    review_schema = tool_schema["$defs"]["RecentReview"]["properties"]

    assert properties["evidence_url"]["type"] == "string"
    assert "format" not in properties["evidence_url"]
    assert properties["source_urls"]["items"] == {"type": "string"}
    assert properties["recent_reviews"]["minItems"] == 5
    assert properties["recent_reviews"]["maxItems"] == 10
    assert "at least two source sites" in properties["recent_reviews"]["description"]
    assert review_schema["review_url"]["type"] == "string"
    assert review_schema["published_at"]["type"] == "string"
    assert "YYYY-MM-01" in review_schema["published_at"]["description"]
    assert review_schema["summary"]["maxLength"] == 500
    assert review_schema["positive_points"]["items"]["minLength"] == 2
    assert review_schema["positive_points"]["items"]["maxLength"] == 30
    assert '"format"' not in json.dumps(tool_schema)


def test_run_item_audit_requires_search_function_tools_and_handoff(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
        candidate_save_calls=1,
        evaluation_tool_calls=1,
        handoff_calls=1,
    )
    research_result = SimpleNamespace(
        new_items=[
            SimpleNamespace(
                type="tool_call_item", raw_item={"type": "web_search_call"}
            ),
        ],
    )
    result = SimpleNamespace(
        last_agent=SimpleNamespace(name="Katsuo Evaluation Agent"),
        new_items=[
            SimpleNamespace(
                type="tool_call_item",
                raw_item={
                    "type": "function_call",
                    "name": "save_restaurant_candidates",
                },
            ),
            SimpleNamespace(
                type="handoff_call_item", raw_item={"type": "function_call"}
            ),
            SimpleNamespace(
                type="handoff_output_item", raw_item={"type": "function_call_output"}
            ),
            SimpleNamespace(
                type="tool_call_item",
                raw_item={
                    "type": "function_call",
                    "name": "evaluate_and_render_top_five",
                },
            ),
        ],
    )

    audit = audit_run_items(result, context, prior_results=(research_result,))

    assert audit.runner_calls == 2
    assert audit.web_search_calls == 1
    assert audit.rejected_research_candidates == 0
    assert audit.function_tool_calls == 2
    assert audit.handoff_items == 2


def test_workflow_continues_after_web_research_final_output(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    agents = build_agents()
    web_researcher = agents.web_researcher
    evaluator = agents.evaluator
    candidates = []
    for index in range(1, 6):
        candidate = make_candidate(index)
        candidates.append(
            RestaurantCandidateInput.model_validate(
                candidate.model_dump(exclude={"distance_km", "within_range"})
            )
        )

    monkeypatch.setattr(
        "katsuo_tabetai.workflow.build_agents",
        lambda model=None: agents,
    )
    monkeypatch.setattr(
        "katsuo_tabetai.workflow.trace",
        lambda **kwargs: nullcontext(),
    )
    called_agents: list[str] = []

    async def fake_run(*, starting_agent, input, context, max_turns):
        called_agents.append(starting_agent.name)
        if starting_agent is web_researcher:
            return SimpleNamespace(
                final_output=ResearchBatch(candidates=candidates),
                last_agent=web_researcher,
                new_items=[
                    SimpleNamespace(
                        type="tool_call_item",
                        raw_item={"type": "web_search_call"},
                    )
                ],
            )

        context.candidate_save_calls = 1
        context.candidates_saved = True
        context.evaluation_tool_calls = 1
        context.handoff_calls = 1
        return SimpleNamespace(
            final_output="completed",
            last_agent=evaluator,
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item={
                        "type": "function_call",
                        "name": "save_restaurant_candidates",
                    },
                ),
                SimpleNamespace(
                    type="handoff_call_item",
                    raw_item={"type": "function_call"},
                ),
                SimpleNamespace(
                    type="handoff_output_item",
                    raw_item={"type": "function_call_output"},
                ),
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item={
                        "type": "function_call",
                        "name": "evaluate_and_render_top_five",
                    },
                ),
            ],
        )

    monkeypatch.setattr("katsuo_tabetai.workflow.Runner.run", fake_run)

    outcome = asyncio.run(run_katsuo_workflow(context))

    assert called_agents == ["Katsuo Web Research Agent", "Katsuo Research Agent"]
    assert context.pending_candidates == candidates
    assert outcome.last_agent == "Katsuo Evaluation Agent"
    assert outcome.audit.runner_calls == 2


def test_workflow_retries_web_research_when_candidate_pool_is_insufficient(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    agents = build_agents()
    web_researcher = agents.web_researcher
    evaluator = agents.evaluator
    duplicate_candidate = RestaurantCandidateInput.model_validate(
        make_candidate(1).model_dump(exclude={"distance_km", "within_range"})
    )
    first_batch = [duplicate_candidate] * 5
    second_batch = [
        RestaurantCandidateInput.model_validate(
            make_candidate(index).model_dump(exclude={"distance_km", "within_range"})
        )
        for index in range(2, 7)
    ]

    monkeypatch.setattr(
        "katsuo_tabetai.workflow.build_agents",
        lambda model=None: agents,
    )
    monkeypatch.setattr(
        "katsuo_tabetai.workflow.trace",
        lambda **kwargs: nullcontext(),
    )
    called_agents: list[str] = []
    web_inputs: list[str] = []

    async def fake_run(*, starting_agent, input, context, max_turns):
        called_agents.append(starting_agent.name)
        if starting_agent is web_researcher:
            web_inputs.append(input)
            candidates = first_batch if len(web_inputs) == 1 else second_batch
            return SimpleNamespace(
                final_output=ResearchBatch(candidates=candidates),
                last_agent=web_researcher,
                new_items=[
                    SimpleNamespace(
                        type="tool_call_item",
                        raw_item={"type": "web_search_call"},
                    )
                ],
            )

        context.candidate_save_calls = 1
        context.candidates_saved = True
        context.evaluation_tool_calls = 1
        context.handoff_calls = 1
        return SimpleNamespace(
            final_output="completed",
            last_agent=evaluator,
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item={
                        "type": "function_call",
                        "name": "save_restaurant_candidates",
                    },
                ),
                SimpleNamespace(
                    type="handoff_call_item", raw_item={"type": "function_call"}
                ),
                SimpleNamespace(
                    type="handoff_output_item",
                    raw_item={"type": "function_call_output"},
                ),
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item={
                        "type": "function_call",
                        "name": "evaluate_and_render_top_five",
                    },
                ),
            ],
        )

    monkeypatch.setattr("katsuo_tabetai.workflow.Runner.run", fake_run)

    outcome = asyncio.run(run_katsuo_workflow(context, research_attempts=2))

    assert called_agents == [
        "Katsuo Web Research Agent",
        "Katsuo Web Research Agent",
        "Katsuo Research Agent",
    ]
    assert len(context.pending_candidates) == 6
    assert "保存条件を満たしていません" in web_inputs[1]
    assert "範囲内の有効店舗があと 4 店必要" in web_inputs[1]
    assert "IN RANGE" in web_inputs[1]
    assert outcome.last_agent == "Katsuo Evaluation Agent"
    assert outcome.audit.runner_calls == 3
    assert outcome.audit.web_search_calls == 2


def test_workflow_caches_partial_pool_when_attempt_limit_is_reached(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    agents = build_agents()
    web_researcher = agents.web_researcher
    candidates = [
        RestaurantCandidateInput.model_validate(
            make_candidate(index).model_dump(exclude={"distance_km", "within_range"})
        )
        for index in range(1, 5)
    ]
    research_batch = ResearchBatch(candidates=[*candidates, candidates[0]])

    monkeypatch.setattr(
        "katsuo_tabetai.workflow.build_agents",
        lambda model=None: agents,
    )
    monkeypatch.setattr(
        "katsuo_tabetai.workflow.trace",
        lambda **kwargs: nullcontext(),
    )

    async def fake_run(*, starting_agent, input, context, max_turns):
        assert starting_agent is web_researcher
        return SimpleNamespace(
            final_output=research_batch,
            last_agent=web_researcher,
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item={"type": "web_search_call"},
                )
            ],
        )

    monkeypatch.setattr("katsuo_tabetai.workflow.Runner.run", fake_run)

    with pytest.raises(InsufficientResearchCandidatesError, match="Received 4"):
        asyncio.run(run_katsuo_workflow(context, research_attempts=1))

    assert context.cached_candidates_written == 4
    assert len(list(context.restaurant_cache_dir.glob("*.json"))) == 4


def test_workflow_retries_web_research_after_malformed_structured_output(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    agents = build_agents()
    web_researcher = agents.web_researcher
    evaluator = agents.evaluator
    candidates = [
        RestaurantCandidateInput.model_validate(
            make_candidate(index).model_dump(exclude={"distance_km", "within_range"})
        )
        for index in range(1, 6)
    ]

    monkeypatch.setattr(
        "katsuo_tabetai.workflow.build_agents",
        lambda model=None: agents,
    )
    monkeypatch.setattr(
        "katsuo_tabetai.workflow.trace",
        lambda **kwargs: nullcontext(),
    )
    called_agents: list[str] = []
    web_inputs: list[str] = []

    async def fake_run(*, starting_agent, input, context, max_turns):
        called_agents.append(starting_agent.name)
        if starting_agent is web_researcher:
            web_inputs.append(input)
            if len(web_inputs) == 1:
                raise ModelBehaviorError("Invalid JSON when parsing {broken")
            return SimpleNamespace(
                final_output=ResearchBatch(candidates=candidates),
                last_agent=web_researcher,
                new_items=[
                    SimpleNamespace(
                        type="tool_call_item",
                        raw_item={"type": "web_search_call"},
                    )
                ],
            )

        context.candidate_save_calls = 1
        context.candidates_saved = True
        context.evaluation_tool_calls = 1
        context.handoff_calls = 1
        return SimpleNamespace(
            final_output="completed",
            last_agent=evaluator,
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item={
                        "type": "function_call",
                        "name": "save_restaurant_candidates",
                    },
                ),
                SimpleNamespace(
                    type="handoff_call_item", raw_item={"type": "function_call"}
                ),
                SimpleNamespace(
                    type="handoff_output_item",
                    raw_item={"type": "function_call_output"},
                ),
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item={
                        "type": "function_call",
                        "name": "evaluate_and_render_top_five",
                    },
                ),
            ],
        )

    monkeypatch.setattr("katsuo_tabetai.workflow.Runner.run", fake_run)

    outcome = asyncio.run(run_katsuo_workflow(context, research_attempts=2))

    assert called_agents == [
        "Katsuo Web Research Agent",
        "Katsuo Web Research Agent",
        "Katsuo Research Agent",
    ]
    assert "ResearchBatchとして解釈できない壊れたJSON" in web_inputs[1]
    assert outcome.last_agent == "Katsuo Evaluation Agent"
    assert outcome.audit.runner_calls == 2
    assert outcome.audit.web_search_calls == 1


def test_workflow_reports_malformed_structured_output_after_retry_limit(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    agents = build_agents()
    web_researcher = agents.web_researcher

    monkeypatch.setattr(
        "katsuo_tabetai.workflow.build_agents",
        lambda model=None: agents,
    )
    monkeypatch.setattr(
        "katsuo_tabetai.workflow.trace",
        lambda **kwargs: nullcontext(),
    )
    called_agents: list[str] = []

    async def fake_run(*, starting_agent, input, context, max_turns):
        called_agents.append(starting_agent.name)
        if starting_agent is web_researcher:
            raise ModelBehaviorError("Invalid JSON when parsing " + ("x" * 1000))
        raise AssertionError("storage phase should not run")

    monkeypatch.setattr("katsuo_tabetai.workflow.Runner.run", fake_run)

    with pytest.raises(
        InvalidResearchOutputError,
        match="malformed structured JSON after 2 attempt",
    ) as exc_info:
        asyncio.run(run_katsuo_workflow(context, research_attempts=2))

    assert called_agents == [
        "Katsuo Web Research Agent",
        "Katsuo Web Research Agent",
    ]
    assert len(str(exc_info.value)) < 340


def test_web_research_phase_excludes_invalid_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    web_researcher = build_agents().web_researcher
    candidates = []
    for index in range(1, 7):
        candidate = make_candidate(index)
        candidates.append(
            RestaurantCandidateInput.model_validate(
                candidate.model_dump(exclude={"distance_km", "within_range"})
            )
        )
    invalid_candidate = candidates[-1]
    invalid_reviews = [
        review.model_copy(
            update={"published_at": review.published_at.replace(year=2020)}
        )
        for review in invalid_candidate.recent_reviews
    ]
    candidates[-1] = invalid_candidate.model_copy(
        update={"recent_reviews": invalid_reviews}
    )
    captured_input = ""

    async def fake_run(*, starting_agent, input, context, max_turns):
        nonlocal captured_input
        captured_input = input
        return SimpleNamespace(final_output=ResearchBatch(candidates=candidates))

    monkeypatch.setattr("katsuo_tabetai.workflow.Runner.run", fake_run)

    asyncio.run(
        run_web_research_phase(
            web_researcher,
            "research",
            context,
            max_turns=24,
        )
    )

    assert context.pending_candidates == candidates[:-1]
    assert len(context.candidate_rejections) == 1
    assert context.cached_candidates_written == 5
    assert len(list(context.restaurant_cache_dir.glob("*.json"))) == 5
    assert invalid_candidate.name in context.candidate_rejections[0]
    assert "両端を含む" in captured_input
    assert "YYYY-MM-01" in captured_input


def test_web_researcher_requires_reviews_from_at_least_two_platforms() -> None:
    instructions = " ".join(build_agents().web_researcher.instructions.split())

    assert "at least two independent review platforms" in instructions
    assert "at least two distinct domains" in instructions
