from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from datetime import date
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
from katsuo_tabetai.scraping import canonical_url, scrape_reference_page
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
from katsuo_tabetai import workflow as workflow_module
from test_scoring import HOTEL, make_candidate
from helpers import populate_scraped_pages


def web_research_items(context: KatsuoContext) -> list[SimpleNamespace]:
    context.scrape_calls += 1
    return [
        SimpleNamespace(type="tool_call_item", raw_item={"type": "web_search_call"}),
        SimpleNamespace(
            type="tool_call_item",
            raw_item={"type": "function_call", "name": "scrape_reference_page"},
        ),
    ]


def test_progress_heartbeat_reports_while_awaiting(tmp_path, monkeypatch) -> None:
    messages: list[str] = []
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
        progress_callback=messages.append,
    )
    monkeypatch.setattr(workflow_module, "PROGRESS_HEARTBEAT_SECONDS", 0.01)

    async def slow_result() -> str:
        await asyncio.sleep(0.03)
        return "done"

    result = asyncio.run(
        workflow_module._await_with_progress(slow_result(), context, "Web調査")
    )

    assert result == "done"
    assert any("Web調査 実行中" in message for message in messages)


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
    assert scrape_reference_page in web_researcher.tools
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
        match="none passed evaluation validation",
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
    assert "independent domains" in properties["source_urls"]["description"]
    assert "Do not include review-list" in properties["source_urls"]["description"]
    assert "minItems" not in properties["recent_reviews"]
    assert properties["recent_reviews"]["maxItems"] == 10
    assert "still collected" in properties["recent_reviews"]["description"]
    assert "recent_reviews" not in candidate_schema["required"]
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
    research_result = SimpleNamespace(new_items=web_research_items(context))
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
    populate_scraped_pages(context, candidates)

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
                new_items=web_research_items(context),
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

    outcome = asyncio.run(
        run_katsuo_workflow(
            context,
            discovery_attempts=3,
            review_enrichment_attempts=0,
        )
    )

    assert called_agents == [
        "Katsuo Web Research Agent",
        "Katsuo Web Research Agent",
        "Katsuo Web Research Agent",
        "Katsuo Research Agent",
    ]
    assert context.pending_candidates == candidates
    assert outcome.last_agent == "Katsuo Evaluation Agent"
    assert outcome.model == "gpt-5.6-luna"
    assert outcome.audit.runner_calls == 4
    run_manifest = json.loads(context.run_manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["model"] == "gpt-5.6-luna"
    assert run_manifest["trace_id"] == outcome.trace_id
    assert run_manifest["audit"]["scrape_tool_calls"] == 3


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
    first_batch = [
        RestaurantCandidateInput.model_validate(
            make_candidate(index).model_dump(exclude={"distance_km", "within_range"})
        )
        for index in range(1, 21)
    ]
    first_batch = [
        first_batch[0],
        *[
            candidate.model_copy(update={"recent_reviews": []})
            for candidate in first_batch[1:]
        ],
    ]
    second_batch = [
        RestaurantCandidateInput.model_validate(
            make_candidate(index).model_dump(exclude={"distance_km", "within_range"})
        )
        for index in range(2, 8)
    ]
    populate_scraped_pages(context, [*first_batch, *second_batch])

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
                new_items=web_research_items(context),
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

    outcome = asyncio.run(
        run_katsuo_workflow(
            context,
            discovery_attempts=1,
            review_enrichment_attempts=1,
        )
    )

    assert called_agents == [
        "Katsuo Web Research Agent",
        "Katsuo Web Research Agent",
        "Katsuo Research Agent",
    ]
    assert len(context.pending_candidates) == 7
    assert "RESEARCH MODE: DISCOVERY" in web_inputs[0]
    assert "RESEARCH MODE: ENRICHMENT" in web_inputs[1]
    assert "Enrichment targets (return each exactly once)" in web_inputs[1]
    assert "missing_reviews=5" in web_inputs[1]
    assert "評価条件を満たす店舗が不足" in web_inputs[1]
    assert "評価可能候補は 1 店で、あと 4 店必要" in web_inputs[1]
    assert "IN RANGE" in web_inputs[1]
    assert outcome.last_agent == "Katsuo Evaluation Agent"
    assert outcome.audit.runner_calls == 3
    assert outcome.audit.web_search_calls == 2


def test_research_starts_with_enrichment_for_incomplete_cached_pool(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidates = [
        RestaurantCandidateInput.model_validate(
            make_candidate(index).model_dump(exclude={"distance_km", "within_range"})
        )
        for index in range(1, 21)
    ]
    repair_target = candidates[1]
    candidates = [
        candidates[0],
        repair_target,
        *[
            candidate.model_copy(update={"recent_reviews": []})
            for candidate in candidates[2:]
        ],
    ]
    populate_scraped_pages(context, candidates)
    invalid_review = repair_target.recent_reviews[0]
    invalid_key = canonical_url(invalid_review.review_url)
    invalid_page = context.scraped_pages[invalid_key]
    context.scraped_pages[invalid_key] = invalid_page.model_copy(
        update={
            "title": "Another restaurant",
            "content": "\n".join(
                [
                    "Another restaurant",
                    "Kochi 999",
                    invalid_review.reviewer_name,
                    invalid_review.published_at.isoformat(),
                    f"{invalid_review.rating:g} / 5",
                    "A review of a different venue",
                ]
            ),
        }
    )
    context.collected_candidates = candidates
    context.pending_candidates = [candidates[0]]
    prompts: list[str] = []

    async def fake_research_phase(
        agent,
        prompt,
        context,
        max_turns,
        progress_label,
    ):
        prompts.append(prompt)
        return SimpleNamespace()

    monkeypatch.setattr(
        workflow_module,
        "run_web_research_phase",
        fake_research_phase,
    )

    asyncio.run(
        workflow_module._run_research_attempts(
            web_researcher=build_agents().web_researcher,
            context=context,
            base_prompt="research",
            max_turns=24,
            discovery_attempts=0,
            review_enrichment_attempts=1,
            as_of=date.today(),
        )
    )

    assert len(prompts) == 1
    assert "RESEARCH MODE: ENRICHMENT" in prompts[0]
    assert "current_reviews=4" in prompts[0]
    assert "missing_reviews=1" in prompts[0]
    assert "current_evidence_domains=" in prompts[0]
    assert "missing_evidence_domains=1" in prompts[0]
    assert "existing-evidence:" in prompts[0]
    assert str(invalid_review.review_url) not in prompts[0]


def test_enrichment_attempts_rotate_across_untried_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidates = [
        RestaurantCandidateInput.model_validate(
            make_candidate(index).model_dump(exclude={"distance_km", "within_range"})
        ).model_copy(update={"recent_reviews": []})
        for index in range(1, 22)
    ]
    populate_scraped_pages(context, candidates)
    context.collected_candidates = candidates
    prompts: list[str] = []

    async def fake_research_phase(
        agent,
        prompt,
        context,
        max_turns,
        progress_label,
    ):
        prompts.append(prompt)
        return SimpleNamespace()

    monkeypatch.setattr(
        workflow_module,
        "run_web_research_phase",
        fake_research_phase,
    )

    asyncio.run(
        workflow_module._run_research_attempts(
            web_researcher=build_agents().web_researcher,
            context=context,
            base_prompt="research",
            max_turns=24,
            discovery_attempts=0,
            review_enrichment_attempts=3,
            as_of=date.today(),
        )
    )

    target_sets: list[set[str]] = []
    for prompt in prompts:
        target_section = prompt.split(
            "Enrichment targets (return each exactly once):",
            maxsplit=1,
        )[1].split("Previously evaluation-eligible candidates:", maxsplit=1)[0]
        target_sets.append(
            {
                line.removeprefix("- ").split(" / ", maxsplit=1)[0]
                for line in target_section.splitlines()
                if line.startswith("- ")
            }
        )

    assert len(prompts) == 3
    assert all(len(targets) == 5 for targets in target_sets)
    assert target_sets[0].isdisjoint(target_sets[1])
    assert len(set.union(*target_sets)) == 15


def test_nine_collected_candidates_get_three_discovery_attempts_before_enrichment(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidates = [
        RestaurantCandidateInput.model_validate(
            make_candidate(index).model_dump(exclude={"distance_km", "within_range"})
        ).model_copy(update={"recent_reviews": []})
        for index in range(1, 10)
    ]
    populate_scraped_pages(context, candidates)
    context.collected_candidates = candidates
    prompts: list[str] = []
    progress_labels: list[str] = []

    async def fake_research_phase(
        agent,
        prompt,
        context,
        max_turns,
        progress_label,
    ):
        prompts.append(prompt)
        progress_labels.append(progress_label)
        return SimpleNamespace()

    monkeypatch.setattr(
        workflow_module,
        "run_web_research_phase",
        fake_research_phase,
    )

    asyncio.run(
        workflow_module._run_research_attempts(
            web_researcher=build_agents().web_researcher,
            context=context,
            base_prompt="research",
            max_turns=24,
            discovery_attempts=3,
            review_enrichment_attempts=1,
            as_of=date.today(),
        )
    )

    assert [
        "DISCOVERY" if "RESEARCH MODE: DISCOVERY" in prompt else "ENRICHMENT"
        for prompt in prompts
    ] == ["DISCOVERY", "DISCOVERY", "DISCOVERY", "ENRICHMENT"]
    assert progress_labels == [
        "店舗発見 1/3",
        "店舗発見 2/3",
        "店舗発見 3/3",
        "口コミ補完 1/1",
    ]


def test_ready_top_five_does_not_stop_discovery_below_collection_target(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidates = [
        RestaurantCandidateInput.model_validate(
            make_candidate(index).model_dump(exclude={"distance_km", "within_range"})
        )
        for index in range(1, 10)
    ]
    populate_scraped_pages(context, candidates)
    context.collected_candidates = candidates
    context.pending_candidates = candidates[:5]
    prompts: list[str] = []

    async def fake_research_phase(
        agent,
        prompt,
        context,
        max_turns,
        progress_label,
    ):
        prompts.append(prompt)
        return SimpleNamespace()

    monkeypatch.setattr(
        workflow_module,
        "run_web_research_phase",
        fake_research_phase,
    )

    asyncio.run(
        workflow_module._run_research_attempts(
            web_researcher=build_agents().web_researcher,
            context=context,
            base_prompt="research",
            max_turns=24,
            discovery_attempts=3,
            review_enrichment_attempts=7,
            as_of=date.today(),
        )
    )

    assert len(prompts) == 10
    assert all("RESEARCH MODE: DISCOVERY" in prompt for prompt in prompts[:3])
    assert all("RESEARCH MODE: ENRICHMENT" in prompt for prompt in prompts[3:])
    assert "すでに到達済みでも" in prompts[0]
    assert "missing_evidence_domains=1" in prompts[3]


def test_evidence_enrichment_runs_all_attempts_after_domain_target_is_reached(
    tmp_path,
    monkeypatch,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidates = [
        RestaurantCandidateInput.model_validate(
            make_candidate(index).model_dump(exclude={"distance_km", "within_range"})
        )
        for index in range(1, 21)
    ]
    populate_scraped_pages(context, candidates)
    context.collected_candidates = candidates
    context.pending_candidates = candidates[:5]
    prompts: list[str] = []

    async def fake_research_phase(
        agent,
        prompt,
        context,
        max_turns,
        progress_label,
    ):
        prompts.append(prompt)
        enriched = [
            candidate.model_copy(
                update={
                    "source_urls": [
                        *candidate.source_urls,
                        type(candidate.evidence_url)(
                            f"https://reservation-{index}.example/menu"
                        ),
                    ]
                }
            )
            for index, candidate in enumerate(
                context.pending_candidates,
                start=1,
            )
        ]
        context.pending_candidates = enriched
        return SimpleNamespace()

    monkeypatch.setattr(
        workflow_module,
        "run_web_research_phase",
        fake_research_phase,
    )

    asyncio.run(
        workflow_module._run_research_attempts(
            web_researcher=build_agents().web_researcher,
            context=context,
            base_prompt="research",
            max_turns=24,
            discovery_attempts=0,
            review_enrichment_attempts=10,
            as_of=date.today(),
        )
    )

    assert len(prompts) == 10
    assert "RESEARCH MODE: ENRICHMENT" in prompts[0]
    assert "missing_evidence_domains=1" in prompts[0]
    assert all(
        candidate.name in prompts[0] for candidate in candidates[:5]
    )


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
    populate_scraped_pages(context, candidates)

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
            new_items=web_research_items(context),
        )

    monkeypatch.setattr("katsuo_tabetai.workflow.Runner.run", fake_run)

    with pytest.raises(InsufficientResearchCandidatesError, match="Received 4"):
        asyncio.run(
            run_katsuo_workflow(
                context,
                discovery_attempts=1,
                review_enrichment_attempts=0,
            )
        )

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
    populate_scraped_pages(context, candidates)

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
                new_items=web_research_items(context),
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

    outcome = asyncio.run(
        run_katsuo_workflow(
            context,
            discovery_attempts=2,
            review_enrichment_attempts=0,
        )
    )

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
        asyncio.run(
            run_katsuo_workflow(
                context,
                discovery_attempts=2,
                review_enrichment_attempts=0,
            )
        )

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
        hotel=HOTEL,
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
    populate_scraped_pages(context, candidates[:-1])
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
    assert context.collected_candidates == candidates
    assert len(list(context.restaurant_cache_dir.glob("*.json"))) == 6
    discovery_store = json.loads(
        context.discovered_candidates_path.read_text(encoding="utf-8")
    )
    assert discovery_store["collected_count"] == 6
    assert discovery_store["evaluation_eligible_count"] == 5
    assert len(context.candidate_rejections) == 1
    assert context.cached_candidates_written == 6
    assert len(list(context.restaurant_cache_dir.glob("*.json"))) == 6
    assert invalid_candidate.name in context.candidate_rejections[0]
    assert "両端を含む" in captured_input
    assert "YYYY-MM-01" in captured_input


def test_web_researcher_requires_reviews_from_at_least_two_platforms() -> None:
    instructions = " ".join(build_agents().web_researcher.instructions.split())

    assert "at least two independent review platforms" in instructions
    assert "at least two distinct domains" in instructions
    assert "Review platforms drive candidate discovery" in instructions
    assert "official sources strengthen dish evidence" in instructions
    assert "official sources" in instructions and "never count as reviews" in instructions
    assert "Aim for at least 20 candidates" in instructions
    assert "at least 3 independent domains" in instructions
    assert "up to 5 domains" in instructions
