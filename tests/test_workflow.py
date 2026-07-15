from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from agents import RunContextWrapper, WebSearchTool

from katsuo_tabetai.context import KatsuoContext
from katsuo_tabetai.models import HotelLocation
from katsuo_tabetai.tools import save_restaurant_candidates
from katsuo_tabetai.workflow import audit_run_items, build_agents


def test_workflow_has_real_handoff_and_required_tools() -> None:
    researcher, evaluator = build_agents()

    assert researcher.name == "Katsuo Research Agent"
    assert evaluator.name == "Katsuo Evaluation Agent"
    assert researcher.handoffs
    assert any(isinstance(tool, WebSearchTool) for tool in researcher.tools)
    assert any(
        getattr(tool, "name", "") == "save_restaurant_candidates"
        for tool in researcher.tools
    )
    assert any(
        getattr(tool, "name", "") == "evaluate_and_render_top_five"
        for tool in evaluator.tools
    )
    assert researcher.model_settings.tool_choice == "required"
    assert researcher.reset_tool_choice is False
    assert evaluator.tool_use_behavior == "stop_on_first_tool"


def test_evaluation_handoff_is_hidden_until_candidates_are_saved(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    researcher, _ = build_agents()
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


def test_candidate_tool_schema_uses_supported_url_strings() -> None:
    tool_schema = save_restaurant_candidates.params_json_schema
    candidate_schema = tool_schema["$defs"]["RestaurantCandidateInput"]
    properties = candidate_schema["properties"]
    review_schema = tool_schema["$defs"]["RecentReview"]["properties"]

    assert properties["evidence_url"]["type"] == "string"
    assert "format" not in properties["evidence_url"]
    assert properties["source_urls"]["items"] == {"type": "string"}
    assert properties["recent_reviews"]["minItems"] == 5
    assert properties["recent_reviews"]["maxItems"] == 10
    assert review_schema["review_url"]["type"] == "string"
    assert review_schema["published_at"]["type"] == "string"
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
    result = SimpleNamespace(
        last_agent=SimpleNamespace(name="Katsuo Evaluation Agent"),
        new_items=[
            SimpleNamespace(
                type="tool_call_item", raw_item={"type": "web_search_call"}
            ),
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

    audit = audit_run_items(result, context)

    assert audit.web_search_calls == 1
    assert audit.function_tool_calls == 2
    assert audit.handoff_items == 2
