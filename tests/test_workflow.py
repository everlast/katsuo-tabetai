from __future__ import annotations

from types import SimpleNamespace

from agents import WebSearchTool

from katsuo_ai.context import KatsuoContext
from katsuo_ai.models import HotelLocation
from katsuo_ai.workflow import audit_run_items, build_agents


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
    assert evaluator.tool_use_behavior == "stop_on_first_tool"


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
