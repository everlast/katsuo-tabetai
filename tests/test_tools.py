from __future__ import annotations

from katsuo_ai.context import KatsuoContext
from katsuo_ai.models import HotelLocation, RestaurantCandidateInput, TopFiveStore
from katsuo_ai.tools import create_top_five_report, persist_restaurant_candidates

from test_scoring import make_candidate


def candidate_input(index: int) -> RestaurantCandidateInput:
    candidate = make_candidate(index)
    return RestaurantCandidateInput.model_validate(
        candidate.model_dump(exclude={"distance_km", "within_range"})
    )


def test_function_tool_core_saves_structured_data_and_html(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(
            name="Test Hotel",
            latitude=33.566927593644714,
            longitude=133.54104073018118,
        ),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )

    save_result = persist_restaurant_candidates(
        context,
        [candidate_input(index) for index in range(1, 7)],
    )
    report_result = create_top_five_report(context)

    assert save_result["within_range"] == 6
    assert context.candidates_path.exists()
    assert context.top_five_path.exists()
    assert context.html_path.exists()
    top_five = TopFiveStore.model_validate_json(
        context.top_five_path.read_text(encoding="utf-8")
    )
    assert len(top_five.restaurants) == 5
    assert report_result["status"] == "completed"
