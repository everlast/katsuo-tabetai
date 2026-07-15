from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from katsuo_tabetai.context import KatsuoContext
from katsuo_tabetai.models import (
    HotelLocation,
    RestaurantCandidateInput,
    TopFiveStore,
)
from katsuo_tabetai.tools import (
    MIN_REVIEW_SOURCE_SITES,
    RECENT_REVIEW_MAX_AGE_DAYS,
    create_top_five_report,
    partition_candidates_by_review_validity,
    persist_restaurant_candidates,
)

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
    assert all(item.recommendation_reason for item in top_five.restaurants)
    assert all(len(item.recent_reviews) >= 5 for item in top_five.restaurants)
    assert report_result["status"] == "completed"


def test_candidate_save_keeps_same_name_at_distinct_locations(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(
            name="Test Hotel",
            latitude=33.566927593644714,
            longitude=133.54104073018118,
        ),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    branch_a = candidate_input(1).model_copy(update={"name": "Same Chain"})
    branch_b = candidate_input(2).model_copy(
        update={
            "name": "Same Chain",
            "address": "Kochi distinct branch",
            "latitude": branch_a.latitude + 0.01,
        }
    )
    candidates = [
        branch_a,
        branch_b,
        *[candidate_input(index) for index in range(3, 6)],
    ]

    save_result = persist_restaurant_candidates(context, candidates)

    assert save_result["unique_candidates"] == 5
    assert save_result["within_range"] == 5


def test_candidate_save_rejects_reviews_outside_recent_window(tmp_path) -> None:
    candidate = candidate_input(1)
    stale_reviews = [
        review.model_copy(
            update={
                "published_at": date.today()
                - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS + 1)
            }
        )
        for review in candidate.recent_reviews
    ]
    candidate = candidate.model_copy(update={"recent_reviews": stale_reviews})
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="older than"):
        persist_restaurant_candidates(context, [candidate] * 5)


def test_research_partition_rejects_only_candidate_with_stale_review() -> None:
    accepted_candidate = candidate_input(1)
    rejected_candidate = candidate_input(2)
    stale_review = rejected_candidate.recent_reviews[0].model_copy(
        update={
            "published_at": date.today()
            - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS + 1)
        }
    )
    rejected_candidate = rejected_candidate.model_copy(
        update={
            "recent_reviews": [
                stale_review,
                *rejected_candidate.recent_reviews[1:],
            ]
        }
    )

    accepted, rejections = partition_candidates_by_review_validity(
        [accepted_candidate, rejected_candidate],
        date.today(),
    )

    assert accepted == [accepted_candidate]
    assert len(rejections) == 1
    assert rejected_candidate.name in rejections[0]
    assert "older than" in rejections[0]


def test_candidate_input_requires_at_least_five_reviews() -> None:
    candidate = candidate_input(1)
    payload = candidate.model_dump()
    payload["recent_reviews"] = payload["recent_reviews"][:4]

    with pytest.raises(ValidationError):
        RestaurantCandidateInput.model_validate(payload)


def test_candidate_save_rejects_duplicate_reviews(tmp_path) -> None:
    candidate = candidate_input(1)
    duplicate = candidate.recent_reviews[0]
    candidate = candidate.model_copy(
        update={
            "recent_reviews": [
                duplicate,
                duplicate,
                *candidate.recent_reviews[2:],
            ]
        }
    )
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="duplicate review"):
        persist_restaurant_candidates(context, [candidate] * 5)


def test_candidate_save_rejects_reviews_from_only_one_site(tmp_path) -> None:
    candidate = candidate_input(1)
    single_site_reviews = [
        review.model_copy(
            update={
                "review_url": type(review.review_url)(
                    f"https://reviews.example/restaurant/1/review/{index}"
                )
            }
        )
        for index, review in enumerate(candidate.recent_reviews, start=1)
    ]
    candidate = candidate.model_copy(update={"recent_reviews": single_site_reviews})
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match=f"fewer than {MIN_REVIEW_SOURCE_SITES} source sites",
    ):
        persist_restaurant_candidates(context, [candidate] * 5)


def test_research_partition_rejects_candidate_with_only_one_review_site() -> None:
    accepted_candidate = candidate_input(1)
    rejected_candidate = candidate_input(2)
    single_site_reviews = [
        review.model_copy(
            update={
                "review_url": type(review.review_url)(
                    f"https://reviews.example/restaurant/2/review/{index}"
                )
            }
        )
        for index, review in enumerate(rejected_candidate.recent_reviews, start=1)
    ]
    rejected_candidate = rejected_candidate.model_copy(
        update={"recent_reviews": single_site_reviews}
    )

    accepted, rejections = partition_candidates_by_review_validity(
        [accepted_candidate, rejected_candidate],
        date.today(),
    )

    assert accepted == [accepted_candidate]
    assert len(rejections) == 1
    assert rejected_candidate.name in rejections[0]
    assert "fewer than 2 source sites" in rejections[0]
