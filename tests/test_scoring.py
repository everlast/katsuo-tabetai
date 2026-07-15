from __future__ import annotations

from datetime import date, timedelta

from katsuo_tabetai.models import (
    EvidenceSourceType,
    HotelLocation,
    RecentReview,
    RestaurantCandidateInput,
)
from katsuo_tabetai.scoring import (
    apply_range_rule,
    haversine_km,
    rank_top_five,
    score_candidate,
)


HOTEL = HotelLocation(
    name="Test Hotel",
    latitude=33.566927593644714,
    longitude=133.54104073018118,
)


def make_reviews(index: int, rating: float | None = None) -> list[RecentReview]:
    ratings = [rating] * 3 if rating is not None else [4.8, 4.4, 4.6]
    return [
        RecentReview(
            source_name="Review A" if review_index < 2 else "Review B",
            review_url=(
                f"https://reviews-{review_index % 2}.example/"
                f"restaurant/{index}/review/{review_index}"
            ),
            published_at=date.today() - timedelta(days=review_index * 30),
            rating=review_rating,
            summary=f"Recent review {review_index} for restaurant {index}",
            positive_points=["カツオの鮮度", "藁の香り"],
            caution_points=["混雑"] if review_index == 3 else [],
        )
        for review_index, review_rating in enumerate(ratings, start=1)
    ]


def make_candidate(
    index: int,
    latitude_offset: float = 0.001,
    review_rating: float | None = None,
):
    candidate = RestaurantCandidateInput(
        name=f"Restaurant {index}",
        address=f"Kochi {index}",
        latitude=HOTEL.latitude + latitude_offset * index,
        longitude=HOTEL.longitude,
        katsuo_dish=f"Katsuo dish {index}",
        evidence_url=f"https://restaurant{index}.example/menu",
        evidence_source_type=EvidenceSourceType.OFFICIAL_RESTAURANT,
        source_urls=[f"https://tourism.example/restaurant/{index}"],
        recent_reviews=make_reviews(index, review_rating),
        has_warayaki=index % 2 == 0,
        has_shio_tataki=index % 3 == 0,
        has_seasonal_katsuo=index % 4 == 0,
    )
    return apply_range_rule(candidate, HOTEL, max_distance_km=2.5)


def test_haversine_returns_zero_for_same_point() -> None:
    assert haversine_km(33.5, 133.5, 33.5, 133.5) == 0.0


def test_range_rule_is_decided_in_code() -> None:
    nearby = make_candidate(1)
    far_away = make_candidate(1, latitude_offset=0.1)

    assert nearby.within_range is True
    assert far_away.within_range is False
    assert nearby.distance_km < 2.5 < far_away.distance_km


def test_score_is_deterministic() -> None:
    candidate = make_candidate(2)

    first = score_candidate(candidate, max_distance_km=2.5)
    second = score_candidate(candidate, max_distance_km=2.5)

    assert first == second
    assert first.total == second.total


def test_recent_review_reputation_changes_score_deterministically() -> None:
    highly_rated = make_candidate(1, review_rating=5.0)
    poorly_rated = make_candidate(1, review_rating=2.0)

    high_score = score_candidate(highly_rated, max_distance_km=2.5)
    low_score = score_candidate(poorly_rated, max_distance_km=2.5)

    assert high_score.recent_reviews == 23.8
    assert low_score.recent_reviews == 11.8
    assert high_score.total > low_score.total


def test_rank_top_five_is_sorted_and_stable() -> None:
    ranked = rank_top_five([make_candidate(i) for i in range(1, 7)], 2.5)

    assert len(ranked) == 5
    assert [item.rank for item in ranked] == [1, 2, 3, 4, 5]
    assert [item.score for item in ranked] == sorted(
        [item.score for item in ranked], reverse=True
    )
    for item in ranked:
        assert "直近レビュー3件" in item.recommendation_reason
        assert "カツオの鮮度" in item.recommendation_reason
        assert item.review_reputation.review_count == 3
