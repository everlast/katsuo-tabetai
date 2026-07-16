from __future__ import annotations

from datetime import date, timedelta

from katsuo_tabetai.models import (
    EvidenceSourceType,
    HotelLocation,
    RecentReview,
    RestaurantCandidateInput,
)
from katsuo_tabetai.scoring import (
    DISTANCE_MAX_POINTS,
    EVIDENCE_MAX_POINTS,
    INDEPENDENT_SOURCES_MAX_POINTS,
    KATSUO_FEATURES_MAX_POINTS,
    RECENT_REVIEWS_MAX_POINTS,
    TOTAL_MAX_POINTS,
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
    ratings = [rating] * 5 if rating is not None else [4.8, 4.4, 4.6, 4.7, 4.5]
    return [
        RecentReview(
            source_name="Review A" if review_index < 2 else "Review B",
            reviewer_name=f"Reviewer {index}-{review_index}",
            review_url=(
                f"https://reviews-{review_index % 2}.example/"
                f"restaurant/{index}/review/{review_index}"
            ),
            published_at=date.today() - timedelta(days=review_index * 30),
            rating=review_rating,
            summary=f"Recent review {review_index} for restaurant {index}",
            positive_points=[
                "カツオの鮮度と旨味が高い",
                "藁焼きの香りが豊かでよい",
            ],
            caution_points=["混雑"] if review_index == 3 else [],
        )
        for review_index, review_rating in enumerate(ratings, start=1)
    ]


def test_review_points_remove_structured_output_artifacts() -> None:
    review = RecentReview(
        source_name="Review A",
        reviewer_name="Reviewer 1",
        review_url="https://reviews.example/restaurant/1/review/1",
        published_at=date.today(),
        rating=4.0,
        summary="Recent review",
        positive_points=[
            "鰹が本格的さに満足.",
            "観光向きの安心感. 2-3],",
            "caution_points':['混雑しやすい']},{",
        ],
        caution_points=["人気で予約推奨. 2-3", "caution_points:["],
    )

    assert review.positive_points == [
        "鰹が本格的さに満足",
        "観光向きの安心感",
    ]
    assert review.caution_points == ["人気で予約推奨"]


def test_review_points_split_concatenated_array_values() -> None:
    review = RecentReview(
        source_name="Review A",
        reviewer_name="Reviewer 1",
        review_url="https://reviews.example/restaurant/1/review/1",
        published_at=date.today(),
        rating=4.0,
        summary="Recent review",
        positive_points=['駅近. 2-3","期待通り. 2-3'],
        caution_points=[],
    )

    assert review.positive_points == ["駅近", "期待通り"]


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


def test_score_category_maximums_total_one_hundred_points() -> None:
    assert (
        EVIDENCE_MAX_POINTS,
        KATSUO_FEATURES_MAX_POINTS,
        INDEPENDENT_SOURCES_MAX_POINTS,
        RECENT_REVIEWS_MAX_POINTS,
        DISTANCE_MAX_POINTS,
    ) == (20, 15, 10, 40, 15)
    assert TOTAL_MAX_POINTS == 100


def test_score_uses_configured_evidence_points() -> None:
    candidate = make_candidate(1)
    expected_points = {
        EvidenceSourceType.OFFICIAL_RESTAURANT: 20.0,
        EvidenceSourceType.OFFICIAL_TOURISM: 17.0,
        EvidenceSourceType.RESERVATION_SITE: 13.0,
        EvidenceSourceType.REVIEW_SITE: 8.0,
    }

    for source_type, expected in expected_points.items():
        scored_candidate = candidate.model_copy(
            update={"evidence_source_type": source_type}
        )

        assert score_candidate(scored_candidate, 2.5).evidence == expected


def test_katsuo_feature_points_have_a_fifteen_point_maximum() -> None:
    candidate = make_candidate(1).model_copy(
        update={
            "has_warayaki": False,
            "has_shio_tataki": False,
            "has_seasonal_katsuo": False,
        }
    )

    assert score_candidate(candidate, 2.5).katsuo_features == 6.0

    feature_points = {
        "has_warayaki": 4.0,
        "has_shio_tataki": 3.0,
        "has_seasonal_katsuo": 2.0,
    }
    for field, expected_increment in feature_points.items():
        featured = candidate.model_copy(update={field: True})
        assert (
            score_candidate(featured, 2.5).katsuo_features == 6.0 + expected_increment
        )

    all_features = candidate.model_copy(
        update={field: True for field in feature_points}
    )
    assert score_candidate(all_features, 2.5).katsuo_features == 15.0


def test_independent_source_points_are_capped_at_five_domains() -> None:
    candidate = make_candidate(1)
    five_additional_domains = [
        candidate.evidence_url,
        *[
            type(candidate.evidence_url)(f"https://source-{index}.example/menu")
            for index in range(1, 5)
        ],
    ]

    five_domains = candidate.model_copy(update={"source_urls": five_additional_domains})
    assert score_candidate(five_domains, 2.5).independent_sources == 10.0

    six_domains = candidate.model_copy(
        update={
            "source_urls": [
                *five_additional_domains,
                type(candidate.evidence_url)("https://source-5.example/menu"),
            ]
        }
    )
    assert score_candidate(six_domains, 2.5).independent_sources == 10.0


def test_recent_review_points_have_a_forty_point_maximum() -> None:
    candidate = make_candidate(1, review_rating=5.0)

    assert score_candidate(candidate, 2.5).recent_reviews == 40.0


def test_distance_points_decrease_linearly_to_zero_at_the_limit() -> None:
    candidate = make_candidate(1)

    assert (
        score_candidate(candidate.model_copy(update={"distance_km": 0.0}), 2.5).distance
        == 15.0
    )
    assert (
        score_candidate(
            candidate.model_copy(update={"distance_km": 1.25}), 2.5
        ).distance
        == 7.5
    )
    assert (
        score_candidate(candidate.model_copy(update={"distance_km": 2.5}), 2.5).distance
        == 0.0
    )


def test_recent_review_reputation_changes_score_deterministically() -> None:
    highly_rated = make_candidate(1, review_rating=5.0)
    poorly_rated = make_candidate(1, review_rating=2.0)

    high_score = score_candidate(highly_rated, max_distance_km=2.5)
    low_score = score_candidate(poorly_rated, max_distance_km=2.5)

    assert high_score.recent_reviews == 40.0
    assert low_score.recent_reviews == 20.8
    assert high_score.total > low_score.total


def test_rank_top_five_is_sorted_and_stable() -> None:
    ranked = rank_top_five([make_candidate(i) for i in range(1, 7)], 2.5)

    assert len(ranked) == 5
    assert [item.rank for item in ranked] == [1, 2, 3, 4, 5]
    assert [item.score for item in ranked] == sorted(
        [item.score for item in ranked], reverse=True
    )
    for item in ranked:
        assert "直近レビュー5件" in item.recommendation_reason
        assert "カツオの鮮度と旨味が高い" in item.recommendation_reason
        assert item.review_reputation.review_count == 5
