from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from math import asin, cos, radians, sin, sqrt
from typing import Final

from .models import (
    EvidenceSourceType,
    HotelLocation,
    RankedRestaurant,
    RestaurantCandidate,
    RestaurantCandidateInput,
    ReviewReputation,
    ScoreBreakdown,
    selected_feature_labels,
)

EARTH_RADIUS_KM = 6371.0088
TWO_PLACES = Decimal("0.01")

EVIDENCE_POINTS: Final = {
    EvidenceSourceType.OFFICIAL_RESTAURANT: Decimal("20"),
    EvidenceSourceType.OFFICIAL_TOURISM: Decimal("17"),
    EvidenceSourceType.RESERVATION_SITE: Decimal("13"),
    EvidenceSourceType.REVIEW_SITE: Decimal("8"),
}
KATSUO_DISH_NAME_POINTS: Final = Decimal("6")
WARAYAKI_POINTS: Final = Decimal("4")
SHIO_TATAKI_POINTS: Final = Decimal("3")
SEASONAL_KATSUO_POINTS: Final = Decimal("2")

INDEPENDENT_SOURCE_POINTS_PER_DOMAIN: Final = Decimal("2")
INDEPENDENT_SOURCE_MAX_DOMAINS: Final = 5

REVIEW_RATING_MAX_POINTS: Final = Decimal("32")
REVIEW_RATING_SCALE_MAX: Final = Decimal("5")
REVIEW_COUNT_MAX_POINTS: Final = Decimal("5")
REVIEW_COUNT_FOR_MAX_POINTS: Final = 5
REVIEW_SOURCE_MAX_POINTS: Final = Decimal("3")
REVIEW_SOURCE_COUNT_FOR_MAX_POINTS: Final = 2

DISTANCE_MAX_POINTS: Final = Decimal("15")

EVIDENCE_MAX_POINTS: Final = max(EVIDENCE_POINTS.values())
KATSUO_FEATURES_MAX_POINTS: Final = (
    KATSUO_DISH_NAME_POINTS
    + WARAYAKI_POINTS
    + SHIO_TATAKI_POINTS
    + SEASONAL_KATSUO_POINTS
)
INDEPENDENT_SOURCES_MAX_POINTS: Final = (
    INDEPENDENT_SOURCE_POINTS_PER_DOMAIN * INDEPENDENT_SOURCE_MAX_DOMAINS
)
RECENT_REVIEWS_MAX_POINTS: Final = (
    REVIEW_RATING_MAX_POINTS + REVIEW_COUNT_MAX_POINTS + REVIEW_SOURCE_MAX_POINTS
)
TOTAL_MAX_POINTS: Final = (
    EVIDENCE_MAX_POINTS
    + KATSUO_FEATURES_MAX_POINTS
    + INDEPENDENT_SOURCES_MAX_POINTS
    + RECENT_REVIEWS_MAX_POINTS
    + DISTANCE_MAX_POINTS
)


def _round(value: Decimal) -> float:
    return float(value.quantize(TWO_PLACES, rounding=ROUND_HALF_UP))


def haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return great-circle distance in kilometers."""
    lat_a, lon_a, lat_b, lon_b = map(
        radians,
        (latitude_a, longitude_a, latitude_b, longitude_b),
    )
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    haversine = (
        sin(delta_lat / 2) ** 2 + cos(lat_a) * cos(lat_b) * sin(delta_lon / 2) ** 2
    )
    return round(2 * EARTH_RADIUS_KM * asin(sqrt(haversine)), 4)


def apply_range_rule(
    candidate: RestaurantCandidateInput,
    hotel: HotelLocation,
    max_distance_km: float,
) -> RestaurantCandidate:
    distance = haversine_km(
        hotel.latitude,
        hotel.longitude,
        candidate.latitude,
        candidate.longitude,
    )
    return RestaurantCandidate(
        **candidate.model_dump(),
        distance_km=distance,
        within_range=distance <= max_distance_km,
    )


def normalized_url_host(url) -> str:
    return (url.host or "").removeprefix("www.")


def _rank_review_points(
    candidate: RestaurantCandidate,
    field: str,
) -> list[str]:
    counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    for review in candidate.recent_reviews:
        values = getattr(review, field)
        normalized_in_review: set[str] = set()
        for value in values:
            label = " ".join(value.split())
            key = label.casefold()
            if not key or key in normalized_in_review:
                continue
            normalized_in_review.add(key)
            counts[key] += 1
            labels.setdefault(key, label)
    ranked = sorted(counts, key=lambda key: (-counts[key], labels[key].casefold()))
    return [labels[key] for key in ranked[:2]]


def summarize_review_reputation(
    candidate: RestaurantCandidate,
) -> ReviewReputation:
    review_count = len(candidate.recent_reviews)
    average = sum(Decimal(str(review.rating)) for review in candidate.recent_reviews)
    average /= Decimal(review_count)
    review_hosts = {
        normalized_url_host(review.review_url) for review in candidate.recent_reviews
    }
    return ReviewReputation(
        average_rating=_round(average),
        review_count=review_count,
        source_count=len(review_hosts),
        top_positive_points=_rank_review_points(candidate, "positive_points"),
        caution_points=_rank_review_points(candidate, "caution_points"),
    )


def build_recommendation_reason(
    candidate: RestaurantCandidate,
    reputation: ReviewReputation,
) -> str:
    features = selected_feature_labels(
        candidate,
        warayaki="藁焼き",
        shio_tataki="塩たたき",
        seasonal_katsuo="旬のカツオ",
    )
    feature_text = "・".join(features) if features else "カツオ料理"
    positives = "・".join(reputation.top_positive_points)
    reason = (
        f"{candidate.katsuo_dish}の提供根拠があり、{feature_text}を楽しめます。"
        f"直近レビュー{reputation.review_count}件の平均評価は"
        f"{reputation.average_rating:.2f}/5で、{positives}が好評です。"
        f"ホテルから{candidate.distance_km:.2f} kmの範囲内です。"
    )
    if reputation.caution_points:
        cautions = "・".join(reputation.caution_points)
        reason += f"一方、{cautions}には注意が必要です。"
    return reason


def score_candidate(
    candidate: RestaurantCandidate,
    max_distance_km: float,
) -> ScoreBreakdown:
    """Calculate a deterministic 100-point score from stored facts."""
    evidence = EVIDENCE_POINTS[candidate.evidence_source_type]

    katsuo_features = KATSUO_DISH_NAME_POINTS
    if candidate.has_warayaki:
        katsuo_features += WARAYAKI_POINTS
    if candidate.has_shio_tataki:
        katsuo_features += SHIO_TATAKI_POINTS
    if candidate.has_seasonal_katsuo:
        katsuo_features += SEASONAL_KATSUO_POINTS

    unique_sources = {
        normalized_url_host(candidate.evidence_url),
        *(normalized_url_host(url) for url in candidate.source_urls),
    }
    independent_sources = (
        Decimal(min(len(unique_sources), INDEPENDENT_SOURCE_MAX_DOMAINS))
        * INDEPENDENT_SOURCE_POINTS_PER_DOMAIN
    )

    reputation = summarize_review_reputation(candidate)
    rating_points = (
        Decimal(str(reputation.average_rating))
        / REVIEW_RATING_SCALE_MAX
        * REVIEW_RATING_MAX_POINTS
    )
    volume_points = (
        Decimal(min(reputation.review_count, REVIEW_COUNT_FOR_MAX_POINTS))
        / Decimal(REVIEW_COUNT_FOR_MAX_POINTS)
        * REVIEW_COUNT_MAX_POINTS
    )
    diversity_points = (
        Decimal(min(reputation.source_count, REVIEW_SOURCE_COUNT_FOR_MAX_POINTS))
        / Decimal(REVIEW_SOURCE_COUNT_FOR_MAX_POINTS)
        * REVIEW_SOURCE_MAX_POINTS
    )
    recent_reviews = rating_points + volume_points + diversity_points

    distance_ratio = Decimal(str(candidate.distance_km)) / Decimal(str(max_distance_km))
    distance = max(
        Decimal("0"),
        DISTANCE_MAX_POINTS * (Decimal("1") - distance_ratio),
    )

    return ScoreBreakdown(
        evidence=_round(evidence),
        katsuo_features=_round(katsuo_features),
        independent_sources=_round(independent_sources),
        recent_reviews=_round(recent_reviews),
        distance=_round(distance),
    )


def rank_restaurants(
    candidates: list[RestaurantCandidate],
    max_distance_km: float,
) -> list[RankedRestaurant]:
    """Rank every in-range restaurant with deterministic tie breaking."""
    scored = [
        (
            candidate,
            score_candidate(candidate, max_distance_km),
            summarize_review_reputation(candidate),
        )
        for candidate in candidates
        if candidate.within_range
    ]
    if len(scored) < 5:
        raise ValueError(
            f"At least 5 in-range restaurants are required; found {len(scored)}."
        )

    scored.sort(
        key=lambda item: (
            -item[1].total,
            item[0].distance_km,
            item[0].name.casefold(),
        )
    )
    return [
        RankedRestaurant(
            **candidate.model_dump(),
            rank=index,
            score=breakdown.total,
            score_breakdown=breakdown,
            review_reputation=reputation,
            recommendation_reason=build_recommendation_reason(
                candidate,
                reputation,
            ),
        )
        for index, (candidate, breakdown, reputation) in enumerate(scored, start=1)
    ]


def rank_top_five(
    candidates: list[RestaurantCandidate],
    max_distance_km: float,
) -> list[RankedRestaurant]:
    """Return the first five restaurants from the full deterministic ranking."""
    return rank_restaurants(candidates, max_distance_km)[:5]
