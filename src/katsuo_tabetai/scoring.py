from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from math import asin, cos, radians, sin, sqrt

from .models import (
    EvidenceSourceType,
    HotelLocation,
    RankedRestaurant,
    RestaurantCandidate,
    RestaurantCandidateInput,
    ReviewReputation,
    ScoreBreakdown,
)

EARTH_RADIUS_KM = 6371.0088
TWO_PLACES = Decimal("0.01")

EVIDENCE_POINTS = {
    EvidenceSourceType.OFFICIAL_RESTAURANT: Decimal("25"),
    EvidenceSourceType.OFFICIAL_TOURISM: Decimal("21"),
    EvidenceSourceType.RESERVATION_SITE: Decimal("16"),
    EvidenceSourceType.REVIEW_SITE: Decimal("10"),
}


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


def _normalized_host(url) -> str:
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
        _normalized_host(review.review_url) for review in candidate.recent_reviews
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
    features: list[str] = []
    if candidate.has_warayaki:
        features.append("藁焼き")
    if candidate.has_shio_tataki:
        features.append("塩たたき")
    if candidate.has_seasonal_katsuo:
        features.append("旬のカツオ")
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

    katsuo_features = Decimal("8")
    if candidate.has_warayaki:
        katsuo_features += Decimal("5")
    if candidate.has_shio_tataki:
        katsuo_features += Decimal("4")
    if candidate.has_seasonal_katsuo:
        katsuo_features += Decimal("3")

    unique_sources = {
        _normalized_host(candidate.evidence_url),
        *(_normalized_host(url) for url in candidate.source_urls),
    }
    independent_sources = Decimal(min(len(unique_sources), 5) * 2)

    reputation = summarize_review_reputation(candidate)
    rating_points = (
        Decimal(str(reputation.average_rating)) / Decimal("5") * Decimal("20")
    )
    volume_points = (
        Decimal(min(reputation.review_count, 5)) / Decimal("5") * Decimal("3")
    )
    diversity_points = (
        Decimal(min(reputation.source_count, 2)) / Decimal("2") * Decimal("2")
    )
    recent_reviews = rating_points + volume_points + diversity_points

    distance_ratio = Decimal(str(candidate.distance_km)) / Decimal(str(max_distance_km))
    distance = max(Decimal("0"), Decimal("20") * (Decimal("1") - distance_ratio))

    return ScoreBreakdown(
        evidence=_round(evidence),
        katsuo_features=_round(katsuo_features),
        independent_sources=_round(independent_sources),
        recent_reviews=_round(recent_reviews),
        distance=_round(distance),
    )


def rank_top_five(
    candidates: list[RestaurantCandidate],
    max_distance_km: float,
) -> list[RankedRestaurant]:
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
        for index, (candidate, breakdown, reputation) in enumerate(scored[:5], start=1)
    ]
