from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from math import asin, cos, radians, sin, sqrt

from .models import (
    EvidenceSourceType,
    HotelLocation,
    RankedRestaurant,
    RestaurantCandidate,
    RestaurantCandidateInput,
    ScoreBreakdown,
)

EARTH_RADIUS_KM = 6371.0088
TWO_PLACES = Decimal("0.01")

EVIDENCE_POINTS = {
    EvidenceSourceType.OFFICIAL_RESTAURANT: Decimal("35"),
    EvidenceSourceType.OFFICIAL_TOURISM: Decimal("29"),
    EvidenceSourceType.RESERVATION_SITE: Decimal("22"),
    EvidenceSourceType.REVIEW_SITE: Decimal("14"),
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


def score_candidate(
    candidate: RestaurantCandidate,
    max_distance_km: float,
) -> ScoreBreakdown:
    """Calculate a deterministic 100-point score from stored facts."""
    evidence = EVIDENCE_POINTS[candidate.evidence_source_type]

    katsuo_features = Decimal("10")
    if candidate.has_warayaki:
        katsuo_features += Decimal("8")
    if candidate.has_shio_tataki:
        katsuo_features += Decimal("5")
    if candidate.has_seasonal_katsuo:
        katsuo_features += Decimal("5")

    unique_sources = {
        candidate.evidence_url.host,
        *(url.host for url in candidate.source_urls),
    }
    independent_sources = Decimal(min(len(unique_sources), 3) * 4)

    distance_ratio = Decimal(str(candidate.distance_km)) / Decimal(str(max_distance_km))
    distance = max(Decimal("0"), Decimal("25") * (Decimal("1") - distance_ratio))

    return ScoreBreakdown(
        evidence=_round(evidence),
        katsuo_features=_round(katsuo_features),
        independent_sources=_round(independent_sources),
        distance=_round(distance),
    )


def rank_top_five(
    candidates: list[RestaurantCandidate],
    max_distance_km: float,
) -> list[RankedRestaurant]:
    scored = [
        (candidate, score_candidate(candidate, max_distance_km))
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
        )
        for index, (candidate, breakdown) in enumerate(scored[:5], start=1)
    ]
