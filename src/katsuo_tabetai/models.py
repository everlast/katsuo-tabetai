from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class EvidenceSourceType(StrEnum):
    OFFICIAL_RESTAURANT = "official_restaurant"
    OFFICIAL_TOURISM = "official_tourism"
    RESERVATION_SITE = "reservation_site"
    REVIEW_SITE = "review_site"


class HotelLocation(BaseModel):
    name: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class RestaurantCandidateInput(BaseModel):
    name: str = Field(min_length=1)
    address: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    katsuo_dish: str = Field(
        min_length=1,
        description="The exact katsuo dish named by the evidence page.",
    )
    evidence_url: HttpUrl = Field(
        description="A page that explicitly names this restaurant's katsuo dish.",
    )
    evidence_source_type: EvidenceSourceType
    source_urls: list[HttpUrl] = Field(
        default_factory=list,
        description="Additional independent pages used to verify the restaurant.",
    )
    has_warayaki: bool
    has_shio_tataki: bool
    has_seasonal_katsuo: bool


class RestaurantCandidate(RestaurantCandidateInput):
    distance_km: float = Field(ge=0)
    within_range: bool


class CandidateStore(BaseModel):
    schema_version: int = 1
    generated_at: datetime
    hotel: HotelLocation
    max_distance_km: float = Field(gt=0)
    candidates: list[RestaurantCandidate]


class ScoreBreakdown(BaseModel):
    evidence: float
    katsuo_features: float
    independent_sources: float
    distance: float

    @property
    def total(self) -> float:
        return round(
            self.evidence
            + self.katsuo_features
            + self.independent_sources
            + self.distance,
            2,
        )


class RankedRestaurant(RestaurantCandidate):
    rank: int = Field(ge=1)
    score: float = Field(ge=0, le=100)
    score_breakdown: ScoreBreakdown


class TopFiveStore(BaseModel):
    schema_version: int = 1
    generated_at: datetime
    hotel: HotelLocation
    max_distance_km: float = Field(gt=0)
    restaurants: list[RankedRestaurant] = Field(min_length=5, max_length=5)
