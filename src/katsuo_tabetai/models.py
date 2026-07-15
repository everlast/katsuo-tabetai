from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, HttpUrl, WithJsonSchema


# Responses function tools reject Pydantic's `format: uri`, while HttpUrl still
# provides the runtime validation and parsed URL attributes used by this app.
FunctionToolHttpUrl = Annotated[
    HttpUrl,
    WithJsonSchema({"type": "string"}, mode="validation"),
]
FunctionToolDate = Annotated[
    date,
    WithJsonSchema({"type": "string"}, mode="validation"),
]
ReviewPoint = Annotated[str, Field(min_length=1, max_length=60)]


class EvidenceSourceType(StrEnum):
    OFFICIAL_RESTAURANT = "official_restaurant"
    OFFICIAL_TOURISM = "official_tourism"
    RESERVATION_SITE = "reservation_site"
    REVIEW_SITE = "review_site"


class HotelLocation(BaseModel):
    name: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class RecentReview(BaseModel):
    source_name: str = Field(
        min_length=1,
        max_length=80,
        description="The review platform or publication name.",
    )
    review_url: FunctionToolHttpUrl = Field(
        description="A page where this specific review can be verified.",
    )
    published_at: FunctionToolDate = Field(
        description="The explicitly displayed publication date in YYYY-MM-DD form.",
    )
    rating: float = Field(
        ge=1,
        le=5,
        description="The review's displayed rating normalized to a 1-5 scale.",
    )
    summary: str = Field(
        min_length=1,
        max_length=240,
        description="A short paraphrase of the review, not a verbatim quotation.",
    )
    positive_points: list[ReviewPoint] = Field(
        min_length=1,
        max_length=3,
        description="One to three short aspects explicitly praised by the reviewer.",
    )
    caution_points: list[ReviewPoint] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three short cautions explicitly mentioned by the reviewer.",
    )


class RestaurantCandidateInput(BaseModel):
    name: str = Field(min_length=1)
    address: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    katsuo_dish: str = Field(
        min_length=1,
        description="The exact katsuo dish named by the evidence page.",
    )
    evidence_url: FunctionToolHttpUrl = Field(
        description="A page that explicitly names this restaurant's katsuo dish.",
    )
    evidence_source_type: EvidenceSourceType
    source_urls: list[FunctionToolHttpUrl] = Field(
        default_factory=list,
        description="Additional independent pages used to verify the restaurant.",
    )
    recent_reviews: list[RecentReview] = Field(
        min_length=3,
        max_length=8,
        description=(
            "Three to eight distinct recent reviews with dates, ratings, paraphrased "
            "summaries, and verifiable URLs."
        ),
    )
    has_warayaki: bool
    has_shio_tataki: bool
    has_seasonal_katsuo: bool


class RestaurantCandidate(RestaurantCandidateInput):
    distance_km: float = Field(ge=0)
    within_range: bool


class CandidateStore(BaseModel):
    schema_version: int = 2
    generated_at: datetime
    hotel: HotelLocation
    max_distance_km: float = Field(gt=0)
    candidates: list[RestaurantCandidate]


class ScoreBreakdown(BaseModel):
    evidence: float
    katsuo_features: float
    independent_sources: float
    recent_reviews: float
    distance: float

    @property
    def total(self) -> float:
        return round(
            self.evidence
            + self.katsuo_features
            + self.independent_sources
            + self.recent_reviews
            + self.distance,
            2,
        )


class ReviewReputation(BaseModel):
    average_rating: float = Field(ge=1, le=5)
    review_count: int = Field(ge=3)
    source_count: int = Field(ge=1)
    top_positive_points: list[str]
    caution_points: list[str]


class RankedRestaurant(RestaurantCandidate):
    rank: int = Field(ge=1)
    score: float = Field(ge=0, le=100)
    score_breakdown: ScoreBreakdown
    review_reputation: ReviewReputation
    recommendation_reason: str = Field(min_length=1)


class TopFiveStore(BaseModel):
    schema_version: int = 2
    generated_at: datetime
    hotel: HotelLocation
    max_distance_km: float = Field(gt=0)
    restaurants: list[RankedRestaurant] = Field(min_length=5, max_length=5)
