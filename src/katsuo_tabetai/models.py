from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl, WithJsonSchema, field_validator


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
PositiveReviewPoint = Annotated[str, Field(min_length=2, max_length=30)]
ReviewPoint = Annotated[str, Field(min_length=1, max_length=60)]

_POINT_LIST_SEPARATOR = re.compile(r'["\']\s*,\s*["\']')
_POINT_INSTRUCTION_SUFFIX = re.compile(
    r"\s*(?:[.。]\s*)?\d+\s*(?:-|~|〜|～)\s*\d+\s*$"
)
_POINT_FIELD_NAME = re.compile(r"(?:positive|caution)_points", re.IGNORECASE)
_POINT_WRAPPER_CHARS = " \t\r\n\"'[]{}(),.、。"


def _normalize_review_points(value: object) -> object:
    """Remove structured-output artifacts occasionally leaked into point text."""
    if not isinstance(value, list):
        return value

    normalized: list[object] = []
    for item in value:
        if not isinstance(item, str):
            normalized.append(item)
            continue
        for fragment in _POINT_LIST_SEPARATOR.split(item):
            fragment = fragment.strip(_POINT_WRAPPER_CHARS)
            fragment = _POINT_INSTRUCTION_SUFFIX.sub("", fragment)
            fragment = fragment.strip(_POINT_WRAPPER_CHARS)
            if not fragment or _POINT_FIELD_NAME.search(fragment):
                continue
            if any(character in fragment for character in "[]{}"):
                continue
            if fragment not in normalized:
                normalized.append(fragment)
    return normalized[:3]


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
    reviewer_name: str = Field(
        min_length=1,
        max_length=120,
        description="The reviewer name displayed next to this review.",
    )
    published_at: FunctionToolDate = Field(
        description=(
            "The explicitly displayed review publication or visit date. If the "
            "source displays only YYYY-MM, normalize it to YYYY-MM-01. Never infer "
            "the displayed year or month."
        ),
    )
    rating: float = Field(
        ge=1,
        le=5,
        description="The review's displayed rating normalized to a 1-5 scale.",
    )
    summary: str = Field(
        min_length=1,
        max_length=500,
        description="A short paraphrase of the review, not a verbatim quotation.",
    )
    positive_points: list[PositiveReviewPoint] = Field(
        min_length=1,
        max_length=3,
        description="One to three short aspects explicitly praised by the reviewer.",
    )
    caution_points: list[ReviewPoint] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three short cautions explicitly mentioned by the reviewer.",
    )

    @field_validator("positive_points", "caution_points", mode="before")
    @classmethod
    def normalize_points(cls, value: object) -> object:
        return _normalize_review_points(value)


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
        default_factory=list,
        max_length=10,
        description=(
            "Zero to ten reviews discovered during collection. A restaurant is "
            "still collected when fewer than five reviews are available. Ranking "
            "eligibility later requires at least five verified recent reviews "
            "from at least two source sites."
        ),
    )
    has_warayaki: bool
    has_shio_tataki: bool
    has_seasonal_katsuo: bool


class ResearchBatch(BaseModel):
    candidates: list[RestaurantCandidateInput] = Field(min_length=5, max_length=30)


class ScrapedPage(BaseModel):
    requested_url: HttpUrl
    final_url: HttpUrl
    fetched_at: datetime
    status_code: int = Field(ge=200, lt=300)
    title: str = Field(default="", max_length=500)
    content: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RestaurantCacheEntry(BaseModel):
    schema_version: Literal[2] = 2
    updated_at: datetime
    candidate: RestaurantCandidateInput
    scraped_pages: list[ScrapedPage]


class RestaurantCandidate(RestaurantCandidateInput):
    distance_km: float = Field(ge=0)
    within_range: bool


class CandidateStore(BaseModel):
    schema_version: Literal[3] = 3
    generated_at: datetime
    model: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    context_markdown: str = "context.md"
    hotel: HotelLocation
    max_distance_km: float = Field(gt=0)
    candidates: list[RestaurantCandidate]
    scraped_pages: list[ScrapedPage]


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
    review_count: int = Field(ge=5)
    source_count: int = Field(ge=2)
    top_positive_points: list[str]
    caution_points: list[str]


class RankedRestaurant(RestaurantCandidate):
    rank: int = Field(ge=1)
    score: float = Field(ge=0, le=100)
    score_breakdown: ScoreBreakdown
    review_reputation: ReviewReputation
    recommendation_reason: str = Field(min_length=1)


class TopFiveStore(BaseModel):
    schema_version: Literal[3] = 3
    generated_at: datetime
    model: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    context_markdown: str = "context.md"
    hotel: HotelLocation
    max_distance_km: float = Field(gt=0)
    restaurants: list[RankedRestaurant] = Field(min_length=5, max_length=5)
