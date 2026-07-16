from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from katsuo_tabetai.context import KatsuoContext
from katsuo_tabetai.models import RestaurantCandidateInput, ScrapedPage
from katsuo_tabetai.scraping import canonical_url


def _page(url: object, title: str, content: str) -> ScrapedPage:
    return ScrapedPage(
        requested_url=str(url),
        final_url=str(url),
        fetched_at=datetime.now(timezone.utc),
        status_code=200,
        title=title,
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def populate_scraped_pages(
    context: KatsuoContext,
    candidates: list[RestaurantCandidateInput],
) -> None:
    for candidate in candidates:
        feature_evidence = []
        if candidate.has_warayaki:
            feature_evidence.append("藁焼き")
        if candidate.has_shio_tataki:
            feature_evidence.append("塩たたき")
        if candidate.has_seasonal_katsuo:
            feature_evidence.append("旬のカツオを入荷")
        evidence_content = "\n".join(
            [
                candidate.name,
                candidate.address,
                candidate.katsuo_dish,
                *feature_evidence,
                "Official restaurant menu and location information.",
            ]
        )
        evidence_page = _page(candidate.evidence_url, candidate.name, evidence_content)
        context.scraped_pages[canonical_url(candidate.evidence_url)] = evidence_page
        for source_url in candidate.source_urls:
            source_page = _page(source_url, candidate.name, evidence_content)
            context.scraped_pages[canonical_url(source_url)] = source_page
        for review in candidate.recent_reviews:
            review_content = "\n".join(
                [
                    candidate.name,
                    candidate.address,
                    review.reviewer_name,
                    review.published_at.isoformat(),
                    f"{review.rating:g} / 5",
                    review.summary,
                ]
            )
            review_page = _page(review.review_url, candidate.name, review_content)
            context.scraped_pages[canonical_url(review.review_url)] = review_page
