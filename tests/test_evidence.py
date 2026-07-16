from __future__ import annotations

from datetime import date

from katsuo_tabetai.context import KatsuoContext
from katsuo_tabetai.evidence import (
    is_specific_review_url,
    sanitize_candidate_claims,
    validate_candidate_references,
)
from katsuo_tabetai.models import HotelLocation, RestaurantCandidateInput
from katsuo_tabetai.scraping import canonical_url

from helpers import populate_scraped_pages
from test_scoring import make_candidate


def _candidate(index: int = 1) -> RestaurantCandidateInput:
    candidate = make_candidate(index)
    return RestaurantCandidateInput.model_validate(
        candidate.model_dump(exclude={"distance_km", "within_range"})
    )


def _context(tmp_path) -> KatsuoContext:
    return KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )


def test_verified_candidate_references_pass(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate()
    populate_scraped_pages(context, [candidate])

    assert validate_candidate_references(candidate, context.scraped_pages) == []


def test_generic_review_homepage_is_rejected(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate()
    generic_review = candidate.recent_reviews[0].model_copy(
        update={"review_url": "https://www.google.com/maps"}
    )
    candidate = candidate.model_copy(
        update={"recent_reviews": [generic_review, *candidate.recent_reviews[1:]]}
    )
    populate_scraped_pages(context, [candidate])

    issues = validate_candidate_references(candidate, context.scraped_pages)

    assert any("generic review URL" in issue for issue in issues)
    assert is_specific_review_url("https://tabelog.com/") is False
    assert is_specific_review_url("https://retty.me/") is False


def test_review_for_another_restaurant_is_rejected(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate()
    populate_scraped_pages(context, [candidate])
    review = candidate.recent_reviews[0]
    page = context.scraped_pages[canonical_url(review.review_url)]
    context.scraped_pages[canonical_url(review.review_url)] = page.model_copy(
        update={
            "title": "Another restaurant",
            "content": "\n".join(
                [
                    "Another restaurant",
                    candidate.address,
                    review.reviewer_name,
                    review.published_at.isoformat(),
                    f"{review.rating:g} / 5",
                    "A review of a different venue.",
                ]
            ),
        }
    )

    issues = validate_candidate_references(candidate, context.scraped_pages)

    assert any("does not name" in issue for issue in issues)


def test_wrong_branch_address_is_rejected(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate()
    populate_scraped_pages(context, [candidate])
    page = context.scraped_pages[canonical_url(candidate.evidence_url)]
    context.scraped_pages[canonical_url(candidate.evidence_url)] = page.model_copy(
        update={"content": page.content.replace(candidate.address, "Kochi 999")}
    )

    issues = validate_candidate_references(candidate, context.scraped_pages)

    assert any("does not confirm this address" in issue for issue in issues)


def test_unscraped_additional_source_is_rejected(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate()
    populate_scraped_pages(context, [candidate])
    context.scraped_pages.pop(canonical_url(candidate.source_urls[0]))

    sanitized = sanitize_candidate_claims(candidate, context.scraped_pages)

    assert sanitized.source_urls == []
    assert validate_candidate_references(sanitized, context.scraped_pages) == []


def test_unverified_feature_claim_is_rejected(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate().model_copy(update={"has_warayaki": True})
    populate_scraped_pages(context, [candidate])
    claim_urls = [candidate.evidence_url, *candidate.source_urls]
    for claim_url in claim_urls:
        key = canonical_url(claim_url)
        page = context.scraped_pages[key]
        context.scraped_pages[key] = page.model_copy(
            update={"content": page.content.replace("藁焼き", "炭火焼き")}
        )

    sanitized = sanitize_candidate_claims(candidate, context.scraped_pages)

    assert sanitized.has_warayaki is False
    assert validate_candidate_references(sanitized, context.scraped_pages) == []


def test_review_for_another_branch_is_rejected(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate().model_copy(update={"name": "Same Chain Main Branch"})
    populate_scraped_pages(context, [candidate])
    review = candidate.recent_reviews[0]
    key = canonical_url(review.review_url)
    page = context.scraped_pages[key]
    context.scraped_pages[key] = page.model_copy(
        update={
            "title": "Same Chain Other Branch",
            "content": page.content.replace(
                candidate.name, "Same Chain Other Branch"
            ).replace(candidate.address, "Kochi 999"),
        }
    )

    issues = validate_candidate_references(candidate, context.scraped_pages)

    assert any("does not confirm the branch or address" in issue for issue in issues)


def test_review_requires_reviewer_date_and_rating_in_same_window(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate()
    populate_scraped_pages(context, [candidate])
    review = candidate.recent_reviews[0]
    page = context.scraped_pages[canonical_url(review.review_url)]
    context.scraped_pages[canonical_url(review.review_url)] = page.model_copy(
        update={"content": page.content.replace(review.reviewer_name, "Unknown reviewer")}
    )

    issues = validate_candidate_references(candidate, context.scraped_pages)

    assert any("cannot be verified together" in issue for issue in issues)


def test_review_facts_allow_tabelog_style_metadata_block(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate()
    populate_scraped_pages(context, [candidate])
    review = candidate.recent_reviews[0].model_copy(
        update={
            "reviewer_name": "Tabelog Reviewerさん",
            "published_at": date(2026, 6, 1),
            "rating": 4.0,
        }
    )
    candidate = candidate.model_copy(
        update={"recent_reviews": [review, *candidate.recent_reviews[1:]]}
    )
    key = canonical_url(review.review_url)
    page = context.scraped_pages[key]
    metadata = [
        "Tabelog Reviewer",
        "口コミ115件",
        "フォロワー17人",
        "4.0",
        "予算",
        "1人",
        "詳細",
        "料理・味",
        "-",
        "サービス",
        "-",
        "雰囲気",
        "-",
        "CP",
        "-",
        "酒・ドリンク",
        "-",
        "2026/06訪問",
    ]
    context.scraped_pages[key] = page.model_copy(
        update={
            "content": "\n".join(
                [candidate.name, candidate.address, *metadata, review.summary]
            )
        }
    )

    assert validate_candidate_references(candidate, context.scraped_pages) == []


def test_hirome_market_branch_alias_is_accepted(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate().model_copy(update={"name": "明神丸 ひろめ市場店"})
    populate_scraped_pages(context, [candidate])
    review = candidate.recent_reviews[0]
    key = canonical_url(review.review_url)
    page = context.scraped_pages[key]
    context.scraped_pages[key] = page.model_copy(
        update={
            "title": "明神丸 ひろめ店",
            "content": page.content.replace(candidate.name, "明神丸 ひろめ店").replace(
                candidate.address, "address omitted"
            ),
        }
    )

    assert validate_candidate_references(candidate, context.scraped_pages) == []


def test_month_only_review_date_accepts_first_day_normalization(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate()
    review = candidate.recent_reviews[0].model_copy(update={"published_at": date(2026, 6, 1)})
    candidate = candidate.model_copy(
        update={"recent_reviews": [review, *candidate.recent_reviews[1:]]}
    )
    populate_scraped_pages(context, [candidate])
    page = context.scraped_pages[canonical_url(review.review_url)]
    context.scraped_pages[canonical_url(review.review_url)] = page.model_copy(
        update={"content": page.content.replace("2026-06-01", "2026-06")}
    )

    assert validate_candidate_references(candidate, context.scraped_pages) == []
