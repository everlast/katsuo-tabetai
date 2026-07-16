from __future__ import annotations

from datetime import date

from katsuo_tabetai.context import KatsuoContext
from katsuo_tabetai.evidence import (
    _BoundedPageTextCache,
    is_specific_review_url,
    normalize_text,
    sanitize_candidate_claims,
    validate_candidate_references,
)
from katsuo_tabetai.models import HotelLocation, RecentReview, RestaurantCandidateInput
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


def test_page_text_cache_reuses_normalization_and_keeps_output(tmp_path) -> None:
    from helpers import _page

    cache = _BoundedPageTextCache(max_total_chars=1_000_000, max_entries=1_024)
    page = _page("https://example.com/menu", "店舗ページ", "カツオのたたき\n" * 500)

    first = cache.normalized_page_text(page, include_title=True)
    second = cache.normalized_page_text(page, include_title=True)

    assert first == normalize_text(f"{page.title}\n{page.content}")
    assert second is first
    assert cache.normalized_page_text(page, include_title=False) == normalize_text(
        page.content
    )


def test_page_text_cache_does_not_serve_stale_text_for_modified_content(tmp_path) -> None:
    from helpers import _page

    cache = _BoundedPageTextCache(max_total_chars=1_000_000, max_entries=1_024)
    page = _page("https://example.com/menu", "店舗ページ", "住所は高知市本町1-1")
    assert "本町11" in cache.normalized_page_text(page, include_title=False)

    # content_sha256 が本文と食い違う model_copy 由来のページでも、
    # 正規化結果は必ず実際の本文に追随する（古いキャッシュを返さない）。
    modified = page.model_copy(update={"content": "住所は高知市帯屋町9-9"})
    modified_text = cache.normalized_page_text(modified, include_title=False)
    assert "帯屋町99" in modified_text
    assert "本町11" not in modified_text


def test_page_text_cache_never_exceeds_total_char_ceiling(tmp_path) -> None:
    from helpers import _page

    ceiling = 2_000
    cache = _BoundedPageTextCache(max_total_chars=ceiling, max_entries=1_024)
    for index in range(50):
        # 各ページの正規化結果は約600文字。上限を跨ぐと保持分が破棄される。
        page = _page(
            f"https://example.com/{index}",
            f"店舗{index}",
            f"かつお{index}" * 150,
        )
        result = cache.normalized_page_text(page, include_title=False)
        assert result == normalize_text(page.content)
        assert cache.total_chars <= ceiling

    oversized = _page(
        "https://example.com/oversized",
        "巨大ページ",
        "鰹" * (ceiling + 100),
    )
    before = cache.total_chars
    assert cache.normalized_page_text(oversized, include_title=False) == normalize_text(
        oversized.content
    )
    # 上限を単体で超える本文はキャッシュへ保持しない。
    assert cache.total_chars == before


def test_page_text_cache_bounds_entry_count_for_tiny_normalizations(tmp_path) -> None:
    from helpers import _page

    max_entries = 64
    cache = _BoundedPageTextCache(max_total_chars=1_000_000, max_entries=max_entries)
    for index in range(500):
        # 正規化結果が空になる記号だけの本文は total_chars を増やさないため、
        # SHA-256キーとdictエントリの増加は件数上限で頭打ちにする。
        page = _page(
            f"https://example.com/{index}",
            "記号のみ",
            "★" * (index + 1),
        )
        assert cache.normalized_page_text(page, include_title=False) == ""
        assert cache.entry_count <= max_entries
    assert cache.total_chars == 0


def test_duplicate_review_identity_is_rejected(tmp_path) -> None:
    context = _context(tmp_path)
    candidate = _candidate()
    original = candidate.recent_reviews[0]
    duplicate = RecentReview.model_validate(
        {
            **original.model_dump(mode="json"),
            # 表記ゆれ（大文字・全角空白・末尾スラッシュ）でも同一指紋として弾く。
            "reviewer_name": f"{original.reviewer_name.upper()}　",
            "review_url": f"{original.review_url}/",
        }
    )
    candidate = candidate.model_copy(
        update={"recent_reviews": [original, duplicate, *candidate.recent_reviews[1:]]}
    )
    populate_scraped_pages(context, [candidate])

    issues = validate_candidate_references(candidate, context.scraped_pages)

    assert issues == [
        "a duplicate review identity for "
        f"{duplicate.reviewer_name} on {duplicate.published_at.isoformat()}"
    ]


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
