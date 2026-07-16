from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from datetime import date
from functools import lru_cache
from urllib.parse import urlsplit

from .models import RecentReview, RestaurantCandidateInput, ScrapedPage
from .scraping import canonical_url

KATSUO_TERMS = ("カツオ", "かつお", "鰹", "katsuo")
_NAME_NOISE = (
    "高知",
    "本店",
    "支店",
    "ひろめ店",
    "土佐料理",
    "居酒屋",
    "酒場",
    "藁焼き鰹たたき",
    "わら焼き",
    "Restaurant",
)
_LOCATION_NAME_NOISE = (
    "高知",
    "土佐料理",
    "居酒屋",
    "酒場",
    "藁焼き鰹たたき",
    "わら焼き",
    "Restaurant",
)
_DASHES = str.maketrans({character: "-" for character in "‐‑‒–—―−ーｰ－"})
_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


# Validation passes re-normalize the same page bodies (up to 100k characters)
# once per checked claim, so memoizing this pure function removes the dominant
# repeated cost without changing any output.
@lru_cache(maxsize=4096)
def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(_DASHES).casefold()
    return re.sub(r"[^0-9a-zぁ-んァ-ヶ一-龠]+", "", normalized)


def _name_aliases(name: str) -> set[str]:
    aliases = {normalize_text(name)}
    without_noise = name
    for noise in _NAME_NOISE:
        without_noise = without_noise.replace(noise, "")
    aliases.add(normalize_text(without_noise))
    aliases.update(
        normalize_text(part)
        for part in re.split(r"[\s　・（）()]+", name)
        if len(normalize_text(part)) >= 3 and part not in _NAME_NOISE
    )
    return {alias for alias in aliases if len(alias) >= 3}


def _page_names_restaurant(candidate: RestaurantCandidateInput, page: ScrapedPage) -> bool:
    page_text = normalize_text(f"{page.title}\n{page.content}")
    return any(alias in page_text for alias in _name_aliases(candidate.name))


def _location_name_aliases(name: str) -> set[str]:
    aliases = {
        normalize_text(name),
        normalize_text(name.replace("ひろめ市場店", "ひろめ店")),
    }
    without_noise = name
    for noise in _LOCATION_NAME_NOISE:
        without_noise = without_noise.replace(noise, "")
    aliases.add(normalize_text(without_noise))
    return {alias for alias in aliases if len(alias) >= 4}


def _page_identifies_location(
    candidate: RestaurantCandidateInput,
    page: ScrapedPage,
) -> bool:
    if _page_names_address(candidate, page):
        return True
    page_text = normalize_text(f"{page.title}\n{page.content}")
    return any(alias in page_text for alias in _location_name_aliases(candidate.name))


def _address_anchor(address: str) -> str:
    value = unicodedata.normalize("NFKC", address).translate(_DASHES)
    value = re.sub(r"^.*?高知市", "", value)
    value = re.sub(r"\s+", "", value)
    match = re.match(r"(.+?\d+(?:-\d+){1,4})", value)
    return normalize_text(match.group(1) if match else value)


def _page_names_address(candidate: RestaurantCandidateInput, page: ScrapedPage) -> bool:
    anchor = _address_anchor(candidate.address)
    return len(anchor) >= 5 and anchor in normalize_text(page.content)


def _page_names_katsuo_dish(candidate: RestaurantCandidateInput, page: ScrapedPage) -> bool:
    page_text = normalize_text(page.content)
    if not any(normalize_text(term) in page_text for term in KATSUO_TERMS):
        return False
    exact_dish = normalize_text(candidate.katsuo_dish)
    if exact_dish and exact_dish in page_text:
        return True
    modifiers = {
        normalize_text(term)
        for term in ("藁焼き", "わら焼き", "塩たたき", "タタキ", "たたき")
        if normalize_text(term) in normalize_text(candidate.katsuo_dish)
    }
    return not modifiers or any(modifier in page_text for modifier in modifiers)


def _page_supports_feature(page: ScrapedPage, terms: tuple[str, ...]) -> bool:
    page_text = normalize_text(page.content)
    return any(normalize_text(term) in page_text for term in terms)


def _validated_claim_pages(
    candidate: RestaurantCandidateInput,
    pages: Mapping[str, ScrapedPage],
) -> tuple[list[ScrapedPage], list[object]]:
    claim_pages: list[ScrapedPage] = []
    valid_source_urls: list[object] = []
    evidence_page = find_scraped_page(pages, candidate.evidence_url)
    if (
        evidence_page is not None
        and _page_names_restaurant(candidate, evidence_page)
        and _page_names_address(candidate, evidence_page)
        and _page_names_katsuo_dish(candidate, evidence_page)
    ):
        claim_pages.append(evidence_page)
    for source_url in candidate.source_urls:
        source_page = find_scraped_page(pages, source_url)
        if (
            source_page is not None
            and _page_names_restaurant(candidate, source_page)
            and _page_identifies_location(candidate, source_page)
            and _page_names_katsuo_dish(candidate, source_page)
        ):
            claim_pages.append(source_page)
            valid_source_urls.append(source_url)
    return claim_pages, valid_source_urls


def sanitize_candidate_claims(
    candidate: RestaurantCandidateInput,
    pages: Mapping[str, ScrapedPage],
) -> RestaurantCandidateInput:
    """Remove optional sources and feature claims that scraped text cannot prove."""
    claim_pages, valid_source_urls = _validated_claim_pages(candidate, pages)

    def supports(terms: tuple[str, ...]) -> bool:
        return any(_page_supports_feature(page, terms) for page in claim_pages)

    return candidate.model_copy(
        update={
            "source_urls": valid_source_urls,
            "has_warayaki": candidate.has_warayaki
            and supports(("藁焼き", "藁焼", "わら焼き", "わら焼")),
            "has_shio_tataki": candidate.has_shio_tataki
            and supports(("塩たたき", "塩タタキ", "塩で食べ", "塩でいただ")),
            "has_seasonal_katsuo": candidate.has_seasonal_katsuo
            and supports(
                ("旬", "季節", "初鰹", "初かつお", "戻り鰹", "戻りかつお", "入荷")
            ),
        }
    )


def is_specific_review_url(value: object) -> bool:
    parts = urlsplit(str(value))
    host = (parts.hostname or "").casefold().removeprefix("www.")
    path = parts.path.rstrip("/")
    if not path:
        return False
    if host.endswith("google.com") and path in {"/maps", "/search"}:
        return False
    if host == "tabelog.com" and path in {"/kochi", "/rstlst"}:
        return False
    if host == "retty.me" and path in {"/area", "/restaurant"}:
        return False
    return len([part for part in path.split("/") if part]) >= 2


def find_scraped_page(
    pages: Mapping[str, ScrapedPage],
    url: object,
) -> ScrapedPage | None:
    key = canonical_url(url)
    page = pages.get(key)
    if page is not None:
        return page
    for candidate_page in pages.values():
        if canonical_url(candidate_page.final_url) == key:
            return candidate_page
    return None


def _date_tokens(value: date) -> set[str]:
    year = value.year
    month = value.month
    day = value.day
    if day == 1:
        tokens = {
            f"{year}-{month:02d}",
            f"{year}/{month:02d}",
            f"{year}年{month}月",
            f"{_MONTH_NAMES[month - 1]}{year}",
        }
    else:
        tokens = {
            f"{year}-{month:02d}-{day:02d}",
            f"{year}/{month:02d}/{day:02d}",
            f"{year}年{month}月{day}日",
            f"{_MONTH_NAMES[month - 1]}{day}{year}",
            f"{_MONTH_NAMES[month - 1]}{day:02d}{year}",
        }
    return {normalize_text(token) for token in tokens}


def _rating_is_present(text: str, rating: float) -> bool:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    value = f"{rating:g}"
    decimal_value = f"{rating:.1f}"
    patterns = {
        rf"(?<!\d){re.escape(value)}\s*(?:/\s*5|点|stars?|★)(?!\d)",
        rf"(?<!\d){re.escape(decimal_value)}(?:\s*/\s*5|\s*点|\s*stars?|\s*★)?(?!\d)",
    }
    if not rating.is_integer():
        patterns.add(rf"(?<!\d){re.escape(value)}(?!\d)")
    return any(re.search(pattern, normalized) for pattern in patterns)


def _review_facts_share_window(review: RecentReview, page: ScrapedPage) -> bool:
    lines = [line for line in page.content.splitlines() if line.strip()]
    reviewer = normalize_text(review.reviewer_name)
    for suffix in ("さん", "様", "氏"):
        reviewer = reviewer.removesuffix(normalize_text(suffix))
    date_tokens = _date_tokens(review.published_at)
    for index, line in enumerate(lines):
        if reviewer not in normalize_text(line):
            continue
        start = max(0, index - 4)
        end = min(len(lines), index + 25)
        window = "\n".join(lines[start:end])
        normalized_window = normalize_text(window)
        if any(token in normalized_window for token in date_tokens) and _rating_is_present(
            window, review.rating
        ):
            return True
    return False


def validate_candidate_references(
    candidate: RestaurantCandidateInput,
    pages: Mapping[str, ScrapedPage],
) -> list[str]:
    issues: list[str] = []
    evidence_page = find_scraped_page(pages, candidate.evidence_url)
    if evidence_page is None:
        issues.append("the katsuo evidence URL was not scraped")
    else:
        if not _page_names_restaurant(candidate, evidence_page):
            issues.append("the katsuo evidence page does not name this restaurant")
        if not _page_names_address(candidate, evidence_page):
            issues.append("the katsuo evidence page does not confirm this address")
        if not _page_names_katsuo_dish(candidate, evidence_page):
            issues.append("the katsuo evidence page does not confirm the stated dish")

    for source_url in candidate.source_urls:
        source_page = find_scraped_page(pages, source_url)
        if source_page is None:
            issues.append(f"an additional source URL was not scraped ({source_url})")
            continue
        if not _page_names_restaurant(candidate, source_page):
            issues.append(
                f"an additional source does not name {candidate.name} ({source_url})"
            )
        if not _page_identifies_location(candidate, source_page):
            issues.append(
                f"an additional source does not confirm the branch or address ({source_url})"
            )
        if not _page_names_katsuo_dish(candidate, source_page):
            issues.append(
                f"an additional source does not confirm the katsuo dish ({source_url})"
            )

    review_fingerprints: set[tuple[str, str, date, float]] = set()
    for review in candidate.recent_reviews:
        url_key = canonical_url(review.review_url)
        fingerprint = (
            url_key,
            normalize_text(review.reviewer_name),
            review.published_at,
            review.rating,
        )
        if fingerprint in review_fingerprints:
            issues.append(
                f"a duplicate review identity for {review.reviewer_name} "
                f"on {review.published_at.isoformat()}"
            )
            continue
        review_fingerprints.add(fingerprint)

        if not is_specific_review_url(review.review_url):
            issues.append(f"a generic review URL ({review.review_url})")
            continue
        review_page = find_scraped_page(pages, review.review_url)
        if review_page is None:
            issues.append(f"an unscraped review URL ({review.review_url})")
            continue
        if not is_specific_review_url(review_page.final_url):
            issues.append(
                f"a review URL redirected to a generic page ({review_page.final_url})"
            )
            continue
        if not _page_names_restaurant(candidate, review_page):
            issues.append(
                f"a review page that does not name {candidate.name} ({review.review_url})"
            )
            continue
        if not _page_identifies_location(candidate, review_page):
            issues.append(
                "a review page that does not confirm the branch or address "
                f"({review.review_url})"
            )
            continue
        if not _review_facts_share_window(review, review_page):
            issues.append(
                "a review whose reviewer, date, and rating cannot be verified together "
                f"({review.review_url})"
            )
    return issues


def scraped_pages_for_candidate(
    candidate: RestaurantCandidateInput,
    pages: Mapping[str, ScrapedPage],
) -> list[ScrapedPage]:
    urls = [
        candidate.evidence_url,
        *candidate.source_urls,
        *(review.review_url for review in candidate.recent_reviews),
    ]
    selected: dict[str, ScrapedPage] = {}
    for url in urls:
        page = find_scraped_page(pages, url)
        if page is not None:
            selected[canonical_url(page.requested_url)] = page
    return list(selected.values())
