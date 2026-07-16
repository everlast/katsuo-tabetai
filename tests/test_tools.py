from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from katsuo_tabetai.candidates import merge_candidate_observations
from katsuo_tabetai.context import KatsuoContext
from katsuo_tabetai.models import (
    HotelLocation,
    RestaurantCandidateInput,
    TopFiveStore,
)
from katsuo_tabetai.scraping import canonical_url
from katsuo_tabetai.tools import (
    MIN_REVIEW_SOURCE_SITES,
    RECENT_REVIEW_MAX_AGE_DAYS,
    cache_restaurant_candidates,
    create_top_five_report,
    deduplicate_restaurant_candidates,
    load_cached_restaurant_candidates,
    partition_candidates_by_review_validity,
    persist_restaurant_candidates,
)

from test_scoring import HOTEL, make_candidate
from helpers import populate_scraped_pages


def test_legacy_tools_namespace_is_reexported() -> None:
    """モジュール分割前に katsuo_tabetai.tools から import できた公開名を維持する。"""
    import katsuo_tabetai.tools as tools_module

    legacy_names = (
        "CandidateStore",
        "KatsuoContext",
        "RecentReview",
        "RestaurantCacheEntry",
        "RestaurantCandidate",
        "RestaurantCandidateInput",
        "ScrapedPage",
        "TopFiveStore",
        "apply_range_rule",
        "canonical_url",
        "haversine_km",
        "normalized_url_host",
        "rank_top_five",
        "render_context_markdown",
        "render_top_five_html",
        "sanitize_candidate_claims",
        "scraped_pages_for_candidate",
        "validate_candidate_references",
    )
    for name in legacy_names:
        assert hasattr(tools_module, name), f"tools.{name} must stay importable"
        assert name in tools_module.__all__, f"tools.{name} must stay in __all__"


def candidate_input(index: int) -> RestaurantCandidateInput:
    candidate = make_candidate(index)
    return RestaurantCandidateInput.model_validate(
        candidate.model_dump(exclude={"distance_km", "within_range"})
    )


def test_function_tool_core_saves_structured_data_and_html(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(
            name="Test Hotel",
            latitude=33.566927593644714,
            longitude=133.54104073018118,
        ),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )

    candidates = [candidate_input(index) for index in range(1, 7)]
    populate_scraped_pages(context, candidates)
    save_result = persist_restaurant_candidates(context, candidates)
    report_result = create_top_five_report(context)

    assert save_result["within_range"] == 6
    assert context.candidates_path.exists()
    assert context.context_markdown_path.exists()
    assert context.scrape_manifest_path.exists()
    assert context.top_five_path.exists()
    assert context.html_path.exists()
    top_five = TopFiveStore.model_validate_json(
        context.top_five_path.read_text(encoding="utf-8")
    )
    assert len(top_five.restaurants) == 5
    assert all(item.recommendation_reason for item in top_five.restaurants)
    assert all(len(item.recent_reviews) >= 5 for item in top_five.restaurants)
    assert report_result["status"] == "completed"
    markdown = context.context_markdown_path.read_text(encoding="utf-8")
    assert "# Katsuo Restaurant Context" in markdown
    assert "gpt-5.6-luna" in markdown
    assert "Reviewer 1-1" in markdown
    assert str(candidates[0].source_urls[0]) in markdown
    assert "Verified features" in markdown
    manifest = context.scrape_manifest_path.read_text(encoding="utf-8")
    assert '"content"' not in manifest


def test_candidate_save_keeps_same_name_at_distinct_locations(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(
            name="Test Hotel",
            latitude=33.566927593644714,
            longitude=133.54104073018118,
        ),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    branch_a = candidate_input(1).model_copy(update={"name": "Same Chain"})
    branch_b = candidate_input(2).model_copy(
        update={
            "name": "Same Chain",
            "address": "Kochi distinct branch",
            "latitude": branch_a.latitude + 0.01,
        }
    )
    candidates = [
        branch_a,
        branch_b,
        *[candidate_input(index) for index in range(3, 6)],
    ]
    populate_scraped_pages(context, candidates)

    save_result = persist_restaurant_candidates(context, candidates)

    assert save_result["unique_candidates"] == 5
    assert save_result["within_range"] == 5


def test_candidate_save_rejects_reviews_outside_recent_window(tmp_path) -> None:
    candidate = candidate_input(1)
    stale_reviews = [
        review.model_copy(
            update={
                "published_at": datetime.now(timezone.utc).date()
                - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS + 1)
            }
        )
        for review in candidate.recent_reviews
    ]
    candidate = candidate.model_copy(update={"recent_reviews": stale_reviews})
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="older than"):
        persist_restaurant_candidates(context, [candidate] * 5)


def test_research_partition_rejects_only_candidate_with_stale_review() -> None:
    accepted_candidate = candidate_input(1)
    rejected_candidate = candidate_input(2)
    stale_review = rejected_candidate.recent_reviews[0].model_copy(
        update={
            "published_at": date.today()
            - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS + 1)
        }
    )
    rejected_candidate = rejected_candidate.model_copy(
        update={
            "recent_reviews": [
                stale_review,
                *rejected_candidate.recent_reviews[1:],
            ]
        }
    )

    accepted, rejections = partition_candidates_by_review_validity(
        [accepted_candidate, rejected_candidate],
        date.today(),
    )

    assert accepted == [accepted_candidate]
    assert len(rejections) == 1
    assert rejected_candidate.name in rejections[0]
    assert "older than" in rejections[0]


def test_research_partition_keeps_candidate_with_five_reviews_after_filtering() -> None:
    candidate = candidate_input(1)
    stale_review = candidate.recent_reviews[0].model_copy(
        update={
            "published_at": date.today()
            - timedelta(days=RECENT_REVIEW_MAX_AGE_DAYS + 1)
        }
    )
    extra_review = candidate.recent_reviews[-1].model_copy(
        update={
            "review_url": type(candidate.recent_reviews[-1].review_url)(
                "https://extra-reviews.example/restaurant/1/review/6"
            ),
            "published_at": date.today() - timedelta(days=10),
            "summary": "An additional recent review",
        }
    )
    candidate = candidate.model_copy(
        update={
            "recent_reviews": [
                stale_review,
                *candidate.recent_reviews[1:],
                extra_review,
            ]
        }
    )

    accepted, rejections = partition_candidates_by_review_validity(
        [candidate],
        date.today(),
    )

    assert rejections == []
    assert len(accepted) == 1
    assert len(accepted[0].recent_reviews) == 5
    assert stale_review not in accepted[0].recent_reviews


def test_research_partition_drops_only_review_with_unverified_reference(tmp_path) -> None:
    candidate = candidate_input(1)
    extra_review = candidate.recent_reviews[-1].model_copy(
        update={
            "review_url": type(candidate.recent_reviews[-1].review_url)(
                "https://extra-reviews.example/restaurant/1/review/6"
            ),
            "reviewer_name": "Unverified reviewer",
            "published_at": date.today() - timedelta(days=10),
            "summary": "An additional review with an invalid source page",
        }
    )
    candidate = candidate.model_copy(
        update={"recent_reviews": [*candidate.recent_reviews, extra_review]}
    )
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    populate_scraped_pages(context, [candidate])
    review_key = canonical_url(extra_review.review_url)
    review_page = context.scraped_pages[review_key]
    context.scraped_pages[review_key] = review_page.model_copy(
        update={
            "title": "Another restaurant",
            "content": "\n".join(
                [
                    "Another restaurant",
                    "Kochi 999",
                    extra_review.reviewer_name,
                    extra_review.published_at.isoformat(),
                    f"{extra_review.rating:g} / 5",
                    "A review of a different venue",
                ]
            ),
        }
    )

    accepted, rejections = partition_candidates_by_review_validity(
        [candidate],
        date.today(),
        context.scraped_pages,
    )

    assert rejections == []
    assert len(accepted) == 1
    assert len(accepted[0].recent_reviews) == 5
    assert extra_review not in accepted[0].recent_reviews


def test_restaurant_cache_writes_and_reloads_one_file_per_store(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(
            name="Test Hotel",
            latitude=33.566927593644714,
            longitude=133.54104073018118,
        ),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidates = [candidate_input(index) for index in range(1, 4)]
    populate_scraped_pages(context, candidates)

    written = cache_restaurant_candidates(context, candidates)
    updated_candidate = candidates[0].model_copy(
        update={"katsuo_dish": "Updated katsuo dish"}
    )
    populate_scraped_pages(context, [updated_candidate])
    cache_restaurant_candidates(context, [updated_candidate])
    loaded, rejections = load_cached_restaurant_candidates(context, date.today())

    cache_files = list(context.restaurant_cache_dir.glob("*.json"))
    assert written == 3
    assert len(cache_files) == 3
    assert rejections == []
    assert sorted(candidate.name for candidate in loaded) == sorted(
        candidate.name for candidate in candidates
    )
    updated_loaded = next(
        candidate for candidate in loaded if candidate.name == candidates[0].name
    )
    assert updated_loaded.katsuo_dish == "Updated katsuo dish"
    assert all('"candidate"' in path.read_text(encoding="utf-8") for path in cache_files)


def test_restaurant_cache_bootstraps_from_existing_aggregate_store(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(
            name="Test Hotel",
            latitude=33.566927593644714,
            longitude=133.54104073018118,
        ),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidates = [candidate_input(index) for index in range(1, 7)]
    populate_scraped_pages(context, candidates)
    persist_restaurant_candidates(context, candidates)

    loaded, rejections = load_cached_restaurant_candidates(context, date.today())

    assert rejections == []
    assert len(loaded) == 6
    assert len(list(context.restaurant_cache_dir.glob("*.json"))) == 6


def test_restaurant_cache_ignores_legacy_schema_without_scraped_evidence(
    tmp_path,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidate = candidate_input(1)
    populate_scraped_pages(context, [candidate])
    cache_restaurant_candidates(context, [candidate])
    cache_path = next(context.restaurant_cache_dir.glob("*.json"))
    cache_path.write_text(
        cache_path.read_text(encoding="utf-8").replace(
            '"schema_version": 2', '"schema_version": 1'
        ),
        encoding="utf-8",
    )

    loaded, rejections = load_cached_restaurant_candidates(context, date.today())

    assert loaded == []
    assert any("Cache ignored" in rejection for rejection in rejections)


def test_candidate_input_allows_incomplete_reviews_for_discovery() -> None:
    candidate = candidate_input(1)
    payload = candidate.model_dump()
    payload["recent_reviews"] = []

    discovered = RestaurantCandidateInput.model_validate(payload)

    assert discovered.recent_reviews == []


def test_candidate_observations_accumulate_independent_evidence_urls() -> None:
    candidate = candidate_input(1)
    url_type = type(candidate.evidence_url)
    first_observation = candidate.model_copy(
        update={
            "source_urls": [url_type("https://tourism.example/restaurant/1")],
        }
    )
    second_observation = candidate.model_copy(
        update={
            "source_urls": [url_type("https://reservation.example/restaurant/1")],
        }
    )

    merged = merge_candidate_observations(first_observation, second_observation)

    assert {url.host for url in merged.source_urls} == {
        "tourism.example",
        "reservation.example",
    }


def test_discovery_cache_saves_in_range_candidate_without_reviews_or_pages(
    tmp_path,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidate = candidate_input(1).model_copy(update={"recent_reviews": []})

    written = cache_restaurant_candidates(context, [candidate])
    loaded, rejections = load_cached_restaurant_candidates(context, date.today())

    assert written == 1
    assert rejections == []
    assert loaded == [candidate]
    assert len(list(context.restaurant_cache_dir.glob("*.json"))) == 1


def test_discovery_cache_skips_candidate_outside_location_range(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=0.01,
        output_dir=tmp_path,
    )
    candidate = candidate_input(1).model_copy(update={"recent_reviews": []})

    written = cache_restaurant_candidates(context, [candidate])

    assert written == 0
    assert not context.restaurant_cache_dir.exists()


def test_discovery_cache_accumulates_reviews_across_research_runs(tmp_path) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidate = candidate_input(1)
    first_observation = candidate.model_copy(
        update={"recent_reviews": candidate.recent_reviews[:2]}
    )
    second_observation = candidate.model_copy(
        update={"recent_reviews": candidate.recent_reviews[2:]}
    )

    cache_restaurant_candidates(context, [first_observation])
    cache_restaurant_candidates(context, [second_observation])
    loaded, rejections = load_cached_restaurant_candidates(context, date.today())

    assert rejections == []
    assert len(loaded) == 1
    assert len(loaded[0].recent_reviews) == 5
    assert {review.reviewer_name for review in loaded[0].recent_reviews} == {
        review.reviewer_name for review in candidate.recent_reviews
    }


def test_discovery_cache_keeps_richer_observation_when_refresh_is_incomplete(
    tmp_path,
) -> None:
    context = KatsuoContext(
        hotel=HOTEL,
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    candidate = candidate_input(1)
    incomplete_refresh = candidate.model_copy(
        update={
            "katsuo_dish": "Incomplete refresh dish",
            "source_urls": [],
            "recent_reviews": [],
        }
    )

    cache_restaurant_candidates(context, [candidate])
    cache_restaurant_candidates(context, [incomplete_refresh])
    loaded, rejections = load_cached_restaurant_candidates(context, date.today())

    assert rejections == []
    assert len(loaded) == 1
    assert loaded[0].katsuo_dish == candidate.katsuo_dish
    assert loaded[0].recent_reviews == candidate.recent_reviews


def test_deduplication_merges_qualified_name_alias_at_same_address() -> None:
    candidate = candidate_input(1)
    qualified_alias = candidate.model_copy(
        update={"name": f"四季料理 {candidate.name}"}
    )
    unrelated_store = candidate.model_copy(update={"name": "Unrelated Store"})

    deduplicated = deduplicate_restaurant_candidates(
        [candidate, qualified_alias, unrelated_store]
    )

    assert deduplicated == [candidate, unrelated_store]


def test_candidate_save_rejects_duplicate_reviews(tmp_path) -> None:
    candidate = candidate_input(1)
    duplicate = candidate.recent_reviews[0]
    candidate = candidate.model_copy(
        update={
            "recent_reviews": [
                duplicate,
                duplicate,
                *candidate.recent_reviews[2:],
            ]
        }
    )
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="duplicate review"):
        persist_restaurant_candidates(context, [candidate] * 5)


def test_candidate_save_rejects_reviews_from_only_one_site(tmp_path) -> None:
    candidate = candidate_input(1)
    single_site_reviews = [
        review.model_copy(
            update={
                "review_url": type(review.review_url)(
                    f"https://reviews.example/restaurant/1/review/{index}"
                )
            }
        )
        for index, review in enumerate(candidate.recent_reviews, start=1)
    ]
    candidate = candidate.model_copy(update={"recent_reviews": single_site_reviews})
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match=f"fewer than {MIN_REVIEW_SOURCE_SITES} source sites",
    ):
        persist_restaurant_candidates(context, [candidate] * 5)


def test_research_partition_rejects_candidate_with_only_one_review_site() -> None:
    accepted_candidate = candidate_input(1)
    rejected_candidate = candidate_input(2)
    single_site_reviews = [
        review.model_copy(
            update={
                "review_url": type(review.review_url)(
                    f"https://reviews.example/restaurant/2/review/{index}"
                )
            }
        )
        for index, review in enumerate(rejected_candidate.recent_reviews, start=1)
    ]
    rejected_candidate = rejected_candidate.model_copy(
        update={"recent_reviews": single_site_reviews}
    )

    accepted, rejections = partition_candidates_by_review_validity(
        [accepted_candidate, rejected_candidate],
        date.today(),
    )

    assert accepted == [accepted_candidate]
    assert len(rejections) == 1
    assert rejected_candidate.name in rejections[0]
    assert "fewer than 2 source sites" in rejections[0]
