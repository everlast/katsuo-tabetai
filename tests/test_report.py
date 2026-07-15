from __future__ import annotations

from datetime import datetime, timezone

from katsuo_tabetai.models import HotelLocation, TopFiveStore
from katsuo_tabetai.report import render_top_five_html
from katsuo_tabetai.scoring import rank_top_five

from test_scoring import make_candidate


def test_html_contains_top_five_and_evidence_links(tmp_path) -> None:
    hotel = HotelLocation(
        name="Test Hotel",
        latitude=33.566927593644714,
        longitude=133.54104073018118,
    )
    restaurants = rank_top_five([make_candidate(i) for i in range(1, 7)], 2.5)
    report = TopFiveStore(
        generated_at=datetime.now(timezone.utc),
        hotel=hotel,
        max_distance_km=2.5,
        restaurants=restaurants,
    )
    output = tmp_path / "top5.html"

    render_top_five_html(report, output)

    html = output.read_text(encoding="utf-8")
    assert "ホテル周辺" in html
    assert "100 POINTS" in html
    assert "/ 25点" in html
    assert html.count('class="restaurant"') == 5
    for restaurant in restaurants:
        assert restaurant.name in html
        assert str(restaurant.evidence_url) in html
        assert restaurant.recommendation_reason in html
        assert "新着レビューから見た評判" in html
        for review in restaurant.recent_reviews:
            assert str(review.review_url) in html
            assert review.summary in html
