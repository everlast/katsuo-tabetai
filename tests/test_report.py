from __future__ import annotations

from datetime import datetime, timezone

from bs4 import BeautifulSoup

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
        model="gpt-5.6-luna",
        trace_id="trace_test",
        hotel=hotel,
        max_distance_km=2.5,
        restaurants=restaurants,
    )
    output = tmp_path / "top5.html"

    render_top_five_html(report, output)

    html = output.read_text(encoding="utf-8")
    document = BeautifulSoup(html, "html.parser")
    assert "ホテル周辺" in html
    assert "100 POINTS" in html
    assert "/ 40点" in html
    assert 'aria-label="総合スコアの評価項目別内訳"' in html
    assert 'aria-label="レビュー評判スコアの評価項目別内訳"' in html
    for label in (
        "カツオ料理の根拠種別",
        "カツオ料理の特徴",
        "独立した料理根拠ドメイン",
        "新着レビューの評判",
        "ホテルからの距離",
        "平均評価",
        "確認件数による加点",
        "情報源数",
    ):
        assert label in html
    assert "独立した料理根拠URL" not in html
    assert html.count('class="restaurant"') == 5
    assert "--rank: #004AAD;" in html
    assert "background: var(--rank);" in html
    assert 'class="ranking-index"' in html
    assert "掲載店へ移動" in html
    assert 'class="score-note"' in html
    assert 'class="score-note-items"' in html
    assert "スコアはどう決まる？" in html
    for explanation in (
        "店舗公式 20点、観光公式 17点、予約サイト 13点、レビューサイト 8点",
        "料理名の掲載 6点を基礎に、藁焼き 4点、塩たたき 3点、旬の案内 2点",
        "1ドメインにつき 2点、最大 5ドメイン",
        "平均評価 32点、確認件数 5点、情報源数 3点",
        "検索距離の上限で0点",
    ):
        assert explanation in html
    assert html.rfind('class="restaurant"') < html.index('class="score-note"')
    assert html.index('class="ranking-index"') < html.index('class="restaurant"')
    review_details = document.select("section.review-section > details.reviews-details")
    assert len(review_details) == len(restaurants)
    assert all("open" not in details.attrs for details in review_details)
    assert all(details.select_one("summary.reviews-toggle") for details in review_details)
    for restaurant in restaurants:
        assert f'href="#restaurant-{restaurant.rank}"' in html
        assert restaurant.name in html
        assert str(restaurant.evidence_url) in html
        for source_url in restaurant.source_urls:
            assert str(source_url) in html
        assert restaurant.recommendation_reason in html
        assert "新着レビューから見た評判" in html
        assert f"{restaurant.score_breakdown.evidence:.2f} / 20" in html
        assert f"{restaurant.score_breakdown.recent_reviews:.2f} / 40" in html
        assert (
            f"{restaurant.review_reputation.review_count}件を確認"
            "（5件で満点）"
        ) in html
        details = review_details[restaurant.rank - 1]
        assert (
            details.select_one(".reviews-toggle-closed").get_text(strip=True)
            == f"レビュー全{restaurant.review_reputation.review_count}件を表示"
        )
        assert (
            details.select_one(".reviews-toggle-open").get_text(strip=True)
            == f"レビュー全{restaurant.review_reputation.review_count}件を閉じる"
        )
        assert len(details.select("ol.reviews > li.review-item")) == len(
            restaurant.recent_reviews
        )
        for review in restaurant.recent_reviews:
            assert str(review.review_url) in html
            assert review.summary in html
            assert (
                f"{review.source_name} · {review.reviewer_name} · "
                f"{review.published_at:%Y-%m}</span>"
            ) in html
            assert (
                f"{review.source_name} · {review.reviewer_name} · "
                f"{review.published_at.isoformat()}</span>"
                not in html
            )
