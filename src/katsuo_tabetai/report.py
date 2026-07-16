from __future__ import annotations

from html import escape
from pathlib import Path

from .models import EvidenceSourceType, TopFiveStore, selected_feature_labels
from .scoring import (
    DISTANCE_MAX_POINTS,
    EVIDENCE_POINTS,
    EVIDENCE_MAX_POINTS,
    INDEPENDENT_SOURCE_MAX_DOMAINS,
    INDEPENDENT_SOURCE_POINTS_PER_DOMAIN,
    INDEPENDENT_SOURCES_MAX_POINTS,
    KATSUO_DISH_NAME_POINTS,
    KATSUO_FEATURES_MAX_POINTS,
    RECENT_REVIEWS_MAX_POINTS,
    REVIEW_COUNT_FOR_MAX_POINTS,
    REVIEW_COUNT_MAX_POINTS,
    REVIEW_RATING_MAX_POINTS,
    REVIEW_RATING_SCALE_MAX,
    REVIEW_SOURCE_COUNT_FOR_MAX_POINTS,
    REVIEW_SOURCE_MAX_POINTS,
    SEASONAL_KATSUO_POINTS,
    SHIO_TATAKI_POINTS,
    TOTAL_MAX_POINTS,
    WARAYAKI_POINTS,
)

SOURCE_LABELS = {
    EvidenceSourceType.OFFICIAL_RESTAURANT: "店舗公式",
    EvidenceSourceType.OFFICIAL_TOURISM: "観光公式",
    EvidenceSourceType.RESERVATION_SITE: "予約サイト",
    EvidenceSourceType.REVIEW_SITE: "レビューサイト",
}


def _score_breakdown_row(
    label: str,
    value: float,
    maximum,
    detail: str | None = None,
) -> str:
    maximum_float = float(maximum)
    width = min(100.0, max(0.0, value / maximum_float * 100))
    detail_html = (
        f'<span class="score-breakdown-detail">{escape(detail)}</span>'
        if detail
        else ""
    )
    return f"""
              <div class="score-breakdown-row">
                <dt>{escape(label)}</dt>
                <dd>
                  <span class="score-breakdown-value">{value:.2f} / {maximum_float:g}点</span>
                  <span class="mini-track" aria-hidden="true"><span style="width:{width:.2f}%"></span></span>
                  {detail_html}
                </dd>
              </div>"""


def _score_breakdown(restaurant) -> str:
    breakdown = restaurant.score_breakdown
    rows = [
        ("カツオ料理の根拠種別", breakdown.evidence, EVIDENCE_MAX_POINTS),
        ("カツオ料理の特徴", breakdown.katsuo_features, KATSUO_FEATURES_MAX_POINTS),
        (
            "独立した料理根拠URL",
            breakdown.independent_sources,
            INDEPENDENT_SOURCES_MAX_POINTS,
        ),
        ("新着レビューの評判", breakdown.recent_reviews, RECENT_REVIEWS_MAX_POINTS),
        ("ホテルからの距離", breakdown.distance, DISTANCE_MAX_POINTS),
    ]
    return "".join(
        _score_breakdown_row(label, value, maximum) for label, value, maximum in rows
    )


def _review_score_breakdown(restaurant) -> str:
    reputation = restaurant.review_reputation
    rating_points = (
        reputation.average_rating
        / float(REVIEW_RATING_SCALE_MAX)
        * float(REVIEW_RATING_MAX_POINTS)
    )
    volume_points = (
        min(reputation.review_count, REVIEW_COUNT_FOR_MAX_POINTS)
        / REVIEW_COUNT_FOR_MAX_POINTS
        * float(REVIEW_COUNT_MAX_POINTS)
    )
    source_points = (
        min(reputation.source_count, REVIEW_SOURCE_COUNT_FOR_MAX_POINTS)
        / REVIEW_SOURCE_COUNT_FOR_MAX_POINTS
        * float(REVIEW_SOURCE_MAX_POINTS)
    )
    rows = [
        ("平均評価による加点", rating_points, REVIEW_RATING_MAX_POINTS, None),
        (
            "確認件数による加点",
            volume_points,
            REVIEW_COUNT_MAX_POINTS,
            (
                f"{reputation.review_count}件を確認"
                f"（{REVIEW_COUNT_FOR_MAX_POINTS}件で満点）"
            ),
        ),
        ("情報源数による加点", source_points, REVIEW_SOURCE_MAX_POINTS, None),
    ]
    return "".join(
        _score_breakdown_row(label, value, maximum, detail)
        for label, value, maximum, detail in rows
    )


def _review_row(review) -> str:
    review_url = escape(str(review.review_url), quote=True)
    review_month = review.published_at.strftime("%Y-%m")
    positives = "".join(
        f'<span class="positive-point">{escape(point)}</span>'
        for point in review.positive_points
    )
    cautions = ""
    if review.caution_points:
        caution_text = "・".join(review.caution_points)
        cautions = f'<p class="review-caution">注意: {escape(caution_text)}</p>'
    return f"""
              <li class="review-item">
                <div class="review-meta">
                  <strong>{review.rating:.1f} / 5</strong>
                  <span>{escape(review.source_name)} · {escape(review.reviewer_name)} · {review_month}</span>
                </div>
                <p>{escape(review.summary)}</p>
                <div class="review-points">{positives}</div>
                {cautions}
                <a href="{review_url}" target="_blank" rel="noreferrer">レビューの根拠を開く</a>
              </li>"""


def _restaurant_row(restaurant) -> str:
    evidence_url = escape(str(restaurant.evidence_url), quote=True)
    additional_sources = "".join(
        '<li><a href="{}" target="_blank" rel="noreferrer">追加根拠 {}</a></li>'.format(
            escape(str(source_url), quote=True),
            index,
        )
        for index, source_url in enumerate(restaurant.source_urls, start=1)
    )
    features = [
        "カツオ料理の掲載あり",
        *selected_feature_labels(
            restaurant,
            warayaki="藁焼き",
            shio_tataki="塩たたき",
            seasonal_katsuo="旬の案内",
        ),
    ]
    feature_html = "".join(f"<li>{escape(item)}</li>" for item in features)
    review_html = "".join(_review_row(review) for review in restaurant.recent_reviews)
    score_breakdown_html = _score_breakdown(restaurant)
    review_score_breakdown_html = _review_score_breakdown(restaurant)
    score_width = min(
        100.0,
        max(0.0, restaurant.score / float(TOTAL_MAX_POINTS) * 100),
    )
    source_label = SOURCE_LABELS[restaurant.evidence_source_type]
    reputation = restaurant.review_reputation
    return f"""
      <article class="restaurant" aria-labelledby="restaurant-{restaurant.rank}">
        <div class="rank" aria-label="{restaurant.rank}位">
          <span>RANK</span><strong>{restaurant.rank}</strong>
        </div>
        <div class="restaurant-main">
          <div class="restaurant-heading">
            <div>
              <p class="source-type">{escape(source_label)}</p>
              <h2 id="restaurant-{restaurant.rank}">{escape(restaurant.name)}</h2>
            </div>
            <div class="score" aria-label="{TOTAL_MAX_POINTS:g}点満点中 {restaurant.score:.2f}点">
              <strong>{restaurant.score:.2f}</strong><span>/ {TOTAL_MAX_POINTS:g}</span>
            </div>
          </div>
          <div class="score-track" aria-hidden="true"><span style="width:{score_width:.2f}%"></span></div>
          <dl class="score-breakdown" aria-label="総合スコアの評価項目別内訳">{score_breakdown_html}
          </dl>
          <section class="recommendation" aria-labelledby="reason-{restaurant.rank}">
            <h3 id="reason-{restaurant.rank}">この店を推す理由</h3>
            <p>{escape(restaurant.recommendation_reason)}</p>
          </section>
          <p class="dish">{escape(restaurant.katsuo_dish)}</p>
          <p class="address">{escape(restaurant.address)} · ホテルから {restaurant.distance_km:.2f} km</p>
          <ul class="features">{feature_html}</ul>
          <a class="evidence" href="{evidence_url}" target="_blank" rel="noreferrer">カツオ料理の根拠ページを開く</a>
          <ul class="source-links">{additional_sources}</ul>
          <section class="review-section" aria-labelledby="reviews-{restaurant.rank}">
            <div class="review-heading">
              <div>
                <p class="review-label">RECENT REVIEWS</p>
                <h3 id="reviews-{restaurant.rank}">新着レビューから見た評判</h3>
              </div>
              <dl class="review-stats">
                <div><dt>平均評価</dt><dd>{reputation.average_rating:.2f} / 5</dd></div>
                <div><dt>確認件数</dt><dd>{reputation.review_count}件</dd></div>
                <div><dt>情報源</dt><dd>{reputation.source_count}サイト</dd></div>
              </dl>
            </div>
            <p class="review-score">総合点のうちレビュー評判: {restaurant.score_breakdown.recent_reviews:.2f} / {RECENT_REVIEWS_MAX_POINTS:g}点</p>
            <dl class="score-breakdown review-score-breakdown" aria-label="レビュー評判スコアの評価項目別内訳">{review_score_breakdown_html}
            </dl>
            <ol class="reviews">{review_html}</ol>
          </section>
        </div>
      </article>"""


_STYLE_CSS = """\
    :root {
      --paper: #f7f8f5;
      --white: #ffffff;
      --ink: #17211f;
      --muted: #62706c;
      --line: #d5dcd8;
      --ocean: #087f78;
      --bonito: #c94432;
      --rank: #004AAD;
      --market: #f1bf3a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Hiragino Sans", "Yu Gothic", system-ui, sans-serif;
      letter-spacing: 0;
    }
    a { color: inherit; }
    .masthead {
      border-top: 8px solid var(--bonito);
      border-bottom: 1px solid var(--line);
      background: var(--white);
    }
    .masthead-inner, main, footer { width: min(1080px, calc(100% - 40px)); margin: 0 auto; }
    .masthead-inner { padding: 34px 0 28px; display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: end; }
    .eyebrow, .source-type {
      margin: 0 0 8px;
      color: var(--ocean);
      font: 700 12px/1.4 "SFMono-Regular", Menlo, monospace;
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      font-family: "Hiragino Mincho ProN", "Yu Mincho", serif;
      font-size: clamp(32px, 6vw, 64px);
      line-height: 1.05;
      font-weight: 700;
      letter-spacing: 0;
    }
    .stamp {
      min-width: 144px;
      padding: 14px 16px;
      border: 2px solid var(--ink);
      box-shadow: 5px 5px 0 var(--market);
      font: 700 13px/1.6 "SFMono-Regular", Menlo, monospace;
    }
    .stamp strong { display: block; font-size: 20px; }
    .criteria {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      border-bottom: 1px solid var(--line);
      background: var(--ink);
      color: var(--white);
    }
    .criteria div { padding: 16px 20px; border-right: 1px solid #40504c; }
    .criteria div:last-child { border-right: 0; }
    .criteria span { display: block; color: #b9c5c1; font-size: 11px; }
    .criteria strong { display: block; margin-top: 3px; font-size: 14px; }
    main { padding: 32px 0 56px; }
    .ranking-index {
      display: grid;
      grid-template-columns: 160px 1fr;
      margin-bottom: 24px;
      border: 1px solid var(--ink);
      background: var(--white);
    }
    .ranking-index-heading {
      display: grid;
      align-content: center;
      padding: 16px 18px;
      border-right: 1px solid var(--ink);
    }
    .ranking-index-label {
      margin: 0 0 4px;
      color: var(--bonito);
      font: 700 11px/1.4 "SFMono-Regular", Menlo, monospace;
    }
    .ranking-index h2 { margin: 0; font-size: 16px; line-height: 1.4; }
    .ranking-index ol {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .ranking-index li { min-width: 0; border-right: 1px solid var(--line); }
    .ranking-index li:last-child { border-right: 0; }
    .ranking-index a {
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 6px;
      min-height: 94px;
      padding: 12px;
      text-decoration: none;
    }
    .ranking-index a:hover { background: #f3f6f4; }
    .ranking-index a:focus-visible { outline: 3px solid var(--market); outline-offset: -3px; }
    .ranking-index-rank { color: var(--ocean); font: 700 10px/1.3 "SFMono-Regular", Menlo, monospace; }
    .ranking-index strong { min-width: 0; overflow-wrap: anywhere; font-size: 13px; line-height: 1.45; }
    .ranking-index-score { color: var(--muted); font: 700 11px/1.3 "SFMono-Regular", Menlo, monospace; }
    .restaurant {
      display: grid;
      grid-template-columns: 104px 1fr;
      margin-bottom: 16px;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: var(--white);
    }
    .rank {
      display: grid;
      place-content: center;
      min-height: 260px;
      background: var(--rank);
      color: var(--white);
      text-align: center;
    }
    .rank span { font: 700 11px/1 "SFMono-Regular", Menlo, monospace; }
    .rank strong { font: 700 58px/1 "Hiragino Mincho ProN", "Yu Mincho", serif; }
    .restaurant-main { padding: 24px 28px 26px; min-width: 0; }
    .restaurant-heading { display: flex; justify-content: space-between; gap: 24px; align-items: start; }
    .restaurant h2 { margin: 0; font-size: 24px; line-height: 1.35; letter-spacing: 0; }
    .score { display: flex; align-items: baseline; white-space: nowrap; color: var(--ocean); }
    .score strong { font: 700 30px/1 "SFMono-Regular", Menlo, monospace; }
    .score span { margin-left: 4px; color: var(--muted); font-size: 12px; }
    .score-track { height: 4px; margin: 16px 0 20px; background: #e5e9e7; }
    .score-track span { display: block; height: 100%; background: var(--ocean); }
    .score-breakdown {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px 12px;
      margin: 0 0 22px;
    }
    .score-breakdown-row {
      min-width: 0;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }
    .score-breakdown dt {
      color: var(--muted);
      font-size: 10px;
      line-height: 1.45;
    }
    .score-breakdown dd { margin: 5px 0 0; }
    .score-breakdown-value {
      display: block;
      font: 700 12px/1.2 "SFMono-Regular", Menlo, monospace;
    }
    .score-breakdown-detail {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 9px;
      line-height: 1.45;
    }
    .mini-track {
      display: block;
      height: 3px;
      margin-top: 7px;
      background: #e5e9e7;
    }
    .mini-track span { display: block; height: 100%; background: var(--ocean); }
    .recommendation { margin: 0 0 20px; padding: 14px 16px; border-left: 4px solid var(--bonito); background: #fff6f3; }
    .recommendation h3 { margin: 0 0 6px; font-size: 13px; }
    .recommendation p { margin: 0; font-size: 14px; line-height: 1.8; }
    .dish { margin: 0 0 7px; font-weight: 700; font-size: 17px; }
    .address { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.7; }
    .features { display: flex; flex-wrap: wrap; gap: 7px; margin: 17px 0 19px; padding: 0; list-style: none; }
    .features li { padding: 5px 8px; border-left: 3px solid var(--market); background: #f5f5ef; font-size: 12px; }
    .evidence {
      display: inline-block;
      padding-bottom: 3px;
      border-bottom: 2px solid var(--bonito);
      font-weight: 700;
      font-size: 13px;
      text-decoration: none;
    }
    .evidence:hover { color: var(--bonito); }
    .evidence:focus-visible { outline: 3px solid var(--market); outline-offset: 4px; }
    .source-links { display: flex; flex-wrap: wrap; gap: 12px; margin: 10px 0 0; padding: 0; list-style: none; }
    .source-links a { color: var(--muted); font-size: 12px; }
    .review-section { margin-top: 26px; padding-top: 22px; border-top: 1px solid var(--line); }
    .review-heading { display: flex; justify-content: space-between; gap: 24px; align-items: end; }
    .review-label { margin: 0 0 5px; color: var(--bonito); font: 700 11px/1.4 "SFMono-Regular", Menlo, monospace; }
    .review-heading h3 { margin: 0; font-size: 18px; }
    .review-stats { display: flex; gap: 18px; margin: 0; }
    .review-stats div { min-width: 76px; }
    .review-stats dt { color: var(--muted); font-size: 10px; }
    .review-stats dd { margin: 3px 0 0; font: 700 14px/1.3 "SFMono-Regular", Menlo, monospace; }
    .review-score { margin: 12px 0 0; color: var(--muted); font-size: 11px; }
    .review-score-breakdown {
      grid-template-columns: repeat(3, minmax(0, 160px));
      margin-top: 12px;
      margin-bottom: 0;
    }
    .reviews { margin: 18px 0 0; padding: 0; list-style: none; border-top: 1px solid var(--line); }
    .review-item { padding: 16px 0; border-bottom: 1px solid var(--line); }
    .review-meta { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }
    .review-meta strong { color: var(--ocean); font: 700 14px/1.4 "SFMono-Regular", Menlo, monospace; }
    .review-meta span { color: var(--muted); font-size: 11px; text-align: right; }
    .review-item > p { margin: 8px 0; font-size: 13px; line-height: 1.75; }
    .review-points { display: flex; flex-wrap: wrap; gap: 6px; }
    .positive-point { padding: 3px 7px; background: #e9f4f1; color: #155f59; font-size: 11px; }
    .review-item .review-caution { margin: 8px 0; color: #854237; font-size: 11px; }
    .review-item a { display: inline-block; margin-top: 9px; color: var(--ocean); font-size: 11px; font-weight: 700; }
    .score-note {
      margin-top: 32px;
      padding: 18px 20px;
      border-left: 4px solid var(--ocean);
      background: #edf4f1;
    }
    .score-note-label {
      margin: 0 0 4px;
      color: var(--ocean);
      font: 700 11px/1.4 "SFMono-Regular", Menlo, monospace;
    }
    .score-note h2 { margin: 0 0 8px; font-size: 17px; line-height: 1.5; }
    .score-note p { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.8; }
    .score-note strong { color: var(--ink); }
    .score-note-items {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      margin: 14px 0;
      border-top: 1px solid #c9d7d2;
      border-bottom: 1px solid #c9d7d2;
    }
    .score-note-items > div { min-width: 0; padding: 12px; border-right: 1px solid #c9d7d2; }
    .score-note-items > div:first-child { padding-left: 0; }
    .score-note-items > div:last-child { padding-right: 0; border-right: 0; }
    .score-note-items dt { font-size: 12px; line-height: 1.5; }
    .score-note-items dt strong { display: block; }
    .score-note-items dt span { display: block; margin-top: 2px; color: var(--ocean); font-size: 10px; font-weight: 700; }
    .score-note-items dd { margin: 7px 0 0; color: var(--muted); font-size: 11px; line-height: 1.7; }
    footer { padding: 22px 0 36px; color: var(--muted); font-size: 12px; line-height: 1.8; }
    @media (max-width: 700px) {
      .masthead-inner { grid-template-columns: 1fr; }
      .stamp { width: max-content; max-width: 100%; }
      .criteria { grid-template-columns: 1fr; }
      .criteria div { border-right: 0; border-bottom: 1px solid #40504c; }
      .ranking-index { grid-template-columns: 1fr; }
      .ranking-index-heading { border-right: 0; border-bottom: 1px solid var(--ink); }
      .ranking-index ol { grid-template-columns: 1fr; }
      .ranking-index li { border-right: 0; border-bottom: 1px solid var(--line); }
      .ranking-index li:last-child { border-bottom: 0; }
      .ranking-index a {
        grid-template-columns: 56px minmax(0, 1fr) auto;
        grid-template-rows: 1fr;
        align-items: center;
        min-height: 52px;
      }
      .score-note-items { grid-template-columns: 1fr; }
      .score-note-items > div,
      .score-note-items > div:first-child,
      .score-note-items > div:last-child {
        padding: 11px 0;
        border-right: 0;
        border-bottom: 1px solid #c9d7d2;
      }
      .score-note-items > div:last-child { border-bottom: 0; }
      .restaurant { grid-template-columns: 64px 1fr; }
      .rank { min-height: 100%; }
      .rank strong { font-size: 42px; }
      .restaurant-main { padding: 20px 18px 22px; }
      .restaurant-heading { display: block; }
      .score { margin-top: 14px; }
      .score-breakdown { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .review-score-breakdown { grid-template-columns: 1fr; }
      .restaurant h2 { font-size: 20px; }
      .review-heading { display: block; }
      .review-stats { margin-top: 14px; gap: 12px; flex-wrap: wrap; }
      .review-meta { display: block; }
      .review-meta span { display: block; margin-top: 4px; text-align: left; }
    }
    @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }"""


def _score_note_html() -> str:
    return f"""\
    <aside class="score-note" aria-labelledby="score-note-title">
      <p class="score-note-label">SCORING NOTE</p>
      <h2 id="score-note-title">スコアはどう決まる？</h2>
      <p><strong>{TOTAL_MAX_POINTS:g}点満点</strong>で、保存済みの事実を次の5つの観点から評価しています。</p>
      <dl class="score-note-items">
        <div>
          <dt><strong>カツオ料理の根拠種別</strong><span>最大 {EVIDENCE_MAX_POINTS:g}点</span></dt>
          <dd>情報元の信頼性を評価。店舗公式 {EVIDENCE_POINTS[EvidenceSourceType.OFFICIAL_RESTAURANT]:g}点、観光公式 {EVIDENCE_POINTS[EvidenceSourceType.OFFICIAL_TOURISM]:g}点、予約サイト {EVIDENCE_POINTS[EvidenceSourceType.RESERVATION_SITE]:g}点、レビューサイト {EVIDENCE_POINTS[EvidenceSourceType.REVIEW_SITE]:g}点です。</dd>
        </div>
        <div>
          <dt><strong>カツオ料理の特徴</strong><span>最大 {KATSUO_FEATURES_MAX_POINTS:g}点</span></dt>
          <dd>料理名の掲載 {KATSUO_DISH_NAME_POINTS:g}点を基礎に、藁焼き {WARAYAKI_POINTS:g}点、塩たたき {SHIO_TATAKI_POINTS:g}点、旬の案内 {SEASONAL_KATSUO_POINTS:g}点を加算します。</dd>
        </div>
        <div>
          <dt><strong>独立した料理根拠URL</strong><span>最大 {INDEPENDENT_SOURCES_MAX_POINTS:g}点</span></dt>
          <dd>根拠URLをドメイン単位で重複除外し、1ドメインにつき {INDEPENDENT_SOURCE_POINTS_PER_DOMAIN:g}点、最大 {INDEPENDENT_SOURCE_MAX_DOMAINS}ドメインまで加算します。</dd>
        </div>
        <div>
          <dt><strong>新着レビューの評判</strong><span>最大 {RECENT_REVIEWS_MAX_POINTS:g}点</span></dt>
          <dd>平均評価 {REVIEW_RATING_MAX_POINTS:g}点、確認件数 {REVIEW_COUNT_MAX_POINTS:g}点、情報源数 {REVIEW_SOURCE_MAX_POINTS:g}点で評価。件数は {REVIEW_COUNT_FOR_MAX_POINTS}件、情報源は {REVIEW_SOURCE_COUNT_FOR_MAX_POINTS}サイトで満点です。</dd>
        </div>
        <div>
          <dt><strong>ホテルからの距離</strong><span>最大 {DISTANCE_MAX_POINTS:g}点</span></dt>
          <dd>ホテルと同じ位置を {DISTANCE_MAX_POINTS:g}点とし、離れるほど直線的に減点。検索距離の上限で0点になります。</dd>
        </div>
      </dl>
      <p>LLMは採点や順位決定を行わず、同じ入力なら同じ結果になります。</p>
    </aside>"""


def render_top_five_html(report: TopFiveStore, output_path: Path) -> None:
    rows = "\n".join(_restaurant_row(item) for item in report.restaurants)
    index_items = "".join(
        f"""
          <li>
            <a href="#restaurant-{restaurant.rank}">
              <span class="ranking-index-rank">RANK {restaurant.rank}</span>
              <strong>{escape(restaurant.name)}</strong>
              <span class="ranking-index-score">{restaurant.score:.2f}点</span>
            </a>
          </li>"""
        for restaurant in report.restaurants
    )
    generated = report.generated_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    html = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ホテル周辺 カツオ TOP 5</title>
  <style>
{_STYLE_CSS}
  </style>
</head>
<body>
  <header class="masthead">
    <div class="masthead-inner">
      <div>
        <p class="eyebrow">Kochi hotel katsuo guide</p>
        <h1>ホテル周辺<br>カツオ TOP 5</h1>
      </div>
      <div class="stamp">DETERMINISTIC SCORE<strong>{TOTAL_MAX_POINTS:g} POINTS</strong></div>
    </div>
    <div class="criteria" aria-label="検索条件">
      <div><span>基準地点</span><strong>{escape(report.hotel.name)}</strong></div>
      <div><span>直線距離の上限</span><strong>{report.max_distance_km:.2f} km</strong></div>
      <div><span>実行モデル</span><strong>{escape(report.model)}</strong></div>
      <div><span>生成日時</span><strong>{escape(generated)}</strong></div>
    </div>
  </header>
  <main>
    <nav class="ranking-index" aria-labelledby="ranking-index-title">
      <div class="ranking-index-heading">
        <p class="ranking-index-label">QUICK INDEX</p>
        <h2 id="ranking-index-title">掲載店へ移動</h2>
      </div>
      <ol>{index_items}
      </ol>
    </nav>
    {rows}
{_score_note_html()}
  </main>
  <footer>
    <p>Model: {escape(report.model)} · Trace: {escape(report.trace_id)} · <a href="{escape(report.context_markdown, quote=True)}">検証済みMarkdownコンテキスト</a></p>
    <p>距離は緯度経度からHaversine式で計算した直線距離です。レビューは生成日時から365日以内に公開・訪問されたものを要約し、原文ではなく根拠リンクを掲載しています。根拠ページに年月までしかない場合は月初として新着判定し、画面には年月まで表示します。営業日・提供メニュー・評判は変わるため、来店前に各ページで最新情報を確認してください。</p>
  </footer>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
