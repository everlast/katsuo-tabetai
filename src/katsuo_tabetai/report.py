from __future__ import annotations

from html import escape
from pathlib import Path

from .models import EvidenceSourceType, TopFiveStore
from .scoring import RECENT_REVIEWS_MAX_POINTS, TOTAL_MAX_POINTS

SOURCE_LABELS = {
    EvidenceSourceType.OFFICIAL_RESTAURANT: "店舗公式",
    EvidenceSourceType.OFFICIAL_TOURISM: "観光公式",
    EvidenceSourceType.RESERVATION_SITE: "予約サイト",
    EvidenceSourceType.REVIEW_SITE: "レビューサイト",
}


def _review_row(review) -> str:
    review_url = escape(str(review.review_url), quote=True)
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
                  <span>{escape(review.source_name)} · {review.published_at.isoformat()}</span>
                </div>
                <p>{escape(review.summary)}</p>
                <div class="review-points">{positives}</div>
                {cautions}
                <a href="{review_url}" target="_blank" rel="noreferrer">レビューの根拠を開く</a>
              </li>"""


def _restaurant_row(restaurant) -> str:
    evidence_url = escape(str(restaurant.evidence_url), quote=True)
    features = ["カツオ料理の掲載あり"]
    if restaurant.has_warayaki:
        features.append("藁焼き")
    if restaurant.has_shio_tataki:
        features.append("塩たたき")
    if restaurant.has_seasonal_katsuo:
        features.append("旬の案内")
    feature_html = "".join(f"<li>{escape(item)}</li>" for item in features)
    review_html = "".join(_review_row(review) for review in restaurant.recent_reviews)
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
          <section class="recommendation" aria-labelledby="reason-{restaurant.rank}">
            <h3 id="reason-{restaurant.rank}">この店を推す理由</h3>
            <p>{escape(restaurant.recommendation_reason)}</p>
          </section>
          <p class="dish">{escape(restaurant.katsuo_dish)}</p>
          <p class="address">{escape(restaurant.address)} · ホテルから {restaurant.distance_km:.2f} km</p>
          <ul class="features">{feature_html}</ul>
          <a class="evidence" href="{evidence_url}" target="_blank" rel="noreferrer">カツオ料理の根拠ページを開く</a>
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
            <ol class="reviews">{review_html}</ol>
          </section>
        </div>
      </article>"""


def render_top_five_html(report: TopFiveStore, output_path: Path) -> None:
    rows = "\n".join(_restaurant_row(item) for item in report.restaurants)
    generated = report.generated_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    html = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>高知駅前 カツオ TOP 5</title>
  <style>
    :root {{
      --paper: #f7f8f5;
      --white: #ffffff;
      --ink: #17211f;
      --muted: #62706c;
      --line: #d5dcd8;
      --ocean: #087f78;
      --bonito: #c94432;
      --market: #f1bf3a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Hiragino Sans", "Yu Gothic", system-ui, sans-serif;
      letter-spacing: 0;
    }}
    a {{ color: inherit; }}
    .masthead {{
      border-top: 8px solid var(--bonito);
      border-bottom: 1px solid var(--line);
      background: var(--white);
    }}
    .masthead-inner, main, footer {{ width: min(1080px, calc(100% - 40px)); margin: 0 auto; }}
    .masthead-inner {{ padding: 34px 0 28px; display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: end; }}
    .eyebrow, .source-type {{
      margin: 0 0 8px;
      color: var(--ocean);
      font: 700 12px/1.4 "SFMono-Regular", Menlo, monospace;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-family: "Hiragino Mincho ProN", "Yu Mincho", serif;
      font-size: clamp(32px, 6vw, 64px);
      line-height: 1.05;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .stamp {{
      min-width: 144px;
      padding: 14px 16px;
      border: 2px solid var(--ink);
      box-shadow: 5px 5px 0 var(--market);
      font: 700 13px/1.6 "SFMono-Regular", Menlo, monospace;
    }}
    .stamp strong {{ display: block; font-size: 20px; }}
    .criteria {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      border-bottom: 1px solid var(--line);
      background: var(--ink);
      color: var(--white);
    }}
    .criteria div {{ padding: 16px 20px; border-right: 1px solid #40504c; }}
    .criteria div:last-child {{ border-right: 0; }}
    .criteria span {{ display: block; color: #b9c5c1; font-size: 11px; }}
    .criteria strong {{ display: block; margin-top: 3px; font-size: 14px; }}
    main {{ padding: 32px 0 56px; }}
    .restaurant {{
      display: grid;
      grid-template-columns: 104px 1fr;
      margin-bottom: 16px;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: var(--white);
    }}
    .rank {{
      display: grid;
      place-content: center;
      min-height: 260px;
      background: var(--bonito);
      color: var(--white);
      text-align: center;
    }}
    .rank span {{ font: 700 11px/1 "SFMono-Regular", Menlo, monospace; }}
    .rank strong {{ font: 700 58px/1 "Hiragino Mincho ProN", "Yu Mincho", serif; }}
    .restaurant-main {{ padding: 24px 28px 26px; min-width: 0; }}
    .restaurant-heading {{ display: flex; justify-content: space-between; gap: 24px; align-items: start; }}
    .restaurant h2 {{ margin: 0; font-size: 24px; line-height: 1.35; letter-spacing: 0; }}
    .score {{ display: flex; align-items: baseline; white-space: nowrap; color: var(--ocean); }}
    .score strong {{ font: 700 30px/1 "SFMono-Regular", Menlo, monospace; }}
    .score span {{ margin-left: 4px; color: var(--muted); font-size: 12px; }}
    .score-track {{ height: 4px; margin: 16px 0 20px; background: #e5e9e7; }}
    .score-track span {{ display: block; height: 100%; background: var(--ocean); }}
    .recommendation {{ margin: 0 0 20px; padding: 14px 16px; border-left: 4px solid var(--bonito); background: #fff6f3; }}
    .recommendation h3 {{ margin: 0 0 6px; font-size: 13px; }}
    .recommendation p {{ margin: 0; font-size: 14px; line-height: 1.8; }}
    .dish {{ margin: 0 0 7px; font-weight: 700; font-size: 17px; }}
    .address {{ margin: 0; color: var(--muted); font-size: 13px; line-height: 1.7; }}
    .features {{ display: flex; flex-wrap: wrap; gap: 7px; margin: 17px 0 19px; padding: 0; list-style: none; }}
    .features li {{ padding: 5px 8px; border-left: 3px solid var(--market); background: #f5f5ef; font-size: 12px; }}
    .evidence {{
      display: inline-block;
      padding-bottom: 3px;
      border-bottom: 2px solid var(--bonito);
      font-weight: 700;
      font-size: 13px;
      text-decoration: none;
    }}
    .evidence:hover {{ color: var(--bonito); }}
    .evidence:focus-visible {{ outline: 3px solid var(--market); outline-offset: 4px; }}
    .review-section {{ margin-top: 26px; padding-top: 22px; border-top: 1px solid var(--line); }}
    .review-heading {{ display: flex; justify-content: space-between; gap: 24px; align-items: end; }}
    .review-label {{ margin: 0 0 5px; color: var(--bonito); font: 700 11px/1.4 "SFMono-Regular", Menlo, monospace; }}
    .review-heading h3 {{ margin: 0; font-size: 18px; }}
    .review-stats {{ display: flex; gap: 18px; margin: 0; }}
    .review-stats div {{ min-width: 76px; }}
    .review-stats dt {{ color: var(--muted); font-size: 10px; }}
    .review-stats dd {{ margin: 3px 0 0; font: 700 14px/1.3 "SFMono-Regular", Menlo, monospace; }}
    .review-score {{ margin: 12px 0 0; color: var(--muted); font-size: 11px; }}
    .reviews {{ margin: 18px 0 0; padding: 0; list-style: none; border-top: 1px solid var(--line); }}
    .review-item {{ padding: 16px 0; border-bottom: 1px solid var(--line); }}
    .review-meta {{ display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }}
    .review-meta strong {{ color: var(--ocean); font: 700 14px/1.4 "SFMono-Regular", Menlo, monospace; }}
    .review-meta span {{ color: var(--muted); font-size: 11px; text-align: right; }}
    .review-item > p {{ margin: 8px 0; font-size: 13px; line-height: 1.75; }}
    .review-points {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .positive-point {{ padding: 3px 7px; background: #e9f4f1; color: #155f59; font-size: 11px; }}
    .review-item .review-caution {{ margin: 8px 0; color: #854237; font-size: 11px; }}
    .review-item a {{ display: inline-block; margin-top: 9px; color: var(--ocean); font-size: 11px; font-weight: 700; }}
    footer {{ padding: 22px 0 36px; color: var(--muted); font-size: 12px; line-height: 1.8; }}
    @media (max-width: 700px) {{
      .masthead-inner {{ grid-template-columns: 1fr; }}
      .stamp {{ width: max-content; max-width: 100%; }}
      .criteria {{ grid-template-columns: 1fr; }}
      .criteria div {{ border-right: 0; border-bottom: 1px solid #40504c; }}
      .restaurant {{ grid-template-columns: 64px 1fr; }}
      .rank {{ min-height: 100%; }}
      .rank strong {{ font-size: 42px; }}
      .restaurant-main {{ padding: 20px 18px 22px; }}
      .restaurant-heading {{ display: block; }}
      .score {{ margin-top: 14px; }}
      .restaurant h2 {{ font-size: 20px; }}
      .review-heading {{ display: block; }}
      .review-stats {{ margin-top: 14px; gap: 12px; flex-wrap: wrap; }}
      .review-meta {{ display: block; }}
      .review-meta span {{ display: block; margin-top: 4px; text-align: left; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior: auto; }} }}
  </style>
</head>
<body>
  <header class="masthead">
    <div class="masthead-inner">
      <div>
        <p class="eyebrow">Kochi station katsuo guide</p>
        <h1>高知駅前<br>カツオ TOP 5</h1>
      </div>
      <div class="stamp">DETERMINISTIC SCORE<strong>{TOTAL_MAX_POINTS:g} POINTS</strong></div>
    </div>
    <div class="criteria" aria-label="検索条件">
      <div><span>基準地点</span><strong>{escape(report.hotel.name)}</strong></div>
      <div><span>直線距離の上限</span><strong>{report.max_distance_km:.2f} km</strong></div>
      <div><span>生成日時</span><strong>{escape(generated)}</strong></div>
    </div>
  </header>
  <main>{rows}</main>
  <footer>
    距離は緯度経度からHaversine式で計算した直線距離です。レビューは生成日時から18か月以内に公開されたものを要約し、原文ではなく根拠リンクを掲載しています。営業日・提供メニュー・評判は変わるため、来店前に各ページで最新情報を確認してください。
  </footer>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
