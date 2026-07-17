# katsuo-tabetai

高知市内の指定ホテル周辺でカツオ料理店TOP 5を作る、OpenAI Agents SDK for Pythonの学習用エージェントです。Web検索、参照ページのスクレイピング、Markdownコンテキスト保存、実handoff、決定論的評価、HTML出力までを1つのtraceで実行します。既定モデルは`gpt-5.6-luna`です。

```text
Katsuo Web Research Agent
  ├─ 店舗発見: WebSearchTool + scrape_reference_page ──> 範囲内の候補すべて
  └─ 口コミ補完: WebSearchTool + scrape_reference_page ──> 候補5店ずつ

Collection layer
  ├─ outputs/restaurants/（店舗別の累積観測）
  └─ outputs/discovered_restaurants.json（全候補と評価除外理由）

Katsuo Research Agent
  ├─ save_restaurant_candidates（評価可能候補JSON保存）
  └─ handoff ──> Katsuo Evaluation Agent
                    └─ evaluate_and_render_top_five（採点・JSON/HTML出力）
```

## 受入条件の実装

- OpenAI Agents SDK: `Agent`, `Runner`, `WebSearchTool`, `function_tool`, `handoff`, `trace`を使用します。
- モデル: 全エージェントの既定値を`gpt-5.6-luna`に固定し、実行時のモデルIDをJSON、HTML、Markdown、run manifestへ記録します。
- 3エージェント: Web調査、候補保存・制御移譲、評価を分離します。
- 2フェーズ: Hosted `WebSearchTool`が検索callと最終messageを1レスポンスで返せるため、Web調査と後段のワークフローを別のSDK runとし、同じ`trace()`内で実行します。
- 段階的なデータ収集: 口コミサイトの個別店舗・口コミページを起点に、検証可能な口コミを持つ候補の発見と即時保存を優先します。その後、店舗公式・観光公式ページで店名、住所、カツオ料理、料理特徴を補完します。範囲内20店を収集目標として店舗発見を既定で3回繰り返し、その後に収集済み候補を5店ずつ選んで口コミを補完します。多数の店舗と全口コミを1応答へ詰め込んで口コミが欠落することを防ぎます。
- 実handoff: 後段runの`researcher.handoffs=[handoff(...)]`で評価担当へ制御を移します。`Agent.as_tool()`は使用しません。
- 最終エージェント: 実行後に`result.last_agent is evaluator`を検証し、違えば失敗させます。
- 検索・スクレイプツール: Web調査フェーズで`WebSearchTool`と`scrape_reference_page`を必須とします。食べログ、ホットペッパーグルメ、Google Maps、Yahoo!マップ、Rettyなどの口コミサイトを候補発見の起点とし、店舗公式・観光公式は料理・所在地・特徴の補完根拠として扱います。公式情報をレビュー件数には数えません。スクレイパーは公開HTTP(S)ページだけを取得し、各リダイレクト先にも公開ネットワーク制約を再適用します。HTMLから抽出した可読テキスト（最大10万文字）、取得時刻、最終URL、抽出テキストのSHA-256を保存します。
- 独自Function Tool: ページ取得、候補保存、評価・HTML生成の3ツールを`@function_tool`で定義しています。
- トレース: Web調査から候補保存・評価までを`trace()`で囲み、CLIにtrace IDとダッシュボードURLを出力します。起動時の店舗キャッシュ読込はトレース開始前に行います。
- コンテキスト: 検証済み店舗、料理根拠、追加情報源、レビューを番号付きMarkdownリストとして`outputs/context.md`へ保存します。
- 構造化保存: 調査で発見した範囲内店舗は、レビュー件数や本文照合の合否にかかわらず`outputs/restaurants/`へ即時保存します。全収集候補と評価可否・除外理由を`outputs/discovered_restaurants.json`へ集約し、評価条件を通った候補だけを`outputs/restaurant_candidates.json`へ分離します。
- 根拠URL: 収集層では、未取得または本文未確認のURLを含む候補も`outputs/restaurants/`へ保持します。評価候補に採用する`evidence_url`、全`source_urls`、全`review_url`は実際にスクレイピングし、取得本文で検証します。主根拠は店舗名・住所・料理名を照合し、不一致なら評価候補から除外します。追加根拠は主根拠と異なるドメインの店舗公式メニュー、観光公式、予約サイトの料理ページを優先し、店名、住所または支店名、料理名を照合します。不一致なら候補から削って採点対象にしません。主根拠を含む3独立ドメインを収集目標とし、見つかる場合は採点上限の5ドメインまで収集します。藁焼き、塩たたき、季節性のフラグも本文で確認できないものは`false`へ落とします。
- 新着レビュー: 各店5〜10件かつレビューURLの異なるドメインを2サイト以上必須とし、生成日から365日以内の公開日または訪問月・5点評価・投稿者名・要約・好評点・注意点を保存します。レビューURLは個別店舗のレビューを確認できるページに限定し、住所または支店名込みの店舗識別に加えて、同じ本文範囲で投稿者名、日付、評価を照合します。年月表示だけの場合は月初日として新着判定し、HTMLには年月だけを表示します。未来日、期間外、重複、本文照合できない口コミだけを評価用集合から除外し、検証済み口コミが5件未満になった店舗は不足分の別口コミを次の補完調査で探します。
- 範囲判定: ホテルと店舗の緯度経度からHaversine式で直線距離を計算し、`within_range`をコードで決定します。同じ店名でも住所・座標が異なる支店は別店舗として扱い、同一店名かつ同一住所または約50m以内の候補だけを重複として除外します。
- 決定論スコア: 保存済みの事実とレビュー評価だけから100点満点で計算し、同点時は距離、店舗名の順で整列します。
- 推薦理由: 料理特徴、レビュー平均、好評点、注意点、距離からコードで店舗ごとの推薦理由を生成します。
- HTML: `outputs/top5.html`へ推薦理由とレビュー評判を含むTOP 5の詳細を出力し、5位の直下に6位以下を順位・店名・点数・料理名・距離・平均評価の簡易リストで表示します。TOP 5の店舗ごとのレビュー一覧は初期状態を閉じ、全件をまとめて開閉できます。

## セットアップ

Python 3.11以上と[uv](https://docs.astral.sh/uv/)を前提にしています。

```bash
uv --no-config sync
cp .env.example .env
```

作成した`.env`を開き、`OPENAI_API_KEY`へ実際のキーを設定します。

```dotenv
OPENAI_API_KEY=sk-...
```

CLIは起動時に現在のディレクトリから親へ`.env`を探索して読み込みます。すでにシェル側で`OPENAI_API_KEY`が設定されている場合は、シェル側の値を優先して上書きしません。`.env`は`.gitignore`に登録済みです。

このプロジェクトは検証した公開パッケージのAPI面を固定するため、`openai-agents>=0.18.0,<0.19.0`を指定しています。

この端末ではユーザー共通のuv設定がローカルパッケージのbuildを無効化しているため、`--no-config`を付けています。該当設定がない環境では通常の`uv sync`でも構いません。

## 実行

デフォルトはザ クラウンパレス高知（2026年8月1日からANAクラウンプラザホテル高知 by IHG）を基準に、直線距離5.0 km以内を対象とします。[HMIホテルグループの公式発表](https://prtimes.jp/main/html/rd/p/000000375.000031330.html)と[IHG公式ホテルページ](https://www.ihg.com/crowneplaza/hotels/jp/ja/kochi/kczck/hoteldetail)に記載された所在地（高知県高知市本町4-2-50）を確認し、[ホテルの地図ピン](https://map.yahoo.co.jp/v3/place/K5d3Z8M-olY)の緯度`33.5577702`、経度`133.5339508`を使用しています。

```bash
uv --no-config run katsuo-tabetai
```

実際の宿泊先に変更する場合はホテル名と緯度経度を渡します。

```bash
uv --no-config run katsuo-tabetai \
  --hotel-name "宿泊ホテル名" \
  --hotel-lat 33.5669 \
  --hotel-lon 133.5435 \
  --max-distance-km 2.0
```

既定モデルは`gpt-5.6-luna`です。検証目的で変更する場合だけ`--model`を指定してください。CLIの明示値を全エージェントへ渡すため、`OPENAI_DEFAULT_MODEL`には依存しません。

Web調査はデフォルトで店舗発見3回と口コミ補完27回の合計30回を、この順序で最後まで実行します。途中で範囲内20店や評価可能5店に達しても早期終了しません。店舗発見は検索語・口コミサイト・エリア名を変えながら、範囲内20店を収集目標に新規店舗を探します。口コミ補完は未試行店舗を優先して5店ずつローテーションするため、同じ店舗群だけを繰り返しません。各回で発見した範囲内店舗は評価条件を満たさなくても`outputs/restaurants/`へ残り、同じ店舗の追加レビューや根拠は次回以降の観測と重複排除しながら合算します。回数は`--discovery-attempts`と`--review-enrichment-attempts`で個別に変更でき、どちらか一方を`0`にすることもできます。

```bash
uv --no-config run katsuo-tabetai \
  --discovery-attempts 5 \
  --review-enrichment-attempts 25
```

出力先を変える場合は`--output-dir`、各SDK runの最大ターン数を変える場合は`--max-turns`を指定します。

実行中はWeb調査や評価の進行状況を標準エラーへ表示し、API応答待ちの間も15秒ごとに経過時間を表示します。1リクエストの待機上限は300秒です。接続切断、リクエストタイムアウト、レート制限、サーバーエラーなどの一時障害は同じAPIリクエストを既定で最大5回再試行します。30回の調査を完走できるよう、ワークフロー全体は10800秒を既定値にしています。必要な場合だけ次のオプションで変更できます。

```bash
uv --no-config run katsuo-tabetai \
  --api-timeout-seconds 300 \
  --api-max-retries 5 \
  --workflow-timeout-seconds 10800
```

実行を中断する場合は`Ctrl-C`を押します。トレースバックを表示せず終了し、それまでに検証・保存できた店舗キャッシュは次回へ引き継がれます。

## 出力

```text
outputs/
├── restaurants/                # 範囲内で発見した全店舗の累積JSON
├── discovered_restaurants.json # 全候補、評価可否、評価除外理由
├── restaurant_candidates.json  # 評価条件を通った候補
├── context.md                  # 検証済み情報のMarkdownリスト
├── scrape_manifest.json        # 取得URL、時刻、最終URL、本文SHA-256
├── run_manifest.json           # モデル、trace ID、ツール実行監査
├── top5.json                    # 上位5店の詳細と6位以下の採点結果
└── top5.html                    # TOP 5詳細と6位以下の簡易リスト
```

候補JSONの主要フィールドは`name`, `address`, `latitude`, `longitude`, `katsuo_dish`, `evidence_url`, `evidence_source_type`, `source_urls`, `recent_reviews`, `distance_km`, `within_range`です。`recent_reviews`の各要素には`source_name`, `review_url`, `reviewer_name`, `published_at`, `rating`, `summary`, `positive_points`, `caution_points`が入ります。取得本文は候補JSONと店舗キャッシュに保持し、`scrape_manifest.json`には本文を重複保存せず監査メタデータだけを書きます。

`outputs/restaurants/`の各ファイルは、正規化した店名と住所から作る安定したキーで保存されます。同じ店舗を再調査すると、情報源URLとレビューを重複排除して最大10件まで累積し、住所の異なる支店は別ファイルになります。口コミのない発見結果が後から返っても、既存の充実した観測を上書きしません。起動時には範囲内の全店舗を収集プールへ読み込み、その後に保存済み本文を使って料理根拠、レビュー期間、5件以上、2サイト以上などを検証して評価プールを作ります。

本文証拠を持たない旧スキーマの候補・店舗キャッシュは検証不能として無視します。現行スキーマの個別ファイルがまだない状態で現行`restaurant_candidates.json`がある場合だけ、店舗単位のファイルへ移行します。

## 決定論スコア

同じ`restaurant_candidates.json`と距離上限を入力すれば、常に同じ結果になります。LLMは採点と並べ替えをしません。

- カツオ料理の根拠種別: 最大20点（店舗公式20、観光公式17、予約13、レビュー8）
- カツオ料理の特徴: 最大15点（料理名6、藁焼き4、塩たたき3、旬の案内2）
- 独立した料理根拠ドメイン: 最大10点（1ドメイン2点、最大5ドメイン）
- 新着レビューの評判: 最大40点（平均評価32、確認件数5、情報源数3）
- ホテルからの距離: 最大15点（距離上限まで線形減点）

### 配点の設定方法

配点と満点条件は[`src/katsuo_tabetai/scoring.py`](src/katsuo_tabetai/scoring.py)の先頭にある定数で変更します。

- `EVIDENCE_POINTS`: 店舗公式、観光公式、予約サイト、レビューサイトの配点
- `KATSUO_DISH_NAME_POINTS`, `WARAYAKI_POINTS`, `SHIO_TATAKI_POINTS`, `SEASONAL_KATSUO_POINTS`: カツオ料理の特徴ごとの配点。料理名は必須項目のため、すべての候補に加点されます
- `INDEPENDENT_SOURCE_POINTS_PER_DOMAIN`, `INDEPENDENT_SOURCE_MAX_DOMAINS`: 独立した料理根拠ドメインの単価と加点対象の上限数
- `REVIEW_RATING_MAX_POINTS`, `REVIEW_COUNT_MAX_POINTS`, `REVIEW_SOURCE_MAX_POINTS`: レビューの平均評価、確認件数、情報源数の最大配点
- `REVIEW_COUNT_FOR_MAX_POINTS`, `REVIEW_SOURCE_COUNT_FOR_MAX_POINTS`: 確認件数と情報源数が満点になる件数
- `DISTANCE_MAX_POINTS`: ホテルと同じ地点にある場合の距離配点。距離上限に達するまで線形に減点されます

`EVIDENCE_MAX_POINTS`などのカテゴリ満点と`TOTAL_MAX_POINTS`は、上記の定数から自動計算されるため直接変更しません。`RankedRestaurant.score`は100点満点を前提に検証するため、配点を変更するときは`TOTAL_MAX_POINTS == 100`を保ってください。HTMLの総合満点、レビュー満点、スコアバーは同じ定数に連動します。

変更後は配点テストを実行します。

```bash
uv --no-config run pytest tests/test_scoring.py tests/test_report.py
```

レビューの「新着」は候補JSONの生成日から365日以内です。1サイトだけ、5件未満、根拠未確認の候補も収集層には保存しますが、評価候補からは除外します。根拠ページが訪問年月しか表示しない場合、`published_at`にはその月の1日を保存して期間判定し、HTMLでは`YYYY-MM`までを表示します。採点時は評価条件を通ったレビューの5点評価、件数、URLドメイン数だけを使い、LLMに点数や順位を決めさせません。推薦理由も保存済みの料理特徴、レビュー集約、距離から定型ロジックで生成します。

範囲外の候補は採点対象外です。5店未満しか範囲内に残らない場合はHTMLを作らず、調査不足として失敗します。

## トレース確認

実行後のJSONに`trace_id`と`https://platform.openai.com/traces`が表示されます。該当traceでは次のSDKイベントを確認できます。各runの候補をキャッシュへ累積し、評価条件を検証する処理はrun間にコードで実行されます。

1. 店舗発見と口コミ補完の各SDK runにおける`Katsuo Web Research Agent`のWeb search call
2. 同じrun内で各根拠URLに対して行う`scrape_reference_page` call
3. 各runの構造化候補出力
4. 全調査runの候補を累積・検証した後に行う`Katsuo Research Agent`の`save_restaurant_candidates` Function Tool call
5. `transfer_to_katsuo_evaluation`のhandoff
6. `Katsuo Evaluation Agent`の`evaluate_and_render_top_five` call

SDKのトレースは通常デフォルトで有効です。無効化用の環境変数やカスタムtrace processorを設定している場合は、その設定を外して実行してください。

## 参照した公式Examples

実装時に参照したOpenAI Agents SDK公式Examplesを明示します。

- [examples/tools/web_search.py](https://github.com/openai/openai-agents-python/blob/main/examples/tools/web_search.py): Hosted `WebSearchTool`と`trace()`の基本形
- [examples/basic/tools.py](https://github.com/openai/openai-agents-python/blob/main/examples/basic/tools.py): `@function_tool`による独自ツール
- [examples/handoffs/message_filter.py](https://github.com/openai/openai-agents-python/blob/main/examples/handoffs/message_filter.py): `.as_tool()`ではない実handoffとtrace
- [examples/agent_patterns/deterministic.py](https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/deterministic.py): LLM判断とコードによる決定論的処理の分離
- [公式Examples一覧](https://openai.github.io/openai-agents-python/examples/)

あわせて[Handoffs](https://openai.github.io/openai-agents-python/handoffs/)、[Tools](https://openai.github.io/openai-agents-python/tools/)、[Results](https://openai.github.io/openai-agents-python/results/)、[Tracing](https://openai.github.io/openai-agents-python/tracing/)を参照しています。

## テスト

APIキーなしで、距離・範囲判定、レビュー期間・重複検証、スクレイプ本文照合、別店舗・別支店の排除、加点フラグの裏取り、Markdownコンテキスト、スコアの再現性、TOP 5 HTML、エージェント構成をテストできます。

```bash
uv --no-config run pytest
```
