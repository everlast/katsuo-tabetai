# katsuo-tabetai

高知駅前のカツオ料理店TOP 5を作る、OpenAI Agents SDK for Pythonの学習用エージェントです。Web検索から候補保存、実handoff、決定論的評価、HTML出力までを1回のrunで行います。

```text
Katsuo Research Agent
  ├─ WebSearchTool（店舗・カツオ料理・新着レビュー・根拠URLを検索）
  ├─ save_restaurant_candidates（候補JSON保存・距離判定）
  └─ handoff ──> Katsuo Evaluation Agent
                    └─ evaluate_and_render_top_five（採点・JSON/HTML出力）
```

## 受入条件の実装

- OpenAI Agents SDK: `Agent`, `Runner`, `WebSearchTool`, `function_tool`, `handoff`, `trace`を使用します。
- 2エージェント: 調査担当と評価担当が同じrun内で動作します。
- 実handoff: `researcher.handoffs=[handoff(...)]`で評価担当へ制御を移します。`Agent.as_tool()`は使用しません。
- 最終エージェント: 実行後に`result.last_agent is evaluator`を検証し、違えば失敗させます。
- 検索ツール: 調査担当の最初の行動として`WebSearchTool`を要求し、`result.new_items`に検索呼び出しがなければ失敗させます。
- 独自Function Tool: 候補保存と評価・HTML生成の2ツールを`@function_tool`で定義しています。
- トレース: ワークフロー全体を`trace()`で囲み、CLIにtrace IDとダッシュボードURLを出力します。
- 構造化保存: `outputs/restaurant_candidates.json`へ店舗情報とレビュー証拠をPydanticモデルから保存します。
- 根拠URL: 全候補の`evidence_url`とレビューごとの`review_url`を必須にし、HTMLにもリンクを表示します。
- 新着レビュー: 各店3〜8件を必須とし、生成日から18か月以内の投稿日・5点評価・要約・好評点・注意点を保存します。未来日、期間外、重複レビューはコードで拒否します。
- 範囲判定: ホテルと店舗の緯度経度からHaversine式で直線距離を計算し、`within_range`をコードで決定します。
- 決定論スコア: 保存済みの事実とレビュー評価だけから100点満点で計算し、同点時は距離、店舗名の順で整列します。
- 推薦理由: 料理特徴、レビュー平均、好評点、注意点、距離からコードで店舗ごとの推薦理由を生成します。
- HTML: `outputs/top5.html`へ推薦理由とレビュー評判を含むTOP 5を出力します。

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

デフォルトはJRクレメントイン高知を基準に、直線距離2.5 km以内を対象とします。デフォルト座標は[ホテル公式アクセスページ](https://www.jrclement.co.jp/inn/kochi/access/)の埋め込み地図中心値です。

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

モデルを明示する場合は`--model`、またはAgents SDK標準の`OPENAI_DEFAULT_MODEL`を使えます。`WebSearchTool`に対応するOpenAIモデルを指定してください。

## 出力

```text
outputs/
├── restaurant_candidates.json  # 全候補、距離、料理・レビュー根拠URL
├── top5.json                    # 採点、レビュー評判、推薦理由を含む上位5店
└── top5.html                    # 推薦理由と新着レビューを読めるランキング
```

候補JSONの主要フィールドは`name`, `address`, `latitude`, `longitude`, `katsuo_dish`, `evidence_url`, `evidence_source_type`, `source_urls`, `recent_reviews`, `distance_km`, `within_range`です。`recent_reviews`の各要素には`source_name`, `review_url`, `published_at`, `rating`, `summary`, `positive_points`, `caution_points`が入ります。

## 決定論スコア

同じ`restaurant_candidates.json`と距離上限を入力すれば、常に同じ結果になります。LLMは採点と並べ替えをしません。

- カツオ料理の根拠種別: 最大25点（店舗公式25、観光公式21、予約16、レビュー10）
- カツオ料理の特徴: 最大20点（料理名8、藁焼き5、塩たたき4、旬の案内3）
- 独立した料理根拠URL: 最大10点（ドメイン単位、最大5件）
- 新着レビューの評判: 最大25点（平均評価20、確認件数3、情報源数2）
- ホテルからの距離: 最大20点（距離上限まで線形減点）

レビューの「新着」は候補JSONの生成日から548日以内です。採点時は保存済みレビューの5点評価、件数、URLドメイン数だけを使い、LLMに点数や順位を決めさせません。推薦理由も保存済みの料理特徴、レビュー集約、距離から定型ロジックで生成します。

範囲外の候補は採点対象外です。5店未満しか範囲内に残らない場合はHTMLを作らず、調査不足として失敗します。

## トレース確認

実行後のJSONに`trace_id`と`https://platform.openai.com/traces`が表示されます。該当traceでは次を確認できます。

1. `Katsuo Research Agent`のWeb search call
2. `save_restaurant_candidates`のFunction Tool call
3. `transfer_to_katsuo_evaluation`のhandoff
4. `Katsuo Evaluation Agent`の`evaluate_and_render_top_five` call

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

APIキーなしで、距離・範囲判定、レビュー期間・重複検証、スコアの再現性、推薦理由、TOP 5整列、HTML内の料理・レビュー根拠URL、エージェント構成をテストできます。

```bash
uv --no-config run pytest
```
