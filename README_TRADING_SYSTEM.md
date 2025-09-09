# 自動取引システム

## 概要

東証グロース市場を対象とした自動株式取引システムです。3つのモジュールで構成されており、企業スクリーニングから取引実行までを自動化します。

## システム構成

### Module 1: 企業スクリーニング
- 東証グロース607社から対象企業を抽出
- 時価総額100億円以下の企業をフィルタリング
- IR公開が1ヶ月以内の企業を選定
- 結果を`target.csv`に保存

### Module 2: 個別銘柄分析・取引推奨
- `target.csv`の企業を個別に分析
- テクニカル分析、ファンダメンタル分析、センチメント分析を実施
- 取引推奨（買い・売り・保有）を生成
- 翌日のTODOリストを作成

### Module 3: 取引記録管理
- 16時の終値で取引を記録
- ポートフォリオ管理
- 日次決済処理
- 損益計算とレポート生成

## インストール

```bash
# 依存関係のインストール
make install

# 開発用依存関係のインストール
make dev-install
```

## 使用方法

### 個別モジュールの実行

```bash
# Module 1: スクリーニング実行
make screening

# Module 2: 分析実行
make analysis

# Module 3: 取引記録
make recording
```

### 定期実行モード

```bash
# 朝の処理（9:00想定）
make morning

# 夕方の処理（16:00想定）
make evening

# フルサイクル実行（テスト用）
make trade-full
```

### スケジューラー起動

```bash
# 自動スケジュール実行
make scheduler
```

これにより以下のスケジュールで自動実行されます：
- 9:00 - 企業スクリーニングと銘柄分析
- 16:00 - 取引記録と決済処理

### カスタムパラメータでの実行

```bash
# 時価総額上限を200億円に設定
python main_trading_system.py --mode screening --market-cap 200

# IR期間を60日に設定
python main_trading_system.py --mode screening --ir-days 60

# 特定日の取引記録
python main_trading_system.py --mode recording --date 2024-01-15
```

## データファイル

システムは以下のデータファイルを使用します：

- `data/target.csv` - スクリーニング結果
- `data/todos.json` - 取引TODOリスト
- `data/portfolio.json` - ポートフォリオ状態
- `data/trade_records.json` - 取引履歴
- `data/settlements.json` - 決済記録

### データクリーンアップ

```bash
# 全データファイルを削除
make clean-data

# ポートフォリオをリセット
make reset-portfolio
```

## 処理フロー

1. **毎日9:00** - 朝の処理
   - 東証グロース全社をスクリーニング
   - 条件（時価総額≤100億円、IR≤30日）でフィルタリング
   - 通過企業を`target.csv`に保存
   - 個別銘柄を分析
   - 取引推奨を生成
   - 翌日のTODOリストを作成

2. **毎日16:00** - 夕方の処理
   - TODOリストを読み込み
   - 各銘柄の終値を取得
   - 終値で取引を記録
   - ポートフォリオを更新
   - 日次決済を作成
   - レポートを生成

## 設定パラメータ

### スクリーニング条件
- `market_cap_limit_billion`: 時価総額上限（デフォルト: 100億円）
- `ir_days_within`: IR公開からの日数（デフォルト: 30日）

### 取引設定
- `commission_rate`: 手数料率（デフォルト: 0.1%）
- `initial_capital`: 初期資金（デフォルト: 1000万円）

## ログ

システムはvibeloggerを使用してログを出力します。各モジュールの実行状況が詳細に記録されます。

## 注意事項

- 本システムはデモ・研究用です
- 実際の取引には使用しないでください
- 市場休業日（土日・祝日）は自動的にスキップされます
- エラー発生時は処理を中断し、ログに記録します

## トラブルシューティング

### データファイルのリセット
```bash
make clean-data
make reset-portfolio
```

### 依存関係の再インストール
```bash
uv sync --dev
```

### ログの確認
各モジュールの実行時にコンソールに詳細なログが出力されます。

## ライセンス

MIT License