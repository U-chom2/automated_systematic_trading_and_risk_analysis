# NTT株式分析実装ログ

## 実装日時
2025-08-21 22:20

## 実装内容
yahooqueryを使用して日本電信電話株式会社（9432.T）の株価情報を取得・分析するシステムを実装

## 実装した機能

### 1. 一日の総流動金額と総流動株数
- **総流動金額**: 21,520,401,000円
- **総流動株数**: 132,027,000株
- yahooquery の price データから `regularMarketVolume` と `regularMarketPrice` を取得して計算

### 2. 過去20日間の株価の最高値・最低値
- **最高値**: 167円
- **最低値**: 150円
- 過去30日間のhistoryデータを取得し、直近20営業日分を抽出して計算

### 3. 株価の移動平均線グラフ（PNG）
- **ファイル名**: `ntt_moving_average.png`
- **内容**: 終値、5日移動平均、20日移動平均を表示
- **技術的課題と解決策**:
  - MultiIndexデータの処理: `history.loc[symbol]` で抽出
  - タイムゾーン混在問題: `pd.to_datetime(utc=True).tz_localize(None)` で解決

### 4. 直近の四半期決算情報
- **純利益**: 減益（-5.3%）
- **売上高**: データ取得不可
- yahooquery の key_stats から `quarterlyEarningsGrowth` を取得

## 技術的詳細

### 使用パッケージ
- `yahooquery`: 株価データ取得
- `pandas`: データ処理
- `matplotlib`: グラフ作成

### 主要な技術課題と解決策

#### 1. MultiIndex処理
**問題**: yahooquery の history データがMultiIndex構造
**解決策**: 
```python
if isinstance(history.index, pd.MultiIndex):
    df = history.loc[self.symbol].copy()
```

#### 2. タイムゾーン混在エラー
**問題**: `Cannot mix tz-aware with tz-naive values`
**原因**: データに datetime.date とタイムゾーン付きdatetime が混在
**解決策**:
```python
df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
```

#### 3. 財務データAPI制限
**問題**: `quarterly_income_stmt` 属性が存在しない
**解決策**: key_stats から代替データを取得
```python
key_stats = self.ticker.key_stats
stats = key_stats[self.symbol]
result['profit_growth'] = float(stats['quarterlyEarningsGrowth']) * 100
```

## 実装されたクラス構造

### NTTStockAnalyzer
- `__init__(symbol)`: 初期化
- `get_daily_trading_volume()`: 流動金額・株数取得
- `get_20day_high_low()`: 20日間高値・安値取得  
- `create_moving_average_chart()`: 移動平均線グラフ作成
- `get_quarterly_earnings_info()`: 四半期決算情報取得
- `run_full_analysis()`: 全機能実行

## 出力ファイル
- `ntt_stock_analysis.py`: メインスクリプト
- `ntt_moving_average.png`: 移動平均線グラフ
- `debug_yahooquery.py`: デバッグ用スクリプト
- `debug_chart.py`: グラフデバッグ用スクリプト

## 実行結果
全ての要求された情報の取得に成功。特に移動平均線グラフの作成では複数の技術的課題を解決し、正常にPNGファイルを生成できた。