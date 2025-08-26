# リスクモデル（ニューラルネットワーク）訓練

このディレクトリには、リスクモデルのニューラルネットワークを訓練するためのスクリプトが含まれています。

## 📋 概要

リスクモデルは、市場の状態から最適な損切り幅（ストップロス％）を予測するニューラルネットワークです。

### 入力特徴量（6次元）
1. **ヒストリカル・ボラティリティ (HV)** - 過去20日間の価格変動率
2. **ATR (Average True Range)** - 平均真の値幅（ボラティリティ指標）
3. **RSI** - 相対力指数（買われすぎ・売られすぎ指標）
4. **出来高比率** - 20日平均に対する現在の出来高比率
5. **移動平均乖離率** - 20日移動平均からの乖離
6. **ベータ値** - 市場全体に対する感応度

### 出力
- **最適な損切り幅（%）** - 0.01（1%）から0.15（15%）の範囲

## 🚀 クイックスタート

最も簡単な方法は、提供されているシェルスクリプトを実行することです：

```bash
cd train
./run_training.sh
```

このスクリプトは以下を自動的に実行します：
1. 必要なパッケージのインストール
2. 訓練データの生成
3. モデルの訓練
4. モデルのテスト

## 📝 手動実行

### 1. 訓練データの生成

```bash
# 米国株データで生成
uv run python generate_training_data.py

# 日本株データで生成（ターゲット企業.xlsxから）
uv run python generate_japan_training_data.py

# テスト用モックデータ生成（動作確認用）
uv run python create_mock_data.py
```

このスクリプトは：
- 米国の主要20銘柄から過去2年分のデータを取得
- 各銘柄から30サンプルを生成（合計約600サンプル）
- 80%を訓練用、20%を検証用に分割
- `data/train_data.json`と`data/val_data.json`に保存

### 2. モデルの訓練

```bash
uv run python train_risk_model.py
```

このスクリプトは：
- 3層のニューラルネットワーク（入力層6→隠れ層32→隠れ層16→出力層1）を構築
- 最大200エポックの訓練（早期停止あり）
- Adam最適化とMSE損失関数を使用
- 最良モデルを`models/best_risk_model.pth`に保存

### 3. モデルのテスト

```bash
uv run python test_model.py
```

このスクリプトは：
- 訓練済みモデルをロード
- 様々な市場シナリオでテスト
- 検証データでの精度を評価
- 本番環境用のモデルを`models/risk_model.pth`に保存

## 📊 生成されるファイル

```
train/
├── data/
│   ├── train_data.json         # 訓練データセット
│   └── val_data.json          # 検証データセット
└── models/
    ├── best_risk_model.pth    # 最良の検証損失を持つモデル
    ├── final_risk_model.pth   # 最終モデル
    ├── risk_model.pth         # 本番用モデル
    ├── training_history.png   # 訓練履歴グラフ
    ├── predictions_vs_actual.png # 予測精度グラフ
    └── metrics.json          # 性能指標

```

## 🎯 期待される性能

訓練後のモデルは以下の性能を目指します：
- **MAE（平均絶対誤差）**: < 2%
- **R²スコア**: > 0.6
- **検証損失**: < 0.001

## 💡 カスタマイズ

### 訓練データに日本株を追加

`generate_training_data.py`の銘柄リストを編集：

```python
training_symbols = [
    '7203.T',  # トヨタ
    '6758.T',  # ソニー
    '9984.T',  # ソフトバンク
    # ... 追加銘柄
]
```

### ハイパーパラメータの調整

`train_risk_model.py`の設定を編集：

```python
config = {
    'epochs': 300,        # エポック数を増やす
    'batch_size': 64,     # バッチサイズを変更
    'learning_rate': 0.0005  # 学習率を調整
}
```

## ⚠️ 注意事項

1. **データ生成には時間がかかります** - 約10-20分
2. **訓練には時間がかかります** - 約10-30分（GPUなしの場合）
3. **インターネット接続が必要** - Yahoo Financeからデータを取得
4. **十分なストレージ** - 約100MB必要

## 🔧 トラブルシューティング

### talib-binaryのインストールエラー
```bash
# macOSの場合
brew install ta-lib

# その後
uv add talib-binary
```

### メモリ不足エラー
バッチサイズを小さくしてください：
```python
config = {'batch_size': 16}
```

### 訓練が収束しない
- 学習率を下げる（例：0.0001）
- エポック数を増やす
- データの正規化を確認

## 📈 本番環境での使用

訓練済みモデルは以下のように使用できます：

```python
from src.analysis_engine.risk_model import RiskModel

# モデルをロード
risk_model = RiskModel()
risk_model.load_model('train/models/risk_model.pth')

# 予測
features = {
    'historical_volatility': 0.25,
    'atr': 14.5,
    'rsi': 65,
    'volume_ratio': 1.5,
    'ma_deviation': 0.08,
    'beta': 1.2
}

prediction = risk_model.predict(features)
print(f"推奨損切り幅: {prediction.stop_loss_percentage * 100:.2f}%")
```