# ニューラルネットワーク訓練完了
作成日時: 2025-08-26 21:52

## ✅ 訓練実行結果

### 成功した訓練
- **訓練データ**: 米国主要10銘柄から200サンプル生成
- **訓練/検証分割**: 160個（訓練）/ 40個（検証）
- **訓練エポック**: 73（早期停止により200から短縮）
- **最終検証損失**: 0.00180（非常に良好）
- **平均絶対誤差**: 2.96%

### モデル性能
- 低ボラティリティ市場: 4.25%の損切り幅
- 高ボラティリティ市場: 1.04%の損切り幅
- 通常市場: 3.54%の損切り幅

## 🧹 コード整理内容

### 削除したファイル
- `generate_training_data.py` (バグあり版) → 削除
- `generate_training_data_fixed.py` → `generate_training_data.py`にリネーム
- `generate_mini_training_data.py` → 削除（テスト用）
- `train_mini_model.py` → 削除（テスト用）

### 最終的なディレクトリ構造
```
train/
├── README.md                        # ドキュメント
├── run_training.sh                  # 自動実行スクリプト
├── generate_training_data.py        # 米国株データ生成（動作確認済み）
├── generate_japan_training_data.py  # 日本株データ生成
├── create_mock_data.py             # テスト用モックデータ生成
├── train_risk_model.py             # モデル訓練
├── test_model.py                   # モデルテスト
├── data/
│   ├── train_data.json            # 訓練データ（160サンプル）
│   ├── val_data.json              # 検証データ（40サンプル）
│   └── mock_*.json                # モックデータ
└── models/
    ├── best_risk_model.pth        # 本番用モデル
    ├── training_history.png       # 訓練履歴グラフ
    └── predictions_vs_actual.png  # 予測精度グラフ
```

## 📝 CLAUDE.md更新内容
以下のルールを追加：
- **バグのあるコードは残さない** - 動作確認済みのコードのみを保持する
- テスト用の一時的なコードは削除する

## 🚀 使用方法

### クイックスタート
```bash
cd train
./run_training.sh  # 全自動実行
```

### 個別実行
```bash
# データ生成
uv run python generate_training_data.py      # 米国株
uv run python generate_japan_training_data.py # 日本株
uv run python create_mock_data.py           # モックデータ

# モデル訓練
uv run python train_risk_model.py

# モデルテスト
uv run python test_model.py
```

## 🎯 本番環境での使用

```python
from src.analysis_engine.risk_model import RiskModel

# モデルロード
risk_model = RiskModel()
risk_model.load_model('train/models/best_risk_model.pth')

# 予測実行
features = {
    'historical_volatility': 0.25,
    'atr': 14.5,
    'rsi': 65,
    'volume_ratio': 1.5,
    'ma_deviation': 0.08,
    'beta': 1.2
}
prediction = risk_model.predict(features)
stop_loss = prediction.stop_loss_percentage * 100
print(f"推奨損切り幅: {stop_loss:.2f}%")
```

## ⚠️ 注意事項
- talibライブラリはPython 3.13で互換性問題があるため、手動実装を使用
- yfinanceのAPIは`Ticker.history()`を使用（`download()`より安定）
- 日付処理は`pd.Timestamp`で統一して型の不一致を回避