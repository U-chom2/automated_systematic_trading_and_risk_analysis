# 株式売買AI - 強化学習ベースの売買判断システム

## 概要
日経225を市場指標として使用し、ターゲット企業の株式売買判断を行うAIシステムです。
直近30日の価格データとIR情報を入力として、最適な売買アクションを推奨します。

## 推論時の入力仕様

### 1. 日経225指数データ（直近30日）
- `nikkei_high`: float[30] - 日次高値
- `nikkei_low`: float[30] - 日次安値  
- `nikkei_close`: float[30] - 日次終値

### 2. ターゲット企業株価データ（直近30日）
- `target_high`: float[30] - 日次高値
- `target_low`: float[30] - 日次安値
- `target_close`: float[30] - 日次終値

### 3. IR情報
- `ir_news`: List[str] - 直近1ヶ月のIRニュース（テキスト）

## 出力仕様
- `action`: 推奨アクション（強売り/ホールド/少量買い/強買い）
- `confidence`: 信頼度（0-1）
- `recommended_position`: 推奨ポジション（-0.33〜1.0）
  - -0.33: 全売却
  - 0: ホールド
  - 1.0: 全力買い

## ディレクトリ構成
```
train/
├── main.py                 # メインスクリプト（推論・デモ）
├── train.py               # 訓練スクリプト
├── models/
│   ├── trading_model.py   # AIモデル定義
│   ├── environment/       # 取引環境
│   └── encoders/         # エンコーダーモジュール
└── README.md
```

## 使用方法

### デモ実行
```bash
# サンプルデータでデモ実行
python main.py --mode demo
```

### 推論実行
```bash
# 学習済みモデルで推論
python main.py --mode inference --model models/trading_model.pth
```

### モデル訓練
```bash
# Yahoo Financeから実データを取得して訓練
python train.py
```

## 必要なライブラリ
```bash
# uvを使用してインストール
uv add torch numpy pandas yfinance gymnasium stable-baselines3
```

## モデルアーキテクチャ

### 1. エンコーダー
- **日経225エンコーダー**: LSTM（2層、隠れ層64）
- **ターゲット株エンコーダー**: LSTM（2層、隠れ層64）
- **IRニュースエンコーダー**: キーワードベース特徴抽出 + 全結合層

### 2. 決定ネットワーク
- 入力: 結合特徴量（日経225 + ターゲット株 + IR）
- 隠れ層: 128 → 64
- 出力: 4クラス分類（売買アクション）

## 学習方法
PPO（Proximal Policy Optimization）アルゴリズムを使用した強化学習

## ライセンス
Private