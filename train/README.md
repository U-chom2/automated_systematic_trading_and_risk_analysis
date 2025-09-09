# 株式売買AI - 強化学習ベースの売買判断システム

## 概要
東証グロース市場605社を対象とした強化学習ベースの株式売買AIシステムです。
日経225を市場指標として使用し、過去1年間のデータで訓練を行い、複数銘柄に対する最適な売買判断を同時実行します。

## データセット
- **対象銘柄**: 東証グロース市場全605社
- **データ期間**: 過去1年間（365日）
- **データソース**: Yahoo Finance API
- **更新頻度**: 日次

## 推論時の入力仕様

### 1. 日経225指数データ（直近30日）
- `nikkei_high`: float[30] - 日次高値
- `nikkei_low`: float[30] - 日次安値  
- `nikkei_close`: float[30] - 日次終値

### 2. グロース605社株価データ（当日）
- `stock_prices`: float[605] - 各銘柄の終値
- `stock_volumes`: float[605] - 各銘柄の出来高

### 3. ポートフォリオ状態
- `cash_ratio`: float - 現金比率
- `position_ratios`: float[605] - 各銘柄のポジション比率

### 4. IR情報（オプション）
- `ir_sentiment`: float[605] - ModernBERT-jaによるセンチメント分析結果

## 出力仕様
- `actions`: float[605] - 各銘柄への推奨アクション
  - -1.0: 全売却
  - 0.0: ホールド  
  - +1.0: 全力買い
- `portfolio_value`: float - 予想ポートフォリオ価値

## ディレクトリ構成
```
train/
├── main.py                     # メインスクリプト（推論・デモ）
├── train.py                   # 訓練スクリプト（東証グロース605社対応）
├── models/
│   ├── trading_model.py       # AIモデル定義
│   ├── agents/
│   │   └── ppo_agent.py      # PPOエージェント実装
│   └── environment/
│       └── trading_env.py    # 取引環境
├── rl/                       # 訓練済みモデル保存先
│   └── ppo_nikkei_model_*.zip
└── README.md
```

## 使用方法

### モデル訓練
```bash
# 東証グロース605社データで訓練
uv run python train/train.py

# 訓練パラメータ:
# - 訓練ステップ数: 80,000
# - 学習率: 1e-4
# - バッチサイズ: 64
# - デバイス: MPS (Apple Silicon) / CUDA / CPU
```

### 推論実行
```bash
# 学習済みモデルで推論
uv run python train/main.py --model train/models/rl/ppo_nikkei_model_*.zip
```

## 必要なライブラリ
```bash
# uvを使用してインストール（CLAUDE.mdの指示に従い、pipは使用禁止）
uv add torch numpy pandas yfinance gymnasium stable-baselines3
uv add transformers  # ModernBERT-ja用
```

## モデルアーキテクチャ

### 1. 観測空間（約7,956次元）
```
観測ベクトル構成:
├── 日経225データ: 90次元（30日 × 3特徴量）
│   ├── 高値 (High) [正規化済み]
│   ├── 安値 (Low) [正規化済み]  
│   └── 終値 (Close) [正規化済み]
│
├── 全605社株式データ: 1,210次元（605社 × 2特徴量）
│   ├── 終値 (Close) [正規化・クリッピング済み]
│   └── 出来高 (Volume) [正規化・クリッピング済み]
│
├── ポートフォリオ状態: 606次元
│   ├── 現金比率: 1次元
│   └── 各銘柄ポジション比率: 605次元
│
└── IR情報: 6,050次元（605社 × 10特徴量）
    └── ModernBERT-jaセンチメント分析結果
```

### 2. ニューラルネットワーク構造

#### 特徴抽出器 (TradingFeaturesExtractor)
```
入力: 7,956次元
├── Linear(7,956 → 1,024) + LayerNorm + ReLU + Dropout(0.2)
├── Linear(1,024 → 512) + LayerNorm + ReLU + Dropout(0.2)
└── Linear(512 → 512) + ReLU
出力: 512次元特徴量
```

#### PPOポリシー・バリューネットワーク
```
共通特徴抽出: 512次元

ポリシーネットワーク (Actor):
├── Linear(512 → 256) + ReLU
├── Linear(256 → 128) + ReLU
└── Linear(128 → 605) → 各銘柄のアクション値

バリューネットワーク (Critic):
├── Linear(512 → 256) + ReLU  
├── Linear(256 → 128) + ReLU
└── Linear(128 → 1) → 状態価値推定
```

### 3. 行動空間
- **タイプ**: 連続行動空間 Box(-1.0, 1.0, shape=(605,))
- **解釈**: 各銘柄に対する売買強度
  - -1.0: 全売却
  - 0.0: ホールド
  - +1.0: 全力買い

### 4. 報酬設計
- **ポートフォリオ価値変化**: メインの報酬シグナル
- **取引コスト**: 手数料を考慮したペナルティ
- **リスク調整**: 過度な集中投資に対する抑制

## 訓練詳細

### 1. PPO (Proximal Policy Optimization) パラメータ
```
total_timesteps: 80,000      # 総訓練ステップ数
learning_rate: 1e-4          # 学習率（低めで安定性重視）
n_steps: 1,024              # ロールアウトステップ数
batch_size: 64              # バッチサイズ
n_epochs: 4                 # PPO更新エポック数
gamma: 0.99                 # 割引率
gae_lambda: 0.95            # GAE係数
clip_range: 0.2             # PPOクリップ範囲
ent_coef: 0.01              # エントロピー係数
vf_coef: 0.5               # 価値関数係数
max_grad_norm: 0.5          # 勾配クリッピング閾値
```

### 2. 正規化・安定化技術
- **Layer Normalization**: 各層で活性化値を正規化
- **Gradient Clipping**: 勾配爆発を防止
- **Dropout**: 過学習を防止（レート: 0.2）
- **Data Clipping**: 入力値の範囲制限で安定性向上
- **NaN Handling**: 欠損値・異常値の自動処理

### 3. 訓練環境
- **初期資金**: 1,000万円
- **取引手数料**: 0.1%
- **ウィンドウサイズ**: 30日（日経225履歴）
- **エピソード長**: 約214ステップ（1年間の取引日数）

### 4. ハードウェア対応
- **Apple Silicon (MPS)**: 最優先
- **NVIDIA GPU (CUDA)**: 次優先  
- **CPU**: フォールバック

## データ前処理

### 1. 価格正規化
```python
normalized_price = np.clip(price / 1000, 0, 50)  # 0-50範囲にクリップ
normalized_volume = np.clip(volume / 1e6, 0, 100)  # 0-100範囲にクリップ
```

### 2. NaN値処理
- **株価**: デフォルト値1000円で補完
- **出来高**: デフォルト値1000株で補完
- **技術指標**: 0で補完

### 3. バッチ処理
- **データ取得**: 50社ずつバッチ処理でAPI制限を回避
- **失敗時**: 自動スキップで訓練継続

## 性能指標
- **年間収益率**: ベンチマーク（日経225）との比較
- **シャープレシオ**: リスク調整後収益
- **最大ドローダウン**: 最大損失期間
- **勝率**: 利益を出した取引の割合

## ライセンス
Private