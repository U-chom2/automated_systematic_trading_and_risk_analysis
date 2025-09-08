# 自動システム売買・リスク分析システム

Automated Systematic Trading and Risk Analysis System

## 🚀 概要

本プロジェクトは、日本株市場（特に東証グロース市場）を対象とした、AI駆動型の自動売買システムです。カタリスト（株価変動要因）を包括的に分析し、データドリブンな投資判断を行い、厳格なリスク管理のもとで自動取引を実行します。

### 主な特徴

- 🔍 **包括的なデータ収集**: TDNet、SNS（X/Twitter）、掲示板から情報を自動収集
- 🧠 **AI分析エンジン**: 自然言語処理とディープラーニングによる市場分析
- 📊 **テクニカル分析**: TA-Libを使用した高度なテクニカル指標の算出
- ⚡ **リアルタイム取引**: 証券会社APIとの連携による自動売買
- 🛡️ **リスク管理**: ニューラルネットワークによる動的なリスクサイジング

## 🏗️ アーキテクチャ

本システムはClean Architecture原則に基づいて設計されており、ビジネスロジックを外部の技術的詳細から独立させています。

```text
┌─────────────────────────────────────────────────────┐
│              Presentation Layer                     │
│         (FastAPI, WebSocket, CLI)                  │
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────┐
│              Application Layer                      │
│    (Use Cases, DTOs, Application Services)         │
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────┐
│                 Domain Layer                        │
│     (Entities, Value Objects, Domain Services)     │
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────┐
│            Infrastructure Layer                     │
│  (Database, External APIs, Message Queue, Cache)   │
└─────────────────────────────────────────────────────┘
```

### レイヤー構成

#### 1. **ドメイン層** (`src/domain/`)
- **Entities**: ポートフォリオ、取引、ポジション、シグナル等のビジネスエンティティ
- **Value Objects**: Money、Price、Quantity等の不変オブジェクト
- **Domain Services**: 取引戦略、リスク管理、シグナル生成等のビジネスロジック
- **Repository Interfaces**: データ永続化の抽象インターフェース

#### 2. **アプリケーション層** (`src/application/`)
- **Use Cases**: ポートフォリオ管理、取引実行、バックテスト等のビジネスユースケース
- **DTOs**: レイヤー間のデータ転送オブジェクト
- **Application Services**: ユースケースのオーケストレーション

#### 3. **インフラストラクチャ層** (`src/infrastructure/`)
- **Database**: PostgreSQL/TimescaleDB接続とSQLAlchemyモデル
- **Repositories**: リポジトリパターンの具体実装
- **External APIs**: Yahoo Finance、TDNet等の外部API連携
- **Cache**: Redis キャッシュ実装
- **AI Models**: PPO、ModernBERT等のAIモデル統合

#### 4. **プレゼンテーション層** (`src/presentation/`)
- **API**: FastAPIによるREST APIエンドポイント
- **WebSocket**: リアルタイム通信
- **CLI**: コマンドラインインターフェース

### 技術スタック

- **フレームワーク**: FastAPI (非同期Web API)
- **データベース**: PostgreSQL + TimescaleDB (時系列データ)
- **キャッシュ**: Redis
- **ORM**: SQLAlchemy 2.0 (async)
- **ML/AI**: PyTorch, Stable-Baselines3, Transformers
- **テクニカル分析**: TA-Lib
- **非同期処理**: asyncio, aiohttp
- **テスト**: pytest, pytest-asyncio
- **コンテナ**: Docker, Docker Compose

## 🔧 セットアップ

### 前提条件

- Python 3.13以上
- uv (Pythonパッケージ管理ツール)
- TA-Lib ライブラリ（システムレベル）

### uvのインストール

```bash
# macOSの場合
brew install uv

# または pipxを使用
pipx install uv

# Linuxの場合
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### プロジェクトのセットアップ

```bash
# リポジトリをクローン
git clone https://github.com/your-username/automated_systematic_trading_and_risk_analysis.git
cd automated_systematic_trading_and_risk_analysis

# Python環境のセットアップ
uv python install 3.13

# 依存パッケージのインストール
uv sync

# TA-Libのインストール（macOSの場合）
brew install ta-lib

# TA-LibのPythonバインディングをインストール
uv add ta-lib
```

### 環境設定

```bash
# 環境変数の設定（.envファイルを作成）
cat <<EOF > .env
# API Keys
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret

# 証券会社API（必要に応じて）
BROKER_API_KEY=your_broker_api_key
BROKER_API_SECRET=your_broker_api_secret

# データベース設定（オプション）
DATABASE_URL=sqlite:///trading.db
EOF
```

## 🎯 使用方法

### Docker環境での実行

```bash
# Dockerコンテナの起動
make docker-up

# または docker-compose直接
docker-compose up -d

# ログの確認
make docker-logs

# コンテナの停止
make docker-down
```

### ローカル環境での実行

```bash
# FastAPIアプリケーションの起動
make run

# または直接実行
uv run uvicorn src.presentation.api.app:app --reload --host 0.0.0.0 --port 8000

# APIドキュメント: http://localhost:8000/docs
```

### API使用例

```bash
# ポートフォリオ作成
curl -X POST "http://localhost:8000/api/v1/portfolios/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Portfolio",
    "initial_capital": 10000000,
    "description": "Test portfolio"
  }'

# 市場データ取得
curl "http://localhost:8000/api/v1/market-data/price/7203.T"

# バックテスト実行
curl -X POST "http://localhost:8000/api/v1/backtest/run" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["7203.T", "9984.T"],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 10000000,
    "strategy_type": "AI_DRIVEN"
  }'
```

### 強化学習AIモデルの訓練と推論

本システムは方針.mdに基づくLSTM + ModernBERT + PPOアーキテクチャを採用しています。

#### 訓練フロー

```bash
# 訓練データの準備と学習実行（Yahoo Finance + TDnet）
cd train
uv run python train.py

# 訓練パラメータ:
# - 市場指標: 日経225指数（^N225）
# - ターゲット銘柄: 
#   * 7203.T（トヨタ）
#   * 9984.T（ソフトバンクグループ）
#   * 6758.T（ソニー）
# - 訓練期間: 2022-01-01 〜 2024-01-01
# - 総ステップ数: 30,000タイムステップ
# - アルゴリズム: PPO（Proximal Policy Optimization）
# - 環境: Gymnasium互換の取引シミュレーター
```

#### 推論フロー

```bash
# デモモード（サンプルデータ使用）
uv run python train/main.py --mode demo

# 推論モード（学習済みモデル使用）
uv run python train/main.py --mode inference --model train/models/trading_model.pth

# リアルタイムデータで推論（実装予定）
uv run python train/main.py --mode inference --data realtime
```

#### モデルアーキテクチャ（方針.md準拠）

- **時系列エンコーダー**: LSTM（2層、隠れ層64次元）
  - 株価パターンとテクニカル指標の抽出
  - 30日間の時系列データを処理
  
- **ニュースエンコーダー**: ModernBERT-ja-130m
  - IRニュースの文脈理解と感情分析
  - 日本語特化の事前学習済みモデル
  
- **意思決定エンジン**: Actor-Critic（PPO）
  - 複数銘柄の同時売買判断
  - リスク調整済み報酬の最大化

#### 入出力仕様

**入力データ（30日分）:**
- 日経225: 高値・安値・終値の日次データ
- ターゲット企業: 高値・安値・終値の日次データ  
- IRニュース: 直近1ヶ月のテキストデータ

**出力:**
- 売買アクション: 強売り(-1) / ホールド(0) / 少量買い(0.5) / 強買い(1)
- 推奨ポジション: -0.33（全売却）〜 1.0（全力買い）
- 信頼度スコア: 0〜1の確率値

### テストの実行

```bash
# 全テストを実行
uv run pytest

# カバレッジレポート付きでテスト実行
uv run pytest --cov=src --cov-report=html

# 特定のテストファイルを実行
uv run pytest tests/test_specific.py

# 統合テストを実行
uv run python test_model_integration.py
```

### 定時実行（本番環境）

```bash
# cronでの定時実行設定例
# 毎日16:15に分析を実行
15 16 * * * cd /path/to/project && uv run python main.py >> logs/daily_analysis.log 2>&1
```

## 📝 開発ガイドライン

### コーディング規約

1. **型ヒント必須**: すべての関数・メソッドに型ヒントを付ける
2. **ドキュメンテーション**: パブリックAPIには必ずdocstringを記載
3. **行長制限**: 最大88文字
4. **エラーハンドリング**: vibeloggerでロギング、例外は必ずraiseする
5. **関数設計**: 単一責任の原則に従い、小さく集中した関数を作る

### パッケージ管理

```bash
# パッケージの追加
uv add package_name

# 開発用パッケージの追加
uv add --dev package_name

# パッケージのアップグレード
uv add --upgrade-package package_name package_name

# 禁止事項
# × uv pip install package_name  # 使用禁止
# × uv add package@latest        # @latest構文は使用禁止
```

### TDD（テスト駆動開発）

```python
# 1. テストを先に書く (Red)
def test_calculate_position_size():
    """ポジションサイズ計算のテスト"""
    assert calculate_position_size(10000, 0.01, 100) == 10

# 2. 最小限の実装 (Green)
def calculate_position_size(capital: float, risk_pct: float, 
                           price: float) -> int:
    """リスクに基づくポジションサイズを計算"""
    return int((capital * risk_pct) / price)

# 3. リファクタリング (Refactor)
```

### ブランチ戦略

```bash
# 機能開発
git checkout -b feature/機能名

# バグ修正
git checkout -b fix/バグ名

# ドキュメント更新
git checkout -b docs/更新内容
```

### 実装ログ

新機能を実装した際は、必ず実装ログを残す：

```bash
# 日付を確認
date

# ログファイルを作成
echo "# 実装内容" > _docs/2025-08-28-17-57_機能名.md
```

## 📊 プロジェクト構造

```text
.
├── README.md                 # このファイル
├── CLAUDE.md                 # 開発規約とガイドライン
├── pyproject.toml            # プロジェクト設定と依存関係
├── uv.lock                   # ロックファイル
├── .env.example              # 環境変数サンプル
├── docker-compose.yml        # Docker構成
├── Dockerfile                # アプリケーションコンテナ
├── Makefile                  # 開発タスク自動化
├── alembic.ini               # データベースマイグレーション設定
├── pytest.ini                # テスト設定
│
├── src/                      # ソースコード（Clean Architecture）
│   ├── domain/               # ドメイン層
│   │   ├── entities/         # ビジネスエンティティ
│   │   ├── value_objects/    # 値オブジェクト
│   │   ├── services/         # ドメインサービス
│   │   └── repositories/     # リポジトリインターフェース
│   │
│   ├── application/          # アプリケーション層
│   │   ├── use_cases/        # ユースケース
│   │   ├── dto/              # データ転送オブジェクト
│   │   └── services/         # アプリケーションサービス
│   │
│   ├── infrastructure/       # インフラストラクチャ層
│   │   ├── database/         # データベース実装
│   │   ├── repositories/     # リポジトリ実装
│   │   ├── external_apis/    # 外部API連携
│   │   ├── cache/            # キャッシュ実装
│   │   └── ai_models/        # AIモデル統合
│   │
│   ├── presentation/         # プレゼンテーション層
│   │   ├── api/              # FastAPI実装
│   │   │   ├── app.py        # アプリケーション設定
│   │   │   └── routers/      # APIエンドポイント
│   │   ├── websocket/        # WebSocket実装
│   │   └── cli/              # CLIインターフェース
│   │
│   └── common/               # 共通モジュール
│       ├── config.py         # 設定管理
│       ├── logging.py        # ロギング設定
│       ├── exceptions.py     # カスタム例外
│       ├── validators.py     # バリデーション
│       ├── utils.py          # ユーティリティ
│       └── constants.py      # 定数定義
│
├── train/                    # 強化学習モデル訓練
│   ├── 方針.md               # モデル設計ドキュメント
│   ├── main.py               # 推論・デモ実行
│   ├── train.py              # 訓練パイプライン
│   └── models/               # モデル実装
│
├── tests/                    # テストコード
│   ├── conftest.py           # テスト共通設定
│   ├── unit/                 # ユニットテスト
│   │   ├── domain/           # ドメイン層テスト
│   │   └── application/      # アプリケーション層テスト
│   ├── integration/          # 統合テスト
│   └── e2e/                  # E2Eテスト
│
├── scripts/                  # スクリプト
│   └── init.sql              # データベース初期化
│
├── docs/                     # ドキュメント
│   └── architecture/         # アーキテクチャ設計書
│
└── alembic/                  # データベースマイグレーション
    └── versions/             # マイグレーションファイル
```

## 🔒 セキュリティとリスク管理

### リスク管理ルール

- **1トレードの最大リスク**: 総資金の1%
- **ポジションサイジング**: ニューラルネットワークによる動的調整
- **OCO注文**: 利益確定と損切りの同時設定
- **ボラティリティベースのストップロス**: ATRを基準とした損切り幅の設定

### セキュリティ

- APIキーは環境変数で管理（.envファイルはgitignore）
- 本番環境ではシークレット管理サービスを使用
- ログにセンシティブ情報を含めない

## 📈 パフォーマンス目標

- **定時分析**: 16:15開始から30分以内に完了
- **稼働率**: 99.9%以上
- **テストカバレッジ**: 80%以上

## 🤝 コントリビューション

1. Issueで機能提案・バグ報告
2. Forkしてブランチを作成
3. 変更を実装（テスト必須）
4. Pull Requestを提出

## 📄 ライセンス

プロプライエタリ - 詳細は要相談

## 🆘 サポート

- Issue: [GitHub Issues](https://github.com/your-username/automated_systematic_trading_and_risk_analysis/issues)
- Email: support@example.com

---

**免責事項**: 本システムは投資判断の参考ツールです。投資は自己責任で行ってください。システムの使用により生じた損失について、開発者は一切の責任を負いません。