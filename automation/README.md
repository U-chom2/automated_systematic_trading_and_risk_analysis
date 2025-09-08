# デイトレードシステム自動実行

macOS launchdを使用したデイトレードシステムの自動実行設定

## 📅 自動実行スケジュール

- **毎日 09:30** - 価格更新・売却判定（東証開始後）
- **毎日 16:00** - 新規推奨銘柄分析・購入（東証クローズ後）
- **平日のみ実行**（土日・祝日は自動スキップ）

## 🚀 セットアップ手順

### 1. 自動実行を有効化

```bash
cd /Users/y.okumura/private_workspace/automated_systematic_trading_and_risk_analysis/automation
./setup_automation.sh
```

### 2. 動作確認

```bash
./manage_automation.sh test
```

### 3. サービス状態確認

```bash
./manage_automation.sh status
```

## 🔧 管理コマンド

### サービス管理
```bash
# 自動実行開始
./manage_automation.sh start

# 自動実行停止
./manage_automation.sh stop

# 状態確認
./manage_automation.sh status
```

### 手動実行
```bash
# 新規推奨分析・購入
./run_daytrading.sh live

# 価格更新・売却判定
./run_daytrading.sh update

# レポート生成
./run_daytrading.sh report
```

### ログ管理
```bash
# 最新ログ表示
./manage_automation.sh logs

# ログファイル削除
./manage_automation.sh clean
```

## 📂 ファイル構成

```
automation/
├── com.daytrading.live.plist      # 16:00実行用launchd設定
├── com.daytrading.update.plist    # 09:30実行用launchd設定
├── run_daytrading.sh              # 実行スクリプト
├── setup_automation.sh            # セットアップスクリプト
├── manage_automation.sh           # 管理スクリプト
└── README.md                      # このファイル
```

## 📋 ログファイル

実行ログは `~/daytrading_logs/` に保存されます：

- `live.log` - 16:00実行の標準出力
- `live_error.log` - 16:00実行のエラー出力
- `update.log` - 09:30実行の標準出力
- `update_error.log` - 09:30実行のエラー出力

## ⚠️ 重要な注意事項

### 安全機能
- 土日・祝日の自動スキップ
- 資金不足時の自動停止
- タイムアウト機能（5分間）
- エラー時の詳細ログ記録

### 監視項目
- 定期的なログ確認
- パフォーマンス監視
- エラー発生時の対応

### 緊急停止
```bash
# 全サービス即座に停止
./manage_automation.sh stop
```

## 🔍 トラブルシューティング

### よくある問題

1. **サービスが開始されない**
   ```bash
   # plistファイルの構文確認
   plutil -lint ~/Library/LaunchAgents/com.daytrading.*.plist
   ```

2. **実行エラーが発生する**
   ```bash
   # エラーログ確認
   tail -f ~/daytrading_logs/live_error.log
   ```

3. **環境変数の問題**
   - スクリプト内のPATH設定を確認
   - 仮想環境のパスを確認

### デバッグ方法

```bash
# 手動でのテスト実行
cd /Users/y.okumura/private_workspace/automated_systematic_trading_and_risk_analysis
./automation/run_daytrading.sh report

# launchdサービスの確認
launchctl list | grep daytrading
```

## 📊 パフォーマンス監視

定期的に以下を確認してください：

```bash
# 取引パフォーマンス確認
cd /Users/y.okumura/private_workspace/automated_systematic_trading_and_risk_analysis
uv run python trading_simulator.py report

# システムリソース確認
./automation/manage_automation.sh status
```

## 🔄 システム更新時の手順

1. サービス停止
2. コード更新
3. テスト実行
4. サービス再開

```bash
./manage_automation.sh stop
# コード更新
./manage_automation.sh test
./manage_automation.sh start
```