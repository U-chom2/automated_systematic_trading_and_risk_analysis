# 🚀 クイックリファレンス - 自動取引シミュレーション

## 基本コマンド一覧

### ▶️ 開始
```bash
# 自動実行を開始
launchctl load ~/Library/LaunchAgents/com.daytrading.update.plist
launchctl load ~/Library/LaunchAgents/com.daytrading.live.plist
```

### ⏸️ 停止
```bash
# 自動実行を停止
launchctl unload ~/Library/LaunchAgents/com.daytrading.update.plist
launchctl unload ~/Library/LaunchAgents/com.daytrading.live.plist
```

### 🔄 再起動
```bash
# 停止してから開始
launchctl unload ~/Library/LaunchAgents/com.daytrading.live.plist && \
launchctl load ~/Library/LaunchAgents/com.daytrading.live.plist
```

### 📊 手動実行
```bash
# ライブ取引を今すぐ実行
cd /Users/y.okumura/private_workspace/automated_systematic_trading_and_risk_analysis
./automation/run_daytrading.sh live

# 価格更新のみ
./automation/run_daytrading.sh update

# レポート生成
./automation/run_daytrading.sh report
```

### 🔍 状態確認
```bash
# 実行状態を確認
launchctl list | grep daytrading

# ログを確認（最新10行）
tail ~/Library/Logs/daytrading_live.log

# 現在のポジション確認
ls -lt simulation_data/positions_*.json | head -1 | xargs cat | jq '.active'
```

### 🆘 トラブル時
```bash
# エラーログ確認
grep ERROR ~/Library/Logs/daytrading_*.log | tail -10

# プロセス強制終了
pkill -f "python.*trading_simulator"

# 設定リセット
launchctl unload ~/Library/LaunchAgents/com.daytrading.*.plist
launchctl load ~/Library/LaunchAgents/com.daytrading.*.plist
```

## 実行スケジュール

| 時刻 | 処理 | 自動/手動 |
|------|------|-----------|
| 09:30 | 価格更新 | 自動（平日） |
| 16:00 | 取引実行 | 自動（平日） |
| 任意 | レポート | 手動 |

## PPOモデル情報

- **使用モデル**: `ppo_nikkei_model_20250907_090550.zip`
- **分析手法**: PPO強化学習
- **切り替え**: `use_ppo=True/False` in `InvestmentAnalyzer`

## 重要ファイル

- 📁 `simulation_data/` - 取引データ
- 📁 `automation/` - 自動化スクリプト
- 📁 `~/Library/Logs/` - ログファイル
- 📁 `train/models/rl/` - PPOモデル

---
💡 **ヒント**: `tab`キーで自動補完、`↑`キーで履歴検索が使えます