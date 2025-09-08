# 自動取引シミュレーション 運用マニュアル

## 📊 システム概要

本システムは、PPO強化学習モデルを使用した自動株式取引シミュレーションを実行します。
macOSのlaunchdを使用して、市場時間に合わせた自動実行が設定されています。

---

## 🚀 自動実行の開始方法

### 1. 自動化設定の有効化

```bash
# 朝9:30の価格更新を有効化
launchctl load ~/Library/LaunchAgents/com.daytrading.update.plist

# 夕方16:00のライブ取引を有効化
launchctl load ~/Library/LaunchAgents/com.daytrading.live.plist
```

### 2. 手動実行（テスト用）

```bash
# 価格更新のみ実行
cd /Users/y.okumura/private_workspace/automated_systematic_trading_and_risk_analysis
./automation/run_daytrading.sh update

# ライブ取引シミュレーション実行
./automation/run_daytrading.sh live

# レポート生成
./automation/run_daytrading.sh report
```

---

## 🛑 自動実行の停止方法

### 1. 一時停止（設定は残す）

```bash
# 朝の更新を停止
launchctl unload ~/Library/LaunchAgents/com.daytrading.update.plist

# 夕方の取引を停止
launchctl unload ~/Library/LaunchAgents/com.daytrading.live.plist
```

### 2. 完全停止（設定も削除）

```bash
# 自動実行を停止
launchctl unload ~/Library/LaunchAgents/com.daytrading.update.plist
launchctl unload ~/Library/LaunchAgents/com.daytrading.live.plist

# 設定ファイルを削除
rm ~/Library/LaunchAgents/com.daytrading.update.plist
rm ~/Library/LaunchAgents/com.daytrading.live.plist
```

---

## 📅 実行スケジュール

| 時刻 | 処理内容 | ファイル |
|------|----------|----------|
| 09:30 | 価格更新・ポジション管理 | com.daytrading.update.plist |
| 16:00 | ライブ取引シミュレーション | com.daytrading.live.plist |

**注意**: 土日祝日は自動的にスキップされます。

---

## 📂 重要なファイル・ディレクトリ

### 設定ファイル
- `~/Library/LaunchAgents/com.daytrading.update.plist` - 朝の更新設定
- `~/Library/LaunchAgents/com.daytrading.live.plist` - 夕方の取引設定

### 実行スクリプト
- `automation/run_daytrading.sh` - メイン実行スクリプト
- `automation/manage_schedule.sh` - スケジュール管理スクリプト

### データ保存先
- `simulation_data/` - シミュレーション結果
  - `positions_*.json` - ポジション情報
  - `trades_*.json` - 取引履歴
  - `snapshots_*.json` - スナップショット

### ログファイル
- `~/Library/Logs/daytrading_update.log` - 更新処理ログ
- `~/Library/Logs/daytrading_live.log` - 取引実行ログ

---

## 🔍 状態確認方法

### 1. 自動実行の状態確認

```bash
# 登録されているジョブを確認
launchctl list | grep daytrading

# 実行状態の詳細確認
launchctl print gui/$(id -u)/com.daytrading.update
launchctl print gui/$(id -u)/com.daytrading.live
```

### 2. ログ確認

```bash
# 最新のログを表示（更新処理）
tail -f ~/Library/Logs/daytrading_update.log

# 最新のログを表示（取引実行）
tail -f ~/Library/Logs/daytrading_live.log

# エラーのみ確認
grep ERROR ~/Library/Logs/daytrading_*.log
```

### 3. ポートフォリオ確認

```bash
# 最新のポジション確認
cat simulation_data/positions_*.json | jq '.active'

# 本日の取引確認
cat simulation_data/trades_*.json | jq '.'
```

---

## 🛠️ トラブルシューティング

### 問題: 自動実行されない

1. launchdサービスの状態確認
```bash
launchctl list | grep daytrading
```

2. 権限の確認
```bash
ls -la ~/Library/LaunchAgents/*.plist
# 権限が644であることを確認
```

3. 手動実行でエラー確認
```bash
./automation/run_daytrading.sh live
```

### 問題: PPOモデルが読み込めない

1. モデルファイルの存在確認
```bash
ls -la train/models/rl/ppo_nikkei_model_*.zip
```

2. Python環境の確認
```bash
uv run python -c "from ppo_scoring_adapter import create_ppo_adapter; print('OK')"
```

### 問題: 取引が実行されない

1. 市場営業日の確認
```bash
# 今日が営業日か確認（手動実行してメッセージを確認）
./automation/run_daytrading.sh live
```

2. ポートフォリオの資金確認
```bash
# 現在の残高確認
cat simulation_data/snapshots_*.json | jq '.[-1].cash'
```

---

## 🔧 設定変更

### 実行時刻の変更

1. plistファイルを編集
```bash
# エディタで開く
nano ~/Library/LaunchAgents/com.daytrading.live.plist
```

2. `<key>Hour</key>`と`<key>Minute</key>`の値を変更

3. 設定を再読み込み
```bash
launchctl unload ~/Library/LaunchAgents/com.daytrading.live.plist
launchctl load ~/Library/LaunchAgents/com.daytrading.live.plist
```

### 投資パラメータの変更

`trading_simulator.py`の`SimulationConfig`を編集:

```python
@dataclass
class SimulationConfig:
    initial_capital: float = 100000.0  # 初期資金
    max_positions: int = 5  # 最大同時保有銘柄数
    max_investment_per_stock: float = 30000.0  # 1銘柄あたり最大投資額
    target_profit_pct: float = 2.0  # 目標利益率
    stop_loss_pct: float = -1.5  # 損切りライン
```

---

## 📊 PPOモデル切り替え

### 従来のテクニカル分析に戻す

```python
# investment_analyzer.py の InvestmentAnalyzer 初期化時
analyzer = InvestmentAnalyzer(config, use_ppo=False)  # PPOを無効化
```

### 新しいPPOモデルを使用

1. 新しいモデルを配置
```bash
cp new_model.zip train/models/rl/
```

2. システムは自動的に最新のモデルを検出・使用

---

## 📈 パフォーマンス確認

### 日次レポート生成

```bash
# 本日のパフォーマンスレポート
./automation/run_daytrading.sh report

# 過去のデータ分析
python analyze_performance.py --start 2025-09-01 --end 2025-09-07
```

### 主要指標

- **総資産**: ポートフォリオ全体の価値
- **実現損益**: 確定した損益
- **未実現損益**: 含み損益
- **勝率**: 利益が出た取引の割合
- **平均保有期間**: ポジションの平均保有日数

---

## ⚠️ 注意事項

1. **本番環境での使用**: このシステムはシミュレーション用です。実際の取引には使用しないでください。

2. **データバックアップ**: `simulation_data/`ディレクトリは定期的にバックアップしてください。

3. **システム負荷**: 取引時間中は他の重い処理を避けてください。

4. **ネットワーク**: 安定したインターネット接続が必要です。

5. **PPOモデル**: モデルは定期的に再学習することを推奨します。

---

## 📞 サポート

問題が解決しない場合は、以下の情報を準備してください：

1. エラーログ（`~/Library/Logs/daytrading_*.log`）
2. 実行コマンドと出力
3. システム情報（macOSバージョン、Pythonバージョン）
4. `simulation_data/`の最新ファイル

---

最終更新: 2025-09-07
バージョン: 1.0.0 (PPO統合版)