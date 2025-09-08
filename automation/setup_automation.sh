#!/bin/bash

# デイトレードシステム自動実行セットアップスクリプト

set -e

echo "🚀 デイトレードシステム自動実行セットアップ"
echo "=============================================="

# 設定
PROJECT_DIR="/Users/y.okumura/private_workspace/automated_systematic_trading_and_risk_analysis"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

# LaunchAgentsディレクトリ作成
mkdir -p "$LAUNCH_AGENTS_DIR"

# ログディレクトリ作成
mkdir -p ~/daytrading_logs

echo "📁 ディレクトリセットアップ完了"

# plistファイルをコピー
echo "📋 launchd設定ファイルをコピー中..."

cp "$PROJECT_DIR/automation/com.daytrading.live.plist" "$LAUNCH_AGENTS_DIR/"
cp "$PROJECT_DIR/automation/com.daytrading.update.plist" "$LAUNCH_AGENTS_DIR/"

echo "✅ 設定ファイルコピー完了"

# サービス登録
echo "🔧 launchctlサービス登録中..."

# 既存のサービスがあれば停止・削除
launchctl unload "$LAUNCH_AGENTS_DIR/com.daytrading.live.plist" 2>/dev/null || true
launchctl unload "$LAUNCH_AGENTS_DIR/com.daytrading.update.plist" 2>/dev/null || true

# 新しいサービスを登録
launchctl load "$LAUNCH_AGENTS_DIR/com.daytrading.live.plist"
launchctl load "$LAUNCH_AGENTS_DIR/com.daytrading.update.plist"

echo "✅ サービス登録完了"

# 設定確認
echo "📊 サービス状態確認:"
echo "Live Service (16:00):"
launchctl list | grep com.daytrading.live || echo "  サービスが見つかりません"

echo "Update Service (09:30):"
launchctl list | grep com.daytrading.update || echo "  サービスが見つかりません"

echo ""
echo "=============================================="
echo "🎉 セットアップ完了！"
echo "=============================================="
echo ""
echo "📅 自動実行スケジュール:"
echo "  • 毎日 09:30 - 価格更新・売却判定"
echo "  • 毎日 16:00 - 新規推奨銘柄分析・購入"
echo "  • 平日のみ実行（土日・祝日は自動スキップ）"
echo ""
echo "📋 手動操作コマンド:"
echo "  • 手動実行: $PROJECT_DIR/automation/run_daytrading.sh [live|update|report]"
echo "  • レポート確認: cd $PROJECT_DIR && uv run python trading_simulator.py report"
echo "  • サービス停止: launchctl unload ~/Library/LaunchAgents/com.daytrading.*.plist"
echo "  • サービス開始: launchctl load ~/Library/LaunchAgents/com.daytrading.*.plist"
echo ""
echo "📂 ログ確認:"
echo "  • 実行ログ: ~/daytrading_logs/"
echo "  • エラーログ: ~/daytrading_logs/*_error.log"
echo ""
echo "⚠️  重要な注意事項:"
echo "  • 初回実行前にシステムが正常動作することを確認してください"
echo "  • 定期的にログとパフォーマンスを確認してください"
echo "  • 予期しない動作があれば直ちに停止してください"
echo ""
echo "🧪 テスト実行:"
echo "  $PROJECT_DIR/automation/run_daytrading.sh report"
echo "=============================================="