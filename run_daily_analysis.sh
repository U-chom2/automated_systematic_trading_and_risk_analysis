#!/bin/bash
#
# 日次分析実行スクリプト
# 毎日16:15に実行することを推奨
#

echo "=================================================="
echo "🚀 AI駆動型株式分析システム"
echo "   Daily Analysis Runner v1.0"
echo "=================================================="
echo ""

# 実行日時を記録
DATE=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "📅 実行日時: $(date '+%Y年%m月%d日 %H:%M:%S')"
echo ""

# 分析実行
echo "🔍 株式分析を開始します..."
uv run python analyze_today.py 2>&1 | tee "$LOG_DIR/analysis_${DATE}.log"

echo ""
echo "=================================================="
echo "✅ 分析完了"
echo "   ログファイル: $LOG_DIR/analysis_${DATE}.log"
echo "=================================================="

# 結果をSlackやメールに送信する場合はここに追加
# curl -X POST -H 'Content-type: application/json' \
#   --data "{\"text\":\"株式分析完了: $(date)\"}" \
#   YOUR_SLACK_WEBHOOK_URL