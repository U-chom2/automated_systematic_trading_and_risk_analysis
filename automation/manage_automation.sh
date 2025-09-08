#!/bin/bash

# デイトレードシステム自動実行管理スクリプト
# Usage: ./manage_automation.sh [start|stop|status|test|logs]

set -e

LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PROJECT_DIR="/Users/y.okumura/private_workspace/automated_systematic_trading_and_risk_analysis"
LOG_DIR="~/daytrading_logs"

show_usage() {
    echo "🔧 デイトレードシステム管理ツール"
    echo "=============================================="
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start   - 自動実行サービスを開始"
    echo "  stop    - 自動実行サービスを停止"
    echo "  status  - サービス状態を確認"
    echo "  test    - テスト実行（レポート生成）"
    echo "  logs    - 最新ログを表示"
    echo "  clean   - ログファイルを削除"
    echo "=============================================="
}

start_services() {
    echo "🚀 自動実行サービスを開始中..."
    
    launchctl load "$LAUNCH_AGENTS_DIR/com.daytrading.live.plist" 2>/dev/null || echo "Live service already loaded"
    launchctl load "$LAUNCH_AGENTS_DIR/com.daytrading.update.plist" 2>/dev/null || echo "Update service already loaded"
    
    echo "✅ サービス開始完了"
    show_status
}

stop_services() {
    echo "🛑 自動実行サービスを停止中..."
    
    launchctl unload "$LAUNCH_AGENTS_DIR/com.daytrading.live.plist" 2>/dev/null || echo "Live service already unloaded"
    launchctl unload "$LAUNCH_AGENTS_DIR/com.daytrading.update.plist" 2>/dev/null || echo "Update service already unloaded"
    
    echo "✅ サービス停止完了"
}

show_status() {
    echo "📊 サービス状態:"
    echo "=============================================="
    
    echo "Live Service (16:00 新規推奨):"
    if launchctl list | grep -q com.daytrading.live; then
        echo "  ✅ 実行中"
        launchctl list com.daytrading.live | head -3
    else
        echo "  ❌ 停止中"
    fi
    
    echo ""
    echo "Update Service (09:30 価格更新):"
    if launchctl list | grep -q com.daytrading.update; then
        echo "  ✅ 実行中"
        launchctl list com.daytrading.update | head -3
    else
        echo "  ❌ 停止中"
    fi
    
    echo ""
    echo "📂 ログファイル:"
    ls -la ~/daytrading_logs/ 2>/dev/null || echo "  ログファイルなし"
    
    echo "=============================================="
}

test_execution() {
    echo "🧪 テスト実行中..."
    echo "=============================================="
    
    cd "$PROJECT_DIR"
    
    echo "📋 レポート生成テスト:"
    if "$PROJECT_DIR/automation/run_daytrading.sh" report; then
        echo "✅ レポート生成成功"
    else
        echo "❌ レポート生成失敗"
        return 1
    fi
    
    echo ""
    echo "🔍 システム状態確認:"
    echo "  Python: $(cd "$PROJECT_DIR" && .venv/bin/python --version)"
    echo "  UV: $(which uv && uv --version)"
    echo "  Working Directory: $(pwd)"
    
    echo "=============================================="
    echo "✅ テスト実行完了"
}

show_logs() {
    echo "📋 最新ログ表示"
    echo "=============================================="
    
    local log_files=(
        "~/daytrading_logs/live.log"
        "~/daytrading_logs/update.log"
        "~/daytrading_logs/live_error.log"
        "~/daytrading_logs/update_error.log"
    )
    
    for log_file in "${log_files[@]}"; do
        expanded_path=$(eval echo "$log_file")
        if [ -f "$expanded_path" ]; then
            echo ""
            echo "📄 $log_file (最新10行):"
            echo "--------------------------------------------"
            tail -10 "$expanded_path" || echo "ログ読み取り失敗"
        fi
    done
    
    echo "=============================================="
}

clean_logs() {
    echo "🧹 ログファイル削除中..."
    
    rm -f ~/daytrading_logs/*.log 2>/dev/null || true
    
    echo "✅ ログファイル削除完了"
}

# メイン処理
case "${1:-help}" in
    "start")
        start_services
        ;;
    "stop")
        stop_services
        ;;
    "status")
        show_status
        ;;
    "test")
        test_execution
        ;;
    "logs")
        show_logs
        ;;
    "clean")
        clean_logs
        ;;
    "help"|*)
        show_usage
        ;;
esac