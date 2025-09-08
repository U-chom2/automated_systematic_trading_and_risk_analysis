#!/bin/bash

# デイトレードシステム自動実行スクリプト
# Usage: ./run_daytrading.sh [live|update|report]

set -e  # エラー時に停止

# 設定
PROJECT_DIR="/Users/y.okumura/private_workspace/automated_systematic_trading_and_risk_analysis"
LOG_DIR="/Users/y.okumura/daytrading_logs"
VENV_DIR="$PROJECT_DIR/.venv"

# ログディレクトリ作成
mkdir -p "$LOG_DIR"

# 実行モード（デフォルト：live）
MODE="${1:-live}"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "============================================="
echo "🚀 デイトレードシステム自動実行"
echo "============================================="
echo "実行時刻: $TIMESTAMP"
echo "実行モード: $MODE"
echo "プロジェクト: $PROJECT_DIR"

# プロジェクトディレクトリに移動
cd "$PROJECT_DIR"

# 仮想環境の確認
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "❌ 仮想環境が見つかりません: $VENV_DIR"
    exit 1
fi

# 祝日・休日チェック関数
check_market_open() {
    local today=$(date "+%Y-%m-%d")
    local weekday=$(date "+%u")  # 1=月曜, 7=日曜
    
    # 土日チェック
    if [ "$weekday" -gt 5 ]; then
        echo "📅 週末のため市場は休場です"
        return 1
    fi
    
    # 基本的な日本の祝日チェック（簡易版）
    local month_day=$(date "+%m-%d")
    case $month_day in
        "01-01"|"01-02"|"01-03")  # 正月
            echo "📅 正月休みのため市場は休場です"
            return 1
            ;;
        "12-29"|"12-30"|"12-31")  # 年末
            echo "📅 年末休みのため市場は休場です" 
            return 1
            ;;
    esac
    
    return 0
}

# 資金チェック関数
check_funds() {
    echo "💰 資金状況チェック中..."
    
    # レポート生成で現在の資金を確認
    local report_output
    if report_output=$("$VENV_DIR/bin/python" trading_simulator.py report 2>&1); then
        # 現金残高を抽出（簡易的な方法）
        if echo "$report_output" | grep -q "現金.*¥0"; then
            echo "❌ 資金不足：現金残高が0円です"
            return 1
        fi
        echo "✅ 資金チェック完了"
    else
        echo "⚠️ 資金チェックをスキップ（初回実行の可能性）"
    fi
    
    return 0
}

# エラーハンドリング関数
handle_error() {
    local exit_code=$1
    echo "❌ 実行エラーが発生しました (終了コード: $exit_code)"
    echo "エラー時刻: $(date "+%Y-%m-%d %H:%M:%S")"
    
    # エラーログに記録
    {
        echo "============================================="
        echo "ERROR LOG - $(date "+%Y-%m-%d %H:%M:%S")"
        echo "============================================="
        echo "Mode: $MODE"
        echo "Exit Code: $exit_code"
        echo "Working Directory: $(pwd)"
        echo "Python Version: $($VENV_DIR/bin/python --version 2>&1 || echo 'Python not found')"
        echo "UV Version: $(which uv && uv --version 2>&1 || echo 'UV not found')"
        echo "============================================="
    } >> "$LOG_DIR/error_$(date "+%Y%m%d").log"
    
    exit $exit_code
}

# メイン実行部分
main() {
    case $MODE in
        "live")
            echo "🔴 新規推奨銘柄分析・購入モード"
            
            # 市場開場チェック
            if ! check_market_open; then
                echo "✅ 市場休場のため処理をスキップしました"
                return 0
            fi
            
            # 資金チェック
            if ! check_funds; then
                echo "❌ 資金不足のため処理を停止しました"
                return 1
            fi
            
            echo "📊 推奨銘柄分析を開始..."
            "$VENV_DIR/bin/python" trading_simulator.py live
            ;;
            
        "update")
            echo "📈 価格更新・売却判定モード"
            
            # 市場開場チェック
            if ! check_market_open; then
                echo "✅ 市場休場のため処理をスキップしました"
                return 0
            fi
            
            echo "📊 価格更新を開始..."
            "$VENV_DIR/bin/python" trading_simulator.py update
            ;;
            
        "report")
            echo "📋 レポート生成モード"
            "$VENV_DIR/bin/python" trading_simulator.py report
            ;;
            
        *)
            echo "❌ 不明なモード: $MODE"
            echo "利用可能なモード: live, update, report"
            return 1
            ;;
    esac
    
    echo "✅ $MODE モードの実行が完了しました"
    return 0
}

# トラップでエラーをキャッチ
trap 'handle_error $?' ERR

# メイン実行
main "$@"

echo "============================================="
echo "🎉 デイトレードシステム実行完了"
echo "終了時刻: $(date "+%Y-%m-%d %H:%M:%S")"
echo "============================================="