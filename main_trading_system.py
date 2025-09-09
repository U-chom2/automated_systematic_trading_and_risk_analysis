#!/usr/bin/env python3
"""
自動取引システムメインスクリプト

3つのモジュールを統合して実行する
"""
import argparse
import sys
from datetime import datetime, date
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent))

from src.utils.logger_utils import create_dual_logger
from src.modules.screening import GrowthScreener
from src.modules.analysis import StockAnalyzer
from src.modules.recording import TradeRecorder


logger = create_dual_logger(__name__, console_output=True)


def run_screening(market_cap_limit: float = 100, ir_days: int = 30):
    """Module 1: 企業スクリーニングを実行"""
    logger.info("="*60)
    logger.info("Module 1: 企業スクリーニング")
    logger.info("="*60)
    
    screener = GrowthScreener(
        market_cap_limit_billion=market_cap_limit,
        ir_days_within=ir_days
    )
    
    result = screener.execute()
    
    logger.info(f"スクリーニング完了: {result.filtered_companies}社が条件をクリア")
    
    return result


def run_analysis():
    """Module 2: 個別銘柄分析・取引推奨を実行"""
    logger.info("="*60)
    logger.info("Module 2: 個別銘柄分析・取引推奨")
    logger.info("="*60)
    
    analyzer = StockAnalyzer(target_csv_path="data/target.csv")
    
    todos = analyzer.execute()
    
    logger.info(f"分析完了: {len(todos)}件の取引TODOを生成")
    
    return todos


def run_recording(target_date: date = None):
    """Module 3: 取引記録を実行"""
    logger.info("="*60)
    logger.info("Module 3: 取引記録管理")
    logger.info("="*60)
    
    recorder = TradeRecorder()
    
    settlement = recorder.execute_daily_settlement(target_date)
    
    logger.info(f"取引記録完了: {settlement.total_trades}件の取引を記録")
    
    return settlement


def run_morning_tasks():
    """朝の処理（9:00実行想定）"""
    logger.info("="*70)
    logger.info(f"朝の処理開始: {datetime.now()}")
    logger.info("="*70)
    
    try:
        # 1. 企業スクリーニング
        screening_result = run_screening()
        
        if screening_result.filtered_companies == 0:
            logger.warning("スクリーニング条件を満たす企業がありません")
            return
        
        # 2. 個別銘柄分析
        todos = run_analysis()
        
        if not todos:
            logger.info("取引推奨はありません")
            return
        
        logger.info(f"翌日の取引TODO: {len(todos)}件")
        
    except Exception as e:
        logger.error(f"朝の処理でエラーが発生: {e}")
        raise
    
    logger.info("朝の処理完了")


def run_evening_tasks(target_date: date = None):
    """夕方の処理（16:00実行想定）"""
    logger.info("="*70)
    logger.info(f"夕方の処理開始: {datetime.now()}")
    logger.info("="*70)
    
    try:
        # 3. 取引記録（終値での決済）
        settlement = run_recording(target_date)
        
        logger.info(settlement.get_summary())
        
    except Exception as e:
        logger.error(f"夕方の処理でエラーが発生: {e}")
        raise
    
    logger.info("夕方の処理完了")


def run_full_cycle():
    """フルサイクル実行（テスト用）"""
    logger.info("="*70)
    logger.info("フルサイクル実行開始")
    logger.info("="*70)
    
    # 朝の処理
    run_morning_tasks()
    
    # 夕方の処理（当日実行として）
    run_evening_tasks(date.today())
    
    logger.info("フルサイクル実行完了")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="自動取引システム")
    
    parser.add_argument(
        "--mode",
        choices=["screening", "analysis", "recording", "morning", "evening", "full"],
        default="full",
        help="実行モード"
    )
    
    parser.add_argument(
        "--market-cap",
        type=float,
        default=100,
        help="時価総額上限（億円）"
    )
    
    parser.add_argument(
        "--ir-days",
        type=int,
        default=30,
        help="IR公開日からの日数制限"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        help="処理対象日（YYYY-MM-DD形式）"
    )
    
    args = parser.parse_args()
    
    # 日付パース
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"無効な日付形式: {args.date}")
            sys.exit(1)
    
    logger.info("="*70)
    logger.info("自動取引システム起動")
    logger.info(f"モード: {args.mode}")
    logger.info("="*70)
    
    try:
        if args.mode == "screening":
            # スクリーニングのみ
            run_screening(args.market_cap, args.ir_days)
            
        elif args.mode == "analysis":
            # 分析のみ
            run_analysis()
            
        elif args.mode == "recording":
            # 記録のみ
            run_recording(target_date)
            
        elif args.mode == "morning":
            # 朝の処理
            run_morning_tasks()
            
        elif args.mode == "evening":
            # 夕方の処理
            run_evening_tasks(target_date)
            
        elif args.mode == "full":
            # フルサイクル
            run_full_cycle()
        
        logger.info("="*70)
        logger.info("処理正常終了")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        logger.error("処理を中断します")
        sys.exit(1)


if __name__ == "__main__":
    main()