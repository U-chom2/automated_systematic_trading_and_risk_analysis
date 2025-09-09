#!/usr/bin/env python3
"""
自動取引システムスケジューラー

定期実行のスケジュール管理
"""
import schedule
import time
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent))

from vibelogger import create_file_logger
from main_trading_system import run_morning_tasks, run_evening_tasks


logger = create_file_logger(__name__)


def is_weekday():
    """平日かどうかをチェック"""
    # 月曜日=0, 日曜日=6
    return datetime.now().weekday() < 5


def is_market_day():
    """市場営業日かどうかをチェック"""
    # TODO: 祝日カレンダーと連携
    # 現在は平日のみチェック
    return is_weekday()


def morning_job():
    """朝のジョブ（9:00実行）"""
    if not is_market_day():
        logger.info("本日は市場休業日のため、朝の処理をスキップします")
        return
    
    logger.info("朝のジョブを開始します")
    
    try:
        run_morning_tasks()
        logger.info("朝のジョブが正常に完了しました")
        
    except Exception as e:
        logger.error(f"朝のジョブでエラーが発生: {e}")
        # エラー通知（Slack、メールなど）を送信
        # notify_error("Morning job failed", str(e))


def evening_job():
    """夕方のジョブ（16:00実行）"""
    if not is_market_day():
        logger.info("本日は市場休業日のため、夕方の処理をスキップします")
        return
    
    logger.info("夕方のジョブを開始します")
    
    try:
        run_evening_tasks()
        logger.info("夕方のジョブが正常に完了しました")
        
    except Exception as e:
        logger.error(f"夕方のジョブでエラーが発生: {e}")
        # エラー通知（Slack、メールなど）を送信
        # notify_error("Evening job failed", str(e))


def setup_schedule():
    """スケジュール設定"""
    logger.info("スケジュール設定中...")
    
    # 朝の処理（9:00）
    schedule.every().day.at("09:00").do(morning_job)
    
    # 夕方の処理（16:00）
    schedule.every().day.at("16:00").do(evening_job)
    
    logger.info("スケジュール設定完了:")
    logger.info("  - 朝の処理: 毎日 09:00")
    logger.info("  - 夕方の処理: 毎日 16:00")


def run_scheduler():
    """スケジューラー実行"""
    logger.info("="*70)
    logger.info("自動取引システムスケジューラー起動")
    logger.info(f"開始時刻: {datetime.now()}")
    logger.info("="*70)
    
    setup_schedule()
    
    logger.info("スケジューラーが稼働中です... (Ctrl+C で終了)")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1分ごとにチェック
            
    except KeyboardInterrupt:
        logger.info("\nスケジューラーを停止します")
        
    except Exception as e:
        logger.error(f"スケジューラーエラー: {e}")
        raise
    
    logger.info("スケジューラー終了")


if __name__ == "__main__":
    run_scheduler()