"""
ロガーユーティリティ: 標準出力とファイル出力を同時に行うロガーの設定
"""
import sys
import logging
from typing import Optional
from pathlib import Path
from vibelogger import VibeLogger, VibeLoggerConfig


def create_dual_logger(project_name: str = "automated_trading", 
                      log_file: Optional[str] = None,
                      console_output: bool = True) -> VibeLogger:
    """
    ファイルと標準出力の両方にログを出力するVibeLoggerを作成
    
    Args:
        project_name: プロジェクト名
        log_file: ログファイルパス（Noneの場合はデフォルト）
        console_output: 標準出力への出力を有効にするか
    
    Returns:
        設定済みのVibeLoggerインスタンス
    """
    # VibeLoggerの設定
    config = VibeLoggerConfig(
        log_file=log_file,
        auto_save=True,
        create_dirs=True,
        log_level='INFO'
    )
    
    # VibeLoggerインスタンスを作成
    vibe_logger = VibeLogger(config=config)
    
    if console_output:
        # 標準のPythonロガーも設定して標準出力に出力
        standard_logger = logging.getLogger(project_name)
        standard_logger.setLevel(logging.INFO)
        
        # 既存のハンドラーをクリア
        standard_logger.handlers.clear()
        
        # コンソールハンドラーを追加
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # フォーマッターを設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        standard_logger.addHandler(console_handler)
        
        # VibeLoggerの各メソッドをラップして標準出力にも出力
        original_info = vibe_logger.info
        original_error = vibe_logger.error
        original_warning = vibe_logger.warning
        original_debug = vibe_logger.debug
        original_critical = vibe_logger.critical
        
        def wrapped_info(*args, **kwargs):
            result = original_info(*args, **kwargs)
            # メッセージを標準出力にも出力
            if args:
                if len(args) == 1:
                    standard_logger.info(args[0])
                elif len(args) >= 2:
                    standard_logger.info(f"{args[0]}: {args[1]}")
            return result
        
        def wrapped_error(*args, **kwargs):
            result = original_error(*args, **kwargs)
            if args:
                if len(args) == 1:
                    standard_logger.error(args[0])
                elif len(args) >= 2:
                    standard_logger.error(f"{args[0]}: {args[1]}")
            return result
        
        def wrapped_warning(*args, **kwargs):
            result = original_warning(*args, **kwargs)
            if args:
                if len(args) == 1:
                    standard_logger.warning(args[0])
                elif len(args) >= 2:
                    standard_logger.warning(f"{args[0]}: {args[1]}")
            return result
        
        def wrapped_debug(*args, **kwargs):
            result = original_debug(*args, **kwargs)
            if args:
                if len(args) == 1:
                    standard_logger.debug(args[0])
                elif len(args) >= 2:
                    standard_logger.debug(f"{args[0]}: {args[1]}")
            return result
        
        def wrapped_critical(*args, **kwargs):
            result = original_critical(*args, **kwargs)
            if args:
                if len(args) == 1:
                    standard_logger.critical(args[0]) 
                elif len(args) >= 2:
                    standard_logger.critical(f"{args[0]}: {args[1]}")
            return result
        
        # メソッドを置き換え
        vibe_logger.info = wrapped_info
        vibe_logger.error = wrapped_error
        vibe_logger.warning = wrapped_warning
        vibe_logger.debug = wrapped_debug
        vibe_logger.critical = wrapped_critical
    
    return vibe_logger