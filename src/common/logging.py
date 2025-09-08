"""ロギング設定モジュール"""
import logging
import logging.config
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
from functools import lru_cache

from .config import settings


class CustomJSONFormatter(logging.Formatter):
    """カスタムJSON形式フォーマッター"""
    
    def format(self, record: logging.LogRecord) -> str:
        """ログレコードをJSON形式にフォーマット"""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # エラーの場合はスタックトレースを追加
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # 追加のフィールドがある場合
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)
        
        return json.dumps(log_obj, ensure_ascii=False)


class StructuredLogger:
    """構造化ログラッパー"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def _log(self, level: int, message: str, **kwargs):
        """構造化ログを出力"""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """デバッグログ"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """情報ログ"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告ログ"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """エラーログ"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """クリティカルログ"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """例外ログ（スタックトレース付き）"""
        kwargs["exception"] = True
        self.logger.exception(message, extra={"extra_fields": kwargs})


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    use_json: bool = False,
) -> None:
    """ロギングを設定
    
    Args:
        log_level: ログレベル
        log_file: ログファイルパス
        use_json: JSON形式を使用するか
    """
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    
    # ロギング設定
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": settings.log_format,
            },
            "json": {
                "()": CustomJSONFormatter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json" if use_json else "default",
                "level": log_level,
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "src": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
    }
    
    # ファイルハンドラーを追加
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json" if use_json else "default",
            "filename": str(log_file),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "level": log_level,
        }
        config["root"]["handlers"].append("file")
        for logger_name in config["loggers"]:
            config["loggers"][logger_name]["handlers"].append("file")
    
    # ロギング設定を適用
    logging.config.dictConfig(config)


@lru_cache(maxsize=None)
def get_logger(name: str, structured: bool = False) -> StructuredLogger | logging.Logger:
    """ロガーを取得
    
    Args:
        name: ロガー名
        structured: 構造化ログを使用するか
    
    Returns:
        ロガー
    """
    logger = logging.getLogger(name)
    
    if structured:
        return StructuredLogger(logger)
    
    return logger


class LogContext:
    """ログコンテキストマネージャー"""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(
            f"Starting {self.operation}",
            extra={"extra_fields": {**self.context, "operation": self.operation}}
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(
                f"Failed {self.operation}",
                extra={
                    "extra_fields": {
                        **self.context,
                        "operation": self.operation,
                        "duration": duration,
                        "error": str(exc_val),
                        "error_type": exc_type.__name__,
                    }
                }
            )
        else:
            self.logger.info(
                f"Completed {self.operation}",
                extra={
                    "extra_fields": {
                        **self.context,
                        "operation": self.operation,
                        "duration": duration,
                    }
                }
            )
        
        return False  # 例外を再発生させる


# デフォルトのロギング設定を適用
setup_logging()