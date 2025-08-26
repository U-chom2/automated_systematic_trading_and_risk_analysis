"""
高度な取引システム用ログイング機能

全てのトリガー、分析結果、取引記録を構造化して記録し、
システムの稼働状況を詳細に監視する。
"""

import logging
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
import threading


class LogLevel(Enum):
    """ログレベル"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class TradingEvent:
    """取引イベントログ"""
    
    timestamp: datetime
    event_type: str  # "trigger", "analysis", "order", "position", "system"
    symbol: Optional[str] = None
    event_data: Optional[Dict[str, Any]] = None
    level: LogLevel = LogLevel.INFO
    message: str = ""
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        return data


class TradingLogger:
    """
    取引システム専用ロガー
    
    構造化されたログを出力し、イベント追跡とパフォーマンス監視を行う。
    """
    
    def __init__(self, 
                 log_dir: str = "_docs/logs",
                 max_log_files: int = 30,
                 max_file_size_mb: int = 50):
        """
        TradingLoggerを初期化
        
        Args:
            log_dir: ログディレクトリ
            max_log_files: 最大ログファイル数
            max_file_size_mb: ログファイルの最大サイズ（MB）
        """
        self.log_dir = Path(log_dir)
        self.max_log_files = max_log_files
        self.max_file_size_mb = max_file_size_mb
        
        # ログディレクトリを作成
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # ログファイルパス
        today = datetime.now().strftime("%Y-%m-%d")
        self.system_log_file = self.log_dir / f"system_{today}.log"
        self.trade_log_file = self.log_dir / f"trading_{today}.log"
        self.performance_log_file = self.log_dir / f"performance_{today}.log"
        
        # ログフォーマッター設定
        self._setup_loggers()
        
        # メトリクス保存用
        self.event_counts: Dict[str, int] = {}
        self.performance_metrics: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        self.log_system_event("SYSTEM", "TradingLogger initialized", LogLevel.INFO)
    
    def _setup_loggers(self) -> None:
        """ログ設定を初期化"""
        # システムログ
        self.system_logger = logging.getLogger("trading_system")
        self.system_logger.setLevel(logging.DEBUG)
        
        # ファイルハンドラー
        system_handler = logging.FileHandler(self.system_log_file, encoding='utf-8')
        system_handler.setLevel(logging.DEBUG)
        
        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        system_handler.setFormatter(formatter)
        
        # ハンドラーを追加（重複防止）
        if not self.system_logger.handlers:
            self.system_logger.addHandler(system_handler)
            
        # コンソールハンドラーも追加
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        if len([h for h in self.system_logger.handlers if isinstance(h, logging.StreamHandler)]) == 0:
            self.system_logger.addHandler(console_handler)
    
    def log_trigger_event(self, 
                         symbol: str,
                         trigger_type: str,
                         trigger_data: Dict[str, Any],
                         level: LogLevel = LogLevel.INFO) -> None:
        """
        トリガーイベントをログ
        
        Args:
            symbol: 銘柄コード
            trigger_type: トリガータイプ（IR, SNS, etc.）
            trigger_data: トリガーデータ
            level: ログレベル
        """
        event = TradingEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="trigger",
            symbol=symbol,
            event_data=trigger_data,
            level=level,
            message=f"Trigger detected: {trigger_type} for {symbol}"
        )
        
        self._write_trading_log(event)
        self._update_metrics("trigger", trigger_type)
    
    def log_analysis_result(self,
                           symbol: str,
                           analysis_type: str,
                           analysis_result: Dict[str, Any],
                           execution_time_ms: Optional[float] = None,
                           level: LogLevel = LogLevel.INFO) -> None:
        """
        分析結果をログ
        
        Args:
            symbol: 銘柄コード
            analysis_type: 分析タイプ（NLP, Technical, etc.）
            analysis_result: 分析結果
            execution_time_ms: 実行時間（ミリ秒）
            level: ログレベル
        """
        event = TradingEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="analysis",
            symbol=symbol,
            event_data=analysis_result,
            level=level,
            message=f"Analysis completed: {analysis_type} for {symbol}",
            execution_time_ms=execution_time_ms
        )
        
        self._write_trading_log(event)
        self._update_metrics("analysis", analysis_type)
    
    def log_order_event(self,
                       symbol: str,
                       order_type: str,
                       order_data: Dict[str, Any],
                       level: LogLevel = LogLevel.INFO) -> None:
        """
        注文イベントをログ
        
        Args:
            symbol: 銘柄コード
            order_type: 注文タイプ
            order_data: 注文データ
            level: ログレベル
        """
        event = TradingEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="order",
            symbol=symbol,
            event_data=order_data,
            level=level,
            message=f"Order {order_type}: {symbol}"
        )
        
        self._write_trading_log(event)
        self._update_metrics("order", order_type)
    
    def log_position_event(self,
                          symbol: str,
                          position_action: str,
                          position_data: Dict[str, Any],
                          level: LogLevel = LogLevel.INFO) -> None:
        """
        ポジションイベントをログ
        
        Args:
            symbol: 銘柄コード
            position_action: ポジション操作（open, close, update）
            position_data: ポジションデータ
            level: ログレベル
        """
        event = TradingEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="position",
            symbol=symbol,
            event_data=position_data,
            level=level,
            message=f"Position {position_action}: {symbol}"
        )
        
        self._write_trading_log(event)
        self._update_metrics("position", position_action)
    
    def log_system_event(self,
                        subsystem: str,
                        message: str,
                        level: LogLevel = LogLevel.INFO,
                        additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        システムイベントをログ
        
        Args:
            subsystem: サブシステム名
            message: ログメッセージ
            level: ログレベル
            additional_data: 追加データ
        """
        event = TradingEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="system",
            symbol=None,
            event_data=additional_data,
            level=level,
            message=f"[{subsystem}] {message}"
        )
        
        # システムログとトレーディングログ両方に出力
        log_level = getattr(logging, level.value)
        self.system_logger.log(log_level, event.message)
        self._write_trading_log(event)
        self._update_metrics("system", subsystem)
    
    def log_performance_metric(self,
                             metric_name: str,
                             metric_value: float,
                             context: Optional[Dict[str, Any]] = None) -> None:
        """
        パフォーマンスメトリクスをログ
        
        Args:
            metric_name: メトリクス名
            metric_value: メトリクス値
            context: コンテキスト情報
        """
        metric_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "context": context or {}
        }
        
        with self._lock:
            self.performance_metrics.append(metric_data)
            
        # ファイルにも書き込み
        with open(self.performance_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metric_data, ensure_ascii=False) + '\n')
    
    def measure_execution_time(self, operation_name: str):
        """
        実行時間測定用デコレータ
        
        Args:
            operation_name: 操作名
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    execution_time = (time.perf_counter() - start_time) * 1000
                    self.log_performance_metric(
                        f"{operation_name}_execution_time_ms",
                        execution_time,
                        {"function": func.__name__}
                    )
                    return result
                except Exception as e:
                    execution_time = (time.perf_counter() - start_time) * 1000
                    self.log_system_event(
                        "PERFORMANCE",
                        f"Error in {operation_name}: {str(e)}",
                        LogLevel.ERROR,
                        {"execution_time_ms": execution_time}
                    )
                    raise
            return wrapper
        return decorator
    
    def _write_trading_log(self, event: TradingEvent) -> None:
        """取引ログをファイルに書き込み"""
        with open(self.trade_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + '\n')
    
    def _update_metrics(self, category: str, subcategory: str) -> None:
        """メトリクス更新"""
        with self._lock:
            key = f"{category}_{subcategory}"
            self.event_counts[key] = self.event_counts.get(key, 0) + 1
    
    def get_event_summary(self) -> Dict[str, Any]:
        """イベント統計の取得"""
        with self._lock:
            return {
                "event_counts": self.event_counts.copy(),
                "total_events": sum(self.event_counts.values()),
                "performance_metrics_count": len(self.performance_metrics)
            }
    
    def get_recent_performance_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """最近のパフォーマンスメトリクスを取得"""
        with self._lock:
            return self.performance_metrics[-limit:]
    
    def cleanup_old_logs(self) -> None:
        """古いログファイルをクリーンアップ"""
        try:
            log_files = list(self.log_dir.glob("*.log"))
            if len(log_files) > self.max_log_files:
                # 古いファイルから削除
                log_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in log_files[:-self.max_log_files]:
                    old_file.unlink()
                    self.log_system_event("CLEANUP", f"Deleted old log file: {old_file.name}")
        except Exception as e:
            self.log_system_event("CLEANUP", f"Error cleaning up logs: {str(e)}", LogLevel.ERROR)
    
    def rotate_logs_if_needed(self) -> None:
        """ログローテーション"""
        for log_file in [self.system_log_file, self.trade_log_file, self.performance_log_file]:
            if log_file.exists() and log_file.stat().st_size > self.max_file_size_mb * 1024 * 1024:
                # ファイル名に時刻を追加してローテーション
                timestamp = datetime.now().strftime("%H%M%S")
                rotated_name = f"{log_file.stem}_{timestamp}{log_file.suffix}"
                rotated_path = log_file.parent / rotated_name
                log_file.rename(rotated_path)
                
                self.log_system_event("ROTATION", f"Log rotated: {log_file.name} -> {rotated_name}")
    
    def search_logs(self, 
                   symbol: Optional[str] = None,
                   event_type: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """
        ログ検索
        
        Args:
            symbol: 銘柄コード
            event_type: イベントタイプ
            start_time: 開始時間
            end_time: 終了時間
            limit: 結果上限
            
        Returns:
            マッチしたログエントリのリスト
        """
        results = []
        
        try:
            with open(self.trade_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(results) >= limit:
                        break
                    
                    try:
                        log_entry = json.loads(line.strip())
                        
                        # フィルタリング
                        if symbol and log_entry.get('symbol') != symbol:
                            continue
                        if event_type and log_entry.get('event_type') != event_type:
                            continue
                        
                        entry_time = datetime.fromisoformat(log_entry['timestamp'])
                        if start_time and entry_time < start_time:
                            continue
                        if end_time and entry_time > end_time:
                            continue
                        
                        results.append(log_entry)
                        
                    except json.JSONDecodeError:
                        continue
                        
        except FileNotFoundError:
            self.log_system_event("SEARCH", "Log file not found for search", LogLevel.WARNING)
        
        return results


# グローバルロガーインスタンス
_trading_logger: Optional[TradingLogger] = None

def get_trading_logger() -> TradingLogger:
    """グローバル取引ロガーを取得"""
    global _trading_logger
    if _trading_logger is None:
        _trading_logger = TradingLogger()
    return _trading_logger

def init_trading_logger(log_dir: str = "_docs/logs", 
                       max_log_files: int = 30,
                       max_file_size_mb: int = 50) -> TradingLogger:
    """取引ロガーを初期化"""
    global _trading_logger
    _trading_logger = TradingLogger(log_dir, max_log_files, max_file_size_mb)
    return _trading_logger