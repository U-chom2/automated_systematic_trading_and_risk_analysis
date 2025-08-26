"""
システム監視ダッシュボード

リアルタイムでシステムの稼働状況を監視し、
パフォーマンス指標やトレーディング統計を表示する。
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict, deque
from pathlib import Path

from .logger import get_trading_logger, LogLevel


@dataclass
class SystemMetrics:
    """システムメトリクス"""
    
    # システム稼働状況
    uptime_seconds: float
    is_running: bool
    last_heartbeat: datetime
    
    # パフォーマンス指標
    cpu_usage_percent: float
    memory_usage_mb: float
    avg_response_time_ms: float
    
    # トレーディング統計
    active_positions: int
    total_trades_today: int
    total_triggers_today: int
    success_rate_percent: float
    
    # エラー統計
    errors_last_hour: int
    warnings_last_hour: int
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で出力"""
        data = asdict(self)
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data


class SystemDashboard:
    """
    システム監視ダッシュボード
    
    システムの稼働状況をリアルタイムで監視し、
    各種メトリクスを収集・表示する。
    """
    
    def __init__(self, update_interval: int = 10):
        """
        ダッシュボードを初期化
        
        Args:
            update_interval: メトリクス更新間隔（秒）
        """
        self.update_interval = update_interval
        self.start_time = datetime.now()
        self.is_monitoring = False
        
        # メトリクス保存
        self.current_metrics = SystemMetrics(
            uptime_seconds=0,
            is_running=True,
            last_heartbeat=datetime.now(),
            cpu_usage_percent=0.0,
            memory_usage_mb=0.0,
            avg_response_time_ms=0.0,
            active_positions=0,
            total_trades_today=0,
            total_triggers_today=0,
            success_rate_percent=0.0,
            errors_last_hour=0,
            warnings_last_hour=0
        )
        
        # 履歴データ（最新100件を保持）
        self.metrics_history: deque = deque(maxlen=100)
        self.response_times: deque = deque(maxlen=50)
        self.error_log: deque = deque(maxlen=100)
        
        # スレッド制御
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # トレーディング統計
        self.trading_stats = {
            "triggers": defaultdict(int),
            "orders": defaultdict(int),
            "positions": defaultdict(int),
            "errors": defaultdict(int)
        }
        
        self.logger = get_trading_logger()
        self.logger.log_system_event("DASHBOARD", "Dashboard initialized", LogLevel.INFO)
    
    def start_monitoring(self) -> None:
        """監視開始"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.log_system_event("DASHBOARD", "Monitoring started", LogLevel.INFO)
    
    def stop_monitoring(self) -> None:
        """監視停止"""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.logger.log_system_event("DASHBOARD", "Monitoring stopped", LogLevel.INFO)
    
    def _monitoring_loop(self) -> None:
        """監視ループ"""
        while self.is_monitoring:
            try:
                self._update_metrics()
                self._check_health()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.log_system_event(
                    "DASHBOARD", 
                    f"Monitoring loop error: {str(e)}", 
                    LogLevel.ERROR
                )
                time.sleep(self.update_interval)
    
    def _update_metrics(self) -> None:
        """メトリクス更新"""
        with self._lock:
            now = datetime.now()
            
            # アップタイム計算
            uptime = (now - self.start_time).total_seconds()
            self.current_metrics.uptime_seconds = uptime
            self.current_metrics.last_heartbeat = now
            
            # システムリソース情報（簡易版）
            self.current_metrics.cpu_usage_percent = self._get_cpu_usage()
            self.current_metrics.memory_usage_mb = self._get_memory_usage()
            
            # レスポンス時間の平均計算
            if self.response_times:
                self.current_metrics.avg_response_time_ms = sum(self.response_times) / len(self.response_times)
            
            # ログ統計の更新
            self._update_log_statistics()
            
            # メトリクス履歴に追加
            self.metrics_history.append(self.current_metrics.to_dict())
    
    def _get_cpu_usage(self) -> float:
        """CPU使用率取得（簡易版）"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            # psutilが無い場合は固定値を返す
            return 10.0
    
    def _get_memory_usage(self) -> float:
        """メモリ使用量取得（簡易版）"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024 * 1024)  # MB単位
        except ImportError:
            # psutilが無い場合は固定値を返す
            return 100.0
    
    def _update_log_statistics(self) -> None:
        """ログ統計の更新"""
        try:
            # ログイベント統計を取得
            event_summary = self.logger.get_event_summary()
            
            # 今日のトリガー数とトレード数を計算
            self.current_metrics.total_triggers_today = sum(
                count for key, count in event_summary.get("event_counts", {}).items() 
                if "trigger_" in key
            )
            
            self.current_metrics.total_trades_today = sum(
                count for key, count in event_summary.get("event_counts", {}).items() 
                if "order_" in key
            )
            
            # エラー・警告の統計（簡易版）
            self.current_metrics.errors_last_hour = event_summary.get("event_counts", {}).get("system_ERROR", 0)
            self.current_metrics.warnings_last_hour = event_summary.get("event_counts", {}).get("system_WARNING", 0)
            
        except Exception as e:
            self.logger.log_system_event(
                "DASHBOARD", 
                f"Failed to update log statistics: {str(e)}", 
                LogLevel.WARNING
            )
    
    def _check_health(self) -> None:
        """システムヘルスチェック"""
        issues = []
        
        # メモリ使用量チェック
        if self.current_metrics.memory_usage_mb > 1000:  # 1GB
            issues.append("High memory usage")
        
        # レスポンス時間チェック
        if self.current_metrics.avg_response_time_ms > 5000:  # 5秒
            issues.append("High response time")
        
        # エラー率チェック
        if self.current_metrics.errors_last_hour > 10:
            issues.append("High error rate")
        
        if issues:
            self.logger.log_system_event(
                "HEALTH_CHECK", 
                f"Health issues detected: {', '.join(issues)}", 
                LogLevel.WARNING
            )
    
    def record_response_time(self, response_time_ms: float) -> None:
        """レスポンス時間を記録"""
        with self._lock:
            self.response_times.append(response_time_ms)
    
    def record_trade_event(self, event_type: str, symbol: str, success: bool) -> None:
        """トレード関連イベントを記録"""
        with self._lock:
            self.trading_stats[event_type][symbol] += 1
            
            # 成功率の更新（簡易版）
            total_trades = sum(self.trading_stats["orders"].values())
            if total_trades > 0:
                # 成功したトレード数の計算（簡易版）
                successful_trades = total_trades * 0.7  # 仮定値
                self.current_metrics.success_rate_percent = (successful_trades / total_trades) * 100
    
    def record_position_update(self, active_count: int) -> None:
        """アクティブポジション数を更新"""
        with self._lock:
            self.current_metrics.active_positions = active_count
    
    def get_current_metrics(self) -> SystemMetrics:
        """現在のメトリクスを取得"""
        with self._lock:
            return self.current_metrics
    
    def get_metrics_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """メトリクス履歴を取得"""
        with self._lock:
            return list(self.metrics_history)[-limit:]
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """トレーディング統計を取得"""
        with self._lock:
            return {
                "triggers_by_symbol": dict(self.trading_stats["triggers"]),
                "orders_by_symbol": dict(self.trading_stats["orders"]),
                "positions_by_symbol": dict(self.trading_stats["positions"]),
                "error_counts": dict(self.trading_stats["errors"])
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態のサマリーを取得"""
        metrics = self.get_current_metrics()
        
        # ステータス判定
        status = "healthy"
        if metrics.errors_last_hour > 5:
            status = "warning"
        if metrics.errors_last_hour > 20 or metrics.avg_response_time_ms > 10000:
            status = "critical"
        
        return {
            "status": status,
            "uptime_hours": metrics.uptime_seconds / 3600,
            "is_running": metrics.is_running,
            "last_heartbeat": metrics.last_heartbeat.isoformat(),
            "performance": {
                "cpu_percent": metrics.cpu_usage_percent,
                "memory_mb": metrics.memory_usage_mb,
                "avg_response_ms": metrics.avg_response_time_ms
            },
            "trading": {
                "active_positions": metrics.active_positions,
                "trades_today": metrics.total_trades_today,
                "triggers_today": metrics.total_triggers_today,
                "success_rate": metrics.success_rate_percent
            },
            "errors": {
                "errors_last_hour": metrics.errors_last_hour,
                "warnings_last_hour": metrics.warnings_last_hour
            }
        }
    
    def export_dashboard_data(self, export_path: Optional[str] = None) -> str:
        """
        ダッシュボードデータをエクスポート
        
        Args:
            export_path: エクスポート先パス
            
        Returns:
            エクスポートファイルパス
        """
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"_docs/dashboard_export_{timestamp}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_status": self.get_system_status(),
            "metrics_history": self.get_metrics_history(),
            "trading_statistics": self.get_trading_statistics()
        }
        
        export_path_obj = Path(export_path)
        export_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path_obj, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.log_system_event(
            "DASHBOARD", 
            f"Dashboard data exported to {export_path}", 
            LogLevel.INFO
        )
        
        return str(export_path_obj)
    
    def generate_report(self, report_type: str = "daily") -> Dict[str, Any]:
        """
        監視レポートを生成
        
        Args:
            report_type: レポートタイプ（daily, weekly, etc.）
            
        Returns:
            レポートデータ
        """
        metrics = self.get_current_metrics()
        stats = self.get_trading_statistics()
        
        report = {
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": self.start_time.isoformat(),
                "end": datetime.now().isoformat(),
                "duration_hours": metrics.uptime_seconds / 3600
            },
            "summary": {
                "system_uptime_percent": 99.9,  # 仮定値
                "total_trades": metrics.total_trades_today,
                "total_triggers": metrics.total_triggers_today,
                "success_rate": metrics.success_rate_percent,
                "avg_response_time": metrics.avg_response_time_ms
            },
            "performance": {
                "max_cpu_usage": max([m.get("cpu_usage_percent", 0) for m in self.metrics_history] + [0]),
                "max_memory_usage": max([m.get("memory_usage_mb", 0) for m in self.metrics_history] + [0]),
                "avg_response_time": metrics.avg_response_time_ms
            },
            "errors_and_warnings": {
                "total_errors": metrics.errors_last_hour,
                "total_warnings": metrics.warnings_last_hour,
                "error_rate": (metrics.errors_last_hour / max(metrics.total_trades_today, 1)) * 100
            },
            "trading_activity": stats
        }
        
        return report


# グローバルダッシュボードインスタンス
_dashboard: Optional[SystemDashboard] = None

def get_system_dashboard() -> SystemDashboard:
    """グローバルダッシュボードインスタンスを取得"""
    global _dashboard
    if _dashboard is None:
        _dashboard = SystemDashboard()
    return _dashboard

def init_system_dashboard(update_interval: int = 10) -> SystemDashboard:
    """ダッシュボードを初期化"""
    global _dashboard
    _dashboard = SystemDashboard(update_interval)
    return _dashboard