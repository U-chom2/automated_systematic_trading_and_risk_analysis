"""
パフォーマンス監視モジュール

システムの応答性能、スループット、リソース使用量を監視し、
ボトルネックの特定と最適化の提案を行う。
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import json
from pathlib import Path

from .logger import get_trading_logger, LogLevel


@dataclass
class PerformanceMetric:
    """パフォーマンス指標"""
    
    metric_name: str
    timestamp: datetime
    value: float
    unit: str
    category: str  # "latency", "throughput", "resource", "error"
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で出力"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceAlert:
    """パフォーマンス警告"""
    
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で出力"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PerformanceMonitor:
    """
    パフォーマンス監視クラス
    
    システムの各種パフォーマンス指標を継続的に監視し、
    異常検知とアラートを行う。
    """
    
    def __init__(self, 
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 history_size: int = 1000):
        """
        パフォーマンス監視を初期化
        
        Args:
            alert_thresholds: アラート閾値の設定
            history_size: 保持する履歴データサイズ
        """
        self.history_size = history_size
        
        # デフォルト閾値
        self.alert_thresholds = alert_thresholds or {
            "response_time_ms": 5000,      # 5秒
            "cpu_usage_percent": 80,       # 80%
            "memory_usage_mb": 2000,       # 2GB
            "error_rate_percent": 5,       # 5%
            "throughput_ops_per_second": 1 # 1オペレーション/秒（最低）
        }
        
        # メトリクス保存
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.active_alerts: List[PerformanceAlert] = []
        
        # 統計データ
        self.statistics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # スレッド制御
        self._lock = threading.Lock()
        self._is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # 監視対象関数
        self._monitored_functions: Dict[str, Callable] = {}
        
        self.logger = get_trading_logger()
        self.logger.log_system_event("PERF_MONITOR", "Performance monitor initialized", LogLevel.INFO)
    
    def record_metric(self, 
                     metric_name: str,
                     value: float,
                     unit: str = "ms",
                     category: str = "latency",
                     context: Optional[Dict[str, Any]] = None) -> None:
        """
        パフォーマンス指標を記録
        
        Args:
            metric_name: 指標名
            value: 指標値
            unit: 単位
            category: カテゴリ
            context: コンテキスト情報
        """
        metric = PerformanceMetric(
            metric_name=metric_name,
            timestamp=datetime.now(),
            value=value,
            unit=unit,
            category=category,
            context=context
        )
        
        with self._lock:
            self.metrics_history[metric_name].append(metric)
            self._update_statistics(metric_name)
            self._check_alerts(metric)
    
    def _update_statistics(self, metric_name: str) -> None:
        """統計データを更新"""
        metrics = list(self.metrics_history[metric_name])
        if not metrics:
            return
        
        values = [m.value for m in metrics]
        
        self.statistics[metric_name] = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99)
        }
    
    def _percentile(self, values: List[float], p: int) -> float:
        """パーセンタイル計算"""
        if not values:
            return 0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100.0)
        f = int(k)
        c = k - f
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    
    def _check_alerts(self, metric: PerformanceMetric) -> None:
        """アラートチェック"""
        threshold_key = metric.metric_name
        if threshold_key not in self.alert_thresholds:
            return
        
        threshold = self.alert_thresholds[threshold_key]
        
        # 閾値超過チェック
        if metric.value > threshold:
            severity = self._determine_severity(metric.value, threshold)
            
            # 既存のアラートが解決されていない場合はスキップ
            existing_alert = next(
                (a for a in self.active_alerts 
                 if a.metric_name == metric.metric_name and not a.resolved),
                None
            )
            
            if not existing_alert:
                alert = PerformanceAlert(
                    alert_type="threshold_exceeded",
                    severity=severity,
                    message=f"{metric.metric_name} exceeded threshold: {metric.value}{metric.unit} > {threshold}{metric.unit}",
                    metric_name=metric.metric_name,
                    threshold=threshold,
                    current_value=metric.value,
                    timestamp=metric.timestamp
                )
                
                self.active_alerts.append(alert)
                self._send_alert(alert)
    
    def _determine_severity(self, value: float, threshold: float) -> str:
        """アラートの重要度を決定"""
        ratio = value / threshold
        if ratio >= 3:
            return "critical"
        elif ratio >= 2:
            return "high"
        elif ratio >= 1.5:
            return "medium"
        else:
            return "low"
    
    def _send_alert(self, alert: PerformanceAlert) -> None:
        """アラートを送信"""
        self.logger.log_system_event(
            "PERFORMANCE_ALERT",
            alert.message,
            LogLevel.WARNING if alert.severity in ["low", "medium"] else LogLevel.ERROR,
            {"severity": alert.severity, "metric": alert.metric_name}
        )
    
    def monitor_function(self, func_name: str):
        """
        関数の実行時間を監視するデコレータ
        
        Args:
            func_name: 監視対象関数名
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    execution_time = (time.perf_counter() - start_time) * 1000
                    self.record_metric(
                        f"{func_name}_execution_time",
                        execution_time,
                        "ms",
                        "latency",
                        {"function": func.__name__, "success": True}
                    )
                    return result
                except Exception as e:
                    execution_time = (time.perf_counter() - start_time) * 1000
                    self.record_metric(
                        f"{func_name}_execution_time",
                        execution_time,
                        "ms",
                        "latency",
                        {"function": func.__name__, "success": False, "error": str(e)}
                    )
                    raise
            return wrapper
        return decorator
    
    def record_throughput(self, operation_name: str, count: int, time_window_seconds: float) -> None:
        """
        スループットを記録
        
        Args:
            operation_name: 操作名
            count: 実行回数
            time_window_seconds: 測定時間窓（秒）
        """
        throughput = count / time_window_seconds
        self.record_metric(
            f"{operation_name}_throughput",
            throughput,
            "ops/sec",
            "throughput"
        )
    
    def record_error_rate(self, operation_name: str, error_count: int, total_count: int) -> None:
        """
        エラー率を記録
        
        Args:
            operation_name: 操作名
            error_count: エラー回数
            total_count: 総実行回数
        """
        if total_count > 0:
            error_rate = (error_count / total_count) * 100
            self.record_metric(
                f"{operation_name}_error_rate",
                error_rate,
                "%",
                "error"
            )
    
    def get_metric_statistics(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        指定指標の統計情報を取得
        
        Args:
            metric_name: 指標名
            
        Returns:
            統計情報辞書
        """
        with self._lock:
            return self.statistics.get(metric_name)
    
    def get_recent_metrics(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        最近の指標データを取得
        
        Args:
            metric_name: 指標名
            limit: 取得件数
            
        Returns:
            指標データのリスト
        """
        with self._lock:
            metrics = list(self.metrics_history[metric_name])[-limit:]
            return [m.to_dict() for m in metrics]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """アクティブなアラートを取得"""
        with self._lock:
            return [alert.to_dict() for alert in self.active_alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        アラートを解決済みにする
        
        Args:
            alert_id: アラートID（タイムスタンプ）
            
        Returns:
            解決成功の場合True
        """
        with self._lock:
            for alert in self.active_alerts:
                if alert.timestamp.isoformat() == alert_id:
                    alert.resolved = True
                    self.logger.log_system_event(
                        "PERFORMANCE_ALERT",
                        f"Alert resolved: {alert.message}",
                        LogLevel.INFO
                    )
                    return True
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリーを取得"""
        with self._lock:
            summary = {
                "total_metrics": len(self.metrics_history),
                "active_alerts": len([a for a in self.active_alerts if not a.resolved]),
                "monitoring_period_hours": 24,  # 仮定値
                "key_statistics": {}
            }
            
            # 主要指標の統計
            for metric_name, stats in self.statistics.items():
                if any(key in metric_name.lower() for key in ["response_time", "throughput", "error_rate"]):
                    summary["key_statistics"][metric_name] = {
                        "mean": round(stats.get("mean", 0), 2),
                        "p95": round(stats.get("p95", 0), 2),
                        "max": round(stats.get("max", 0), 2)
                    }
            
            return summary
    
    def export_performance_data(self, export_path: Optional[str] = None) -> str:
        """
        パフォーマンスデータをエクスポート
        
        Args:
            export_path: エクスポート先パス
            
        Returns:
            エクスポートファイルパス
        """
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"_docs/performance_export_{timestamp}.json"
        
        with self._lock:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "summary": self.get_performance_summary(),
                "statistics": dict(self.statistics),
                "recent_metrics": {
                    name: [m.to_dict() for m in list(metrics)[-50:]]
                    for name, metrics in self.metrics_history.items()
                },
                "alerts": [alert.to_dict() for alert in self.active_alerts]
            }
        
        export_path_obj = Path(export_path)
        export_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path_obj, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.log_system_event(
            "PERF_MONITOR",
            f"Performance data exported to {export_path}",
            LogLevel.INFO
        )
        
        return str(export_path_obj)
    
    def analyze_trends(self, metric_name: str, window_hours: int = 24) -> Dict[str, Any]:
        """
        指標のトレンド分析
        
        Args:
            metric_name: 指標名
            window_hours: 分析時間窓（時間）
            
        Returns:
            トレンド分析結果
        """
        with self._lock:
            if metric_name not in self.metrics_history:
                return {"error": "Metric not found"}
            
            now = datetime.now()
            cutoff_time = now - timedelta(hours=window_hours)
            
            # 指定時間窓内のデータを取得
            recent_metrics = [
                m for m in self.metrics_history[metric_name]
                if m.timestamp >= cutoff_time
            ]
            
            if len(recent_metrics) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            values = [m.value for m in recent_metrics]
            
            # トレンド計算（簡易版）
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            trend_direction = "increasing" if second_avg > first_avg else "decreasing"
            trend_magnitude = abs(second_avg - first_avg)
            trend_percentage = (trend_magnitude / first_avg) * 100 if first_avg > 0 else 0
            
            return {
                "metric_name": metric_name,
                "window_hours": window_hours,
                "data_points": len(recent_metrics),
                "trend": {
                    "direction": trend_direction,
                    "magnitude": round(trend_magnitude, 2),
                    "percentage": round(trend_percentage, 2)
                },
                "statistics": {
                    "first_half_avg": round(first_avg, 2),
                    "second_half_avg": round(second_avg, 2),
                    "overall_avg": round(statistics.mean(values), 2),
                    "volatility": round(statistics.stdev(values), 2) if len(values) > 1 else 0
                }
            }
    
    def set_alert_threshold(self, metric_name: str, threshold: float) -> None:
        """
        アラート閾値を設定
        
        Args:
            metric_name: 指標名
            threshold: 閾値
        """
        with self._lock:
            self.alert_thresholds[metric_name] = threshold
            self.logger.log_system_event(
                "PERF_MONITOR",
                f"Alert threshold set for {metric_name}: {threshold}",
                LogLevel.INFO
            )
    
    def get_optimization_suggestions(self) -> List[Dict[str, str]]:
        """最適化の提案を生成"""
        suggestions = []
        
        with self._lock:
            # レスポンス時間の分析
            for metric_name, stats in self.statistics.items():
                if "response_time" in metric_name.lower():
                    if stats.get("p95", 0) > 3000:  # 3秒
                        suggestions.append({
                            "type": "performance",
                            "metric": metric_name,
                            "issue": "High response time",
                            "suggestion": "Consider optimizing database queries or adding caching"
                        })
                
                # スループットの分析
                elif "throughput" in metric_name.lower():
                    if stats.get("mean", 0) < 5:  # 5ops/sec
                        suggestions.append({
                            "type": "throughput",
                            "metric": metric_name,
                            "issue": "Low throughput",
                            "suggestion": "Consider parallel processing or resource scaling"
                        })
                
                # エラー率の分析
                elif "error_rate" in metric_name.lower():
                    if stats.get("mean", 0) > 2:  # 2%
                        suggestions.append({
                            "type": "reliability",
                            "metric": metric_name,
                            "issue": "High error rate",
                            "suggestion": "Review error handling and add more robust validation"
                        })
        
        return suggestions


# グローバルパフォーマンス監視インスタンス
_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """グローバルパフォーマンス監視インスタンスを取得"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def init_performance_monitor(alert_thresholds: Optional[Dict[str, float]] = None) -> PerformanceMonitor:
    """パフォーマンス監視を初期化"""
    global _performance_monitor
    _performance_monitor = PerformanceMonitor(alert_thresholds)
    return _performance_monitor