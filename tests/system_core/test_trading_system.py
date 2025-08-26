"""TradingSystemのテストケース"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from src.system_core.trading_system import TradingSystem, SystemStatus, SystemConfig
from src.system_core.workflow_manager import WorkflowManager
from src.execution_manager.order_manager import OrderManager
from src.execution_manager.position_tracker import PositionTracker


class TestTradingSystem:
    """TradingSystemのテストクラス"""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """テスト用システム設定"""
        return {
            "capital": Decimal("1000000"),
            "risk_per_trade_ratio": 0.01,
            "max_positions": 5,
            "market_hours": {
                "start": "09:00",
                "end": "15:00"
            },
            "watchlist_file": "test_watchlist.xlsx",
            "log_level": "DEBUG",
            "data_sources": {
                "tdnet": {"polling_interval": 1},
                "x_streaming": {"keywords": ["決算", "上方修正"]},
                "price_api": {"provider": "mock"}
            }
        }
    
    @pytest.fixture
    def trading_system(self, sample_config: Dict[str, Any]) -> TradingSystem:
        """テスト用TradingSystemインスタンス"""
        return TradingSystem(config=sample_config)
    
    def test_system_initialization(self, trading_system: TradingSystem, sample_config: Dict[str, Any]) -> None:
        """システム初期化のテスト"""
        assert trading_system is not None
        assert trading_system.config.capital == sample_config["capital"]
        assert trading_system.config.risk_per_trade_ratio == sample_config["risk_per_trade_ratio"]
        assert trading_system.config.max_positions == sample_config["max_positions"]
        assert trading_system.status == SystemStatus.STOPPED
        assert trading_system.is_running is False
        
        # コンポーネントの初期化確認
        assert trading_system.workflow_manager is not None
        assert trading_system.order_manager is not None
        assert trading_system.position_tracker is not None
        assert trading_system.data_collector is not None
        assert trading_system.analysis_engine is not None
    
    def test_system_startup(self, trading_system: TradingSystem) -> None:
        """システム起動のテスト"""
        # 起動前状態確認
        assert trading_system.status == SystemStatus.STOPPED
        assert trading_system.is_running is False
        
        # システム起動
        startup_result = trading_system.start_system()
        
        assert startup_result is True
        assert trading_system.status == SystemStatus.RUNNING
        assert trading_system.is_running is True
        assert trading_system.start_time is not None
    
    def test_system_shutdown(self, trading_system: TradingSystem) -> None:
        """システム停止のテスト"""
        # システムを起動してから停止テスト
        trading_system.start_system()
        assert trading_system.is_running is True
        
        # システム停止
        shutdown_result = trading_system.stop_system()
        
        assert shutdown_result is True
        assert trading_system.status == SystemStatus.STOPPED
        assert trading_system.is_running is False
        assert trading_system.stop_time is not None
    
    def test_system_restart(self, trading_system: TradingSystem) -> None:
        """システム再起動のテスト"""
        # 初回起動
        trading_system.start_system()
        first_start_time = trading_system.start_time
        
        # 停止
        trading_system.stop_system()
        
        # 再起動
        trading_system.restart_system()
        
        assert trading_system.is_running is True
        assert trading_system.status == SystemStatus.RUNNING
        assert trading_system.start_time != first_start_time  # 起動時間が更新される
    
    def test_market_hours_validation(self, trading_system: TradingSystem) -> None:
        """市場時間バリデーションのテスト"""
        # 市場時間内のテスト
        with patch('datetime.datetime') as mock_datetime:
            # 10:00（市場時間内）
            mock_datetime.now.return_value = datetime.now().replace(hour=10, minute=0)
            assert trading_system.is_market_open() is True
        
        # 市場時間外のテスト
        with patch('datetime.datetime') as mock_datetime:
            # 17:00（市場時間外）
            mock_datetime.now.return_value = datetime.now().replace(hour=17, minute=0)
            assert trading_system.is_market_open() is False
        
        # 境界値テスト
        with patch('datetime.datetime') as mock_datetime:
            # 09:00（開始時刻）
            mock_datetime.now.return_value = datetime.now().replace(hour=9, minute=0)
            assert trading_system.is_market_open() is True
            
            # 15:00（終了時刻）
            mock_datetime.now.return_value = datetime.now().replace(hour=15, minute=0)
            assert trading_system.is_market_open() is True
            
            # 15:01（終了後）
            mock_datetime.now.return_value = datetime.now().replace(hour=15, minute=1)
            assert trading_system.is_market_open() is False
    
    def test_emergency_stop(self, trading_system: TradingSystem) -> None:
        """緊急停止のテスト"""
        # システム起動
        trading_system.start_system()
        assert trading_system.is_running is True
        
        # 緊急停止実行
        emergency_result = trading_system.emergency_stop("Test emergency stop")
        
        assert emergency_result is True
        assert trading_system.status == SystemStatus.EMERGENCY_STOPPED
        assert trading_system.is_running is False
        assert trading_system.emergency_reason == "Test emergency stop"
    
    def test_system_health_check(self, trading_system: TradingSystem) -> None:
        """システムヘルスチェックのテスト"""
        # システム起動前
        health = trading_system.get_system_health()
        assert health["status"] == "STOPPED"
        assert health["uptime"] == 0
        assert "components" in health
        
        # システム起動後
        trading_system.start_system()
        health = trading_system.get_system_health()
        assert health["status"] == "RUNNING"
        assert health["uptime"] > 0
        assert health["components"]["workflow_manager"]["status"] == "OK"
        assert health["components"]["order_manager"]["status"] == "OK"
        assert health["components"]["position_tracker"]["status"] == "OK"
    
    def test_performance_statistics(self, trading_system: TradingSystem) -> None:
        """パフォーマンス統計のテスト"""
        stats = trading_system.get_performance_statistics()
        
        # 必須フィールドの確認
        required_fields = [
            "total_trades", "winning_trades", "losing_trades", "win_rate",
            "total_profit_loss", "average_profit", "average_loss", 
            "max_drawdown", "sharpe_ratio", "profit_factor"
        ]
        
        for field in required_fields:
            assert field in stats
        
        # 初期値の確認
        assert stats["total_trades"] == 0
        assert stats["winning_trades"] == 0
        assert stats["losing_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["total_profit_loss"] == Decimal("0")
    
    def test_configuration_validation(self) -> None:
        """設定バリデーションのテスト"""
        # 正常な設定
        valid_config = {
            "capital": Decimal("1000000"),
            "risk_per_trade_ratio": 0.01,
            "max_positions": 5,
            "market_hours": {"start": "09:00", "end": "15:00"}
        }
        
        system = TradingSystem(config=valid_config)
        assert system is not None
        
        # 無効な設定のテスト
        invalid_configs = [
            # 資金がゼロ
            {**valid_config, "capital": Decimal("0")},
            # リスク率が負
            {**valid_config, "risk_per_trade_ratio": -0.01},
            # 最大ポジション数がゼロ
            {**valid_config, "max_positions": 0},
            # 市場時間の形式が不正
            {**valid_config, "market_hours": {"start": "invalid", "end": "15:00"}}
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                TradingSystem(config=invalid_config)
    
    def test_component_integration(self, trading_system: TradingSystem) -> None:
        """コンポーネント統合のテスト"""
        # 各コンポーネントが正しく統合されていることを確認
        assert trading_system.workflow_manager.trading_system == trading_system
        assert trading_system.order_manager.trading_system == trading_system
        assert trading_system.position_tracker.trading_system == trading_system
    
    def test_error_handling_and_recovery(self, trading_system: TradingSystem) -> None:
        """エラーハンドリングと復旧のテスト"""
        # システム起動
        trading_system.start_system()
        
        # コンポーネントエラーのシミュレーション
        with patch.object(trading_system.workflow_manager, 'process_trigger') as mock_process:
            mock_process.side_effect = Exception("Component error")
            
            # エラーが発生してもシステムが継続すること
            with pytest.raises(Exception):
                trading_system.workflow_manager.process_trigger({"test": "data"})
            
            # システムはまだ動作中
            assert trading_system.is_running is True
    
    def test_logging_functionality(self, trading_system: TradingSystem) -> None:
        """ログ機能のテスト"""
        # ログ設定の確認
        assert trading_system.logger is not None
        
        # ログレベルのテスト
        trading_system.set_log_level("DEBUG")
        assert trading_system.logger.level == 10  # DEBUG level
        
        trading_system.set_log_level("INFO")
        assert trading_system.logger.level == 20  # INFO level
    
    def test_concurrent_operations(self, trading_system: TradingSystem) -> None:
        """並行操作のテスト"""
        # システム起動
        trading_system.start_system()
        
        # 複数の操作を並行実行（モック）
        operations = []
        for i in range(5):
            operation = {
                "type": "trigger",
                "data": {"symbol": f"test{i}", "importance": 30 + i}
            }
            operations.append(operation)
        
        # 並行処理のテスト（実装時にはasyncioを使用）
        results = trading_system.process_concurrent_operations(operations)
        
        assert len(results) == 5
        for result in results:
            assert "processed" in result
    
    def test_system_metrics_collection(self, trading_system: TradingSystem) -> None:
        """システムメトリクス収集のテスト"""
        # システム起動
        trading_system.start_system()
        
        # メトリクス収集
        metrics = trading_system.collect_system_metrics()
        
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "active_threads" in metrics
        assert "queue_size" in metrics
        assert "processing_time" in metrics
        
        # メトリクスが数値であることを確認
        assert isinstance(metrics["cpu_usage"], (int, float))
        assert isinstance(metrics["memory_usage"], (int, float))
        assert isinstance(metrics["active_threads"], int)
    
    def test_system_state_persistence(self, trading_system: TradingSystem) -> None:
        """システム状態永続化のテスト"""
        # システム起動と状態設定
        trading_system.start_system()
        
        # 状態保存
        save_result = trading_system.save_system_state()
        assert save_result is True
        
        # 新しいインスタンスで状態復元
        new_system = TradingSystem(config=trading_system.config.__dict__)
        restore_result = new_system.restore_system_state()
        
        assert restore_result is True
        # 復元された状態の確認（実装次第）
    
    def test_system_shutdown_graceful(self, trading_system: TradingSystem) -> None:
        """グレースフルシャットダウンのテスト"""
        # システム起動
        trading_system.start_system()
        
        # 進行中のタスクをシミュレート
        with patch.object(trading_system, '_has_pending_operations') as mock_pending:
            mock_pending.return_value = True
            
            # グレースフルシャットダウン実行
            shutdown_result = trading_system.graceful_shutdown(timeout=5)
            
            assert shutdown_result is True
            assert trading_system.is_running is False
    
    def test_system_configuration_update(self, trading_system: TradingSystem) -> None:
        """システム設定更新のテスト"""
        original_capital = trading_system.config.capital
        
        # 設定更新
        new_config = {
            "capital": Decimal("2000000"),  # 200万円に変更
            "risk_per_trade_ratio": 0.02    # 2%に変更
        }
        
        update_result = trading_system.update_configuration(new_config)
        
        assert update_result is True
        assert trading_system.config.capital == Decimal("2000000")
        assert trading_system.config.risk_per_trade_ratio == 0.02
        assert trading_system.config.capital != original_capital