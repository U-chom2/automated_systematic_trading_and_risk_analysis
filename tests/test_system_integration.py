"""システム統合テストケース"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from src.system_core.trading_system import TradingSystem
from src.system_core.workflow_manager import WorkflowManager, TradingPhase, WorkflowState, TriggerEvent
from src.execution_manager.order_manager import OrderManager, OrderType, OrderStatus
from src.execution_manager.position_tracker import PositionTracker, Position, PositionStatus


class TestSystemIntegration:
    """システム統合テストクラス"""
    
    @pytest.fixture
    def mock_trading_system(self) -> TradingSystem:
        """モックTradingSystemを作成"""
        return TradingSystem(config={
            "capital": Decimal("1000000"),  # 100万円
            "risk_per_trade_ratio": 0.01,   # 1%リスク
            "max_positions": 5,
            "market_hours": {"start": "09:00", "end": "15:00"}
        })
    
    @pytest.fixture 
    def sample_market_data(self) -> Dict[str, Any]:
        """テスト用市場データ"""
        return {
            "symbol": "7203",
            "company_name": "トヨタ自動車",
            "current_price": Decimal("2500"),
            "volume": 1000000,
            "timestamp": datetime.now()
        }
    
    @pytest.fixture
    def sample_ir_trigger(self) -> Dict[str, Any]:
        """テスト用IRトリガーデータ"""
        return {
            "symbol": "7203",
            "title": "業績の上方修正に関するお知らせ",
            "content": "当社は、業績予想の上方修正を発表いたします。",
            "timestamp": datetime.now(),
            "importance_score": 50
        }
    
    def test_system_initialization(self, mock_trading_system: TradingSystem) -> None:
        """システム初期化のテスト"""
        assert mock_trading_system is not None
        assert mock_trading_system.config.capital == Decimal("1000000")
        assert mock_trading_system.is_running is False
        assert mock_trading_system.workflow_manager is not None
        assert mock_trading_system.order_manager is not None
    
    def test_system_startup_and_shutdown(self, mock_trading_system: TradingSystem) -> None:
        """システム起動・停止のテスト"""
        # 起動テスト
        assert mock_trading_system.start_system() is True
        assert mock_trading_system.is_running is True
        
        # 停止テスト
        assert mock_trading_system.stop_system() is True
        assert mock_trading_system.is_running is False
    
    @pytest.mark.asyncio
    async def test_end_to_end_trading_workflow(self, mock_trading_system: TradingSystem, 
                                               sample_ir_trigger: Dict[str, Any],
                                               sample_market_data: Dict[str, Any]) -> None:
        """エンドツーエンド取引ワークフローのテスト"""
        # TriggerEventオブジェクトを作成
        trigger_event = TriggerEvent(
            trigger_type="IR",
            symbol=sample_ir_trigger["symbol"],
            title=sample_ir_trigger["title"],
            content=sample_ir_trigger["content"],
            timestamp=sample_ir_trigger["timestamp"],
            source="TDnet",
            importance_score=sample_ir_trigger["importance_score"]
        )
        
        # フェーズ2: トリガー検知
        trigger_result = await mock_trading_system.workflow_manager.process_trigger(trigger_event)
        assert trigger_result.trigger_activated is True
        assert trigger_result.symbol == "7203"
        
        # フェーズ3: 分析と判断
        analysis_result = await mock_trading_system.workflow_manager.analyze_and_decide(
            trigger_event, sample_market_data
        )
        assert "total_score" in analysis_result
        assert "buy_decision" in analysis_result
        assert analysis_result["total_score"] >= 0
        
        # フェーズ4: 実行準備（買い判断の場合）
        if analysis_result["buy_decision"]:
            execution_result = await mock_trading_system.workflow_manager.prepare_execution(
                sample_market_data, analysis_result
            )
            assert "position_size" in execution_result
            assert "entry_price" in execution_result
            assert execution_result["position_size"] > 0
    
    def test_position_sizing_calculation(self, mock_trading_system: TradingSystem) -> None:
        """ポジションサイジング計算のテスト"""
        # テストパラメータ
        capital = Decimal("1000000")  # 100万円
        risk_per_trade_ratio = 0.01   # 1%
        entry_price = Decimal("2500")
        stop_loss_percentage = 0.08   # 8%
        
        position_sizing_result = mock_trading_system.order_manager.calculate_position_size(
            capital=capital,
            risk_per_trade_ratio=risk_per_trade_ratio,
            entry_price=entry_price,
            stop_loss_percentage=stop_loss_percentage
        )
        
        # 期待値計算
        max_loss = capital * Decimal(str(risk_per_trade_ratio))  # 10,000円
        stop_loss_price = entry_price * (Decimal("1") - Decimal(str(stop_loss_percentage)))
        risk_per_share = entry_price - stop_loss_price
        expected_size = int(max_loss / risk_per_share) 
        expected_size = (expected_size // 100) * 100  # 単元株調整
        
        if expected_size == 0:
            expected_size = 100  # Minimum unit
        
        assert position_sizing_result["position_size"] == expected_size
        assert position_sizing_result["position_size"] > 0
    
    def test_oco_order_creation(self, mock_trading_system: TradingSystem) -> None:
        """OCO注文作成のテスト"""
        # エントリー注文
        entry_order = {
            "symbol": "7203",
            "quantity": 500,
            "price": Decimal("2500"),
            "order_type": OrderType.MARKET_BUY
        }
        
        # OCO注文パラメータ
        take_profit_price = Decimal("2700")  # +8%
        stop_loss_price = Decimal("2300")    # -8%
        
        oco_orders = mock_trading_system.order_manager.create_oco_order(
            entry_order=entry_order,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price
        )
        
        assert len(oco_orders) == 2
        assert any(order["order_type"] == OrderType.LIMIT_SELL for order in oco_orders)
        assert any(order["order_type"] == OrderType.STOP_LOSS for order in oco_orders)
        
        # 利確注文の確認
        tp_order = next(order for order in oco_orders if order["order_type"] == OrderType.LIMIT_SELL)
        assert tp_order["price"] == take_profit_price
        assert tp_order["quantity"] == entry_order["quantity"]
        
        # 損切り注文の確認
        sl_order = next(order for order in oco_orders if order["order_type"] == OrderType.STOP_LOSS)
        assert sl_order["price"] == stop_loss_price
        assert sl_order["quantity"] == entry_order["quantity"]
    
    def test_re_entry_prohibition_rule(self, mock_trading_system: TradingSystem) -> None:
        """再エントリー禁止ルールのテスト"""
        symbol = "7203"
        
        # 最初のトレードを終了
        mock_trading_system.order_manager.record_trade_exit(
            symbol=symbol,
            exit_time=datetime.now()
        )
        
        # 3取引時間以内の再エントリー試行
        can_enter = mock_trading_system.order_manager.can_enter_position(symbol)
        assert can_enter is False
        
        # 3取引時間後の再エントリー試行（モック）
        with patch('src.execution_manager.order_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(hours=4)
            can_enter = mock_trading_system.order_manager.can_enter_position(symbol)
            assert can_enter is True
    
    def test_risk_management_filter(self, mock_trading_system: TradingSystem) -> None:
        """リスク管理フィルターのテスト"""
        # 過熱状態の市場データ
        overheated_data = {
            "symbol": "7203",
            "rsi": 85.0,           # RSI > 75
            "ma_deviation": 30.0,  # 乖離率 > 25%
            "current_price": Decimal("2500")
        }
        
        # フィルターチェック
        filter_result = mock_trading_system.workflow_manager.check_risk_filters(overheated_data)
        assert filter_result["passed"] is False
        assert "rsi_overheated" in filter_result["reasons"]
        assert "ma_deviation_overheated" in filter_result["reasons"]
        
        # 正常状態の市場データ
        normal_data = {
            "symbol": "7203", 
            "rsi": 55.0,          # RSI < 75
            "ma_deviation": 10.0, # 乖離率 < 25%
            "current_price": Decimal("2500")
        }
        
        filter_result = mock_trading_system.workflow_manager.check_risk_filters(normal_data)
        assert filter_result["passed"] is True
        assert len(filter_result["reasons"]) == 0
    
    @pytest.mark.asyncio
    async def test_system_error_handling(self, mock_trading_system: TradingSystem) -> None:
        """システムエラーハンドリングのテスト"""
        # システムエラー処理のテスト：正常な処理フローを確認
        valid_trigger = TriggerEvent(
            trigger_type="IR",
            symbol="7203",
            title="テストIR",
            content="テスト内容",
            timestamp=datetime.now(),
            source="TDnet",
            importance_score=50
        )
        
        # 正常処理が例外を発生させないことを確認
        try:
            result = await mock_trading_system.workflow_manager.process_trigger(valid_trigger)
            assert result is not None
            assert result.symbol == "7203"
        except Exception as e:
            # 予期しないエラーの場合、テストを失敗させる
            pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")
        
        # エラー復旧機能のテスト
        # まずエラー状態に設定
        mock_trading_system.workflow_manager.state = WorkflowState.ERROR
        mock_trading_system.workflow_manager.error_message = "Test error"
        
        # エラーからの復旧
        recovery_result = mock_trading_system.workflow_manager.recover_from_error()
        assert recovery_result is True
        assert mock_trading_system.workflow_manager.state == WorkflowState.WAITING
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, mock_trading_system: TradingSystem) -> None:
        """並行処理のテスト"""
        # 複数のTriggerEventを作成
        triggers = [
            TriggerEvent(
                trigger_type="IR",
                symbol="7203",
                title="IR発表1",
                content="テスト内容1",
                timestamp=datetime.now(),
                source="TDnet",
                importance_score=40
            ),
            TriggerEvent(
                trigger_type="IR",
                symbol="6758",
                title="IR発表2",
                content="テスト内容2",
                timestamp=datetime.now(),
                source="TDnet",
                importance_score=45
            ),
            TriggerEvent(
                trigger_type="IR",
                symbol="9984",
                title="IR発表3",
                content="テスト内容3",
                timestamp=datetime.now(),
                source="TDnet",
                importance_score=35
            )
        ]
        
        # 並行処理実行
        results = await mock_trading_system.workflow_manager.process_multiple_triggers(triggers)
        
        assert len(results) == 3
        for result in results:
            assert result.trigger_activated is not None
            assert result.symbol in ["7203", "6758", "9984"]
    
    def test_position_tracking(self, mock_trading_system: TradingSystem) -> None:
        """ポジション追跡のテスト"""
        # ポジション作成
        position = Position(
            symbol="7203",
            quantity=500,
            entry_price=Decimal("2500"),
            entry_time=datetime.now(),
            position_type="LONG",
            status=PositionStatus.OPEN
        )
        
        # ポジション登録
        mock_trading_system.position_tracker.add_position(position)
        
        # ポジション確認
        active_positions = mock_trading_system.position_tracker.get_active_positions()
        assert len(active_positions) == 1
        assert active_positions[0].symbol == "7203"
        assert active_positions[0].quantity == 500
    
    def test_system_performance_monitoring(self, mock_trading_system: TradingSystem) -> None:
        """システムパフォーマンス監視のテスト"""
        # パフォーマンス統計取得
        stats = mock_trading_system.get_performance_statistics()
        
        assert "total_trades" in stats
        assert "win_rate" in stats
        assert "total_profit_loss" in stats
        assert "max_drawdown" in stats
        assert "sharpe_ratio" in stats
        
        # 初期値確認
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["total_profit_loss"] == Decimal("0")
    
    @pytest.mark.asyncio
    async def test_market_hours_validation(self, mock_trading_system: TradingSystem) -> None:
        """市場時間バリデーションのテスト"""
        # 市場時間外のテスト
        with patch('src.system_core.trading_system.datetime') as mock_datetime:
            # 17:00（市場時間外）
            mock_datetime.now.return_value = datetime.now().replace(hour=17, minute=0)
            
            is_market_open = mock_trading_system.is_market_open()
            assert is_market_open is False
        
        # 市場時間内のテスト
        with patch('src.system_core.trading_system.datetime') as mock_datetime:
            # 10:00（市場時間内）
            mock_datetime.now.return_value = datetime.now().replace(hour=10, minute=0)
            
            is_market_open = mock_trading_system.is_market_open()
            assert is_market_open is True