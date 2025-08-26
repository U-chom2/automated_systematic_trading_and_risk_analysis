"""ExecutionManagerの拡張テストケース"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from enum import Enum

from src.execution_manager.order_manager import (
    OrderManager, OrderType, OrderStatus, PositionSizing, OCOOrder
)
from src.execution_manager.position_tracker import (
    PositionTracker, Position, PositionStatus, TradeRecord
)


class TestExecutionManagerExtended:
    """ExecutionManagerの拡張テストクラス"""
    
    @pytest.fixture
    def order_manager(self) -> OrderManager:
        """テスト用OrderManagerインスタンス"""
        config = {
            "broker_api_key": "test_key",
            "broker_secret": "test_secret",
            "paper_trading": True,
            "max_positions": 5,
            "risk_per_trade_ratio": 0.01
        }
        return OrderManager(config=config)
    
    @pytest.fixture
    def position_tracker(self) -> PositionTracker:
        """テスト用PositionTrackerインスタンス"""
        return PositionTracker()
    
    def test_position_sizing_calculation_detailed(self, order_manager: OrderManager) -> None:
        """詳細なポジションサイジング計算のテスト"""
        # 要件定義書の計算式を実装
        capital = Decimal("1000000")      # 100万円
        risk_per_trade_ratio = 0.01       # 1%
        entry_price = Decimal("2500")     # エントリー価格
        stop_loss_percentage = 0.08       # 8%損切り
        
        result = order_manager.calculate_position_size(
            capital=capital,
            risk_per_trade_ratio=risk_per_trade_ratio,
            entry_price=entry_price,
            stop_loss_percentage=stop_loss_percentage
        )
        
        # 手動計算による期待値
        max_loss_per_trade = capital * Decimal(str(risk_per_trade_ratio))  # 10,000円
        stop_loss_price = entry_price * (Decimal("1") - Decimal(str(stop_loss_percentage)))  # 2,300円
        risk_per_share = entry_price - stop_loss_price  # 200円
        expected_position_size = int(max_loss_per_trade / risk_per_share)  # 50株
        expected_position_size = (expected_position_size // 100) * 100  # 単元株調整: 0株
        
        if expected_position_size == 0:
            expected_position_size = 100  # 最小単位
        
        assert result["position_size"] == expected_position_size
        assert result["max_loss"] == max_loss_per_trade
        assert result["stop_loss_price"] == stop_loss_price
        assert result["risk_per_share"] == risk_per_share
    
    def test_position_sizing_edge_cases(self, order_manager: OrderManager) -> None:
        """ポジションサイジングの境界値テスト"""
        # ケース1: 非常に小さなリスク（1株未満になる場合）
        result1 = order_manager.calculate_position_size(
            capital=Decimal("100000"),
            risk_per_trade_ratio=0.001,  # 0.1%
            entry_price=Decimal("10000"),
            stop_loss_percentage=0.05
        )
        assert result1["position_size"] == 0  # リスクが小さすぎて取引不可
        
        # ケース2: 非常に大きなリスク
        result2 = order_manager.calculate_position_size(
            capital=Decimal("1000000"),
            risk_per_trade_ratio=0.01,
            entry_price=Decimal("100"),
            stop_loss_percentage=0.05
        )
        assert result2["position_size"] > 0
        assert result2["position_size"] % 100 == 0  # 単元株
        
        # ケース3: 損切り幅がゼロ
        result3 = order_manager.calculate_position_size(
            capital=Decimal("1000000"),
            risk_per_trade_ratio=0.01,
            entry_price=Decimal("2500"),
            stop_loss_percentage=0.0
        )
        assert result3["position_size"] == 0  # リスクゼロは取引不可
    
    def test_oco_order_creation_detailed(self, order_manager: OrderManager) -> None:
        """詳細なOCO注文作成のテスト"""
        # エントリー注文パラメータ
        entry_params = {
            "symbol": "7203",
            "quantity": 500,
            "price": Decimal("2500"),
            "order_type": OrderType.MARKET_BUY
        }
        
        # 利確・損切り価格
        take_profit_price = Decimal("2700")  # +8%
        stop_loss_price = Decimal("2300")    # -8%
        
        oco_orders = order_manager.create_oco_order(
            entry_order=entry_params,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price
        )
        
        assert len(oco_orders) == 2
        
        # 利確注文の詳細確認
        tp_order = next((o for o in oco_orders if o["order_type"] == OrderType.LIMIT_SELL), None)
        assert tp_order is not None
        assert tp_order["symbol"] == "7203"
        assert tp_order["quantity"] == 500
        assert tp_order["price"] == take_profit_price
        assert tp_order["linked_order_id"] is not None
        
        # 損切り注文の詳細確認
        sl_order = next((o for o in oco_orders if o["order_type"] == OrderType.STOP_LOSS), None)
        assert sl_order is not None
        assert sl_order["symbol"] == "7203"
        assert sl_order["quantity"] == 500
        assert sl_order["price"] == stop_loss_price
        assert sl_order["linked_order_id"] is not None
        
        # OCOリンクの確認
        assert tp_order["linked_order_id"] == sl_order["order_id"]
        assert sl_order["linked_order_id"] == tp_order["order_id"]
    
    def test_re_entry_prohibition_rule_detailed(self, order_manager: OrderManager) -> None:
        """詳細な再エントリー禁止ルールのテスト"""
        symbol = "7203"
        
        # トレード終了を記録
        exit_time = datetime.now()
        order_manager.record_trade_exit(symbol=symbol, exit_time=exit_time)
        
        # 直後の再エントリー試行（禁止されるべき）
        can_enter = order_manager.can_enter_position(symbol)
        assert can_enter is False
        
        # 1時間後（まだ禁止期間内）
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = exit_time + timedelta(hours=1)
            can_enter = order_manager.can_enter_position(symbol)
            assert can_enter is False
        
        # 2時間後（まだ禁止期間内）
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = exit_time + timedelta(hours=2)
            can_enter = order_manager.can_enter_position(symbol)
            assert can_enter is False
        
        # 3取引時間経過後（再エントリー可能）
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = exit_time + timedelta(hours=3, minutes=1)
            can_enter = order_manager.can_enter_position(symbol)
            assert can_enter is True
        
        # 他の銘柄は影響を受けない
        can_enter_other = order_manager.can_enter_position("6758")
        assert can_enter_other is True
    
    def test_risk_management_rules(self, order_manager: OrderManager) -> None:
        """リスク管理ルールのテスト"""
        # 最大ポジション数制限
        assert order_manager.max_positions == 5
        
        # 現在のポジション数確認
        current_positions = order_manager.get_current_position_count()
        assert current_positions >= 0
        
        # 新しいポジションが許可されるかチェック
        can_open_new = order_manager.can_open_new_position()
        if current_positions < 5:
            assert can_open_new is True
        else:
            assert can_open_new is False
    
    def test_position_tracking_lifecycle(self, position_tracker: PositionTracker) -> None:
        """ポジション追跡ライフサイクルのテスト"""
        # 新しいポジション作成
        position = Position(
            symbol="7203",
            quantity=500,
            entry_price=Decimal("2500"),
            entry_time=datetime.now(),
            position_type="LONG",
            status=PositionStatus.OPEN,
            stop_loss_price=Decimal("2300"),
            take_profit_price=Decimal("2700")
        )
        
        # ポジション追加
        position_tracker.add_position(position)
        
        # アクティブポジション確認
        active_positions = position_tracker.get_active_positions()
        assert len(active_positions) == 1
        assert active_positions[0].symbol == "7203"
        assert active_positions[0].status == PositionStatus.OPEN
        
        # ポジション更新（部分決済）
        position_tracker.update_position(
            position.position_id,
            {"quantity": 300, "status": PositionStatus.PARTIALLY_CLOSED}
        )
        
        # 更新確認
        updated_position = position_tracker.get_position(position.position_id)
        assert updated_position.quantity == 300
        assert updated_position.status == PositionStatus.PARTIALLY_CLOSED
        
        # ポジション完全決済
        exit_time = datetime.now()
        position_tracker.close_position(
            position.position_id,
            exit_price=Decimal("2600"),
            exit_time=exit_time,
            exit_reason="TAKE_PROFIT"
        )
        
        # 決済確認
        closed_position = position_tracker.get_position(position.position_id)
        assert closed_position.status == PositionStatus.CLOSED
        assert closed_position.exit_price == Decimal("2600")
        assert closed_position.exit_time == exit_time
    
    def test_trade_record_generation(self, position_tracker: PositionTracker) -> None:
        """トレード記録生成のテスト"""
        # ポジション作成と決済
        position = Position(
            symbol="7203",
            quantity=500,
            entry_price=Decimal("2500"),
            entry_time=datetime.now(),
            position_type="LONG",
            status=PositionStatus.OPEN
        )
        
        position_tracker.add_position(position)
        
        # 決済
        exit_time = datetime.now() + timedelta(hours=2)
        position_tracker.close_position(
            position.position_id,
            exit_price=Decimal("2600"),
            exit_time=exit_time,
            exit_reason="TAKE_PROFIT"
        )
        
        # トレード記録取得
        trade_record = position_tracker.get_trade_record(position.position_id)
        
        assert trade_record is not None
        assert trade_record.symbol == "7203"
        assert trade_record.entry_price == Decimal("2500")
        assert trade_record.exit_price == Decimal("2600")
        assert trade_record.profit_loss == Decimal("50000")  # (2600-2500) * 500
        assert trade_record.profit_loss_percentage > 0
        assert trade_record.holding_time == timedelta(hours=2)
    
    def test_portfolio_statistics(self, position_tracker: PositionTracker) -> None:
        """ポートフォリオ統計のテスト"""
        # 複数のトレード記録を作成
        trades = [
            {"symbol": "7203", "entry": 2500, "exit": 2600, "qty": 500, "result": "WIN"},
            {"symbol": "6758", "entry": 3000, "exit": 2850, "qty": 300, "result": "LOSS"},
            {"symbol": "9984", "entry": 8000, "exit": 8400, "qty": 100, "result": "WIN"}
        ]
        
        for trade in trades:
            position = Position(
                symbol=trade["symbol"],
                quantity=trade["qty"],
                entry_price=Decimal(str(trade["entry"])),
                entry_time=datetime.now(),
                position_type="LONG",
                status=PositionStatus.CLOSED,
                exit_price=Decimal(str(trade["exit"])),
                exit_time=datetime.now() + timedelta(hours=1)
            )
            position_tracker.add_position(position)
        
        # ポートフォリオ統計取得
        stats = position_tracker.get_portfolio_statistics()
        
        assert stats["total_trades"] == 3
        assert stats["winning_trades"] == 2
        assert stats["losing_trades"] == 1
        assert stats["win_rate"] == pytest.approx(0.6667, rel=1e-3)
        assert stats["total_profit_loss"] > 0  # 全体で利益
        assert stats["average_trade_duration"] > timedelta(0)
    
    def test_order_execution_simulation(self, order_manager: OrderManager) -> None:
        """注文執行シミュレーションのテスト"""
        # 市場買い注文
        buy_order = {
            "symbol": "7203",
            "quantity": 500,
            "order_type": OrderType.MARKET_BUY,
            "price": None  # 成行注文
        }
        
        # 注文実行（モック）
        order_result = order_manager.execute_order(buy_order)
        
        assert order_result["status"] == OrderStatus.FILLED
        assert order_result["filled_quantity"] == 500
        assert order_result["execution_price"] is not None
        assert order_result["order_id"] is not None
        
        # 指値売り注文
        sell_order = {
            "symbol": "7203",
            "quantity": 500,
            "order_type": OrderType.LIMIT_SELL,
            "price": Decimal("2700")
        }
        
        sell_result = order_manager.execute_order(sell_order)
        
        assert sell_result["status"] == OrderStatus.PENDING
        assert sell_result["order_id"] is not None
    
    def test_order_cancellation(self, order_manager: OrderManager) -> None:
        """注文キャンセルのテスト"""
        # 指値注文作成
        order = {
            "symbol": "7203",
            "quantity": 500,
            "order_type": OrderType.LIMIT_BUY,
            "price": Decimal("2400")
        }
        
        order_result = order_manager.execute_order(order)
        order_id = order_result["order_id"]
        
        # 注文キャンセル
        cancel_result = order_manager.cancel_order(order_id)
        
        assert cancel_result["success"] is True
        assert cancel_result["order_id"] == order_id
        
        # 注文状態確認
        order_status = order_manager.get_order_status(order_id)
        assert order_status["status"] == OrderStatus.CANCELLED
    
    def test_slippage_handling(self, order_manager: OrderManager) -> None:
        """スリッページ処理のテスト"""
        # スリッページを考慮した注文
        order = {
            "symbol": "7203",
            "quantity": 500,
            "order_type": OrderType.MARKET_BUY,
            "expected_price": Decimal("2500"),
            "max_slippage_percent": 0.5  # 0.5%まで許容
        }
        
        result = order_manager.execute_order_with_slippage_control(order)
        
        if result["status"] == OrderStatus.FILLED:
            execution_price = result["execution_price"]
            expected_price = order["expected_price"]
            max_acceptable_price = expected_price * Decimal("1.005")
            
            assert execution_price <= max_acceptable_price
        else:
            # スリッページが大きすぎて約定拒否
            assert result["rejection_reason"] == "EXCESSIVE_SLIPPAGE"
    
    def test_order_validation(self, order_manager: OrderManager) -> None:
        """注文バリデーションのテスト"""
        # 有効な注文
        valid_order = {
            "symbol": "7203",
            "quantity": 500,
            "order_type": OrderType.MARKET_BUY,
            "price": None
        }
        
        validation_result = order_manager.validate_order(valid_order)
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
        
        # 無効な注文（数量がゼロ）
        invalid_order1 = {
            "symbol": "7203",
            "quantity": 0,
            "order_type": OrderType.MARKET_BUY,
            "price": None
        }
        
        validation_result = order_manager.validate_order(invalid_order1)
        assert validation_result["valid"] is False
        assert "quantity" in validation_result["errors"]
        
        # 無効な注文（銘柄コードが空）
        invalid_order2 = {
            "symbol": "",
            "quantity": 500,
            "order_type": OrderType.MARKET_BUY,
            "price": None
        }
        
        validation_result = order_manager.validate_order(invalid_order2)
        assert validation_result["valid"] is False
        assert "symbol" in validation_result["errors"]