"""Tests for OrderManager."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from src.execution_manager.order_manager import (
    OrderManager, 
    PositionSizing, 
    OrderType, 
    OrderStatus
)


class TestOrderManager:
    """Test cases for OrderManager."""
    
    @pytest.fixture
    def order_manager(self) -> OrderManager:
        """Create OrderManager instance for testing."""
        return OrderManager(
            broker_api_key="test_key",
            broker_secret="test_secret",
            paper_trading=True
        )
    
    def test_position_sizing_calculation(self, order_manager: OrderManager) -> None:
        """Test position sizing calculation with valid inputs."""
        capital = Decimal("100000")  # 10万円
        entry_price = Decimal("1000")  # 1000円
        stop_loss_percentage = 0.08  # 8%
        
        result = order_manager.calculate_position_size(
            capital, entry_price, stop_loss_percentage
        )
        
        # Expected calculations:
        # max_loss = 100000 * 0.01 = 1000円
        # stop_loss_price = 1000 * (1 - 0.08) = 920円
        # risk_per_share = 1000 - 920 = 80円
        # position_size = 1000 / 80 = 12.5 -> 12 -> 1200株 (unit lot)
        
        assert result.is_valid
        assert result.recommended_shares == 1200
        assert result.risk_per_share == Decimal("80")
        assert result.max_loss_amount == Decimal("1000")
    
    def test_position_sizing_with_zero_risk(self, order_manager: OrderManager) -> None:
        """Test position sizing with zero risk scenario."""
        capital = Decimal("100000")
        entry_price = Decimal("1000")
        stop_loss_percentage = 0.0  # No stop loss
        
        result = order_manager.calculate_position_size(
            capital, entry_price, stop_loss_percentage
        )
        
        assert not result.is_valid
        assert result.recommended_shares == 0
        assert "Risk per share is zero or negative" in result.reason
    
    def test_position_sizing_with_small_capital(self, order_manager: OrderManager) -> None:
        """Test position sizing with small capital."""
        capital = Decimal("10000")  # 1万円
        entry_price = Decimal("5000")  # 高額株
        stop_loss_percentage = 0.05  # 5%
        
        result = order_manager.calculate_position_size(
            capital, entry_price, stop_loss_percentage
        )
        
        # max_loss = 10000 * 0.01 = 100円
        # risk_per_share = 5000 * 0.05 = 250円
        # position_size = 100 / 250 = 0.4 -> 0株
        
        assert not result.is_valid
        assert result.recommended_shares == 0
    
    def test_re_entry_restriction(self, order_manager: OrderManager) -> None:
        """Test re-entry restriction functionality."""
        symbol = "7203"
        
        # Initially no restriction
        assert not order_manager.is_re_entry_restricted(symbol)
        
        # Add restriction
        order_manager.add_re_entry_restriction(symbol, hours=3)
        assert order_manager.is_re_entry_restricted(symbol)
        
        # Mock time passage - this would need time mocking in real tests
        # For now, just test the basic functionality
    
    def test_market_buy_order(self, order_manager: OrderManager) -> None:
        """Test market buy order placement."""
        symbol = "7203"
        quantity = 100
        
        response = order_manager.place_market_buy_order(symbol, quantity)
        
        assert response.order_id.startswith(f"BUY_{symbol}")
        assert response.status == OrderStatus.PENDING
        assert response.timestamp is not None
    
    def test_oco_order(self, order_manager: OrderManager) -> None:
        """Test OCO order placement."""
        symbol = "7203"
        quantity = 100
        take_profit = Decimal("1100")
        stop_loss = Decimal("900")
        
        response = order_manager.place_oco_order(
            symbol, quantity, take_profit, stop_loss
        )
        
        assert response.order_id.startswith(f"OCO_{symbol}")
        assert response.status == OrderStatus.PENDING
    
    def test_complete_trade_execution(self, order_manager: OrderManager) -> None:
        """Test complete trade execution workflow."""
        symbol = "7203"
        capital = Decimal("100000")
        entry_price = Decimal("1000")
        stop_loss_percentage = 0.08
        
        result = order_manager.execute_complete_trade(
            symbol, capital, entry_price, stop_loss_percentage
        )
        
        assert result["success"]
        assert "buy_order" in result
        assert "oco_order" in result
        assert "position_sizing" in result
        assert "stop_loss_price" in result
        assert "take_profit_price" in result
    
    def test_complete_trade_with_restriction(self, order_manager: OrderManager) -> None:
        """Test complete trade execution with re-entry restriction."""
        symbol = "7203"
        capital = Decimal("100000")
        entry_price = Decimal("1000")
        stop_loss_percentage = 0.08
        
        # Add restriction first
        order_manager.add_re_entry_restriction(symbol, hours=3)
        
        result = order_manager.execute_complete_trade(
            symbol, capital, entry_price, stop_loss_percentage
        )
        
        assert not result["success"]
        assert result["reason"] == "Re-entry restriction active"