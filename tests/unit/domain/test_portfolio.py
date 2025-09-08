"""ポートフォリオドメインエンティティのユニットテスト"""
import pytest
from decimal import Decimal
from datetime import datetime
from uuid import uuid4

from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.entities.trade import Trade
from src.domain.value_objects.money import Money
from src.common.exceptions import InsufficientFundsException, PositionNotFoundException


class TestPortfolio:
    """ポートフォリオエンティティのテスト"""
    
    def test_create_portfolio(self):
        """ポートフォリオの作成"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        assert portfolio.name == "Test Portfolio"
        assert portfolio.initial_capital == Decimal("10000000")
        assert portfolio.current_capital == Decimal("10000000")
        assert portfolio.is_active is True
        assert len(portfolio.positions) == 0
    
    def test_add_position(self):
        """ポジションの追加"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        position = Position(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            quantity=100,
            entry_price=Decimal("2500")
        )
        
        portfolio.add_position(position)
        
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].ticker == "7203.T"
    
    def test_remove_position(self):
        """ポジションの削除"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        position = Position(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            quantity=100,
            entry_price=Decimal("2500")
        )
        
        portfolio.add_position(position)
        portfolio.remove_position("7203.T")
        
        assert len(portfolio.positions) == 0
    
    def test_remove_nonexistent_position(self):
        """存在しないポジションの削除"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        with pytest.raises(PositionNotFoundException):
            portfolio.remove_position("9999.T")
    
    def test_get_position(self):
        """ポジションの取得"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        position = Position(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            quantity=100,
            entry_price=Decimal("2500")
        )
        
        portfolio.add_position(position)
        retrieved = portfolio.get_position("7203.T")
        
        assert retrieved is not None
        assert retrieved.ticker == "7203.T"
    
    def test_calculate_total_value(self):
        """総価値の計算"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        # ポジションを追加
        position1 = Position(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            quantity=100,
            entry_price=Decimal("2500")
        )
        position1.current_price = Decimal("2600")
        
        position2 = Position(
            portfolio_id=portfolio.id,
            ticker="9984.T",
            quantity=200,
            entry_price=Decimal("5000")
        )
        position2.current_price = Decimal("5100")
        
        portfolio.add_position(position1)
        portfolio.add_position(position2)
        portfolio.current_capital = Decimal("8750000")  # 残りの現金
        
        total_value = portfolio.calculate_total_value()
        
        # 8750000 + (100 * 2600) + (200 * 5100) = 10030000
        assert total_value == Decimal("10030000")
    
    def test_calculate_return(self):
        """収益率の計算"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        portfolio.current_capital = Decimal("8750000")
        
        position = Position(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            quantity=100,
            entry_price=Decimal("2500")
        )
        position.current_price = Decimal("2600")
        portfolio.add_position(position)
        
        position2 = Position(
            portfolio_id=portfolio.id,
            ticker="9984.T",
            quantity=200,
            entry_price=Decimal("5000")
        )
        position2.current_price = Decimal("5100")
        portfolio.add_position(position2)
        
        returns = portfolio.calculate_return()
        
        # (10030000 - 10000000) / 10000000 = 0.003
        assert returns == Decimal("0.003")
    
    def test_can_afford_trade_buy(self):
        """買い注文が可能かチェック"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        trade = Trade(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            side="BUY",
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250")
        )
        
        assert portfolio.can_afford_trade(trade) is True
    
    def test_cannot_afford_trade_insufficient_funds(self):
        """資金不足で買い注文が不可能"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("1000")
        )
        
        trade = Trade(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            side="BUY",
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250")
        )
        
        assert portfolio.can_afford_trade(trade) is False
    
    def test_execute_buy_trade(self):
        """買い取引の実行"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        trade = Trade(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            side="BUY",
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250")
        )
        
        portfolio.execute_trade(trade)
        
        assert portfolio.current_capital == Decimal("9749750")  # 10000000 - 250250
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].quantity == 100
    
    def test_execute_sell_trade(self):
        """売り取引の実行"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        # まず買いポジションを作成
        position = Position(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            quantity=100,
            entry_price=Decimal("2500")
        )
        portfolio.add_position(position)
        portfolio.current_capital = Decimal("9749750")
        
        # 売り取引を実行
        trade = Trade(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            side="SELL",
            quantity=100,
            price=Decimal("2600"),
            commission=Decimal("260")
        )
        
        portfolio.execute_trade(trade)
        
        assert portfolio.current_capital == Decimal("10009490")  # 9749750 + 259740
        assert len(portfolio.positions) == 0
    
    def test_partial_sell_trade(self):
        """部分売却"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        # まず買いポジションを作成
        position = Position(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            quantity=200,
            entry_price=Decimal("2500")
        )
        portfolio.add_position(position)
        portfolio.current_capital = Decimal("9499500")
        
        # 部分売却
        trade = Trade(
            portfolio_id=portfolio.id,
            ticker="7203.T",
            side="SELL",
            quantity=100,
            price=Decimal("2600"),
            commission=Decimal("260")
        )
        
        portfolio.execute_trade(trade)
        
        assert portfolio.current_capital == Decimal("9759240")  # 9499500 + 259740
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].quantity == 100
    
    def test_portfolio_to_dict(self):
        """辞書形式への変換"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        data = portfolio.to_dict()
        
        assert data["name"] == "Test Portfolio"
        assert data["initial_capital"] == "10000000"
        assert data["current_capital"] == "10000000"
        assert data["is_active"] is True
        assert len(data["positions"]) == 0