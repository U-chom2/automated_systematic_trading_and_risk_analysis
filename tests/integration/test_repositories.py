"""リポジトリの統合テスト"""
import pytest
from decimal import Decimal
from datetime import datetime
from uuid import uuid4

from src.infrastructure.repositories.portfolio_repository_impl import PortfolioRepositoryImpl
from src.infrastructure.repositories.trade_repository_impl import TradeRepositoryImpl
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.trade import Trade


@pytest.mark.integration
class TestPortfolioRepository:
    """ポートフォリオリポジトリの統合テスト"""
    
    @pytest.mark.asyncio
    async def test_save_and_find_portfolio(self, db_connection):
        """ポートフォリオの保存と取得"""
        repo = PortfolioRepositoryImpl()
        repo.db = db_connection
        
        # ポートフォリオを作成
        portfolio = Portfolio(
            name="Integration Test Portfolio",
            initial_capital=Decimal("10000000")
        )
        
        # 保存
        await repo.save(portfolio)
        
        # 取得
        retrieved = await repo.find_by_id(portfolio.id)
        
        # 検証
        assert retrieved is not None
        assert retrieved.id == portfolio.id
        assert retrieved.name == "Integration Test Portfolio"
        assert retrieved.initial_capital == Decimal("10000000")
    
    @pytest.mark.asyncio
    async def test_update_portfolio(self, db_connection):
        """ポートフォリオの更新"""
        repo = PortfolioRepositoryImpl()
        repo.db = db_connection
        
        # ポートフォリオを作成・保存
        portfolio = Portfolio(
            name="Original Name",
            initial_capital=Decimal("10000000")
        )
        await repo.save(portfolio)
        
        # 更新
        portfolio.name = "Updated Name"
        portfolio.current_capital = Decimal("11000000")
        await repo.update(portfolio)
        
        # 取得して検証
        retrieved = await repo.find_by_id(portfolio.id)
        assert retrieved.name == "Updated Name"
        assert retrieved.current_capital == Decimal("11000000")
    
    @pytest.mark.asyncio
    async def test_delete_portfolio(self, db_connection):
        """ポートフォリオの削除"""
        repo = PortfolioRepositoryImpl()
        repo.db = db_connection
        
        # ポートフォリオを作成・保存
        portfolio = Portfolio(
            name="To Be Deleted",
            initial_capital=Decimal("10000000")
        )
        await repo.save(portfolio)
        
        # 削除
        await repo.delete(portfolio.id)
        
        # 取得して検証
        retrieved = await repo.find_by_id(portfolio.id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_find_all_portfolios(self, db_connection):
        """全ポートフォリオの取得"""
        repo = PortfolioRepositoryImpl()
        repo.db = db_connection
        
        # 複数のポートフォリオを作成
        portfolios = []
        for i in range(3):
            portfolio = Portfolio(
                name=f"Portfolio {i}",
                initial_capital=Decimal("10000000")
            )
            await repo.save(portfolio)
            portfolios.append(portfolio)
        
        # 全件取得
        all_portfolios = await repo.find_all()
        
        # 検証
        assert len(all_portfolios) >= 3
        saved_ids = {p.id for p in portfolios}
        retrieved_ids = {p.id for p in all_portfolios}
        assert saved_ids.issubset(retrieved_ids)


@pytest.mark.integration
class TestTradeRepository:
    """取引リポジトリの統合テスト"""
    
    @pytest.mark.asyncio
    async def test_save_and_find_trade(self, db_connection):
        """取引の保存と取得"""
        repo = TradeRepositoryImpl()
        repo.db = db_connection
        
        # 取引を作成
        trade = Trade(
            portfolio_id=uuid4(),
            ticker="7203.T",
            side="BUY",
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250")
        )
        
        # 保存
        await repo.save(trade)
        
        # 取得
        retrieved = await repo.find_by_id(trade.id)
        
        # 検証
        assert retrieved is not None
        assert retrieved.id == trade.id
        assert retrieved.ticker == "7203.T"
        assert retrieved.quantity == 100
        assert retrieved.price == Decimal("2500")
    
    @pytest.mark.asyncio
    async def test_find_trades_by_portfolio(self, db_connection):
        """ポートフォリオIDによる取引の取得"""
        repo = TradeRepositoryImpl()
        repo.db = db_connection
        
        portfolio_id = uuid4()
        
        # 複数の取引を作成
        trades = []
        for i in range(3):
            trade = Trade(
                portfolio_id=portfolio_id,
                ticker=f"700{i}.T",
                side="BUY",
                quantity=100,
                price=Decimal("2500"),
                commission=Decimal("250")
            )
            await repo.save(trade)
            trades.append(trade)
        
        # ポートフォリオIDで取得
        portfolio_trades = await repo.find_by_portfolio_id(portfolio_id)
        
        # 検証
        assert len(portfolio_trades) == 3
        saved_ids = {t.id for t in trades}
        retrieved_ids = {t.id for t in portfolio_trades}
        assert saved_ids == retrieved_ids
    
    @pytest.mark.asyncio
    async def test_find_trades_by_date_range(self, db_connection):
        """日付範囲による取引の取得"""
        repo = TradeRepositoryImpl()
        repo.db = db_connection
        
        portfolio_id = uuid4()
        now = datetime.utcnow()
        
        # 異なる日時の取引を作成
        trade1 = Trade(
            portfolio_id=portfolio_id,
            ticker="7203.T",
            side="BUY",
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250"),
            executed_at=now - timedelta(days=5)
        )
        
        trade2 = Trade(
            portfolio_id=portfolio_id,
            ticker="9984.T",
            side="BUY",
            quantity=100,
            price=Decimal("5000"),
            commission=Decimal("500"),
            executed_at=now - timedelta(days=2)
        )
        
        trade3 = Trade(
            portfolio_id=portfolio_id,
            ticker="6758.T",
            side="SELL",
            quantity=100,
            price=Decimal("3000"),
            commission=Decimal("300"),
            executed_at=now
        )
        
        await repo.save(trade1)
        await repo.save(trade2)
        await repo.save(trade3)
        
        # 過去3日間の取引を取得
        start_date = now - timedelta(days=3)
        end_date = now
        recent_trades = await repo.find_by_date_range(
            portfolio_id, start_date, end_date
        )
        
        # 検証
        assert len(recent_trades) == 2
        trade_ids = {t.id for t in recent_trades}
        assert trade2.id in trade_ids
        assert trade3.id in trade_ids
        assert trade1.id not in trade_ids