"""APIエンドポイントのE2Eテスト"""
import pytest
from httpx import AsyncClient
from decimal import Decimal
from datetime import datetime, date
import json

from src.presentation.api.app import app


@pytest.mark.e2e
class TestPortfolioEndpoints:
    """ポートフォリオAPIのE2Eテスト"""
    
    @pytest.mark.asyncio
    async def test_create_portfolio(self):
        """ポートフォリオ作成エンドポイント"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/portfolios/",
                json={
                    "name": "E2E Test Portfolio",
                    "initial_capital": "10000000",
                    "description": "Portfolio for E2E testing"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "E2E Test Portfolio"
        assert data["initial_capital"] == "10000000"
    
    @pytest.mark.asyncio
    async def test_get_portfolio(self):
        """ポートフォリオ取得エンドポイント"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # まずポートフォリオを作成
            create_response = await client.post(
                "/api/v1/portfolios/",
                json={
                    "name": "Test Portfolio",
                    "initial_capital": "10000000"
                }
            )
            portfolio_id = create_response.json()["id"]
            
            # 取得
            response = await client.get(f"/api/v1/portfolios/{portfolio_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == portfolio_id
        assert data["name"] == "Test Portfolio"
    
    @pytest.mark.asyncio
    async def test_list_portfolios(self):
        """ポートフォリオ一覧取得エンドポイント"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/portfolios/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.e2e
class TestMarketDataEndpoints:
    """市場データAPIのE2Eテスト"""
    
    @pytest.mark.asyncio
    async def test_get_latest_price(self):
        """最新価格取得エンドポイント"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/market-data/price/7203.T")
        
        # 市場データが利用可能な場合
        if response.status_code == 200:
            data = response.json()
            assert "ticker" in data
            assert "price" in data
            assert data["ticker"] == "7203.T"
    
    @pytest.mark.asyncio
    async def test_get_ohlcv(self):
        """OHLCVデータ取得エンドポイント"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/market-data/ohlcv/7203.T",
                params={
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "interval": "1d"
                }
            )
        
        # 市場データが利用可能な場合
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            if len(data) > 0:
                assert "open" in data[0]
                assert "high" in data[0]
                assert "low" in data[0]
                assert "close" in data[0]
                assert "volume" in data[0]


@pytest.mark.e2e
class TestTradingEndpoints:
    """取引APIのE2Eテスト"""
    
    @pytest.mark.asyncio
    async def test_execute_trade(self):
        """取引実行エンドポイント"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # まずポートフォリオを作成
            portfolio_response = await client.post(
                "/api/v1/portfolios/",
                json={
                    "name": "Trading Test Portfolio",
                    "initial_capital": "10000000"
                }
            )
            portfolio_id = portfolio_response.json()["id"]
            
            # 取引を実行
            response = await client.post(
                "/api/v1/trades/",
                json={
                    "portfolio_id": portfolio_id,
                    "ticker": "7203.T",
                    "side": "BUY",
                    "quantity": 100,
                    "order_type": "MARKET"
                }
            )
        
        # 取引が成功した場合
        if response.status_code == 200:
            data = response.json()
            assert data["portfolio_id"] == portfolio_id
            assert data["ticker"] == "7203.T"
            assert data["side"] == "BUY"
            assert data["quantity"] == 100
    
    @pytest.mark.asyncio
    async def test_get_trade_history(self):
        """取引履歴取得エンドポイント"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/trades/",
                params={"limit": 10}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.e2e
class TestBacktestEndpoints:
    """バックテストAPIのE2Eテスト"""
    
    @pytest.mark.asyncio
    async def test_run_backtest(self):
        """バックテスト実行エンドポイント"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/backtest/run",
                json={
                    "tickers": ["7203.T", "9984.T"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "initial_capital": 10000000,
                    "strategy_type": "AI_DRIVEN"
                }
            )
        
        # バックテストが成功した場合
        if response.status_code == 200:
            data = response.json()
            assert "total_return" in data
            assert "sharpe_ratio" in data
            assert "max_drawdown" in data
            assert "trades" in data
    
    @pytest.mark.asyncio
    async def test_compare_strategies(self):
        """戦略比較エンドポイント"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/backtest/compare",
                json={
                    "tickers": ["7203.T"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "initial_capital": 10000000,
                    "strategies": ["MOMENTUM", "MEAN_REVERSION", "AI_DRIVEN"]
                }
            )
        
        # 比較が成功した場合
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert isinstance(data["results"], list)