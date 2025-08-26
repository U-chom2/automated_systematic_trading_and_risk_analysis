"""Test cases for scheduled execution (16:15 daily analysis)."""

import pytest
import asyncio
from datetime import datetime, time, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch, AsyncMock

from src.system_core.workflow_manager import (
    WorkflowManager, TradingPhase, WorkflowState,
    DailyDataCollectionResult, DailyAnalysisResult, NextDayExecutionPlan
)
from src.data_collector.tdnet_scraper import TdnetScraper
from src.data_collector.price_fetcher import PriceFetcher
from src.data_collector.x_streamer import XStreamer
from src.analysis_engine.nlp_analyzer import NlpAnalyzer
from src.analysis_engine.technical_analyzer import TechnicalAnalyzer
from src.execution_manager.order_manager import OrderManager


class TestScheduledExecution:
    """Test scheduled execution functionality."""
    
    @pytest.fixture
    def workflow_manager(self):
        """Create WorkflowManager instance."""
        config = {
            "buy_threshold": 80,
            "rsi_filter_threshold": 75,
            "ma_deviation_filter_threshold": 25,
            "capital": Decimal("1000000"),
            "risk_per_trade_ratio": 0.01,
            "max_concurrent_positions": 5,
            "watchlist": ["7203", "9984", "6758"]
        }
        return WorkflowManager(config)
    
    @pytest.mark.asyncio
    async def test_daily_analysis_flow(self, workflow_manager):
        """Test complete daily analysis flow at 16:15."""
        # Verify initial state
        assert workflow_manager.current_phase == TradingPhase.IDLE
        assert workflow_manager.state == WorkflowState.WAITING
        
        # Run daily analysis
        result = await workflow_manager.run_daily_analysis()
        
        # Verify execution
        assert result is True
        assert workflow_manager.current_phase == TradingPhase.PENDING_EXECUTION
        assert workflow_manager.next_day_plan is not None
        assert workflow_manager.metrics["total_days_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_data_collection_phase(self, workflow_manager):
        """Test data collection phase."""
        collection_result = await workflow_manager.collect_daily_data()
        
        assert collection_result.success is True
        assert len(collection_result.symbols) > 0
        assert collection_result.collection_date is not None
        assert isinstance(collection_result.ir_announcements, list)
        assert isinstance(collection_result.sns_mentions, dict)
        assert isinstance(collection_result.market_data, dict)
    
    @pytest.mark.asyncio
    async def test_analysis_phase(self, workflow_manager):
        """Test analysis phase with collected data."""
        # Prepare test data
        collection_result = DailyDataCollectionResult(
            collection_date=datetime.now(),
            symbols=["7203", "9984"],
            ir_announcements=[{
                "symbol": "7203",
                "title": "業績予想の上方修正に関するお知らせ",
                "content": "売上高を上方修正いたします",
                "has_trigger": True
            }],
            sns_mentions={"7203": 300, "9984": 150},
            market_data={
                "7203": {
                    "close_price": Decimal("1500.00"),
                    "volume": 1000000,
                    "rsi": 65.0,
                    "ma_deviation": 12.0,
                    "volatility": 0.25
                },
                "9984": {
                    "close_price": Decimal("2000.00"),
                    "volume": 800000,
                    "rsi": 70.0,
                    "ma_deviation": 15.0,
                    "volatility": 0.30
                }
            },
            success=True,
            errors=[]
        )
        
        # Run analysis
        analysis_results = await workflow_manager.analyze_collected_data(collection_result)
        
        # Verify results
        assert len(analysis_results) == 2
        for result in analysis_results:
            assert isinstance(result, DailyAnalysisResult)
            assert 0 <= result.catalyst_score <= 50
            assert 0 <= result.sentiment_score <= 30
            assert 0 <= result.market_env_score <= 20
            assert 0 <= result.total_score <= 100
            assert result.filter_passed in [True, False]
    
    @pytest.mark.asyncio
    async def test_trading_decision_phase(self, workflow_manager):
        """Test trading decision phase."""
        # Create analysis results
        analysis_results = [
            DailyAnalysisResult(
                analysis_date=datetime.now(),
                symbol="7203",
                catalyst_score=45,
                sentiment_score=25,
                market_env_score=15,
                total_score=85,
                buy_decision=True,
                filter_passed=True,
                filter_reasons=[],
                position_size=100,
                entry_price=Decimal("1500.00"),
                stop_loss_price=Decimal("1380.00"),
                take_profit_price=Decimal("1680.00")
            ),
            DailyAnalysisResult(
                analysis_date=datetime.now(),
                symbol="9984",
                catalyst_score=20,
                sentiment_score=15,
                market_env_score=10,
                total_score=45,
                buy_decision=False,
                filter_passed=True,
                filter_reasons=[]
            )
        ]
        
        # Make trading decisions
        execution_plan = await workflow_manager.make_trading_decisions(analysis_results)
        
        # Verify execution plan
        assert isinstance(execution_plan, NextDayExecutionPlan)
        assert len(execution_plan.buy_targets) == 1  # Only one met threshold
        assert execution_plan.buy_targets[0].symbol == "7203"
        assert execution_plan.risk_per_trade == Decimal("0.01")
    
    @pytest.mark.asyncio
    async def test_morning_execution(self, workflow_manager):
        """Test morning trade execution at 9:00 AM."""
        # Setup execution plan
        workflow_manager.current_phase = TradingPhase.PENDING_EXECUTION
        workflow_manager.next_day_plan = NextDayExecutionPlan(
            execution_date=datetime.now() + timedelta(days=1),
            buy_targets=[
                DailyAnalysisResult(
                    analysis_date=datetime.now(),
                    symbol="7203",
                    catalyst_score=45,
                    sentiment_score=25,
                    market_env_score=15,
                    total_score=85,
                    buy_decision=True,
                    filter_passed=True,
                    filter_reasons=[],
                    position_size=100,
                    entry_price=Decimal("1500.00"),
                    stop_loss_price=Decimal("1380.00"),
                    take_profit_price=Decimal("1680.00")
                )
            ],
            total_capital_allocation=Decimal("150000"),
            risk_per_trade=Decimal("0.01")
        )
        
        # Execute morning trades
        result = await workflow_manager.execute_morning_trades()
        
        # Verify execution
        assert result is True
        assert workflow_manager.current_phase == TradingPhase.COMPLETED
        assert workflow_manager.next_day_plan is None
    
    def test_scheduled_time_check(self, workflow_manager):
        """Test scheduled time checking."""
        # Mock current time to 16:15
        with patch('src.system_core.workflow_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.combine(
                datetime.today(), time(16, 15, 30)
            )
            mock_datetime.combine = datetime.combine
            mock_datetime.today = datetime.today
            
            assert workflow_manager.is_scheduled_time() is True
        
        # Mock current time to 10:00 (not scheduled time)
        with patch('src.system_core.workflow_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.combine(
                datetime.today(), time(10, 0, 0)
            )
            mock_datetime.combine = datetime.combine
            mock_datetime.today = datetime.today
            
            assert workflow_manager.is_scheduled_time() is False
    
    def test_execution_time_check(self, workflow_manager):
        """Test morning execution time checking."""
        # Mock current time to 9:00
        with patch('src.system_core.workflow_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.combine(
                datetime.today(), time(9, 0, 30)
            )
            mock_datetime.combine = datetime.combine
            mock_datetime.today = datetime.today
            
            assert workflow_manager.is_execution_time() is True
        
        # Mock current time to 16:15 (not execution time)
        with patch('src.system_core.workflow_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.combine(
                datetime.today(), time(16, 15, 0)
            )
            mock_datetime.combine = datetime.combine
            mock_datetime.today = datetime.today
            
            assert workflow_manager.is_execution_time() is False
    
    @pytest.mark.asyncio
    async def test_risk_filter_application(self, workflow_manager):
        """Test risk filter application in analysis."""
        collection_result = DailyDataCollectionResult(
            collection_date=datetime.now(),
            symbols=["7203"],
            ir_announcements=[],
            sns_mentions={"7203": 150},
            market_data={
                "7203": {
                    "close_price": Decimal("1500.00"),
                    "volume": 1000000,
                    "rsi": 80.0,  # Overheated
                    "ma_deviation": 30.0,  # Overheated
                    "volatility": 0.25
                }
            },
            success=True,
            errors=[]
        )
        
        # Run analysis
        analysis_results = await workflow_manager.analyze_collected_data(collection_result)
        
        # Verify filter applied
        assert len(analysis_results) == 1
        result = analysis_results[0]
        assert result.filter_passed is False
        assert "rsi_overheated" in result.filter_reasons
        assert "ma_deviation_overheated" in result.filter_reasons
        assert result.buy_decision is False
    
    def test_workflow_metrics(self, workflow_manager):
        """Test workflow metrics tracking."""
        metrics = workflow_manager.get_workflow_metrics()
        
        assert "total_days_processed" in metrics
        assert "successful_analyses" in metrics
        assert "failed_analyses" in metrics
        assert "average_processing_time_minutes" in metrics
        assert "total_buy_decisions" in metrics
        assert "current_phase" in metrics
        assert "current_state" in metrics
        assert "scheduled_time" in metrics
        assert "execution_time" in metrics
        assert "has_pending_plan" in metrics


class TestDataCollectorBatch:
    """Test data collector batch processing."""
    
    @pytest.mark.asyncio
    async def test_tdnet_daily_collection(self):
        """Test TDnet daily data collection."""
        scraper = TdnetScraper()
        target_date = datetime.now()
        
        # Collect daily releases
        releases = await scraper.collect_daily_releases(target_date)
        
        assert isinstance(releases, list)
        # Verify each release has required fields
        for release in releases:
            assert "has_trigger" in release
            assert "importance" in release
    
    @pytest.mark.asyncio
    async def test_price_fetcher_daily_collection(self):
        """Test price fetcher daily data collection."""
        fetcher = PriceFetcher()
        symbols = ["7203", "9984", "6758"]
        
        # Collect daily market data
        market_data = await fetcher.collect_daily_market_data(symbols)
        
        assert isinstance(market_data, dict)
        assert len(market_data) == len(symbols)
        
        for symbol, data in market_data.items():
            assert "close_price" in data
            assert "volume" in data
            assert "date" in data
    
    @pytest.mark.asyncio
    async def test_x_streamer_daily_collection(self):
        """Test X (Twitter) daily mention collection."""
        streamer = XStreamer()
        symbols = ["7203", "9984"]
        
        # Collect daily mentions
        mention_stats = await streamer.collect_daily_mentions(symbols)
        
        assert isinstance(mention_stats, dict)
        
        for symbol, stats in mention_stats.items():
            assert "total_mentions" in stats
            assert "positive" in stats
            assert "negative" in stats
            assert "sentiment_ratio" in stats
            assert "is_anomaly" in stats


class TestAnalysisEngineBatch:
    """Test analysis engine batch processing."""
    
    @pytest.mark.asyncio
    async def test_nlp_daily_catalyst_analysis(self):
        """Test NLP daily catalyst analysis."""
        analyzer = NlpAnalyzer()
        
        ir_data = [{
            "title": "業績予想の上方修正に関するお知らせ",
            "content": "売上高を上方修正いたします",
            "has_trigger": True
        }]
        
        sns_data = {
            "total_mentions": 300,
            "sentiment_ratio": 0.7,
            "is_anomaly": True
        }
        
        board_data = {
            "sentiment": {
                "positive_ratio": 0.65
            }
        }
        
        # Perform daily catalyst analysis
        result = await analyzer.analyze_daily_catalyst(ir_data, sns_data, board_data)
        
        assert "catalyst_score" in result
        assert 0 <= result["catalyst_score"] <= 50
        assert "ir_score" in result
        assert "sns_score" in result
        assert "board_score" in result
    
    def test_technical_daily_analysis(self):
        """Test technical daily analysis."""
        analyzer = TechnicalAnalyzer()
        
        market_data = {
            "close_price": Decimal("1500.00"),
            "volume": 1000000,
            "rsi": 65.0,
            "ma_deviation": 12.0,
            "volatility": 0.25,
            "avg_volume": 900000
        }
        
        # Perform daily technical analysis
        result = analyzer.analyze_daily_technicals(market_data)
        
        assert "technical_score" in result
        assert 0 <= result["technical_score"] <= 20
        assert "filter_passed" in result
        assert "filter_reasons" in result


class TestExecutionManagerBatch:
    """Test execution manager batch processing."""
    
    @pytest.mark.asyncio
    async def test_morning_batch_execution(self):
        """Test morning batch order execution."""
        manager = OrderManager(paper_trading=True)
        
        execution_plan = {
            "buy_targets": [
                {
                    "symbol": "7203",
                    "position_size": 100,
                    "entry_price": Decimal("1500.00"),
                    "stop_loss_price": Decimal("1380.00"),
                    "take_profit_price": Decimal("1680.00")
                },
                {
                    "symbol": "9984",
                    "position_size": 50,
                    "entry_price": Decimal("2000.00"),
                    "stop_loss_price": Decimal("1900.00"),
                    "take_profit_price": Decimal("2200.00")
                }
            ]
        }
        
        # Execute morning batch orders
        results = await manager.execute_morning_batch_orders(execution_plan)
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result in results:
            assert "symbol" in result
            assert "success" in result
            if result["success"]:
                assert "buy_order" in result
                assert "oco_order" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])