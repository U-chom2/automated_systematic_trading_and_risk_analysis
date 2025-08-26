"""Tests for TechnicalAnalyzer."""

import pytest
from decimal import Decimal
from typing import List, Dict, Any

from src.analysis_engine.technical_analyzer import (
    TechnicalAnalyzer,
    TechnicalIndicators,
    MarketEnvironmentScore
)


class TestTechnicalAnalyzer:
    """Test cases for TechnicalAnalyzer."""
    
    @pytest.fixture
    def technical_analyzer(self) -> TechnicalAnalyzer:
        """Create TechnicalAnalyzer instance for testing."""
        return TechnicalAnalyzer()
    
    @pytest.fixture
    def sample_price_data(self) -> List[Dict[str, Any]]:
        """Create sample price data for testing."""
        return [
            {
                "symbol": "7203",
                "current_price": Decimal("1000"),
                "open_price": Decimal("980"),
                "high_price": Decimal("1020"),
                "low_price": Decimal("975"),
                "volume": 1000000,
                "timestamp": "2024-01-15T09:00:00"
            }
            # More sample data would be added here
        ]
    
    def test_rsi_calculation(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test RSI calculation."""
        prices = [Decimal(str(p)) for p in range(1000, 1015)]  # Increasing prices
        
        rsi = technical_analyzer.calculate_rsi(prices, period=14)
        
        assert 0 <= rsi <= 100
        assert isinstance(rsi, float)
    
    def test_moving_average_deviation_calculation(self, 
                                                technical_analyzer: TechnicalAnalyzer) -> None:
        """Test moving average deviation calculation."""
        current_price = Decimal("1100")
        prices = [Decimal("1000")] * 25  # Constant price history
        
        deviation = technical_analyzer.calculate_moving_average_deviation(
            current_price, prices, period=25
        )
        
        # Should be around 10% since current price is 1100 vs 1000 average
        assert isinstance(deviation, float)
    
    def test_volume_ratio_calculation(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test volume ratio calculation."""
        current_volume = 2000000
        historical_volumes = [1000000] * 20  # Average 1M volume
        
        ratio = technical_analyzer.calculate_volume_ratio(
            current_volume, historical_volumes, period=20
        )
        
        # Should be around 2.0 since current is 2M vs 1M average
        assert isinstance(ratio, float)
        assert ratio >= 0
    
    def test_atr_calculation(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test ATR calculation."""
        highs = [Decimal("1020")] * 14
        lows = [Decimal("980")] * 14
        closes = [Decimal("1000")] * 14
        
        atr = technical_analyzer.calculate_atr(highs, lows, closes, period=14)
        
        assert isinstance(atr, Decimal)
        assert atr >= Decimal("0")
    
    def test_bollinger_bands_calculation(self, 
                                       technical_analyzer: TechnicalAnalyzer) -> None:
        """Test Bollinger Bands calculation."""
        prices = [Decimal("1000")] * 20  # Constant prices for simplicity
        
        upper, middle, lower = technical_analyzer.calculate_bollinger_bands(
            prices, period=20, std_dev=2.0
        )
        
        assert isinstance(upper, float)
        assert isinstance(middle, float)
        assert isinstance(lower, float)
        # For constant prices, upper should equal middle should equal lower
    
    def test_macd_calculation(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test MACD calculation."""
        prices = [Decimal(str(1000 + i)) for i in range(30)]  # Trending up
        
        macd_line, signal_line, histogram = technical_analyzer.calculate_macd(
            prices, fast_period=12, slow_period=26, signal_period=9
        )
        
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)
    
    def test_get_technical_indicators(self, technical_analyzer: TechnicalAnalyzer,
                                    sample_price_data: List[Dict[str, Any]]) -> None:
        """Test comprehensive technical indicators calculation."""
        symbol = "7203"
        
        indicators = technical_analyzer.get_technical_indicators(symbol, sample_price_data)
        
        assert isinstance(indicators, TechnicalIndicators)
        assert hasattr(indicators, 'rsi')
        assert hasattr(indicators, 'moving_avg_deviation')
        assert hasattr(indicators, 'volume_ratio')
        assert hasattr(indicators, 'atr')
        assert hasattr(indicators, 'bollinger_upper')
        assert hasattr(indicators, 'bollinger_lower')
        assert hasattr(indicators, 'macd_line')
        assert hasattr(indicators, 'macd_signal')
    
    def test_analyze_market_environment(self, 
                                      technical_analyzer: TechnicalAnalyzer) -> None:
        """Test market environment analysis."""
        market_env = technical_analyzer.analyze_market_environment()
        
        assert isinstance(market_env, MarketEnvironmentScore)
        assert market_env.nikkei_trend in ["bullish", "bearish", "neutral"]
        assert market_env.topix_trend in ["bullish", "bearish", "neutral"]
        assert market_env.sector_trend in ["bullish", "bearish", "neutral"]
        assert 0 <= market_env.market_score <= 20
        assert market_env.risk_level in ["low", "medium", "high"]
    
    def test_overheating_filter(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """Test overheating filter conditions."""
        # Test with high RSI
        indicators_high_rsi = TechnicalIndicators(
            rsi=80.0,  # Above 75
            moving_avg_deviation=10.0,  # Below 25%
            volume_ratio=1.5,
            atr=0.05,
            bollinger_upper=1050.0,
            bollinger_lower=950.0,
            macd_line=5.0,
            macd_signal=3.0,
            stochastic_k=75.0,
            stochastic_d=70.0
        )
        
        assert technical_analyzer.check_overheating_filter(indicators_high_rsi)
        
        # Test with high moving average deviation
        indicators_high_ma_dev = TechnicalIndicators(
            rsi=60.0,  # Below 75
            moving_avg_deviation=30.0,  # Above 25%
            volume_ratio=1.5,
            atr=0.05,
            bollinger_upper=1050.0,
            bollinger_lower=950.0,
            macd_line=5.0,
            macd_signal=3.0,
            stochastic_k=75.0,
            stochastic_d=70.0
        )
        
        assert technical_analyzer.check_overheating_filter(indicators_high_ma_dev)
        
        # Test with normal conditions
        indicators_normal = TechnicalIndicators(
            rsi=60.0,  # Below 75
            moving_avg_deviation=15.0,  # Below 25%
            volume_ratio=1.5,
            atr=0.05,
            bollinger_upper=1050.0,
            bollinger_lower=950.0,
            macd_line=5.0,
            macd_signal=3.0,
            stochastic_k=75.0,
            stochastic_d=70.0
        )
        
        assert not technical_analyzer.check_overheating_filter(indicators_normal)
    
    def test_technical_score_for_trading(self, technical_analyzer: TechnicalAnalyzer,
                                       sample_price_data: List[Dict[str, Any]]) -> None:
        """Test technical score calculation for trading."""
        symbol = "7203"
        
        score = technical_analyzer.calculate_technical_score_for_trading(
            symbol, sample_price_data
        )
        
        assert isinstance(score, int)
        assert 0 <= score <= 20  # Market environment score range
    
    def test_technical_score_with_overheating(self, 
                                            technical_analyzer: TechnicalAnalyzer,
                                            sample_price_data: List[Dict[str, Any]]) -> None:
        """Test that overheating condition returns 0 score."""
        symbol = "7203"
        
        # Mock the overheating check to return True
        original_method = technical_analyzer.check_overheating_filter
        technical_analyzer.check_overheating_filter = lambda x: True
        
        score = technical_analyzer.calculate_technical_score_for_trading(
            symbol, sample_price_data
        )
        
        # Restore original method
        technical_analyzer.check_overheating_filter = original_method
        
        assert score == 0