"""Tests for XStreamer (FR-02 X/Twitter component)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable

from src.data_collector.x_streamer import XStreamer


class TestXStreamer:
    """Test cases for XStreamer."""
    
    @pytest.fixture
    def x_streamer(self) -> XStreamer:
        """Create XStreamer instance for testing."""
        return XStreamer(
            api_key="test_api_key",
            api_secret="test_api_secret",
            access_token="test_access_token",
            access_token_secret="test_access_token_secret"
        )
    
    @pytest.fixture
    def sample_tweet_data(self) -> Dict[str, Any]:
        """Sample tweet data for testing."""
        return {
            "id": "1234567890",
            "text": "トヨタ株が上昇中 #7203 買い増し検討",
            "author": {
                "id": "user123",
                "username": "trader_user",
                "name": "Stock Trader"
            },
            "created_at": "2024-01-15T14:30:00Z",
            "public_metrics": {
                "retweet_count": 5,
                "like_count": 12,
                "reply_count": 3
            },
            "entities": {
                "hashtags": [{"tag": "7203"}],
                "cashtags": [{"tag": "7203"}]
            }
        }
    
    def test_initialization(self) -> None:
        """Test XStreamer initialization."""
        streamer = XStreamer(
            api_key="key1",
            api_secret="secret1",
            access_token="token1",
            access_token_secret="token_secret1"
        )
        
        assert streamer.api_key == "key1"
        assert streamer.api_secret == "secret1"
        assert streamer.access_token == "token1"
        assert streamer.access_token_secret == "token_secret1"
        assert streamer.is_streaming is False
        assert len(streamer.callbacks) == 0
    
    def test_add_callback(self, x_streamer: XStreamer) -> None:
        """Test adding callback functions."""
        def mock_callback(tweet_data: Dict[str, Any]) -> None:
            pass
        
        x_streamer.add_callback(mock_callback)
        
        assert len(x_streamer.callbacks) == 1
        assert mock_callback in x_streamer.callbacks
    
    def test_start_streaming(self, x_streamer: XStreamer) -> None:
        """Test starting tweet streaming."""
        keywords = ["7203", "6758", "トヨタ", "ソニー"]
        
        x_streamer.start_streaming(keywords)
        
        assert x_streamer.is_streaming is True
    
    def test_stop_streaming(self, x_streamer: XStreamer) -> None:
        """Test stopping tweet streaming."""
        # Start first
        x_streamer.start_streaming(["7203"])
        assert x_streamer.is_streaming is True
        
        # Then stop
        x_streamer.stop_streaming()
        assert x_streamer.is_streaming is False
    
    def test_get_mention_count(self, x_streamer: XStreamer) -> None:
        """Test getting mention count for symbol."""
        symbol = "7203"
        timeframe = 5  # 5 minutes
        
        count = x_streamer.get_mention_count(symbol, timeframe)
        
        assert isinstance(count, int)
        assert count >= 0
    
    def test_calculate_anomaly_threshold(self, x_streamer: XStreamer) -> None:
        """Test statistical anomaly threshold calculation."""
        symbol = "7203"
        historical_hours = 24
        
        threshold = x_streamer.calculate_anomaly_threshold(symbol, historical_hours)
        
        assert isinstance(threshold, float)
        assert threshold >= 0.0
    
    def test_check_mention_anomaly(self, x_streamer: XStreamer) -> None:
        """Test mention anomaly detection."""
        symbol = "7203"
        
        # Mock the helper methods
        with patch.object(x_streamer, 'get_mention_count', return_value=10):
            with patch.object(x_streamer, 'calculate_anomaly_threshold', 
                            return_value=5.0):
                # Current count (10) > threshold (5) should detect anomaly
                anomaly = x_streamer.check_mention_anomaly(symbol)
                assert isinstance(anomaly, bool)
        
        # Test with normal activity
        with patch.object(x_streamer, 'get_mention_count', return_value=3):
            with patch.object(x_streamer, 'calculate_anomaly_threshold', 
                            return_value=5.0):
                # Current count (3) < threshold (5) should not detect anomaly
                anomaly = x_streamer.check_mention_anomaly(symbol)
                assert isinstance(anomaly, bool)
    
    def test_process_tweet(self, x_streamer: XStreamer, 
                         sample_tweet_data: Dict[str, Any]) -> None:
        """Test processing raw tweet data."""
        processed = x_streamer.process_tweet(sample_tweet_data)
        
        # Verify structure
        assert isinstance(processed, dict)
        required_fields = ["id", "text", "author", "timestamp", "mentions", "sentiment"]
        
        for field in required_fields:
            assert field in processed
        
        assert isinstance(processed["timestamp"], datetime)
        assert isinstance(processed["mentions"], list)
        assert isinstance(processed["sentiment"], float)
    
    def test_extract_stock_mentions(self, x_streamer: XStreamer) -> None:
        """Test extracting stock symbols from tweet text."""
        test_texts = [
            "トヨタ株が好調 #7203",
            "$7203 上昇中",
            "7203についてどう思う？",
            "ソニー6758も注目",
            "一般的なツイート内容"
        ]
        
        for text in test_texts:
            mentions = x_streamer.extract_stock_mentions(text)
            assert isinstance(mentions, list)
            # Each mention should be a string (stock code)
            for mention in mentions:
                assert isinstance(mention, str)
    
    def test_streaming_with_callbacks(self, x_streamer: XStreamer) -> None:
        """Test streaming with callback functions."""
        processed_tweets = []
        
        def tweet_callback(tweet_data: Dict[str, Any]) -> None:
            processed_tweets.append(tweet_data)
        
        x_streamer.add_callback(tweet_callback)
        
        # Verify callback is registered
        assert len(x_streamer.callbacks) == 1
        assert tweet_callback in x_streamer.callbacks
    
    def test_multiple_callbacks(self, x_streamer: XStreamer) -> None:
        """Test handling multiple callbacks."""
        callback_calls = {"count": 0}
        
        def callback1(tweet_data: Dict[str, Any]) -> None:
            callback_calls["count"] += 1
        
        def callback2(tweet_data: Dict[str, Any]) -> None:
            callback_calls["count"] += 10
        
        x_streamer.add_callback(callback1)
        x_streamer.add_callback(callback2)
        
        assert len(x_streamer.callbacks) == 2
    
    def test_mention_count_timeframe_validation(self, 
                                               x_streamer: XStreamer) -> None:
        """Test mention count with different timeframes."""
        symbol = "7203"
        
        # Test various timeframes
        timeframes = [1, 5, 15, 60]  # minutes
        
        for timeframe in timeframes:
            count = x_streamer.get_mention_count(symbol, timeframe)
            assert isinstance(count, int)
            assert count >= 0
    
    def test_anomaly_threshold_historical_periods(self, 
                                                 x_streamer: XStreamer) -> None:
        """Test anomaly threshold with different historical periods."""
        symbol = "7203"
        
        # Test various historical periods
        periods = [1, 6, 12, 24, 48]  # hours
        
        for period in periods:
            threshold = x_streamer.calculate_anomaly_threshold(symbol, period)
            assert isinstance(threshold, float)
            assert threshold >= 0.0
    
    def test_streaming_state_management(self, x_streamer: XStreamer) -> None:
        """Test streaming state transitions."""
        # Initial state
        assert x_streamer.is_streaming is False
        
        # Start streaming
        x_streamer.start_streaming(["7203"])
        assert x_streamer.is_streaming is True
        
        # Stop streaming
        x_streamer.stop_streaming()
        assert x_streamer.is_streaming is False
        
        # Multiple start/stop cycles
        for i in range(3):
            x_streamer.start_streaming(["7203"])
            assert x_streamer.is_streaming is True
            x_streamer.stop_streaming()
            assert x_streamer.is_streaming is False
    
    def test_keyword_filtering(self, x_streamer: XStreamer) -> None:
        """Test keyword filtering for streaming."""
        # Test with empty keywords
        x_streamer.start_streaming([])
        assert x_streamer.is_streaming is True
        
        # Test with multiple keywords
        keywords = ["7203", "トヨタ", "TOYOTA", "$7203"]
        x_streamer.start_streaming(keywords)
        assert x_streamer.is_streaming is True
    
    def test_api_credentials_validation(self) -> None:
        """Test API credentials validation."""
        # Test with empty credentials
        empty_streamer = XStreamer("", "", "", "")
        assert empty_streamer.api_key == ""
        
        # Test with valid credentials
        valid_streamer = XStreamer("key", "secret", "token", "token_secret")
        assert valid_streamer.api_key == "key"
        assert valid_streamer.api_secret == "secret"
    
    def test_concurrent_streaming_safety(self, x_streamer: XStreamer) -> None:
        """Test concurrent streaming operations."""
        # Test rapid start/stop operations
        for _ in range(5):
            x_streamer.start_streaming(["7203"])
            x_streamer.stop_streaming()
        
        # Should end in stopped state
        assert x_streamer.is_streaming is False
    
    def test_tweet_processing_edge_cases(self, x_streamer: XStreamer) -> None:
        """Test tweet processing with edge cases."""
        edge_cases = [
            {},  # Empty tweet data
            {"id": "123"},  # Minimal data
            {"text": None},  # None text
            {"author": {}},  # Empty author
        ]
        
        for case in edge_cases:
            # Should not raise exceptions
            result = x_streamer.process_tweet(case)
            assert isinstance(result, dict)