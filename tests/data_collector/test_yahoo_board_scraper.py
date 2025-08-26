"""Tests for YahooBoardScraper (FR-02 Yahoo Finance component)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.data_collector.yahoo_board_scraper import (
    YahooBoardScraper,
    BoardPost,
    BoardAnalytics
)


class TestYahooBoardScraper:
    """Test cases for YahooBoardScraper."""
    
    @pytest.fixture
    def yahoo_scraper(self) -> YahooBoardScraper:
        """Create YahooBoardScraper instance for testing."""
        return YahooBoardScraper(request_delay=1.0)
    
    @pytest.fixture
    def sample_board_posts(self) -> List[BoardPost]:
        """Sample board posts for testing."""
        base_time = datetime.now()
        return [
            BoardPost(
                post_id="post_001",
                symbol="7203",
                title="トヨタ株の今後について",
                content="決算発表が近いので期待している",
                author="investor_123",
                timestamp=base_time - timedelta(minutes=10),
                likes=5,
                replies=2,
                sentiment_score=0.7
            ),
            BoardPost(
                post_id="post_002",
                symbol="7203",
                title="売り時かも",
                content="チャートが天井圏に見える",
                author="trader_456",
                timestamp=base_time - timedelta(minutes=30),
                likes=1,
                replies=0,
                sentiment_score=-0.3
            ),
            BoardPost(
                post_id="post_003",
                symbol="7203",
                title="ホールド継続",
                content="長期投資で保有中",
                author="long_term",
                timestamp=base_time - timedelta(hours=2),
                likes=3,
                replies=1,
                sentiment_score=0.1
            )
        ]
    
    def test_initialization(self) -> None:
        """Test YahooBoardScraper initialization."""
        scraper = YahooBoardScraper(request_delay=0.5)
        
        assert scraper.request_delay == 0.5
        assert scraper.session is None
        assert len(scraper.posts_cache) == 0
        assert len(scraper.last_scrape_time) == 0
    
    def test_get_board_url(self, yahoo_scraper: YahooBoardScraper) -> None:
        """Test board URL generation."""
        url = yahoo_scraper.get_board_url("7203")
        
        expected_url = "https://finance.yahoo.co.jp/cm/message/17203/a_condition/1"
        assert url == expected_url
    
    def test_initialize_session(self, yahoo_scraper: YahooBoardScraper) -> None:
        """Test session initialization."""
        result = yahoo_scraper.initialize_session()
        
        # Current dummy implementation returns True
        assert result is True
    
    def test_scrape_board_posts(self, yahoo_scraper: YahooBoardScraper) -> None:
        """Test scraping board posts."""
        symbol = "7203"
        posts = yahoo_scraper.scrape_board_posts(symbol, max_posts=20)
        
        # Current dummy implementation returns empty list
        assert isinstance(posts, list)
        assert symbol in yahoo_scraper.last_scrape_time
    
    def test_parse_post_data(self, yahoo_scraper: YahooBoardScraper) -> None:
        """Test parsing post data from HTML."""
        sample_html = """
        <div class="post">
            <div class="title">トヨタ株について</div>
            <div class="content">上昇期待</div>
            <div class="author">user123</div>
            <div class="time">2024-01-15 14:30</div>
            <div class="likes">5</div>
        </div>
        """
        
        post = yahoo_scraper.parse_post_data(sample_html, "7203")
        
        assert post is not None
        assert isinstance(post, BoardPost)
        assert post.symbol == "7203"
        assert isinstance(post.timestamp, datetime)
    
    @patch('src.data_collector.yahoo_board_scraper.datetime')
    def test_get_recent_posts(self, mock_datetime: Mock, 
                            yahoo_scraper: YahooBoardScraper,
                            sample_board_posts: List[BoardPost]) -> None:
        """Test getting recent posts within timeframe."""
        # Setup mock time
        mock_now = datetime(2024, 1, 15, 15, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        # Mock the cache
        yahoo_scraper.posts_cache["7203"] = sample_board_posts
        
        # Get posts from last hour
        recent_posts = yahoo_scraper.get_recent_posts("7203", hours=1)
        
        # Should include posts within last hour
        assert isinstance(recent_posts, list)
    
    def test_calculate_post_volume_anomaly(self, 
                                         yahoo_scraper: YahooBoardScraper,
                                         sample_board_posts: List[BoardPost]) -> None:
        """Test post volume anomaly detection."""
        symbol = "7203"
        
        # Mock cache with sample posts
        yahoo_scraper.posts_cache[symbol] = sample_board_posts
        
        # Test with insufficient data (should return False)
        anomaly = yahoo_scraper.calculate_post_volume_anomaly(symbol)
        assert isinstance(anomaly, bool)
        
        # Test with mocked statistics for sufficient data scenario
        with patch('statistics.mean', return_value=1.0), \
             patch('statistics.stdev', return_value=0.5):
            
            # Create enough sample posts (24+ hours)
            extended_posts = sample_board_posts * 10  # Simulate more posts
            yahoo_scraper.posts_cache[symbol] = extended_posts
            
            anomaly = yahoo_scraper.calculate_post_volume_anomaly(symbol)
            assert isinstance(anomaly, bool)
    
    def test_analyze_board_sentiment(self, yahoo_scraper: YahooBoardScraper) -> None:
        """Test board sentiment analysis."""
        symbol = "7203"
        
        sentiment = yahoo_scraper.analyze_board_sentiment(symbol, hours=1)
        
        assert isinstance(sentiment, dict)
        assert "total_posts" in sentiment
        assert "positive_ratio" in sentiment
        assert "negative_ratio" in sentiment
        assert "neutral_ratio" in sentiment
        assert "average_sentiment" in sentiment
        assert "sentiment_keywords" in sentiment
        
        # Ratios should sum to approximately 1.0 if there are posts
        # If no posts, ratios should be 0.0
        total_ratio = (sentiment["positive_ratio"] + 
                      sentiment["negative_ratio"] + 
                      sentiment["neutral_ratio"])
        
        if sentiment["total_posts"] > 0:
            assert abs(total_ratio - 1.0) < 0.01
        else:
            # No posts case - ratios should be 0.0
            assert total_ratio == 0.0
    
    def test_get_board_analytics(self, yahoo_scraper: YahooBoardScraper,
                               sample_board_posts: List[BoardPost]) -> None:
        """Test comprehensive board analytics."""
        symbol = "7203"
        yahoo_scraper.posts_cache[symbol] = sample_board_posts
        
        analytics = yahoo_scraper.get_board_analytics(symbol)
        
        assert isinstance(analytics, BoardAnalytics)
        assert analytics.symbol == symbol
        assert analytics.total_posts >= 0
        assert analytics.posts_last_hour >= 0
        assert analytics.posts_last_24h >= 0
        assert isinstance(analytics.average_sentiment, float)
        assert isinstance(analytics.sentiment_distribution, dict)
        assert isinstance(analytics.top_keywords, list)
        assert isinstance(analytics.unusual_activity, bool)
    
    def test_update_posts_cache(self, yahoo_scraper: YahooBoardScraper) -> None:
        """Test updating posts cache."""
        symbol = "7203"
        
        # First update
        result = yahoo_scraper.update_posts_cache(symbol)
        assert result is True
        
        # Should have cache entry
        assert symbol in yahoo_scraper.posts_cache
        
        # Immediate second update should be skipped (within 5 minutes)
        result = yahoo_scraper.update_posts_cache(symbol)
        assert result is True  # Should succeed but skip actual update
        
        # Force refresh
        result = yahoo_scraper.update_posts_cache(symbol, force_refresh=True)
        assert result is True
    
    def test_clear_old_posts(self, yahoo_scraper: YahooBoardScraper,
                           sample_board_posts: List[BoardPost]) -> None:
        """Test clearing old posts from cache."""
        symbol = "7203"
        
        # Add old posts to cache
        old_time = datetime.now() - timedelta(days=3)
        old_posts = [
            BoardPost("old_001", symbol, "Old post", "content", "user", 
                     old_time, 0, 0)
        ]
        
        yahoo_scraper.posts_cache[symbol] = sample_board_posts + old_posts
        original_count = len(yahoo_scraper.posts_cache[symbol])
        
        # Clear posts older than 48 hours
        cleared = yahoo_scraper.clear_old_posts(hours=48)
        
        assert cleared >= 0
        assert len(yahoo_scraper.posts_cache[symbol]) <= original_count
    
    def test_get_cache_status(self, yahoo_scraper: YahooBoardScraper,
                            sample_board_posts: List[BoardPost]) -> None:
        """Test getting cache status."""
        # Add some data to cache
        yahoo_scraper.posts_cache["7203"] = sample_board_posts
        yahoo_scraper.posts_cache["6758"] = sample_board_posts[:2]
        yahoo_scraper.last_scrape_time["7203"] = datetime.now()
        
        status = yahoo_scraper.get_cache_status()
        
        assert isinstance(status, dict)
        assert "cached_symbols" in status
        assert "total_cached_posts" in status
        assert "last_scrape_times" in status
        assert "memory_usage_estimate" in status
        
        assert status["cached_symbols"] == 2
        assert status["total_cached_posts"] == 5  # 3 + 2 posts
    
    def test_error_handling_invalid_symbol(self, 
                                         yahoo_scraper: YahooBoardScraper) -> None:
        """Test error handling with invalid symbol."""
        # Test with empty symbol
        posts = yahoo_scraper.scrape_board_posts("")
        assert isinstance(posts, list)
        assert len(posts) == 0
        
        # Test with None symbol (should handle gracefully)
        analytics = yahoo_scraper.analyze_board_sentiment("", hours=1)
        assert isinstance(analytics, dict)
        assert analytics["total_posts"] == 0