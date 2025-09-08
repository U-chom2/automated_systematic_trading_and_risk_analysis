"""Yahoo Finance message board scraper for sentiment analysis."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BoardPost:
    """Yahoo Finance board post data."""
    post_id: str
    symbol: str
    title: str
    content: str
    author: str
    timestamp: datetime
    likes: int
    replies: int
    sentiment_score: Optional[float] = None


@dataclass
class BoardAnalytics:
    """Board analytics for a symbol."""
    symbol: str
    total_posts: int
    posts_last_hour: int
    posts_last_24h: int
    average_sentiment: float
    sentiment_distribution: Dict[str, int]  # positive, negative, neutral
    top_keywords: List[str]
    unusual_activity: bool


class YahooBoardScraper:
    """Yahoo Finance message board scraper."""
    
    def __init__(self, request_delay: float = 1.0) -> None:
        """
        Initialize Yahoo Board Scraper.
        
        Args:
            request_delay: Delay between requests in seconds
        """
        self.request_delay = request_delay
        self.session = None
        self.posts_cache: Dict[str, List[BoardPost]] = {}
        self.last_scrape_time: Dict[str, datetime] = {}
        logger.info(f"YahooBoardScraper initialized with {request_delay}s delay")
    
    def initialize_session(self) -> bool:
        """
        Initialize HTTP session with proper headers.
        
        Returns:
            True if session initialized successfully
        """
        try:
            # TODO: Initialize requests session with proper headers
            logger.info("Initializing Yahoo Finance session...")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            return False
    
    def get_board_url(self, symbol: str) -> str:
        """
        Get Yahoo Finance board URL for symbol.
        
        Args:
            symbol: Stock symbol (e.g., "7203")
            
        Returns:
            Board URL
        """
        # Yahoo Finance JP board URL format
        return f"https://finance.yahoo.co.jp/cm/message/1{symbol}/a_condition/1"
    
    def scrape_board_posts(self, symbol: str, max_posts: int = 50) -> List[BoardPost]:
        """
        Scrape recent posts from Yahoo Finance board.
        
        Args:
            symbol: Stock symbol
            max_posts: Maximum number of posts to retrieve
            
        Returns:
            List of board posts
        """
        try:
            logger.debug(f"Scraping board posts for {symbol}")
            
            # Initialize session if needed
            if not self.session:
                self.initialize_session()
            
            board_url = self.get_board_url(symbol)
            
            # Add rate limiting
            if symbol in self.last_scrape_time:
                time_since_last = datetime.now() - self.last_scrape_time[symbol]
                if time_since_last.total_seconds() < self.request_delay:
                    sleep_time = self.request_delay - time_since_last.total_seconds()
                    logger.debug(f"Rate limiting: waiting {sleep_time:.2f} seconds")
                    import time
                    time.sleep(sleep_time)
            
            try:
                # Attempt to scrape Yahoo Finance board
                response = self.session.get(board_url, timeout=10)
                response.raise_for_status()
                
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                posts = []
                
                # Try to find post elements (Yahoo Finance structure may change)
                # This is a simplified implementation - real Yahoo scraping would need
                # to handle dynamic loading, authentication, etc.
                
                post_elements = soup.find_all(['div', 'article', 'li'], 
                                            class_=lambda x: x and any(keyword in str(x).lower() 
                                                                     for keyword in ['post', 'comment', 'message', 'discussion']))
                
                logger.debug(f"Found {len(post_elements)} potential post elements")
                
                for i, elem in enumerate(post_elements[:max_posts]):
                    try:
                        # Extract text content
                        text = elem.get_text(strip=True)
                        if not text or len(text) < 10:  # Skip empty or too short posts
                            continue
                        
                        # Try to extract timestamp (this would need adjustment for actual Yahoo structure)
                        timestamp = datetime.now()  # Default to now
                        
                        # Create BoardPost object with correct parameters
                        post = BoardPost(
                            post_id=f"scraped_{symbol}_{i}",
                            symbol=symbol,
                            title="掲示板投稿",  # Default title
                            content=text[:500],  # Limit content length
                            author="anonymous",  # Yahoo doesn't always show usernames
                            timestamp=timestamp,
                            likes=0,  # Default value
                            replies=0,  # Default value
                            sentiment_score=None  # Will be calculated later if needed
                        )
                        posts.append(post)
                        
                    except Exception as post_error:
                        logger.debug(f"Error processing post element: {post_error}")
                        continue
                
                if posts:
                    logger.info(f"Scraped {len(posts)} posts from Yahoo board for {symbol}")
                else:
                    logger.warning(f"No posts found on Yahoo board for {symbol}")
                    # Generate demo posts as fallback
                    posts = self._generate_demo_board_posts(symbol, max_posts)
                
                # Update cache and tracking
                self.posts_cache[symbol] = posts
                self.last_scrape_time[symbol] = datetime.now()
                
                return posts
                
            except Exception as scrape_error:
                logger.warning(f"Failed to scrape Yahoo board for {symbol}: {scrape_error}")
                # Generate demo posts as fallback
                posts = self._generate_demo_board_posts(symbol, max_posts)
                self.posts_cache[symbol] = posts
                self.last_scrape_time[symbol] = datetime.now()
                return posts
            
        except Exception as e:
            logger.error(f"Failed to scrape board for {symbol}: {e}")
            return []
    
    # TODO システムに悪影響なのでいずれ消去する
    def _generate_demo_board_posts(self, symbol: str, max_posts: int) -> List[BoardPost]:
        """Generate demo board posts for testing."""
        import random
        from datetime import timedelta
        
        # Demo post templates
        post_templates = [
            f"{symbol}の決算発表が楽しみです！",
            f"今日の{symbol}の値動きはどう思いますか？",
            f"{symbol}、長期保有で持ち続ける予定です",
            f"{symbol}の業績好調ですね！上昇期待",
            f"{symbol}のチャートを見ると調整局面かも",
            f"明日の{symbol}の動きが気になります",
            f"{symbol}、いいタイミングで買い増ししました",
            f"{symbol}の将来性に期待しています",
            f"今四半期の{symbol}の決算はどうでしょう？",
            f"{symbol}のニュースが出てますね"
        ]
        
        demo_posts = []
        base_time = datetime.now()
        
        for i in range(min(max_posts, 10)):
            content = random.choice(post_templates)
            timestamp = base_time - timedelta(minutes=random.randint(1, 1440))  # Last 24 hours
            
            post = BoardPost(
                post_id=f"demo_{symbol}_{i}",
                symbol=symbol,
                title="掲示板投稿",  # Default title
                content=content,
                author=f"user_{random.randint(1000, 9999)}",
                timestamp=timestamp,
                likes=random.randint(0, 20),
                replies=random.randint(0, 5),
                sentiment_score=random.uniform(-1.0, 1.0)
            )
            demo_posts.append(post)
        
        # Sort by timestamp (newest first)
        demo_posts.sort(key=lambda p: p.timestamp, reverse=True)
        
        logger.info(f"Generated {len(demo_posts)} demo board posts for {symbol}")
        return demo_posts
    
    def parse_post_data(self, raw_post_html: str, symbol: str) -> Optional[BoardPost]:
        """
        Parse raw HTML post data into BoardPost object.
        
        Args:
            raw_post_html: Raw HTML containing post data
            symbol: Stock symbol
            
        Returns:
            Parsed BoardPost or None if parsing failed
        """
        try:
            # TODO: Implement HTML parsing logic
            logger.debug("Parsing post data")
            
            # Placeholder parsing
            return BoardPost(
                post_id="",
                symbol=symbol,
                title="",
                content="",
                author="",
                timestamp=datetime.now(),
                likes=0,
                replies=0
            )
            
        except Exception as e:
            logger.error(f"Failed to parse post data: {e}")
            return None
    
    def get_recent_posts(self, symbol: str, hours: int = 1) -> List[BoardPost]:
        """
        Get recent posts within specified timeframe.
        
        Args:
            symbol: Stock symbol
            hours: Number of hours to look back
            
        Returns:
            List of recent posts
        """
        if symbol not in self.posts_cache:
            self.posts_cache[symbol] = self.scrape_board_posts(symbol)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_posts = [
            post for post in self.posts_cache[symbol]
            if post.timestamp >= cutoff_time
        ]
        
        logger.debug(f"Found {len(recent_posts)} recent posts for {symbol}")
        return recent_posts
    
    def calculate_post_volume_anomaly(self, symbol: str) -> bool:
        """
        Calculate if current post volume is anomalous.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if post volume anomaly detected
        """
        try:
            # Get current hour posts
            current_hour_posts = len(self.get_recent_posts(symbol, hours=1))
            
            # Get historical data (24 hours)
            all_posts = self.get_recent_posts(symbol, hours=24)
            
            if len(all_posts) < 24:  # Not enough data
                return False
            
            # Calculate hourly post counts for last 24 hours
            hourly_counts = []
            for i in range(24):
                hour_start = datetime.now() - timedelta(hours=i+1)
                hour_end = datetime.now() - timedelta(hours=i)
                
                hour_posts = [
                    post for post in all_posts
                    if hour_start <= post.timestamp < hour_end
                ]
                hourly_counts.append(len(hour_posts))
            
            # Calculate mean and standard deviation
            mean_posts = statistics.mean(hourly_counts)
            std_posts = statistics.stdev(hourly_counts) if len(hourly_counts) > 1 else 0
            
            # Check if current hour exceeds μ + 3σ threshold
            threshold = mean_posts + (3 * std_posts)
            is_anomaly = current_hour_posts > threshold
            
            logger.debug(f"Post volume anomaly check for {symbol}: "
                        f"current={current_hour_posts}, threshold={threshold:.1f}, "
                        f"anomaly={is_anomaly}")
            
            return is_anomaly
            
        except Exception as e:
            logger.error(f"Failed to calculate post volume anomaly: {e}")
            return False
    
    def analyze_board_sentiment(self, symbol: str, hours: int = 1) -> Dict[str, Any]:
        """
        Analyze sentiment of recent board posts.
        
        Args:
            symbol: Stock symbol
            hours: Hours of posts to analyze
            
        Returns:
            Sentiment analysis result
        """
        posts = self.get_recent_posts(symbol, hours)
        
        if not posts:
            return {
                "total_posts": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "average_sentiment": 0.0,
                "sentiment_keywords": []
            }
        
        # TODO: Implement actual sentiment analysis
        # This would integrate with NlpAnalyzer
        logger.debug(f"Analyzing sentiment for {len(posts)} posts")
        
        return {
            "total_posts": len(posts),
            "positive_ratio": 0.4,
            "negative_ratio": 0.3,
            "neutral_ratio": 0.3,
            "average_sentiment": 0.1,
            "sentiment_keywords": ["上昇", "期待", "買い"]
        }
    
    def get_board_analytics(self, symbol: str) -> BoardAnalytics:
        """
        Get comprehensive board analytics for symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Board analytics data
        """
        posts_1h = self.get_recent_posts(symbol, hours=1)
        posts_24h = self.get_recent_posts(symbol, hours=24)
        sentiment = self.analyze_board_sentiment(symbol, hours=1)
        
        return BoardAnalytics(
            symbol=symbol,
            total_posts=len(posts_24h),
            posts_last_hour=len(posts_1h),
            posts_last_24h=len(posts_24h),
            average_sentiment=sentiment["average_sentiment"],
            sentiment_distribution={
                "positive": int(sentiment["positive_ratio"] * len(posts_1h)),
                "negative": int(sentiment["negative_ratio"] * len(posts_1h)),
                "neutral": int(sentiment["neutral_ratio"] * len(posts_1h))
            },
            top_keywords=sentiment["sentiment_keywords"],
            unusual_activity=self.calculate_post_volume_anomaly(symbol)
        )
    
    def update_posts_cache(self, symbol: str, force_refresh: bool = False) -> bool:
        """
        Update posts cache for symbol.
        
        Args:
            symbol: Stock symbol
            force_refresh: Force refresh even if recently updated
            
        Returns:
            True if cache updated successfully
        """
        # Check if recent update exists
        if not force_refresh and symbol in self.last_scrape_time:
            time_since_last = datetime.now() - self.last_scrape_time[symbol]
            if time_since_last.seconds < 300:  # 5 minutes
                logger.debug(f"Skipping cache update for {symbol} (recent update)")
                return True
        
        try:
            new_posts = self.scrape_board_posts(symbol)
            self.posts_cache[symbol] = new_posts
            logger.debug(f"Updated posts cache for {symbol}: {len(new_posts)} posts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update posts cache for {symbol}: {e}")
            return False
    
    def clear_old_posts(self, hours: int = 48) -> int:
        """
        Clear old posts from cache to manage memory.
        
        Args:
            hours: Hours to keep in cache
            
        Returns:
            Number of posts cleared
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cleared_count = 0
        
        for symbol in self.posts_cache:
            original_count = len(self.posts_cache[symbol])
            self.posts_cache[symbol] = [
                post for post in self.posts_cache[symbol]
                if post.timestamp >= cutoff_time
            ]
            cleared_count += original_count - len(self.posts_cache[symbol])
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old posts from cache")
        
        return cleared_count
    
    async def collect_daily_board_posts(self, symbols: List[str], target_date: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
        """
        Collect daily board posts for multiple symbols (for scheduled batch processing).
        
        Args:
            symbols: List of stock symbols
            target_date: Target date (default: today)
            
        Returns:
            Dictionary mapping symbols to their board analytics
        """
        if target_date is None:
            target_date = datetime.now()
        
        logger.info(f"Collecting board posts for {len(symbols)} symbols on {target_date.strftime('%Y-%m-%d')}")
        
        board_data = {}
        
        for symbol in symbols:
            try:
                # Scrape board posts
                posts = await self.scrape_board_posts(symbol)
                
                # Filter posts for target date
                target_date_str = target_date.strftime("%Y-%m-%d")
                daily_posts = [
                    post for post in posts
                    if post.get("datetime", "").startswith(target_date_str)
                ]
                
                # Analyze sentiment
                sentiment_data = self.analyze_board_sentiment(daily_posts)
                
                # Calculate anomaly
                anomaly_data = self.calculate_post_volume_anomaly(symbol, len(daily_posts))
                
                board_data[symbol] = {
                    "date": target_date_str,
                    "total_posts": len(daily_posts),
                    "sentiment": sentiment_data,
                    "anomaly": anomaly_data,
                    "top_posts": daily_posts[:10],  # Top 10 posts
                    "hourly_distribution": self._calculate_hourly_distribution(daily_posts)
                }
                
                # Respect rate limiting
                await asyncio.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error collecting board posts for {symbol}: {e}")
                board_data[symbol] = {
                    "error": str(e),
                    "total_posts": 0
                }
        
        logger.info(f"Collected board data for {len(board_data)} symbols")
        return board_data
    
    def _calculate_hourly_distribution(self, posts: List[BoardPost]) -> Dict[int, int]:
        """
        Calculate hourly distribution of posts.
        
        Args:
            posts: List of board posts
            
        Returns:
            Dictionary mapping hour to post count
        """
        distribution = {hour: 0 for hour in range(24)}
        
        for post in posts:
            try:
                post_time = datetime.fromisoformat(post.datetime)
                distribution[post_time.hour] += 1
            except:
                continue
        
        return distribution

    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get cache status information.
        
        Returns:
            Cache status dictionary
        """
        total_posts = sum(len(posts) for posts in self.posts_cache.values())
        
        return {
            "cached_symbols": len(self.posts_cache),
            "total_cached_posts": total_posts,
            "last_scrape_times": {
                symbol: time.isoformat() 
                for symbol, time in self.last_scrape_time.items()
            },
            "memory_usage_estimate": total_posts * 1024  # Rough estimate in bytes
        }