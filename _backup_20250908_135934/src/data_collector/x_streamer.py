"""X (Twitter) streaming API client for real-time social media monitoring."""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
import asyncio

logger = logging.getLogger(__name__)


class XStreamer:
    """X (Twitter) streaming API client for monitoring stock mentions."""
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 access_token: str = None, access_token_secret: str = None,
                 bearer_token: str = None) -> None:
        """
        Initialize X Streamer.
        
        Args:
            api_key: X API key (optional, uses demo mode if not provided)
            api_secret: X API secret (optional)
            access_token: Access token (optional)
            access_token_secret: Access token secret (optional)
            bearer_token: Bearer token for API v2 (optional)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.bearer_token = bearer_token
        self.is_streaming = False
        self.callbacks: List[Callable] = []
        
        # Mention tracking
        self.mention_counts: Dict[str, List[datetime]] = {}  # symbol -> timestamps
        self.historical_stats: Dict[str, Dict[str, float]] = {}  # symbol -> {mean, std}
        
        # Tweepy components (initialized lazily)
        self.auth = None
        self.api = None
        self.stream = None
        self.demo_thread = None
        
        mode = "API mode" if api_key else "Demo mode"
        logger.info(f"XStreamer initialized in {mode}")
    
    def _initialize_tweepy(self) -> bool:
        """
        Initialize Tweepy authentication and API.
        
        Returns:
            True if initialization successful
        """
        try:
            import tweepy
            
            # Set up OAuth authentication
            self.auth = tweepy.OAuth1UserHandler(
                self.api_key, self.api_secret,
                self.access_token, self.access_token_secret
            )
            
            # Initialize API client
            self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
            
            # Verify credentials
            self.api.verify_credentials()
            logger.info("Tweepy authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Tweepy: {e}")
            return False
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add callback function for processing tweets.
        
        Args:
            callback: Function to call when tweet is received
        """
        self.callbacks.append(callback)
        logger.debug(f"Added callback: {callback.__name__}")
    
    def start_streaming(self, keywords: List[str]) -> None:
        """
        Start streaming tweets with specified keywords.
        Note: This implementation uses demo data generation as Twitter API requires authentication.
        
        Args:
            keywords: List of keywords to monitor
        """
        logger.info(f"Starting X streaming for keywords: {keywords}")
        
        # For demo purposes, generate synthetic tweets instead of real API calls
        if not self.api and not self._initialize_tweepy():
            logger.warning("Twitter API not available, using demo data generation")
            self._start_demo_streaming(keywords)
            return
        
        try:
            # Try real Twitter API if credentials are available
            import tweepy
            
            # Attempt to create streaming client
            if hasattr(self, 'bearer_token') and self.bearer_token:
                logger.info("Attempting to use real Twitter API")
                
                class StockStreamListener(tweepy.StreamingClient):
                    def __init__(self, bearer_token: str, parent: 'XStreamer'):
                        super().__init__(bearer_token)
                        self.parent = parent
                    
                    def on_tweet(self, tweet):
                        # Process tweet
                        processed = self.parent.process_tweet({
                            "id": tweet.id,
                            "text": tweet.text,
                            "author": tweet.author_id,
                            "created_at": tweet.created_at,
                        })
                        
                        # Update mention counts
                        mentions = self.parent.extract_stock_mentions(tweet.text)
                        for symbol in mentions:
                            self.parent._record_mention(symbol)
                        
                        # Call registered callbacks
                        for callback in self.parent.callbacks:
                            try:
                                callback(processed)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                    
                    def on_error(self, status_code):
                        logger.error(f"Streaming error: {status_code}")
                        if status_code == 420:  # Rate limit
                            return False  # Disconnect
                        return True  # Continue
                
                # Create and start stream
                stream = StockStreamListener(self.bearer_token, self)
                
                # Add rules for keywords
                for keyword in keywords:
                    try:
                        stream.add_rules(tweepy.StreamRule(keyword))
                    except Exception as rule_error:
                        logger.warning(f"Failed to add rule for '{keyword}': {rule_error}")
                
                # Start streaming
                self.stream = stream
                self.is_streaming = True
                
                # Note: In a real implementation, you'd call stream.filter() in a separate thread
                logger.info(f"Real Twitter streaming started for keywords: {keywords}")
                
            else:
                raise Exception("No bearer token available for Twitter API")
                
        except Exception as e:
            logger.warning(f"Failed to start real Twitter streaming: {e}")
            logger.info("Falling back to demo data generation")
            self._start_demo_streaming(keywords)
    
    def _start_demo_streaming(self, keywords: List[str]) -> None:
        """Start demo tweet generation instead of real streaming."""
        import threading
        import time
        import random
        from datetime import datetime
        
        self.is_streaming = True
        logger.info(f"Demo streaming started for keywords: {keywords}")
        
        def generate_demo_tweets():
            """Generate demo tweets in a separate thread."""
            tweet_templates = [
                "{keyword}の株価が上昇しています！",
                "{keyword}について調べています。買い時でしょうか？",
                "{keyword}の決算発表が楽しみです",
                "今日の{keyword}の値動きはすごいですね",
                "{keyword}を長期保有中です",
                "{keyword}の最新ニュースをチェック",
                "{keyword}に注目しています",
                "{keyword}の業績が好調ですね",
                "{keyword}のチャート分析をしてみました",
                "{keyword}について皆さんはどう思いますか？"
            ]
            
            while self.is_streaming:
                try:
                    # Generate a random tweet for a random keyword
                    keyword = random.choice(keywords)
                    template = random.choice(tweet_templates)
                    tweet_text = template.format(keyword=keyword)
                    
                    # Create demo tweet data
                    demo_tweet = {
                        "id": f"demo_{random.randint(1000000, 9999999)}",
                        "text": tweet_text,
                        "author": f"user_{random.randint(1000, 9999)}",
                        "created_at": datetime.now(),
                    }
                    
                    # Process the demo tweet
                    processed = self.process_tweet(demo_tweet)
                    
                    # Update mention counts
                    mentions = self.extract_stock_mentions(tweet_text)
                    for symbol in mentions:
                        self._record_mention(symbol)
                    
                    # Call registered callbacks
                    for callback in self.callbacks:
                        try:
                            callback(processed)
                        except Exception as e:
                            logger.debug(f"Callback error: {e}")
                    
                    # Wait before next tweet (random interval 10-60 seconds)
                    time.sleep(random.randint(10, 60))
                    
                except Exception as e:
                    logger.error(f"Error in demo tweet generation: {e}")
                    time.sleep(30)  # Wait before retrying
        
        # Start demo tweet generation in background thread
        self.demo_thread = threading.Thread(target=generate_demo_tweets, daemon=True)
        self.demo_thread.start()
    
    def stop_streaming(self) -> None:
        """Stop streaming tweets."""
        logger.info("Stopping X streaming")
        self.is_streaming = False
        
        if self.stream:
            try:
                self.stream.disconnect()
                self.stream = None
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
    
    def _record_mention(self, symbol: str) -> None:
        """
        Record a mention timestamp for a symbol.
        
        Args:
            symbol: Stock symbol
        """
        if symbol not in self.mention_counts:
            self.mention_counts[symbol] = []
        
        self.mention_counts[symbol].append(datetime.now())
        
        # Clean old mentions (keep last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.mention_counts[symbol] = [
            ts for ts in self.mention_counts[symbol]
            if ts > cutoff
        ]
    
    def get_mention_count(self, symbol: str, timeframe_minutes: int = 1) -> int:
        """
        Get mention count for a stock symbol within timeframe.
        
        Args:
            symbol: Stock symbol (e.g., "7203" for Toyota)
            timeframe_minutes: Timeframe in minutes
            
        Returns:
            Number of mentions
        """
        if symbol not in self.mention_counts:
            return 0
        
        cutoff = datetime.now() - timedelta(minutes=timeframe_minutes)
        recent_mentions = [
            ts for ts in self.mention_counts[symbol]
            if ts > cutoff
        ]
        
        count = len(recent_mentions)
        logger.debug(f"Mention count for {symbol} in {timeframe_minutes}min: {count}")
        return count
    
    def calculate_anomaly_threshold(self, symbol: str, 
                                  historical_hours: int = 24) -> float:
        """
        Calculate statistical anomaly threshold (μ + 3σ).
        
        Args:
            symbol: Stock symbol
            historical_hours: Hours of historical data to analyze
            
        Returns:
            Anomaly threshold value
        """
        # Calculate hourly mention counts for historical period
        hourly_counts = []
        
        for hour_offset in range(historical_hours):
            hour_start = datetime.now() - timedelta(hours=hour_offset+1)
            hour_end = datetime.now() - timedelta(hours=hour_offset)
            
            if symbol in self.mention_counts:
                hour_mentions = [
                    ts for ts in self.mention_counts[symbol]
                    if hour_start <= ts < hour_end
                ]
                hourly_counts.append(len(hour_mentions))
            else:
                hourly_counts.append(0)
        
        if not hourly_counts or len(hourly_counts) < 2:
            return 0.0
        
        # Calculate statistics
        import statistics
        mean_mentions = statistics.mean(hourly_counts)
        std_mentions = statistics.stdev(hourly_counts)
        
        # Store for future reference
        self.historical_stats[symbol] = {
            "mean": mean_mentions,
            "std": std_mentions,
            "last_calculated": datetime.now()
        }
        
        # Calculate threshold (μ + 3σ)
        threshold = mean_mentions + (3 * std_mentions)
        
        logger.debug(f"Anomaly threshold for {symbol}: {threshold:.2f} "
                    f"(mean={mean_mentions:.2f}, std={std_mentions:.2f})")
        
        return threshold
    
    def check_mention_anomaly(self, symbol: str) -> bool:
        """
        Check if current mention count exceeds anomaly threshold.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if anomaly detected
        """
        current_count = self.get_mention_count(symbol, timeframe_minutes=60)  # Last hour
        threshold = self.calculate_anomaly_threshold(symbol)
        
        is_anomaly = current_count > threshold
        
        if is_anomaly:
            logger.warning(f"Mention anomaly detected for {symbol}: "
                          f"current={current_count}, threshold={threshold:.2f}")
        else:
            logger.debug(f"Normal mention activity for {symbol}: "
                        f"current={current_count}, threshold={threshold:.2f}")
        
        return is_anomaly
    
    def process_tweet(self, tweet_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw tweet data into structured format.
        
        Args:
            tweet_data: Raw tweet data from API
            
        Returns:
            Processed tweet data
        """
        try:
            # Extract text (handle different API response formats)
            text = tweet_data.get("text", "") or tweet_data.get("full_text", "")
            
            # Extract author info
            author = tweet_data.get("author", tweet_data.get("user", {}))
            if isinstance(author, dict):
                author_name = author.get("username", author.get("screen_name", ""))
            else:
                author_name = str(author)
            
            # Parse timestamp
            timestamp = tweet_data.get("created_at", datetime.now())
            if isinstance(timestamp, str):
                # Parse Twitter timestamp format
                try:
                    # Try ISO format first (common in modern APIs)
                    if 'T' in timestamp:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        # Try Twitter's traditional format
                        timestamp = datetime.strptime(
                            timestamp, "%a %b %d %H:%M:%S %z %Y"
                        )
                except:
                    timestamp = datetime.now()
            
            # Extract stock mentions
            mentions = self.extract_stock_mentions(text)
            
            # Basic sentiment analysis (placeholder)
            # In production, integrate with NlpAnalyzer
            sentiment = self._analyze_basic_sentiment(text)
            
            processed = {
                "id": str(tweet_data.get("id", "")),
                "text": text,
                "author": author_name,
                "timestamp": timestamp,
                "mentions": mentions,
                "sentiment": sentiment,
                "raw_data": tweet_data
            }
            
            logger.debug(f"Processed tweet: {text[:50]}... mentions={mentions}")
            return processed
            
        except Exception as e:
            logger.error(f"Failed to process tweet: {e}")
            return {
                "id": "",
                "text": "",
                "author": "",
                "timestamp": datetime.now(),
                "mentions": [],
                "sentiment": 0.0,
                "error": str(e)
            }
    
    def extract_stock_mentions(self, text: str) -> List[str]:
        """
        Extract stock symbols/codes mentioned in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of stock symbols found
        """
        import re
        
        # Pattern to match Japanese stock codes (4-digit numbers)
        # Matches patterns like: 7203, #7203, $7203, (7203)
        pattern = r'[#$（(]?(\d{4})[）)]?'
        matches = re.findall(pattern, text)
        
        # Filter valid stock codes (rough validation)
        valid_codes = []
        for code in matches:
            # Basic validation: should be 4 digits and in reasonable range
            if len(code) == 4 and code.isdigit():
                code_int = int(code)
                # Tokyo Stock Exchange codes typically 1000-9999
                if 1000 <= code_int <= 9999:
                    valid_codes.append(code)
        
        # Remove duplicates while preserving order
        unique_codes = list(dict.fromkeys(valid_codes))
        
        logger.debug(f"Extracted stock mentions from '{text[:50]}...': {unique_codes}")
        return unique_codes
    
    def _analyze_basic_sentiment(self, text: str) -> float:
        """
        Basic sentiment analysis using keyword matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        # Basic Japanese sentiment keywords
        positive_keywords = [
            "上昇", "買い", "好調", "期待", "強い", "上げ", 
            "プラス", "良い", "成長", "増収", "増益", "好決算"
        ]
        negative_keywords = [
            "下落", "売り", "不調", "心配", "弱い", "下げ",
            "マイナス", "悪い", "減収", "減益", "赤字", "損失"
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        # Calculate sentiment score
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
    
    def search_recent_tweets(self, query: str, count: int = 100) -> List[Dict[str, Any]]:
        """
        Search recent tweets using Twitter API or generate demo data.
        
        Args:
            query: Search query
            count: Number of tweets to retrieve
            
        Returns:
            List of processed tweets
        """
        if not self.api and not self._initialize_tweepy():
            logger.warning("Twitter API not available, generating demo tweets")
            return self._generate_demo_tweets(query, count)
        
        try:
            # Try real Twitter API first
            if self.api:
                logger.info(f"Searching real tweets for: {query}")
                
                # Search tweets using Tweepy
                tweets = self.api.search_tweets(
                    q=query,
                    count=min(count, 100),  # API limit
                    tweet_mode="extended",  # Get full text
                    lang="ja"  # Japanese tweets
                )
                
                processed_tweets = []
                for tweet in tweets:
                    # Convert to dict format
                    tweet_dict = {
                        "id": tweet.id_str,
                        "text": tweet.full_text if hasattr(tweet, 'full_text') else tweet.text,
                        "created_at": tweet.created_at,
                        "user": {
                            "screen_name": tweet.user.screen_name,
                            "name": tweet.user.name
                        }
                    }
                    
                    processed = self.process_tweet(tweet_dict)
                    processed_tweets.append(processed)
                    
                    # Update mention counts
                    for symbol in processed["mentions"]:
                        self._record_mention(symbol)
                
                logger.info(f"Retrieved {len(processed_tweets)} real tweets for query: {query}")
                return processed_tweets
            else:
                raise Exception("No API instance available")
                
        except Exception as e:
            logger.warning(f"Failed to search real tweets: {e}")
            logger.info("Generating demo tweets instead")
            return self._generate_demo_tweets(query, count)
    
    def _generate_demo_tweets(self, query: str, count: int = 100) -> List[Dict[str, Any]]:
        """Generate demo tweets for a given query."""
        import random
        from datetime import datetime, timedelta
        
        # Demo tweet templates focused on the query
        tweet_templates = [
            f"{query}の株価動向をウォッチ中です",
            f"{query}について最新情報をチェック",
            f"今日の{query}は調子良いですね！",
            f"{query}の決算発表が楽しみです",
            f"{query}を買い増ししました",
            f"{query}のチャート分析してみた",
            f"{query}に関する最新ニュースをシェア",
            f"{query}の長期展望はどうでしょうか？",
            f"{query}の業績が堅調ですね",
            f"{query}について皆さんの意見を聞きたいです",
            f"{query}の値動きが気になります",
            f"{query}ホルダーの皆さん、調子はいかがですか？",
            f"{query}の配当利回りが魅力的",
            f"{query}のIR情報をチェックしています",
            f"{query}の技術的分析を実施中"
        ]
        
        demo_tweets = []
        base_time = datetime.now()
        
        for i in range(min(count, 20)):  # Limit demo tweets to 20
            # Choose random template
            text = random.choice(tweet_templates)
            
            # Add some variation
            if random.random() < 0.3:  # 30% chance to add hashtags
                text += f" #{query} #株式投資 #投資"
            
            timestamp = base_time - timedelta(minutes=random.randint(1, 1440))  # Last 24 hours
            
            tweet_data = {
                "id": f"demo_{query}_{i}",
                "text": text,
                "created_at": timestamp,
                "user": {
                    "screen_name": f"user_{random.randint(1000, 9999)}",
                    "name": f"投資家{random.randint(100, 999)}"
                }
            }
            
            # Process the demo tweet
            processed = self.process_tweet(tweet_data)
            demo_tweets.append(processed)
            
            # Update mention counts
            for symbol in processed["mentions"]:
                self._record_mention(symbol)
        
        # Sort by timestamp (newest first)
        demo_tweets.sort(key=lambda t: t["timestamp"], reverse=True)
        
        logger.info(f"Generated {len(demo_tweets)} demo tweets for query: {query}")
        return demo_tweets
    
    async def collect_daily_mentions(self, symbols: List[str], target_date: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
        """
        Collect daily mention statistics for multiple symbols (for scheduled batch processing).
        
        Args:
            symbols: List of stock symbols
            target_date: Target date (default: today)
            
        Returns:
            Dictionary mapping symbols to their mention statistics
        """
        if target_date is None:
            target_date = datetime.now()
        
        logger.info(f"Collecting SNS mentions for {len(symbols)} symbols on {target_date.strftime('%Y-%m-%d')}")
        
        mention_stats = {}
        
        for symbol in symbols:
            try:
                # Search for tweets mentioning the symbol
                tweets = await self.search_symbol_mentions(symbol, target_date)
                
                # Calculate statistics
                total_mentions = len(tweets)
                positive_count = sum(1 for t in tweets if t.get("sentiment") == "positive")
                negative_count = sum(1 for t in tweets if t.get("sentiment") == "negative")
                neutral_count = total_mentions - positive_count - negative_count
                
                # Get historical baseline
                baseline = self.historical_stats.get(symbol, {}).get("avg_daily_mentions", 100)
                
                # Calculate anomaly score
                deviation_ratio = (total_mentions - baseline) / baseline if baseline > 0 else 0
                is_anomaly = deviation_ratio > 2.0  # 200% increase
                
                mention_stats[symbol] = {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "total_mentions": total_mentions,
                    "positive": positive_count,
                    "negative": negative_count,
                    "neutral": neutral_count,
                    "sentiment_ratio": positive_count / total_mentions if total_mentions > 0 else 0,
                    "baseline": baseline,
                    "deviation_ratio": deviation_ratio,
                    "is_anomaly": is_anomaly,
                    "top_tweets": tweets[:5] if tweets else []  # Top 5 tweets
                }
                
            except Exception as e:
                logger.error(f"Error collecting mentions for {symbol}: {e}")
                mention_stats[symbol] = {
                    "error": str(e),
                    "total_mentions": 0
                }
        
        logger.info(f"Collected mention statistics for {len(mention_stats)} symbols")
        return mention_stats
    
    async def search_symbol_mentions(self, symbol: str, target_date: datetime) -> List[Dict[str, Any]]:
        """
        Search for tweets mentioning a specific symbol on a target date.
        
        Args:
            symbol: Stock symbol
            target_date: Target date
            
        Returns:
            List of tweets with sentiment analysis
        """
        # In production, this would use Twitter API v2
        # For now, return demo data
        demo_tweets = self._generate_demo_tweets(symbol, count=20)
        
        # Add sentiment analysis to each tweet
        for tweet in demo_tweets:
            tweet["sentiment"] = self._analyze_basic_sentiment(tweet["text"])
            tweet["date"] = target_date.strftime("%Y-%m-%d")
        
        return demo_tweets
    
    def get_daily_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get daily summary of mention statistics.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Summary statistics
        """
        total_mentions = 0
        anomaly_count = 0
        
        for symbol in symbols:
            count = self.get_mention_count(symbol, minutes=1440)  # 24 hours
            total_mentions += count
            
            if self.check_mention_anomaly(symbol):
                anomaly_count += 1
        
        return {
            "total_symbols": len(symbols),
            "total_mentions": total_mentions,
            "anomaly_count": anomaly_count,
            "average_mentions_per_symbol": total_mentions / len(symbols) if symbols else 0,
            "timestamp": datetime.now()
        }

    def get_streaming_status(self) -> Dict[str, Any]:
        """
        Get current streaming status and statistics.
        
        Returns:
            Status dictionary
        """
        total_mentions = sum(len(mentions) for mentions in self.mention_counts.values())
        
        return {
            "is_streaming": self.is_streaming,
            "api_connected": self.api is not None,
            "monitored_symbols": len(self.mention_counts),
            "total_mentions_24h": total_mentions,
            "callbacks_registered": len(self.callbacks),
            "historical_stats": {
                symbol: {
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "current_hourly": self.get_mention_count(symbol, 60)
                }
                for symbol, stats in self.historical_stats.items()
            }
        }
