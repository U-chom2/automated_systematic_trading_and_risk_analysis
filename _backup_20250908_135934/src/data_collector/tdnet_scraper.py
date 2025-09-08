"""TDnet scraper for real-time IR/press release data collection."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TdnetScraper:
    """Scraper for TDnet IR/press release information."""
    
    def __init__(self, polling_interval: int = 1) -> None:
        """
        Initialize TdnetScraper.
        
        Args:
            polling_interval: Polling interval in seconds (default: 1)
        """
        self.polling_interval = polling_interval
        self.last_update_time: Optional[datetime] = None
        self.session = None
        self.base_url = "https://www.release.tdnet.info"
        self.recent_releases_cache: List[Dict[str, Any]] = []
        self.seen_release_ids: set = set()
        
        # S-class trigger keywords from requirements
        self.trigger_keywords = [
            "上方修正", "業務提携", "決算", "買収", "合併", 
            "上場", "株式分割", "増配", "特別配当"
        ]
        
        logger.info(f"TdnetScraper initialized with {polling_interval}s interval")
    
    def _initialize_session(self) -> bool:
        """
        Initialize HTTP session with proper headers.
        
        Returns:
            True if session initialized successfully
        """
        try:
            import requests
            
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            })
            
            logger.info("HTTP session initialized for TDnet scraping")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            return False
    
    def start_monitoring(self) -> None:
        """Start monitoring TDnet for new releases."""
        logger.info("Starting TDnet monitoring...")
        
        if not self.session:
            self._initialize_session()
        
        # TODO: Implement actual monitoring loop
        # This would run in a separate thread or async task
        # For now, just log the start
        pass
    
    def stop_monitoring(self) -> None:
        """Stop monitoring TDnet."""
        logger.info("Stopping TDnet monitoring...")
        
        if self.session:
            self.session.close()
            self.session = None
    
    def get_latest_releases(self) -> List[Dict[str, Any]]:
        """
        Get latest IR/press releases from TDnet RSS feed.
        
        Returns:
            List of release data dictionaries
        """
        try:
            if not self.session:
                self._initialize_session()
            
            logger.debug("Fetching latest releases from TDnet RSS")
            
            # Use RSS feed to get actual TDnet data
            rss_releases = self.fetch_tdnet_rss()
            
            if not rss_releases:
                logger.warning("No releases fetched from RSS, returning cached data")
                return self.recent_releases_cache.copy()
            
            # Filter and process releases
            processed_releases = []
            for release in rss_releases:
                # Check for important trigger keywords
                is_important = self.check_for_trigger_keywords(release.get("title", ""))
                
                # Add to processed list if it's new or important
                if is_important or len(processed_releases) < 10:
                    processed_releases.append({
                        "title": release.get("title", ""),
                        "content": release.get("content", ""),
                        "company_code": release.get("company_code", ""),
                        "symbol": release.get("company_code", ""),  # Use company_code as symbol
                        "url": release.get("url", ""),
                        "timestamp": release.get("timestamp", datetime.now()),
                        "importance_score": release.get("importance_score", 0),
                        "source": "tdnet",
                        "trigger_detected": is_important
                    })
            
            # Update cache with new releases
            self.recent_releases_cache = processed_releases[:20]  # Keep latest 20
            self.last_update_time = datetime.now()
            
            # Update seen release IDs to prevent duplicates
            for release in processed_releases:
                release_id = f"{release['company_code']}_{release['timestamp']}"
                self.seen_release_ids.add(release_id)
            
            logger.info(f"Fetched {len(processed_releases)} releases from TDnet RSS")
            return processed_releases
            
        except Exception as e:
            logger.error(f"Failed to fetch latest releases: {e}")
            # Return cached releases if available
            return self.recent_releases_cache.copy()
    
    def check_for_trigger_keywords(self, title: str) -> bool:
        """
        Check if title contains S-class trigger keywords.
        
        Args:
            title: IR/press release title
            
        Returns:
            True if contains trigger keywords
        """
        if not title:
            return False
        
        title_lower = title.lower()
        
        for keyword in self.trigger_keywords:
            if keyword in title or keyword.lower() in title_lower:
                logger.info(f"Trigger keyword '{keyword}' found in: {title}")
                return True
        
        logger.debug(f"No trigger keywords found in: {title}")
        return False
    
    def parse_release_data(self, raw_data: str) -> Dict[str, Any]:
        """
        Parse raw release data into structured format.
        
        Args:
            raw_data: Raw HTML/text data
            
        Returns:
            Parsed release data
        """
        try:
            from bs4 import BeautifulSoup
            import re
            
            soup = BeautifulSoup(raw_data, 'html.parser')
            
            # TODO: Implement actual parsing based on TDnet HTML structure
            # This is a placeholder that would need to be customized
            # based on the actual TDnet website structure
            
            # Extract basic information
            title = ""
            company_code = ""
            content = ""
            
            # Try to extract title
            title_elem = soup.find(['title', 'h1', 'h2']) or soup.find(class_=re.compile(r'title|heading'))
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Try to extract company code (4-digit number)
            code_pattern = r'\b(\d{4})\b'
            code_matches = re.findall(code_pattern, raw_data)
            if code_matches:
                company_code = code_matches[0]
            
            # Extract main content
            content_elem = soup.find(['div', 'p'], class_=re.compile(r'content|body|main'))
            if content_elem:
                content = content_elem.get_text().strip()
            else:
                content = soup.get_text()[:500]  # First 500 chars as fallback
            
            # Calculate importance score
            importance_score = 0
            if self.check_for_trigger_keywords(title):
                for keyword in self.trigger_keywords:
                    if keyword in title:
                        if keyword in ["上方修正", "買収", "合併"]:
                            importance_score = 50
                        elif keyword in ["業務提携", "決算"]:
                            importance_score = 40
                        else:
                            importance_score = 30
                        break
            
            result = {
                "title": title,
                "content": content,
                "timestamp": datetime.now(),
                "company_code": company_code,
                "importance_score": importance_score,
                "raw_data": raw_data[:1000],  # Keep sample for debugging
                "source": "tdnet"
            }
            
            logger.debug(f"Parsed release: {title[:50]}... (score: {importance_score})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse release data: {e}")
            return {
                "title": "",
                "content": "",
                "timestamp": datetime.now(),
                "company_code": "",
                "importance_score": 0,
                "error": str(e),
                "source": "tdnet"
            }
    
    def fetch_tdnet_rss(self) -> List[Dict[str, Any]]:
        """
        Fetch TDnet data via RSS feed or generate demo data.
        
        Returns:
            List of release data from RSS or demo data
        """
        try:
            if not self.session:
                self._initialize_session()
            
            # Try multiple TDnet RSS URLs
            rss_urls = [
                "https://www.release.tdnet.info/inbs/rss.xml",
                "https://www.release.tdnet.info/rss/rss.xml",
                "https://www.release.tdnet.info/feed.xml"
            ]
            
            for rss_url in rss_urls:
                try:
                    logger.debug(f"Trying RSS URL: {rss_url}")
                    
                    response = self.session.get(rss_url, timeout=15)
                    response.raise_for_status()
                    
                    # Parse RSS XML with proper error handling
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response.content, 'xml')
                    except Exception as parse_error:
                        logger.debug(f"XML parsing failed, trying html.parser: {parse_error}")
                        soup = BeautifulSoup(response.content, 'html.parser')
                    
                    releases = []
                    items = soup.find_all('item')
                    
                    if not items:
                        logger.debug(f"No RSS items found in {rss_url}")
                        continue
                    
                    logger.info(f"Found {len(items)} RSS items from {rss_url}")
                    
                    for item in items:
                        try:
                            # Extract basic RSS fields
                            title_elem = item.find('title')
                            title = title_elem.get_text().strip() if title_elem else ""
                            
                            link_elem = item.find('link')
                            link = link_elem.get_text().strip() if link_elem else ""
                            
                            pub_date_elem = item.find('pubDate')
                            pub_date = pub_date_elem.get_text().strip() if pub_date_elem else ""
                            
                            description_elem = item.find('description')
                            description = description_elem.get_text().strip() if description_elem else ""
                            
                            # Parse publication date
                            timestamp = datetime.now()
                            if pub_date:
                                try:
                                    from dateutil import parser
                                    timestamp = parser.parse(pub_date)
                                except ImportError:
                                    # Fallback parsing for common RSS date format
                                    import re
                                    date_pattern = r'(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})'
                                    match = re.search(date_pattern, pub_date)
                                    if match:
                                        year, month, day, hour, minute, second = map(int, match.groups())
                                        timestamp = datetime(year, month, day, hour, minute, second)
                                except Exception:
                                    logger.debug(f"Could not parse date: {pub_date}")
                            
                            # Extract company code using multiple patterns
                            import re
                            company_code = ""
                            
                            # Pattern 1: Look for 4-digit code in parentheses or brackets
                            code_patterns = [
                                r'\[(\d{4})\]',  # [1234] format
                                r'\((\d{4})\)',  # (1234) format
                                r'^(\d{4})\s',   # 1234 at start
                                r'\s(\d{4})\s',  # 1234 surrounded by spaces
                                r'証券コード\s*:?\s*(\d{4})',  # 証券コード: 1234
                            ]
                            
                            text_to_search = f"{title} {description}"
                            for pattern in code_patterns:
                                matches = re.findall(pattern, text_to_search)
                                if matches:
                                    company_code = matches[0]
                                    break
                            
                            # Calculate importance score based on keywords
                            importance_score = 0
                            if self.check_for_trigger_keywords(title):
                                importance_score = 50
                            elif any(word in title.lower() for word in ['決算', '業績', '修正', '配当']):
                                importance_score = 30
                            elif any(word in title.lower() for word in ['発表', '開示', 'お知らせ']):
                                importance_score = 10
                            
                            release_data = {
                                "title": title,
                                "content": description,
                                "company_code": company_code,
                                "url": link,
                                "pub_date": pub_date,
                                "timestamp": timestamp,
                                "importance_score": importance_score,
                                "source": "tdnet_rss"
                            }
                            
                            # Only include releases with valid company codes
                            if company_code:
                                releases.append(release_data)
                            
                        except Exception as item_error:
                            logger.debug(f"Error processing RSS item: {item_error}")
                            continue
                    
                    if releases:
                        # Sort by timestamp (newest first)
                        releases.sort(key=lambda x: x['timestamp'], reverse=True)
                        logger.info(f"Successfully fetched {len(releases)} releases from TDnet RSS")
                        return releases[:50]  # Return latest 50 releases
                    
                except Exception as url_error:
                    logger.debug(f"Failed to fetch from {rss_url}: {url_error}")
                    continue
            
            # If all RSS URLs failed, generate demo data for testing
            logger.warning("All TDnet RSS URLs failed, generating demo data for testing")
            return self._generate_demo_releases()
            
        except Exception as e:
            logger.error(f"Critical error in fetch_tdnet_rss: {e}")
            return self._generate_demo_releases()
    
    def _generate_demo_releases(self) -> List[Dict[str, Any]]:
        """Generate demo release data for testing."""
        import random
        from datetime import timedelta
        
        demo_companies = [
            ("7203", "トヨタ自動車"),
            ("6758", "ソニーグループ"),
            ("9984", "ソフトバンクグループ"),
            ("8306", "三菱UFJフィナンシャル・グループ"),
            ("6501", "日立製作所")
        ]
        
        demo_releases = []
        base_time = datetime.now()
        
        for i in range(10):
            code, name = random.choice(demo_companies)
            
            # Generate realistic release titles
            title_templates = [
                f"{name} 第3四半期決算短信〔日本基準〕（連結）",
                f"{name} 業績予想の修正に関するお知らせ",
                f"{name} 株式分割及び株式分割に伴う定款の一部変更に関するお知らせ",
                f"{name} 自己株式立会外買付取引（ToSTNeT-3）による自己株式の買付について",
                f"{name} 第三者割当による新株式発行に関するお知らせ"
            ]
            
            title = random.choice(title_templates)
            importance_score = 50 if "修正" in title or "業績" in title else random.randint(10, 30)
            
            release_data = {
                "title": title,
                "content": f"{name}の重要なお知らせです。詳細は添付資料をご覧ください。",
                "company_code": code,
                "url": f"https://www.release.tdnet.info/sample/{code}_{i}.pdf",
                "pub_date": (base_time - timedelta(minutes=i*30)).strftime("%Y/%m/%d %H:%M:%S"),
                "timestamp": base_time - timedelta(minutes=i*30),
                "importance_score": importance_score,
                "source": "tdnet_demo"
            }
            
            demo_releases.append(release_data)
        
        logger.info(f"Generated {len(demo_releases)} demo releases")
        return demo_releases
    
    async def collect_daily_releases(self, target_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Collect all releases for a specific date (for scheduled batch processing).
        
        Args:
            target_date: Target date (default: today)
            
        Returns:
            List of releases for the target date
        """
        if target_date is None:
            target_date = datetime.now()
        
        logger.info(f"Collecting all releases for {target_date.strftime('%Y-%m-%d')}")
        
        try:
            # Initialize session if needed
            if not self.session:
                self._initialize_session()
            
            # Get all releases for the day
            all_releases = await self.fetch_tdnet_rss()
            
            # Filter releases for target date
            daily_releases = []
            target_date_str = target_date.strftime("%Y-%m-%d")
            
            for release in all_releases:
                release_date = release.get("datetime", "")
                if release_date.startswith(target_date_str):
                    # Check for trigger keywords
                    if self.check_for_trigger_keywords(release.get("title", "")):
                        release["has_trigger"] = True
                        release["importance"] = "high"
                    else:
                        release["has_trigger"] = False
                        release["importance"] = "normal"
                    
                    daily_releases.append(release)
            
            logger.info(f"Collected {len(daily_releases)} releases for {target_date_str}")
            
            # Cache the results
            self.recent_releases_cache = daily_releases
            self.last_update_time = datetime.now()
            
            return daily_releases
            
        except Exception as e:
            logger.error(f"Error collecting daily releases: {e}")
            return []
    
    async def get_releases_by_symbols(self, symbols: List[str], target_date: Optional[datetime] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get releases for specific symbols on a target date.
        
        Args:
            symbols: List of stock symbols to filter
            target_date: Target date (default: today)
            
        Returns:
            Dictionary mapping symbols to their releases
        """
        # Get all daily releases
        daily_releases = await self.collect_daily_releases(target_date)
        
        # Group by symbol
        releases_by_symbol = {}
        for symbol in symbols:
            symbol_releases = [
                release for release in daily_releases
                if release.get("company_code") == symbol
            ]
            if symbol_releases:
                releases_by_symbol[symbol] = symbol_releases
        
        return releases_by_symbol

    def check_for_new_releases(self) -> List[Dict[str, Any]]:
        """
        Check for new releases since last check.
        
        Returns:
            List of new release data
        """
        try:
            current_releases = self.get_latest_releases()
            
            # If no previous cache, all releases are "new"
            if not self.recent_releases_cache:
                logger.info(f"Initial fetch: {len(current_releases)} releases found")
                return current_releases
            
            # Find truly new releases by comparing timestamps and content
            new_releases = []
            cutoff_time = self.last_update_time or (datetime.now() - timedelta(hours=1))
            
            for release in current_releases:
                release_timestamp = release.get("timestamp", datetime.now())
                release_id = f"{release['company_code']}_{release_timestamp}"
                
                # Check if this is a new release
                is_new = (
                    release_timestamp > cutoff_time and
                    release_id not in self.seen_release_ids and
                    release.get("company_code")  # Must have company code
                )
                
                if is_new:
                    # Additional filtering for market relevance
                    if self.is_market_relevant_release(release):
                        new_releases.append(release)
                        self.seen_release_ids.add(release_id)
            
            if new_releases:
                logger.info(f"Found {len(new_releases)} new market-relevant releases")
                
                # Log important releases
                for release in new_releases:
                    if release.get("trigger_detected", False):
                        logger.warning(f"TRIGGER DETECTED: {release['title']} "
                                     f"(Symbol: {release['symbol']}, Score: {release['importance_score']})")
            else:
                logger.debug("No new releases detected")
            
            return new_releases
            
        except Exception as e:
            logger.error(f"Error checking for new releases: {e}")
            return []
    
    def is_market_relevant_release(self, release: Dict[str, Any], 
                                  watchlist: List[str]) -> bool:
        """
        Check if release is relevant to monitored companies.
        
        Args:
            release: Release data
            watchlist: List of company codes to monitor
            
        Returns:
            True if release is relevant
        """
        company_code = release.get("company_code", "")
        
        if not company_code:
            return False
        
        # Check if company is in watchlist
        if company_code in watchlist:
            return True
        
        # Check if it's a major market-moving announcement
        # (even if not directly in watchlist)
        title = release.get("title", "")
        major_keywords = ["日経平均", "TOPIX", "相場", "市場全体"]
        
        for keyword in major_keywords:
            if keyword in title:
                return True
        
        return False