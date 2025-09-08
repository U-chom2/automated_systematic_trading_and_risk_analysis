"""Target company selector for investment decisions.

Selects companies from target watchlist based on integrated scoring.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
import logging
from pathlib import Path

from ..data_collector.target_watchlist_loader import TargetWatchlistLoader
from ..data_collector.tdnet_real_scraper import TDnetRealScraper
from ..data_collector.yahoo_finance_client import YahooFinanceClient
from ..data_collector.yahoo_board_scraper import YahooBoardScraper
from ..data_collector.x_streamer import XStreamer
from ..analysis_engine.integrated_scorer import IntegratedScorer
from ..analysis_engine.technical_analyzer import TechnicalAnalyzer
from ..analysis_engine.nlp_analyzer import NlpAnalyzer

logger = logging.getLogger(__name__)


class TargetSelector:
    """Selects investment targets from watchlist based on comprehensive analysis."""
    
    def __init__(self, excel_path: Optional[Path] = None) -> None:
        """Initialize target selector.
        
        Args:
            excel_path: Path to Excel watchlist file
        """
        # Initialize components
        self.watchlist_loader = TargetWatchlistLoader(excel_path)
        self.scorer = IntegratedScorer()
        
        # Data collectors
        self.tdnet_scraper = TDnetRealScraper()
        self.yahoo_client = YahooFinanceClient()
        self.board_scraper = YahooBoardScraper()
        self.x_streamer = XStreamer()
        
        # Analyzers
        self.technical_analyzer = TechnicalAnalyzer()
        self.nlp_analyzer = NlpAnalyzer()
        
        # Cache for today's analysis
        self.analysis_cache: Dict[str, Any] = {}
        self.last_analysis_time: Optional[datetime] = None
    
    def load_targets(self) -> List[Dict[str, Any]]:
        """Load target companies from Excel.
        
        Returns:
            List of target companies
        """
        return self.watchlist_loader.load_watchlist()
    
    def collect_catalyst_data(self, symbol: str) -> Dict[str, Any]:
        """Collect catalyst data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Catalyst data including IR releases
        """
        try:
            # Remove .T suffix for TDnet search
            symbol_clean = symbol.replace('.T', '')
            
            # Get today's IR releases
            releases = self.tdnet_scraper.get_todays_releases(symbol_clean)
            
            if releases:
                # Take most important release
                most_important = releases[0]
                
                # Extract keywords from title
                keywords = self.nlp_analyzer.extract_keywords(most_important.get("title", ""))
                
                return {
                    "title": most_important.get("title", ""),
                    "keywords": keywords,
                    "importance": "high" if any(k in most_important.get("title", "") 
                                              for k in ["上方修正", "業務提携", "M&A"]) else "medium",
                    "release_time": most_important.get("datetime"),
                    "url": most_important.get("url")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to collect catalyst data for {symbol}: {e}")
            return None
    
    def collect_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Collect sentiment data from social sources.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sentiment data with scores and momentum
        """
        try:
            sentiment_data = {
                "positive_ratio": 0.5,
                "change_rate": 1.0,
                "mention_count": 0
            }
            
            # Get Yahoo board sentiment
            board_posts = self.board_scraper.collect_daily_posts(symbol)
            if board_posts:
                sentiment_data["positive_ratio"] = board_posts.get("sentiment_ratio", 0.5)
                sentiment_data["mention_count"] = board_posts.get("post_count", 0)
            
            # Get Twitter/X mentions (if configured)
            # Note: Requires API credentials
            # x_data = self.x_streamer.collect_daily_mentions(symbol)
            
            # Calculate change rate (simplified - comparing to yesterday)
            # In production, this would compare to historical average
            if sentiment_data["mention_count"] > 50:
                sentiment_data["change_rate"] = 1.5  # Assume 50% increase for high activity
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Failed to collect sentiment data for {symbol}: {e}")
            return {
                "positive_ratio": 0.5,
                "change_rate": 1.0,
                "mention_count": 0
            }
    
    def collect_technical_data(self, symbol: str) -> Dict[str, Any]:
        """Collect technical analysis data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Technical indicators and market data
        """
        try:
            # Get price data
            price_data = self.yahoo_client.get_stock_data(symbol)
            
            if not price_data or price_data.empty:
                return None
            
            # Calculate technical indicators
            technical_indicators = self.technical_analyzer.calculate_indicators(price_data)
            
            # Determine trend
            sma_20 = technical_indicators.get("sma_20", 0)
            sma_50 = technical_indicators.get("sma_50", 0)
            current_price = float(price_data['Close'].iloc[-1])
            
            trend = "sideways"
            if sma_20 > sma_50 and current_price > sma_20:
                trend = "uptrend"
            elif sma_20 < sma_50 and current_price < sma_20:
                trend = "downtrend"
            
            # Calculate volume ratio
            avg_volume = price_data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = price_data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # MA deviation
            ma_deviation = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
            
            return {
                "trend": trend,
                "sector_performance": 0.01,  # Placeholder - would need sector index
                "volume_ratio": volume_ratio,
                "rsi": technical_indicators.get("rsi", 50),
                "ma_deviation": ma_deviation,
                "current_price": current_price,
                "indicators": technical_indicators
            }
            
        except Exception as e:
            logger.error(f"Failed to collect technical data for {symbol}: {e}")
            return None
    
    def analyze_company(self, company: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis on a single company.
        
        Args:
            company: Company data from watchlist
            
        Returns:
            Analysis result with scores and decision
        """
        symbol = company.get("symbol")
        
        if not symbol:
            return None
        
        logger.info(f"Analyzing {symbol}: {company.get('company_name', 'Unknown')}")
        
        # Collect all data
        catalyst_data = self.collect_catalyst_data(symbol)
        sentiment_data = self.collect_sentiment_data(symbol)
        technical_data = self.collect_technical_data(symbol)
        
        # Skip if no technical data (likely invalid symbol)
        if not technical_data:
            logger.warning(f"No technical data available for {symbol}")
            return None
        
        # Prepare analysis data
        analysis_data = {
            "symbol": symbol,
            "company_name": company.get("company_name", "Unknown"),
            "catalyst": catalyst_data,
            "sentiment": sentiment_data,
            "technical": technical_data,
            "metadata": {
                "market_cap": company.get("時価総額 (百万円)"),
                "theme": company.get("主要テーマ"),
                "performance_trend": company.get("業績トレンド"),
                "ir_history": company.get("主要なir実績")
            }
        }
        
        # Get scores and decision
        result = self.scorer.analyze_with_breakdown(analysis_data)
        result.update(analysis_data["metadata"])
        
        return result
    
    def select_top_targets(self, max_targets: int = 5) -> List[Dict[str, Any]]:
        """Select top investment targets from watchlist.
        
        Args:
            max_targets: Maximum number of targets to select
            
        Returns:
            List of top targets with analysis results
        """
        # Load watchlist
        companies = self.load_targets()
        
        if not companies:
            logger.error("No companies loaded from watchlist")
            return []
        
        logger.info(f"Analyzing {len(companies)} companies from watchlist")
        
        # Analyze all companies
        results = []
        for company in companies:
            try:
                result = self.analyze_company(company)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {company.get('symbol')}: {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x["scores"]["total_score"], reverse=True)
        
        # Filter for buy decisions only
        buy_candidates = [r for r in results if r.get("execute", False)]
        
        # Return top candidates (up to max_targets)
        selected = buy_candidates[:max_targets]
        
        logger.info(f"Selected {len(selected)} targets from {len(buy_candidates)} buy candidates")
        
        # Cache results
        self.analysis_cache = {
            "timestamp": datetime.now(),
            "all_results": results,
            "selected_targets": selected,
            "statistics": {
                "total_analyzed": len(results),
                "buy_candidates": len(buy_candidates),
                "selected": len(selected)
            }
        }
        self.last_analysis_time = datetime.now()
        
        return selected
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of last analysis.
        
        Returns:
            Summary statistics and top performers
        """
        if not self.analysis_cache:
            return {"status": "No analysis performed yet"}
        
        all_results = self.analysis_cache.get("all_results", [])
        
        # Calculate statistics
        scores = [r["scores"]["total_score"] for r in all_results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "timestamp": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "statistics": self.analysis_cache.get("statistics", {}),
            "average_score": avg_score,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "score_distribution": {
                "excellent (90-100)": len([s for s in scores if s >= 90]),
                "good (80-89)": len([s for s in scores if 80 <= s < 90]),
                "moderate (60-79)": len([s for s in scores if 60 <= s < 80]),
                "poor (<60)": len([s for s in scores if s < 60])
            },
            "top_5_companies": [
                {
                    "symbol": r["symbol"],
                    "name": r["company_name"],
                    "score": r["scores"]["total_score"],
                    "decision": r["decision"]
                }
                for r in all_results[:5]
            ]
        }
    
    def export_results(self, output_path: Path) -> bool:
        """Export analysis results to file.
        
        Args:
            output_path: Path for output file
            
        Returns:
            True if successful
        """
        try:
            import json
            
            if not self.analysis_cache:
                logger.error("No analysis results to export")
                return False
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_cache, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Exported results to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False