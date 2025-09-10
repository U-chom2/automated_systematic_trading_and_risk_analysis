"""ニュース・IR情報収集機能 - TDNet、Yahoo Finance統合"""

import sys
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import asyncio
import aiohttp
from dataclasses import dataclass

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger_utils import create_dual_logger
from tdnet.scraper import TDNetScraper
from tdnet.investigator import TDNetInvestigator

logger = create_dual_logger(__name__, console_output=True)


@dataclass
class NewsItem:
    """ニュース項目データクラス"""
    source: str  # 'tdnet', 'yahoo_finance', 'manual'
    ticker: str
    company_name: str
    title: str
    content: str
    published_date: datetime
    url: Optional[str] = None
    importance: float = 0.5  # 0.0-1.0
    category: str = "general"  # earnings, merger, dividend, etc.


class NewsCollector:
    """統合ニュース・IR情報収集システム"""
    
    def __init__(self):
        """初期化"""
        self.tdnet_scraper = TDNetScraper()
        self.tdnet_investigator = TDNetInvestigator()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # ニュースキャッシュ
        self.news_cache: Dict[str, List[NewsItem]] = {}
        self.cache_timeout = 3600  # 1時間
        self.last_cache_time: Dict[str, datetime] = {}
    
    def collect_ir_news(self, ticker: str, company_name: str, days_back: int = 7) -> List[NewsItem]:
        """TDNet IRニュースを収集
        
        Args:
            ticker: ティッカーシンボル
            company_name: 企業名
            days_back: 過去何日分を収集するか
            
        Returns:
            IRニュースリスト
        """
        news_items = []
        end_date = date.today()
        
        try:
            for i in range(days_back):
                check_date = end_date - timedelta(days=i)
                
                # 土日は取引所が休みなのでスキップ
                if check_date.weekday() >= 5:  # 土曜(5), 日曜(6)
                    continue
                
                # TDNet開示情報をチェック
                has_disclosure = self.tdnet_investigator.check_disclosure_by_date(
                    company_name, check_date
                )
                
                if has_disclosure:
                    # 開示情報の詳細を取得（簡易実装）
                    news_item = NewsItem(
                        source="tdnet",
                        ticker=ticker,
                        company_name=company_name,
                        title=f"{company_name} - TDNet開示情報",
                        content=f"{check_date.strftime('%Y-%m-%d')}にTDNetで開示情報が公開されました",
                        published_date=datetime.combine(check_date, datetime.min.time()),
                        url=f"https://www.release.tdnet.info/inbs/I_list_001_{check_date.strftime('%Y%m%d')}.html",
                        importance=0.8,
                        category="disclosure"
                    )
                    news_items.append(news_item)
                    logger.info(f"TDNetニュース発見: {company_name} ({check_date})")
        
        except Exception as e:
            logger.warning(f"TDNetニュース収集エラー {ticker}: {e}")
        
        return news_items
    
    def collect_yahoo_finance_news(self, ticker: str, company_name: str) -> List[NewsItem]:
        """Yahoo Financeニュースを収集
        
        Args:
            ticker: ティッカーシンボル
            company_name: 企業名
            
        Returns:
            ニュースリスト
        """
        news_items = []
        
        try:
            # yfinanceでニュース取得
            stock = yf.Ticker(ticker)
            news_data = stock.news
            
            for news in news_data[:5]:  # 最新5件
                published_timestamp = news.get('providerPublishTime', int(datetime.now().timestamp()))
                published_date = datetime.fromtimestamp(published_timestamp)
                
                news_item = NewsItem(
                    source="yahoo_finance",
                    ticker=ticker,
                    company_name=company_name,
                    title=news.get('title', ''),
                    content=self._extract_news_content(news.get('link', '')),
                    published_date=published_date,
                    url=news.get('link'),
                    importance=0.6,
                    category=self._categorize_news(news.get('title', ''))
                )
                news_items.append(news_item)
            
            logger.info(f"Yahoo Financeニュース収集完了: {ticker} ({len(news_items)}件)")
        
        except Exception as e:
            logger.warning(f"Yahoo Financeニュース収集エラー {ticker}: {e}")
        
        return news_items
    
    def _extract_news_content(self, url: str) -> str:
        """ニュースURLから本文を抽出（簡易実装）
        
        Args:
            url: ニュースURL
            
        Returns:
            抽出されたコンテンツ
        """
        try:
            if not url:
                return ""
            
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return ""
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 一般的なニュース本文の抽出
            content_candidates = [
                soup.find('div', class_='caas-body'),
                soup.find('div', class_='story-body'),
                soup.find('article'),
                soup.find('div', class_='content'),
            ]
            
            for candidate in content_candidates:
                if candidate:
                    text = candidate.get_text(strip=True)
                    if len(text) > 50:  # 最小限の長さチェック
                        return text[:500]  # 最初の500文字
            
            # フォールバック: タイトルのみ
            title_tag = soup.find('title')
            return title_tag.get_text(strip=True) if title_tag else ""
        
        except Exception as e:
            logger.debug(f"コンテンツ抽出エラー {url}: {e}")
            return ""
    
    def _categorize_news(self, title: str) -> str:
        """ニュースタイトルからカテゴリを推定
        
        Args:
            title: ニュースタイトル
            
        Returns:
            カテゴリ名
        """
        title_lower = title.lower()
        
        if any(keyword in title_lower for keyword in ['earnings', '決算', '業績', 'financial results']):
            return 'earnings'
        elif any(keyword in title_lower for keyword in ['dividend', '配当', 'payout']):
            return 'dividend'
        elif any(keyword in title_lower for keyword in ['merger', 'acquisition', 'M&A', '買収', '合併']):
            return 'merger'
        elif any(keyword in title_lower for keyword in ['partnership', 'alliance', '提携', '協業']):
            return 'partnership'
        elif any(keyword in title_lower for keyword in ['launch', 'product', '新製品', 'release']):
            return 'product'
        else:
            return 'general'
    
    def collect_all_news(
        self, 
        ticker: str, 
        company_name: str, 
        days_back: int = 7,
        use_cache: bool = True
    ) -> List[NewsItem]:
        """全ソースからニュースを収集
        
        Args:
            ticker: ティッカーシンボル
            company_name: 企業名
            days_back: 過去何日分を収集するか
            use_cache: キャッシュを使用するか
            
        Returns:
            統合ニュースリスト
        """
        cache_key = f"{ticker}_{days_back}"
        
        # キャッシュチェック
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"ニュースキャッシュを使用: {ticker}")
            return self.news_cache[cache_key]
        
        all_news = []
        
        # TDNet IRニュース
        ir_news = self.collect_ir_news(ticker, company_name, days_back)
        all_news.extend(ir_news)
        
        # Yahoo Financeニュース
        yahoo_news = self.collect_yahoo_finance_news(ticker, company_name)
        all_news.extend(yahoo_news)
        
        # 日付順にソート（新しい順）
        all_news.sort(key=lambda x: x.published_date, reverse=True)
        
        # キャッシュに保存
        if use_cache:
            self.news_cache[cache_key] = all_news
            self.last_cache_time[cache_key] = datetime.now()
        
        logger.info(f"ニュース収集完了: {ticker} ({len(all_news)}件)")
        return all_news
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """キャッシュが有効かチェック
        
        Args:
            cache_key: キャッシュキー
            
        Returns:
            キャッシュが有効ならTrue
        """
        if cache_key not in self.news_cache:
            return False
        
        last_time = self.last_cache_time.get(cache_key)
        if not last_time:
            return False
        
        return (datetime.now() - last_time).seconds < self.cache_timeout
    
    def get_news_summary(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """ニュースサマリーを生成
        
        Args:
            news_items: ニュースリスト
            
        Returns:
            ニュースサマリー
        """
        if not news_items:
            return {
                "total_count": 0,
                "by_source": {},
                "by_category": {},
                "latest_date": None,
                "avg_importance": 0.0
            }
        
        by_source = {}
        by_category = {}
        importance_scores = []
        
        for news in news_items:
            # ソース別集計
            by_source[news.source] = by_source.get(news.source, 0) + 1
            
            # カテゴリ別集計
            by_category[news.category] = by_category.get(news.category, 0) + 1
            
            importance_scores.append(news.importance)
        
        return {
            "total_count": len(news_items),
            "by_source": by_source,
            "by_category": by_category,
            "latest_date": max(news_items, key=lambda x: x.published_date).published_date.isoformat(),
            "avg_importance": sum(importance_scores) / len(importance_scores) if importance_scores else 0.0
        }
    
    def clear_cache(self):
        """ニュースキャッシュをクリア"""
        self.news_cache.clear()
        self.last_cache_time.clear()
        logger.info("ニュースキャッシュをクリアしました")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得
        
        Returns:
            キャッシュ統計
        """
        return {
            "cache_size": len(self.news_cache),
            "cached_tickers": list(self.news_cache.keys()),
            "last_update_times": {k: v.isoformat() for k, v in self.last_cache_time.items()}
        }