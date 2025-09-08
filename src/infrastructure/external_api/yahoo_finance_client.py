"""Yahoo Finance APIクライアント"""
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from decimal import Decimal
import aiohttp
import asyncio
from dataclasses import dataclass

from ...domain.value_objects.price import OHLCV, Price


@dataclass
class YahooFinanceConfig:
    """Yahoo Finance設定"""
    base_url: str = "https://query1.finance.yahoo.com"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1


class YahooFinanceClient:
    """Yahoo Finance APIクライアント"""
    
    def __init__(self, config: Optional[YahooFinanceConfig] = None):
        """初期化
        
        Args:
            config: 設定
        """
        self.config = config or YahooFinanceConfig()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """コンテキストマネージャー開始"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了"""
        if self._session:
            await self._session.close()
    
    async def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """APIリクエスト
        
        Args:
            endpoint: エンドポイント
            params: パラメータ
        
        Returns:
            レスポンスデータ
        """
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                async with self._session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
    
    def _convert_ticker(self, ticker: str) -> str:
        """ティッカーシンボルを変換
        
        日本株の場合は.Tを付加
        """
        if ticker.isdigit() and len(ticker) == 4:
            return f"{ticker}.T"
        return ticker
    
    async def get_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """株価情報を取得
        
        Args:
            ticker: ティッカーシンボル
        
        Returns:
            株価情報
        """
        ticker = self._convert_ticker(ticker)
        
        try:
            data = await self._request(
                "/v8/finance/quote",
                {"symbols": ticker}
            )
            
            quotes = data.get("quoteResponse", {}).get("result", [])
            if quotes:
                quote = quotes[0]
                return {
                    "ticker": ticker,
                    "price": quote.get("regularMarketPrice"),
                    "change": quote.get("regularMarketChange"),
                    "change_percent": quote.get("regularMarketChangePercent"),
                    "volume": quote.get("regularMarketVolume"),
                    "market_cap": quote.get("marketCap"),
                    "pe_ratio": quote.get("trailingPE"),
                    "dividend_yield": quote.get("dividendYield"),
                }
            return None
            
        except Exception as e:
            print(f"Error fetching quote for {ticker}: {e}")
            return None
    
    async def get_historical_data(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> List[OHLCV]:
        """過去データを取得
        
        Args:
            ticker: ティッカーシンボル
            start_date: 開始日
            end_date: 終了日
            interval: インターバル
        
        Returns:
            OHLCVデータのリスト
        """
        ticker = self._convert_ticker(ticker)
        
        # タイムスタンプに変換
        start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        
        try:
            data = await self._request(
                f"/v8/finance/chart/{ticker}",
                {
                    "period1": start_ts,
                    "period2": end_ts,
                    "interval": interval,
                    "includeAdjustedClose": "true"
                }
            )
            
            result = []
            chart = data.get("chart", {}).get("result", [])
            
            if chart:
                timestamps = chart[0].get("timestamp", [])
                quotes = chart[0].get("indicators", {}).get("quote", [])
                
                if quotes and timestamps:
                    quote = quotes[0]
                    opens = quote.get("open", [])
                    highs = quote.get("high", [])
                    lows = quote.get("low", [])
                    closes = quote.get("close", [])
                    volumes = quote.get("volume", [])
                    
                    for i in range(len(timestamps)):
                        if all([
                            i < len(opens) and opens[i] is not None,
                            i < len(highs) and highs[i] is not None,
                            i < len(lows) and lows[i] is not None,
                            i < len(closes) and closes[i] is not None,
                            i < len(volumes) and volumes[i] is not None,
                        ]):
                            ohlcv = OHLCV(
                                open=Price(Decimal(str(opens[i]))),
                                high=Price(Decimal(str(highs[i]))),
                                low=Price(Decimal(str(lows[i]))),
                                close=Price(Decimal(str(closes[i]))),
                                volume=int(volumes[i]),
                                timestamp=datetime.fromtimestamp(timestamps[i])
                            )
                            result.append(ohlcv)
            
            return result
            
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return []
    
    async def get_company_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """企業情報を取得
        
        Args:
            ticker: ティッカーシンボル
        
        Returns:
            企業情報
        """
        ticker = self._convert_ticker(ticker)
        
        try:
            data = await self._request(
                f"/v10/finance/quoteSummary/{ticker}",
                {"modules": "summaryProfile,financialData,defaultKeyStatistics"}
            )
            
            result = data.get("quoteSummary", {}).get("result", [])
            if result:
                summary = result[0]
                profile = summary.get("summaryProfile", {})
                financial = summary.get("financialData", {})
                stats = summary.get("defaultKeyStatistics", {})
                
                return {
                    "ticker": ticker,
                    "company_name": profile.get("longName"),
                    "sector": profile.get("sector"),
                    "industry": profile.get("industry"),
                    "website": profile.get("website"),
                    "description": profile.get("longBusinessSummary"),
                    "employees": profile.get("fullTimeEmployees"),
                    "market_cap": stats.get("marketCap", {}).get("raw"),
                    "pe_ratio": stats.get("trailingPE", {}).get("raw"),
                    "peg_ratio": stats.get("pegRatio", {}).get("raw"),
                    "dividend_yield": financial.get("dividendYield", {}).get("raw"),
                    "profit_margin": financial.get("profitMargins", {}).get("raw"),
                    "revenue_growth": financial.get("revenueGrowth", {}).get("raw"),
                }
            return None
            
        except Exception as e:
            print(f"Error fetching company info for {ticker}: {e}")
            return None
    
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """ティッカーを検索
        
        Args:
            query: 検索クエリ
            limit: 結果の上限
        
        Returns:
            検索結果
        """
        try:
            data = await self._request(
                "/v1/finance/search",
                {"q": query, "quotesCount": limit}
            )
            
            quotes = data.get("quotes", [])
            result = []
            
            for quote in quotes[:limit]:
                result.append({
                    "ticker": quote.get("symbol"),
                    "name": quote.get("longname") or quote.get("shortname"),
                    "exchange": quote.get("exchange"),
                    "type": quote.get("quoteType"),
                })
            
            return result
            
        except Exception as e:
            print(f"Error searching for {query}: {e}")
            return []
    
    async def get_options_chain(self, ticker: str) -> Optional[Dict[str, Any]]:
        """オプションチェーンを取得
        
        Args:
            ticker: ティッカーシンボル
        
        Returns:
            オプションチェーン
        """
        ticker = self._convert_ticker(ticker)
        
        try:
            data = await self._request(
                f"/v7/finance/options/{ticker}",
                {}
            )
            
            result = data.get("optionChain", {}).get("result", [])
            if result:
                options = result[0]
                return {
                    "ticker": ticker,
                    "expiration_dates": options.get("expirationDates", []),
                    "strikes": options.get("strikes", []),
                    "options": options.get("options", []),
                }
            return None
            
        except Exception as e:
            print(f"Error fetching options chain for {ticker}: {e}")
            return None