"""Yahoo Finance APIクライアント

yfinanceライブラリを使用してYahoo Financeからデータを取得します。
"""

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

import yfinance as yf
from yfinance import Ticker

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YahooFinanceClient:
    """Yahoo Finance APIクライアント
    
    株価データ、企業情報、テクニカル指標を取得します。
    """
    
    def __init__(self, enable_cache: bool = False, cache_ttl: int = 60,
                 rate_limit: int = 10) -> None:
        """初期化
        
        Args:
            enable_cache: キャッシュを有効にするか
            cache_ttl: キャッシュの有効期限（秒）
            rate_limit: 秒間の最大リクエスト数
        """
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.rate_limit = rate_limit
        
        # キャッシュ
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # レート制限用
        self._last_request_time = 0.0
        self._request_interval = 1.0 / rate_limit if rate_limit > 0 else 0
        
        logger.info("YahooFinanceClient initialized")
    
    def _apply_rate_limit(self) -> None:
        """レート制限を適用"""
        if self._request_interval > 0:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            if time_since_last < self._request_interval:
                sleep_time = self._request_interval - time_since_last
                time.sleep(sleep_time)
            
            self._last_request_time = time.time()
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """キャッシュから取得
        
        Args:
            key: キャッシュキー
            
        Returns:
            キャッシュデータまたはNone
        """
        if not self.enable_cache:
            return None
        
        if key in self._cache:
            cached = self._cache[key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['data']
        
        return None
    
    def _save_to_cache(self, key: str, data: Dict[str, Any]) -> None:
        """キャッシュに保存
        
        Args:
            key: キャッシュキー
            data: 保存するデータ
        """
        if self.enable_cache:
            self._cache[key] = {
                'data': data,
                'timestamp': time.time()
            }
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """現在価格を取得
        
        Args:
            symbol: 銘柄コード（例: "7203.T"）
            
        Returns:
            価格データまたはNone
        """
        cache_key = f"price_{symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            self._apply_rate_limit()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 価格情報を取得
            price_data = {
                'symbol': symbol,
                'current_price': float(info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)),
                'volume': int(info.get('volume', 0) or info.get('regularMarketVolume', 0)),
                'high': float(info.get('dayHigh', 0) or info.get('regularMarketDayHigh', 0)),
                'low': float(info.get('dayLow', 0) or info.get('regularMarketDayLow', 0)),
                'open': float(info.get('open', 0) or info.get('regularMarketOpen', 0)),
                'previous_close': float(info.get('previousClose', 0) or info.get('regularMarketPreviousClose', 0)),
                'timestamp': datetime.now().isoformat()
            }
            
            # 価格が0の場合は最新の履歴データから取得を試みる
            if price_data['current_price'] == 0:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price_data['current_price'] = float(hist['Close'].iloc[-1])
                    price_data['volume'] = int(hist['Volume'].iloc[-1])
                    price_data['high'] = float(hist['High'].iloc[-1])
                    price_data['low'] = float(hist['Low'].iloc[-1])
                    price_data['open'] = float(hist['Open'].iloc[-1])
            
            self._save_to_cache(cache_key, price_data)
            
            logger.info(f"Fetched current price for {symbol}: {price_data['current_price']}")
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_historical_data(self, symbol: str, period_days: int = 30) -> List[Dict[str, Any]]:
        """過去の価格データを取得
        
        Args:
            symbol: 銘柄コード
            period_days: 取得期間（日数）
            
        Returns:
            過去データのリスト
        """
        try:
            self._apply_rate_limit()
            
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # 過去データを取得
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No historical data found for {symbol}")
                return []
            
            # DataFrameをリストに変換
            historical_data = []
            for index, row in hist.iterrows():
                daily_data = {
                    'date': index.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                }
                historical_data.append(daily_data)
            
            logger.info(f"Fetched {len(historical_data)} days of historical data for {symbol}")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    def calculate_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """テクニカル指標を計算
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            テクニカル指標の辞書
        """
        try:
            self._apply_rate_limit()
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")  # 3ヶ月のデータ
            
            if hist.empty:
                logger.warning(f"No data for technical indicators for {symbol}")
                return {}
            
            close_prices = hist['Close']
            
            # 移動平均
            sma_20 = float(close_prices.rolling(window=20).mean().iloc[-1]) if len(close_prices) >= 20 else None
            sma_50 = float(close_prices.rolling(window=50).mean().iloc[-1]) if len(close_prices) >= 50 else None
            
            # RSI計算
            rsi = self._calculate_rsi(close_prices)
            
            # MACD計算
            macd = self._calculate_macd(close_prices)
            
            # ボリンジャーバンド
            bb = self._calculate_bollinger_bands(close_prices)
            
            indicators = {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'macd': macd,
                'bollinger_bands': bb
            }
            
            logger.info(f"Calculated technical indicators for {symbol}")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """RSI（相対力指数）を計算
        
        Args:
            prices: 価格データ
            period: 計算期間
            
        Returns:
            RSI値（0-100）
        """
        if len(prices) < period:
            return None
        
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
        except:
            return None
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, Optional[float]]:
        """MACD（移動平均収束拡散法）を計算
        
        Args:
            prices: 価格データ
            
        Returns:
            MACD値の辞書
        """
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': float(macd_line.iloc[-1]) if not macd_line.empty else None,
                'signal': float(signal_line.iloc[-1]) if not signal_line.empty else None,
                'histogram': float(histogram.iloc[-1]) if not histogram.empty else None
            }
        except:
            return {'macd': None, 'signal': None, 'histogram': None}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Dict[str, Optional[float]]:
        """ボリンジャーバンドを計算
        
        Args:
            prices: 価格データ
            period: 計算期間
            
        Returns:
            ボリンジャーバンドの値
        """
        if len(prices) < period:
            return {'upper': None, 'middle': None, 'lower': None}
        
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            return {
                'upper': float(upper_band.iloc[-1]),
                'middle': float(sma.iloc[-1]),
                'lower': float(lower_band.iloc[-1])
            }
        except:
            return {'upper': None, 'middle': None, 'lower': None}
    
    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """企業情報を取得
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            企業情報の辞書
        """
        try:
            self._apply_rate_limit()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_info = {
                'symbol': symbol,
                'company_name': info.get('longName', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', None),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
            
            logger.info(f"Fetched company info for {symbol}")
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return None
    
    def get_intraday_data(self, symbol: str, interval: str = '5m') -> List[Dict[str, Any]]:
        """日中データを取得
        
        Args:
            symbol: 銘柄コード
            interval: データ間隔（1m, 5m, 15m, 30m, 60m）
            
        Returns:
            日中データのリスト
        """
        try:
            self._apply_rate_limit()
            
            ticker = yf.Ticker(symbol)
            
            # 日中データを取得（過去1日）
            hist = ticker.history(period="1d", interval=interval)
            
            if hist.empty:
                logger.warning(f"No intraday data found for {symbol}")
                return []
            
            intraday_data = []
            for index, row in hist.iterrows():
                data_point = {
                    'timestamp': index.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                }
                intraday_data.append(data_point)
            
            logger.info(f"Fetched {len(intraday_data)} intraday data points for {symbol}")
            return intraday_data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return []
    
    def get_market_status(self, market: str = 'TSE') -> Dict[str, Any]:
        """市場の状態を取得
        
        Args:
            market: 市場コード（TSE=東京証券取引所）
            
        Returns:
            市場状態の辞書
        """
        try:
            now = datetime.now()
            
            # 東京証券取引所の営業時間
            if market == 'TSE':
                # 平日の9:00-15:00（昼休み11:30-12:30なし）
                weekday = now.weekday()
                is_weekday = weekday < 5  # 月曜(0)から金曜(4)
                
                current_hour = now.hour
                current_minute = now.minute
                current_time = current_hour * 100 + current_minute
                
                is_open = (is_weekday and 
                          900 <= current_time < 1500)
                
                # 次の開場・閉場時刻
                if is_open:
                    next_close = now.replace(hour=15, minute=0, second=0)
                    next_open = None
                else:
                    # 次の営業日を計算
                    if current_time >= 1500 or not is_weekday:
                        days_ahead = 1
                        if weekday == 4:  # 金曜
                            days_ahead = 3
                        elif weekday == 5:  # 土曜
                            days_ahead = 2
                        next_open = (now + timedelta(days=days_ahead)).replace(
                            hour=9, minute=0, second=0
                        )
                    else:
                        next_open = now.replace(hour=9, minute=0, second=0)
                    
                    next_close = None
                
                market_status = {
                    'market': market,
                    'is_open': is_open,
                    'current_time': now.isoformat(),
                    'next_open': next_open.isoformat() if next_open else None,
                    'next_close': next_close.isoformat() if next_close else None
                }
            else:
                # その他の市場（未実装）
                market_status = {
                    'market': market,
                    'is_open': False,
                    'current_time': now.isoformat(),
                    'next_open': None,
                    'next_close': None
                }
            
            logger.info(f"Market status for {market}: {'open' if market_status['is_open'] else 'closed'}")
            return market_status
            
        except Exception as e:
            logger.error(f"Error getting market status for {market}: {e}")
            return {
                'market': market,
                'is_open': False,
                'current_time': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_batch_prices(self, symbols: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """複数銘柄の価格を一括取得
        
        Args:
            symbols: 銘柄コードのリスト
            
        Returns:
            銘柄コードをキーとした価格データの辞書
        """
        batch_data = {}
        
        for symbol in symbols:
            try:
                price_data = self.get_current_price(symbol)
                batch_data[symbol] = price_data
            except Exception as e:
                logger.error(f"Error in batch fetch for {symbol}: {e}")
                batch_data[symbol] = None
        
        logger.info(f"Batch fetched prices for {len(symbols)} symbols")
        return batch_data