"""
Data Fetcher Module
データ取得モジュール - Yahoo Financeから株価データを取得
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import logging
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)


class DataFetcher:
    """データ取得クラス"""
    
    def __init__(self):
        self.config = config.system
        self._apply_warning_filters()
    
    def _apply_warning_filters(self):
        """警告フィルターを適用"""
        for filter_rule in self.config.warning_filters:
            warnings.filterwarnings(*filter_rule.split(":"))
    
    def load_target_companies(self) -> pd.DataFrame:
        """ターゲット企業.xlsxから企業情報を読み込み"""
        try:
            file_path = config.target_companies_path
            if not file_path.exists():
                raise FileNotFoundError(f"Target companies file not found: {file_path}")
            
            df = pd.read_excel(file_path)
            logger.info(f"Loaded {len(df)} target companies from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load target companies: {e}")
            raise
    
    def get_stock_data(
        self, 
        symbol: str, 
        period_days: int = None,
        timeout: int = None
    ) -> Optional[pd.DataFrame]:
        """指定銘柄の株価データを取得
        
        Args:
            symbol: 株価シンボル (例: '7203.T')
            period_days: 取得期間（日数）
            timeout: タイムアウト（秒）
        
        Returns:
            株価データのDataFrame、失敗時はNone
        """
        if period_days is None:
            period_days = self.config.analysis_period_days
        if timeout is None:
            timeout = self.config.data_fetch_timeout
        
        try:
            # 期間設定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # データ取得
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date, 
                end=end_date,
                timeout=timeout
            )
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # データクリーニング
            data = self._clean_stock_data(data)
            
            logger.debug(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def _clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """株価データをクリーニング"""
        if data.empty:
            return data
        
        # NaN値の処理
        data = data.dropna()
        
        # 異常値の処理（極端な価格変動をフィルタ）
        for column in ['Open', 'High', 'Low', 'Close']:
            if column in data.columns:
                # 前日比50%以上の変動を異常値として除外
                pct_change = data[column].pct_change().abs()
                data = data[pct_change <= 0.5]
        
        return data
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """現在の株価を取得"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", timeout=self.config.data_fetch_timeout)
            
            if data.empty:
                return None
            
            return float(data['Close'].iloc[-1])
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    def batch_fetch_stock_data(
        self, 
        symbols: List[str],
        period_days: int = None
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """複数銘柄の株価データを一括取得
        
        Args:
            symbols: 銘柄シンボルのリスト
            period_days: 取得期間（日数）
        
        Returns:
            シンボルをキーとしたデータのDict
        """
        logger.info(f"Batch fetching data for {len(symbols)} symbols...")
        
        results = {}
        current_prices = {}
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"📊 {symbol} ({i}/{len(symbols)}) データ取得中...")
            
            # 株価データ取得
            data = self.get_stock_data(symbol, period_days)
            results[symbol] = data
            
            # 現在価格取得
            if data is not None and not data.empty:
                current_prices[symbol] = float(data['Close'].iloc[-1])
                logger.info(f"✓ {symbol}: ¥{current_prices[symbol]:.0f}")
            else:
                current_prices[symbol] = None
                logger.warning(f"❌ {symbol}: データ取得失敗")
        
        logger.info(f"✅ {len([v for v in results.values() if v is not None])}/{len(symbols)} 銘柄のデータ取得完了")
        
        return results
    
    def get_batch_current_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """複数銘柄の現在価格を一括取得"""
        prices = {}
        
        for symbol in symbols:
            prices[symbol] = self.get_current_price(symbol)
        
        return prices
    
    def validate_symbol_format(self, symbol: str) -> bool:
        """シンボル形式の妥当性を検証"""
        # 日本株式のシンボル形式（例: 7203.T, 9984.T）
        if not symbol.endswith('.T'):
            return False
        
        stock_code = symbol.replace('.T', '')
        if not stock_code.isdigit():
            return False
        
        if len(stock_code) != 4:
            return False
        
        return True
    
    def create_symbol_from_code(self, stock_code: str) -> str:
        """証券コードからシンボルを作成"""
        if isinstance(stock_code, (int, float)):
            stock_code = str(int(stock_code))
        
        # 4桁に正規化
        stock_code = stock_code.zfill(4)
        
        return f"{stock_code}.T"
    
    def get_company_info(self, symbol: str) -> Dict[str, any]:
        """企業情報を取得（基本情報のみ）"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Failed to get company info for {symbol}: {e}")
            return {
                'symbol': symbol,
                'name': symbol,
                'market_cap': 0,
                'sector': 'Unknown',
                'industry': 'Unknown'
            }