"""
日経225インデックスデータ取得モジュール
実際の日経225データをYahoo Financeから取得
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class Nikkei225DataFetcher:
    """日経225データ取得クラス"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        初期化
        
        Args:
            cache_dir: キャッシュディレクトリ（Noneの場合はキャッシュなし）
        """
        self.nikkei_symbol = '^N225'
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Nikkei225DataFetcher initialized")
    
    def fetch_historical_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = '60d'
    ) -> pd.DataFrame:
        """
        日経225の過去データを取得
        
        Args:
            start_date: 開始日 (YYYY-MM-DD形式)
            end_date: 終了日 (YYYY-MM-DD形式)
            period: 期間 (start_date/end_dateが指定されていない場合に使用)
        
        Returns:
            日経225データのDataFrame
        """
        try:
            # キャッシュチェック
            cache_key = self._get_cache_key(start_date, end_date, period)
            cached_data = self._load_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached Nikkei 225 data")
                return cached_data
            
            logger.info(f"Fetching Nikkei 225 data from Yahoo Finance")
            ticker = yf.Ticker(self.nikkei_symbol)
            
            if start_date and end_date:
                hist = ticker.history(start=start_date, end=end_date)
            else:
                hist = ticker.history(period=period)
            
            if hist.empty:
                logger.warning("No Nikkei 225 data retrieved")
                return self._create_fallback_data(period)
            
            # データを整形
            nikkei_data = pd.DataFrame({
                'date': hist.index,
                'open': hist['Open'].values,
                'high': hist['High'].values,
                'low': hist['Low'].values,
                'close': hist['Close'].values,
                'volume': hist['Volume'].values
            })
            
            # データをキャッシュに保存
            if self.cache_dir:
                self._save_cache(cache_key, nikkei_data)
            
            logger.info(f"Fetched {len(nikkei_data)} records for Nikkei 225")
            return nikkei_data
            
        except Exception as e:
            logger.error(f"Error fetching Nikkei 225 data: {e}")
            return self._create_fallback_data(period)
    
    def fetch_realtime_data(self) -> Dict[str, Any]:
        """
        日経225のリアルタイムデータを取得
        
        Returns:
            現在の価格情報
        """
        try:
            ticker = yf.Ticker(self.nikkei_symbol)
            info = ticker.info
            
            # 最新の価格データを取得（1日分）
            recent = ticker.history(period='1d')
            
            if not recent.empty:
                latest = recent.iloc[-1]
                return {
                    'symbol': self.nikkei_symbol,
                    'name': 'Nikkei 225',
                    'current_price': latest['Close'],
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'volume': latest['Volume'],
                    'previous_close': info.get('previousClose', latest['Close']),
                    'change': latest['Close'] - info.get('previousClose', latest['Close']),
                    'change_percent': ((latest['Close'] - info.get('previousClose', latest['Close'])) 
                                     / info.get('previousClose', latest['Close']) * 100 
                                     if info.get('previousClose') else 0),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # フォールバック値
                return self._create_fallback_realtime()
                
        except Exception as e:
            logger.error(f"Error fetching realtime Nikkei 225 data: {e}")
            return self._create_fallback_realtime()
    
    def fetch_for_window(self, window_size: int = 30) -> pd.DataFrame:
        """
        指定されたウィンドウサイズ分のデータを取得
        
        Args:
            window_size: 取得する日数
        
        Returns:
            日経225データのDataFrame
        """
        # ウィンドウサイズ + バッファ日数を取得（休日を考慮）
        fetch_days = window_size + 10
        end_date = datetime.now()
        start_date = end_date - timedelta(days=fetch_days * 2)  # 余裕を持って取得
        
        data = self.fetch_historical_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # 必要な日数分だけ取得
        if len(data) > window_size:
            return data.tail(window_size)
        return data
    
    def _create_fallback_data(self, period: str) -> pd.DataFrame:
        """
        フォールバック用のデータを作成（すべて0で明確にダミーとわかるように）
        
        Args:
            period: 期間
        
        Returns:
            フォールバックデータ（すべて0）
        """
        # 期間から日数を推定
        if period.endswith('d'):
            days = int(period[:-1])
        elif period.endswith('mo'):
            days = int(period[:-2]) * 30
        elif period.endswith('y'):
            days = int(period[:-1]) * 365
        else:
            days = 60
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # すべて0のダミーデータ（明確にダミーとわかるように）
        data = pd.DataFrame({
            'date': dates,
            'open': [0.0] * days,
            'high': [0.0] * days,
            'low': [0.0] * days,
            'close': [0.0] * days,
            'volume': [0.0] * days
        })
        
        logger.error("FALLBACK: Using dummy data (all zeros) for Nikkei 225 - Real data fetch failed")
        return data
    
    def _create_fallback_realtime(self) -> Dict[str, Any]:
        """フォールバック用のリアルタイムデータ（すべて0で明確にダミー）"""
        logger.error("FALLBACK: Using dummy realtime data (all zeros) - Real data fetch failed")
        return {
            'symbol': self.nikkei_symbol,
            'name': 'Nikkei 225 (DUMMY)',
            'current_price': 0.0,
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'volume': 0.0,
            'previous_close': 0.0,
            'change': 0.0,
            'change_percent': 0.0,
            'timestamp': datetime.now().isoformat(),
            'is_dummy': True  # 明確にダミーデータであることを示すフラグ
        }
    
    def _get_cache_key(self, start_date: Optional[str], end_date: Optional[str], period: str) -> str:
        """キャッシュキーを生成"""
        if start_date and end_date:
            return f"nikkei_{start_date}_{end_date}"
        else:
            return f"nikkei_{period}_{datetime.now().strftime('%Y%m%d')}"
    
    def _load_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """キャッシュからデータをロード"""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            # キャッシュが1日以内なら使用
            if (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 1:
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                except Exception as e:
                    logger.error(f"Failed to load cache: {e}")
        return None
    
    def _save_cache(self, cache_key: str, data: pd.DataFrame):
        """データをキャッシュに保存"""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            # DataFrameをJSON形式で保存
            data_dict = data.copy()
            data_dict['date'] = data_dict['date'].astype(str)
            data_dict.to_json(cache_file, orient='records', indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")


# テスト用コード
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # キャッシュディレクトリを作成
    cache_dir = Path("cache/nikkei")
    fetcher = Nikkei225DataFetcher(cache_dir=cache_dir)
    
    # 過去60日分のデータを取得
    print("Fetching historical data...")
    hist_data = fetcher.fetch_historical_data(period='60d')
    print(f"Historical data shape: {hist_data.shape}")
    print(hist_data.head())
    
    # リアルタイムデータを取得
    print("\nFetching realtime data...")
    realtime = fetcher.fetch_realtime_data()
    print(f"Current Nikkei 225: ¥{realtime['current_price']:,.0f}")
    print(f"Change: {realtime['change']:+,.0f} ({realtime['change_percent']:+.2f}%)")
    
    # ウィンドウサイズ分のデータを取得
    print("\nFetching window data...")
    window_data = fetcher.fetch_for_window(30)
    print(f"Window data shape: {window_data.shape}")
    print(f"Date range: {window_data['date'].min()} to {window_data['date'].max()}")