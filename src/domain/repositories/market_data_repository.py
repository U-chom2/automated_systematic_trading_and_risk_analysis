"""市場データリポジトリインターフェース"""
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import List, Optional, Dict
from decimal import Decimal

from ..value_objects.price import OHLCV, Price


class MarketDataRepository(ABC):
    """市場データリポジトリの抽象インターフェース"""
    
    @abstractmethod
    async def get_latest_price(self, ticker: str) -> Optional[Price]:
        """最新価格を取得
        
        Args:
            ticker: ティッカーシンボル
        
        Returns:
            最新価格（見つからない場合はNone）
        """
        pass
    
    @abstractmethod
    async def get_latest_prices(self, tickers: List[str]) -> Dict[str, Price]:
        """複数銘柄の最新価格を取得
        
        Args:
            tickers: ティッカーシンボルのリスト
        
        Returns:
            ticker -> Price のマッピング
        """
        pass
    
    @abstractmethod
    async def get_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> List[OHLCV]:
        """OHLCV データを取得
        
        Args:
            ticker: ティッカーシンボル
            start_date: 開始日
            end_date: 終了日
            interval: 時間間隔（1m, 5m, 1h, 1d, 1w, 1M）
        
        Returns:
            OHLCVデータのリスト
        """
        pass
    
    @abstractmethod
    async def get_multiple_ohlcv(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> Dict[str, List[OHLCV]]:
        """複数銘柄のOHLCVデータを取得
        
        Args:
            tickers: ティッカーシンボルのリスト
            start_date: 開始日
            end_date: 終了日
            interval: 時間間隔
        
        Returns:
            ticker -> OHLCVリスト のマッピング
        """
        pass
    
    @abstractmethod
    async def get_intraday_data(
        self,
        ticker: str,
        date: date,
        interval: str = "5m",
    ) -> List[OHLCV]:
        """日中データを取得
        
        Args:
            ticker: ティッカーシンボル
            date: 対象日
            interval: 時間間隔（1m, 5m, 15m, 30m, 1h）
        
        Returns:
            OHLCVデータのリスト
        """
        pass
    
    @abstractmethod
    async def save_ohlcv(
        self,
        ticker: str,
        data: List[OHLCV],
        interval: str = "1d",
    ) -> bool:
        """OHLCVデータを保存
        
        Args:
            ticker: ティッカーシンボル
            data: OHLCVデータのリスト
            interval: 時間間隔
        
        Returns:
            保存成功の可否
        """
        pass
    
    @abstractmethod
    async def save_batch_ohlcv(
        self,
        data: Dict[str, List[OHLCV]],
        interval: str = "1d",
    ) -> bool:
        """複数銘柄のOHLCVデータを一括保存
        
        Args:
            data: ticker -> OHLCVリスト のマッピング
            interval: 時間間隔
        
        Returns:
            保存成功の可否
        """
        pass
    
    # リアルタイムデータ関連
    
    @abstractmethod
    async def subscribe_realtime(
        self,
        tickers: List[str],
        callback: callable,
    ) -> str:
        """リアルタイムデータを購読
        
        Args:
            tickers: ティッカーシンボルのリスト
            callback: データ受信時のコールバック関数
        
        Returns:
            購読ID
        """
        pass
    
    @abstractmethod
    async def unsubscribe_realtime(self, subscription_id: str) -> bool:
        """リアルタイムデータの購読を解除
        
        Args:
            subscription_id: 購読ID
        
        Returns:
            解除成功の可否
        """
        pass
    
    # 市場情報関連
    
    @abstractmethod
    async def get_market_status(self) -> Dict[str, str]:
        """市場ステータスを取得
        
        Returns:
            市場 -> ステータス のマッピング
        """
        pass
    
    @abstractmethod
    async def get_market_hours(self, exchange: str) -> Dict[str, str]:
        """市場時間を取得
        
        Args:
            exchange: 取引所コード
        
        Returns:
            時間情報
        """
        pass
    
    @abstractmethod
    async def is_market_open(self, exchange: str) -> bool:
        """市場が開いているか確認
        
        Args:
            exchange: 取引所コード
        
        Returns:
            開いている場合True
        """
        pass
    
    # テクニカル指標関連
    
    @abstractmethod
    async def get_technical_indicators(
        self,
        ticker: str,
        indicators: List[str],
        period: int = 20,
    ) -> Dict[str, List[Decimal]]:
        """テクニカル指標を取得
        
        Args:
            ticker: ティッカーシンボル
            indicators: 指標名のリスト（sma, ema, rsi, macd等）
            period: 計算期間
        
        Returns:
            指標名 -> 値リスト のマッピング
        """
        pass
    
    @abstractmethod
    async def get_volume_profile(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> Dict[Decimal, int]:
        """ボリュームプロファイルを取得
        
        Args:
            ticker: ティッカーシンボル
            start_date: 開始日
            end_date: 終了日
        
        Returns:
            価格 -> 取引量 のマッピング
        """
        pass
    
    # 企業情報関連
    
    @abstractmethod
    async def get_company_info(self, ticker: str) -> Optional[Dict[str, any]]:
        """企業情報を取得
        
        Args:
            ticker: ティッカーシンボル
        
        Returns:
            企業情報（見つからない場合はNone）
        """
        pass
    
    @abstractmethod
    async def get_financial_data(
        self,
        ticker: str,
        period: str = "annual",
    ) -> Optional[Dict[str, any]]:
        """財務データを取得
        
        Args:
            ticker: ティッカーシンボル
            period: 期間（annual, quarterly）
        
        Returns:
            財務データ（見つからない場合はNone）
        """
        pass
    
    @abstractmethod
    async def get_dividends(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, any]]:
        """配当情報を取得
        
        Args:
            ticker: ティッカーシンボル
            start_date: 開始日（オプション）
            end_date: 終了日（オプション）
        
        Returns:
            配当情報のリスト
        """
        pass
    
    @abstractmethod
    async def get_splits(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, any]]:
        """株式分割情報を取得
        
        Args:
            ticker: ティッカーシンボル
            start_date: 開始日（オプション）
            end_date: 終了日（オプション）
        
        Returns:
            株式分割情報のリスト
        """
        pass