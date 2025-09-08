"""市場データリポジトリ実装"""
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Callable
from decimal import Decimal
from uuid import uuid4
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

from ...domain.repositories.market_data_repository import MarketDataRepository
from ...domain.value_objects.price import OHLCV, Price
from ..database.models import MarketDataModel, StockModel
from ..database.session import get_session_manager
from ..database.connection import get_db_connection


class MarketDataRepositoryImpl(MarketDataRepository):
    """市場データリポジトリの実装"""
    
    def __init__(self):
        """初期化"""
        self._session_manager = get_session_manager()
        self._connection = get_db_connection()
        self._subscriptions: Dict[str, asyncio.Task] = {}
    
    def _to_ohlcv(self, model: MarketDataModel) -> OHLCV:
        """モデルをOHLCVに変換"""
        return OHLCV(
            open=Price(model.open),
            high=Price(model.high),
            low=Price(model.low),
            close=Price(model.close),
            volume=model.volume,
            timestamp=model.time,
        )
    
    async def get_latest_price(self, ticker: str) -> Optional[Price]:
        """最新価格を取得"""
        async with self._session_manager.session() as session:
            stmt = select(MarketDataModel).where(
                MarketDataModel.ticker == ticker
            ).order_by(MarketDataModel.time.desc()).limit(1)
            
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                return Price(model.close, timestamp=model.time)
            return None
    
    async def get_latest_prices(self, tickers: List[str]) -> Dict[str, Price]:
        """複数銘柄の最新価格を取得"""
        result = {}
        
        async with self._session_manager.session() as session:
            for ticker in tickers:
                stmt = select(MarketDataModel).where(
                    MarketDataModel.ticker == ticker
                ).order_by(MarketDataModel.time.desc()).limit(1)
                
                query_result = await session.execute(stmt)
                model = query_result.scalar_one_or_none()
                
                if model:
                    result[ticker] = Price(model.close, timestamp=model.time)
        
        return result
    
    async def get_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> List[OHLCV]:
        """OHLCV データを取得"""
        async with self._session_manager.session() as session:
            stmt = select(MarketDataModel).where(
                and_(
                    MarketDataModel.ticker == ticker,
                    MarketDataModel.time >= datetime.combine(start_date, datetime.min.time()),
                    MarketDataModel.time <= datetime.combine(end_date, datetime.max.time()),
                    MarketDataModel.interval == interval,
                )
            ).order_by(MarketDataModel.time)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_ohlcv(model) for model in models]
    
    async def get_multiple_ohlcv(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> Dict[str, List[OHLCV]]:
        """複数銘柄のOHLCVデータを取得"""
        result = {}
        
        # 並列処理で取得
        tasks = []
        for ticker in tickers:
            task = self.get_ohlcv(ticker, start_date, end_date, interval)
            tasks.append(task)
        
        ohlcv_lists = await asyncio.gather(*tasks)
        
        for ticker, ohlcv_list in zip(tickers, ohlcv_lists):
            result[ticker] = ohlcv_list
        
        return result
    
    async def get_intraday_data(
        self,
        ticker: str,
        date: date,
        interval: str = "5m",
    ) -> List[OHLCV]:
        """日中データを取得"""
        start_datetime = datetime.combine(date, datetime.min.time())
        end_datetime = datetime.combine(date, datetime.max.time())
        
        async with self._session_manager.session() as session:
            stmt = select(MarketDataModel).where(
                and_(
                    MarketDataModel.ticker == ticker,
                    MarketDataModel.time >= start_datetime,
                    MarketDataModel.time <= end_datetime,
                    MarketDataModel.interval == interval,
                )
            ).order_by(MarketDataModel.time)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_ohlcv(model) for model in models]
    
    async def save_ohlcv(
        self,
        ticker: str,
        data: List[OHLCV],
        interval: str = "1d",
    ) -> bool:
        """OHLCVデータを保存"""
        async with self._session_manager.session() as session:
            # 株式IDを取得
            stmt = select(StockModel.id).where(StockModel.ticker == ticker).limit(1)
            result = await session.execute(stmt)
            stock_id = result.scalar_one_or_none()
            
            if not stock_id:
                # 株式が存在しない場合は仮のIDを使用
                stock_id = uuid4()
            
            models = []
            for ohlcv in data:
                model = MarketDataModel(
                    time=ohlcv.timestamp,
                    stock_id=stock_id,
                    ticker=ticker,
                    open=ohlcv.open.value,
                    high=ohlcv.high.value,
                    low=ohlcv.low.value,
                    close=ohlcv.close.value,
                    volume=ohlcv.volume,
                    interval=interval,
                )
                models.append(model)
            
            session.add_all(models)
            await session.commit()
            return True
    
    async def save_batch_ohlcv(
        self,
        data: Dict[str, List[OHLCV]],
        interval: str = "1d",
    ) -> bool:
        """複数銘柄のOHLCVデータを一括保存"""
        for ticker, ohlcv_list in data.items():
            await self.save_ohlcv(ticker, ohlcv_list, interval)
        return True
    
    async def subscribe_realtime(
        self,
        tickers: List[str],
        callback: Callable,
    ) -> str:
        """リアルタイムデータを購読"""
        subscription_id = str(uuid4())
        
        async def stream_data():
            """データストリーミング（仮実装）"""
            while subscription_id in self._subscriptions:
                # 実際にはWebSocketやストリーミングAPIを使用
                for ticker in tickers:
                    price = await self.get_latest_price(ticker)
                    if price:
                        await callback(ticker, price)
                
                await asyncio.sleep(5)  # 5秒ごとに更新
        
        # タスクを開始
        task = asyncio.create_task(stream_data())
        self._subscriptions[subscription_id] = task
        
        return subscription_id
    
    async def unsubscribe_realtime(self, subscription_id: str) -> bool:
        """リアルタイムデータの購読を解除"""
        if subscription_id in self._subscriptions:
            task = self._subscriptions[subscription_id]
            task.cancel()
            del self._subscriptions[subscription_id]
            return True
        return False
    
    async def get_market_status(self) -> Dict[str, str]:
        """市場ステータスを取得"""
        now = datetime.now()
        
        # 簡易的な実装
        status = {}
        
        # 東京証券取引所
        tse_open = 9
        tse_close = 15
        if now.weekday() < 5:  # 平日
            if tse_open <= now.hour < tse_close:
                status["TSE"] = "OPEN"
            elif now.hour < tse_open:
                status["TSE"] = "PRE_MARKET"
            else:
                status["TSE"] = "AFTER_HOURS"
        else:
            status["TSE"] = "CLOSED"
        
        # NYSE/NASDAQ（日本時間）
        nyse_open = 23  # 前日23:30
        nyse_close = 6   # 翌日6:00
        if now.weekday() < 5:
            if nyse_open <= now.hour or now.hour < nyse_close:
                status["NYSE"] = "OPEN"
            else:
                status["NYSE"] = "CLOSED"
        else:
            status["NYSE"] = "CLOSED"
        
        return status
    
    async def get_market_hours(self, exchange: str) -> Dict[str, str]:
        """市場時間を取得"""
        hours = {}
        
        if exchange == "TSE":
            hours["open"] = "09:00"
            hours["close"] = "15:00"
            hours["timezone"] = "Asia/Tokyo"
        elif exchange in ["NYSE", "NASDAQ"]:
            hours["open"] = "09:30"
            hours["close"] = "16:00"
            hours["timezone"] = "America/New_York"
        
        return hours
    
    async def is_market_open(self, exchange: str) -> bool:
        """市場が開いているか確認"""
        status = await self.get_market_status()
        return status.get(exchange) == "OPEN"
    
    async def get_technical_indicators(
        self,
        ticker: str,
        indicators: List[str],
        period: int = 20,
    ) -> Dict[str, List[Decimal]]:
        """テクニカル指標を取得"""
        # 過去データを取得
        end_date = date.today()
        start_date = end_date - timedelta(days=period * 2)
        ohlcv_data = await self.get_ohlcv(ticker, start_date, end_date)
        
        if not ohlcv_data:
            return {}
        
        result = {}
        
        # 簡易的な指標計算
        if "sma" in indicators:
            # 単純移動平均
            closes = [o.close.value for o in ohlcv_data]
            sma = []
            for i in range(period, len(closes)):
                avg = sum(closes[i-period:i]) / period
                sma.append(avg)
            result["sma"] = sma
        
        if "ema" in indicators:
            # 指数移動平均（簡易版）
            closes = [o.close.value for o in ohlcv_data]
            ema = []
            multiplier = Decimal(2) / (period + 1)
            
            if len(closes) >= period:
                # 初期値はSMA
                ema.append(sum(closes[:period]) / period)
                
                for i in range(period, len(closes)):
                    ema_val = closes[i] * multiplier + ema[-1] * (1 - multiplier)
                    ema.append(ema_val)
            
            result["ema"] = ema
        
        if "rsi" in indicators:
            # RSI（簡易版）
            closes = [o.close.value for o in ohlcv_data]
            rsi = []
            
            for i in range(period, len(closes)):
                gains = []
                losses = []
                
                for j in range(i-period+1, i):
                    change = closes[j] - closes[j-1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(abs(change))
                
                avg_gain = sum(gains) / period if gains else Decimal("0")
                avg_loss = sum(losses) / period if losses else Decimal("0")
                
                if avg_loss == 0:
                    rsi.append(Decimal("100"))
                else:
                    rs = avg_gain / avg_loss
                    rsi.append(100 - (100 / (1 + rs)))
            
            result["rsi"] = rsi
        
        return result
    
    async def get_volume_profile(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> Dict[Decimal, int]:
        """ボリュームプロファイルを取得"""
        ohlcv_data = await self.get_ohlcv(ticker, start_date, end_date)
        
        if not ohlcv_data:
            return {}
        
        # 価格帯ごとの取引量を集計
        profile = {}
        
        for ohlcv in ohlcv_data:
            # 価格を10円単位で丸める
            price_level = (ohlcv.close.value // 10) * 10
            
            if price_level not in profile:
                profile[price_level] = 0
            
            profile[price_level] += ohlcv.volume
        
        return profile
    
    async def get_company_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """企業情報を取得"""
        async with self._session_manager.session() as session:
            stmt = select(StockModel).where(StockModel.ticker == ticker).limit(1)
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                return {
                    "ticker": model.ticker,
                    "company_name": model.company_name,
                    "company_name_jp": model.company_name_jp,
                    "exchange": model.exchange,
                    "sector": model.sector,
                    "industry": model.industry,
                    "market_cap": float(model.market_cap) if model.market_cap else None,
                    "currency": model.currency,
                }
            return None
    
    async def get_financial_data(
        self,
        ticker: str,
        period: str = "annual",
    ) -> Optional[Dict[str, Any]]:
        """財務データを取得"""
        # 仮実装
        return {
            "ticker": ticker,
            "period": period,
            "revenue": 1000000000,
            "net_income": 100000000,
            "eps": 150.5,
            "pe_ratio": 15.2,
            "roe": 12.5,
        }
    
    async def get_dividends(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """配当情報を取得"""
        # 仮実装
        return [
            {
                "ticker": ticker,
                "ex_date": "2024-03-15",
                "payment_date": "2024-03-31",
                "amount": 50.0,
                "currency": "JPY",
            }
        ]
    
    async def get_splits(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """株式分割情報を取得"""
        # 仮実装
        return []