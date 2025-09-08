"""市場データ関連DTO"""
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Optional, Any

from ...domain.value_objects.price import Price, OHLCV


@dataclass
class PriceDTO:
    """価格DTO"""
    ticker: str
    value: float
    currency: str
    timestamp: str
    
    @classmethod
    def from_entity(cls, ticker: str, price: Price) -> "PriceDTO":
        """エンティティからDTOを作成"""
        return cls(
            ticker=ticker,
            value=float(price.value),
            currency=price.currency,
            timestamp=price.timestamp.isoformat() if price.timestamp else datetime.now().isoformat(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "ticker": self.ticker,
            "value": self.value,
            "currency": self.currency,
            "timestamp": self.timestamp,
        }


@dataclass
class OHLCVDTO:
    """OHLCV DTO"""
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: str
    change: float
    change_percent: float
    
    @classmethod
    def from_entity(cls, ticker: str, ohlcv: OHLCV) -> "OHLCVDTO":
        """エンティティからDTOを作成"""
        change = float(ohlcv.close.value - ohlcv.open.value)
        change_percent = (change / float(ohlcv.open.value) * 100) if ohlcv.open.value > 0 else 0.0
        
        return cls(
            ticker=ticker,
            open=float(ohlcv.open.value),
            high=float(ohlcv.high.value),
            low=float(ohlcv.low.value),
            close=float(ohlcv.close.value),
            volume=ohlcv.volume,
            timestamp=ohlcv.timestamp.isoformat(),
            change=change,
            change_percent=change_percent,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "ticker": self.ticker,
            "ohlcv": {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            },
            "change": {
                "value": self.change,
                "percent": self.change_percent,
            },
            "timestamp": self.timestamp,
        }


@dataclass
class MarketDataRequestDTO:
    """市場データリクエストDTO"""
    tickers: List[str]
    start_date: str
    end_date: str
    interval: str = "1d"
    data_types: List[str] = None
    
    def __post_init__(self):
        if self.data_types is None:
            self.data_types = ["ohlcv"]
    
    def validate(self) -> None:
        """バリデーション"""
        if not self.tickers:
            raise ValueError("At least one ticker is required")
        
        # 日付の検証
        try:
            start = datetime.fromisoformat(self.start_date)
            end = datetime.fromisoformat(self.end_date)
            if start > end:
                raise ValueError("Start date must be before end date")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")
        
        # インターバルの検証
        valid_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"]
        if self.interval not in valid_intervals:
            raise ValueError(f"Interval must be one of {valid_intervals}")
        
        # データタイプの検証
        valid_types = ["ohlcv", "price", "volume", "indicators"]
        for dt in self.data_types:
            if dt not in valid_types:
                raise ValueError(f"Data type must be one of {valid_types}")


@dataclass
class MarketDataResponseDTO:
    """市場データレスポンスDTO"""
    request: MarketDataRequestDTO
    data: Dict[str, List[OHLCVDTO]]
    metadata: Dict[str, Any]
    
    @classmethod
    def create(
        cls,
        request: MarketDataRequestDTO,
        ohlcv_data: Dict[str, List[OHLCV]],
        processing_time_ms: int = 0,
    ) -> "MarketDataResponseDTO":
        """OHLCVデータからレスポンスDTOを作成"""
        data = {}
        total_records = 0
        
        for ticker, ohlcv_list in ohlcv_data.items():
            data[ticker] = [OHLCVDTO.from_entity(ticker, ohlcv) for ohlcv in ohlcv_list]
            total_records += len(ohlcv_list)
        
        metadata = {
            "tickers_count": len(data),
            "total_records": total_records,
            "interval": request.interval,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now().isoformat(),
        }
        
        return cls(
            request=request,
            data=data,
            metadata=metadata,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "request": {
                "tickers": self.request.tickers,
                "start_date": self.request.start_date,
                "end_date": self.request.end_date,
                "interval": self.request.interval,
                "data_types": self.request.data_types,
            },
            "data": {
                ticker: [d.to_dict() for d in ohlcv_list]
                for ticker, ohlcv_list in self.data.items()
            },
            "metadata": self.metadata,
        }


@dataclass
class TechnicalIndicatorDTO:
    """テクニカル指標DTO"""
    ticker: str
    indicator_name: str
    period: int
    values: List[float]
    timestamps: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "ticker": self.ticker,
            "indicator": {
                "name": self.indicator_name,
                "period": self.period,
            },
            "data": [
                {"timestamp": ts, "value": val}
                for ts, val in zip(self.timestamps, self.values)
            ]
        }


@dataclass
class MarketStatusDTO:
    """市場ステータスDTO"""
    exchange: str
    status: str  # OPEN, CLOSED, PRE_MARKET, AFTER_HOURS
    current_time: str
    next_open: Optional[str]
    next_close: Optional[str]
    timezone: str
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "exchange": self.exchange,
            "status": self.status,
            "current_time": self.current_time,
            "trading_hours": {
                "next_open": self.next_open,
                "next_close": self.next_close,
            },
            "timezone": self.timezone,
        }


@dataclass
class CompanyInfoDTO:
    """企業情報DTO"""
    ticker: str
    company_name: str
    sector: str
    industry: str
    market_cap: float
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    description: str
    website: Optional[str]
    employees: Optional[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "ticker": self.ticker,
            "company": {
                "name": self.company_name,
                "sector": self.sector,
                "industry": self.industry,
                "description": self.description,
                "website": self.website,
                "employees": self.employees,
            },
            "financials": {
                "market_cap": self.market_cap,
                "pe_ratio": self.pe_ratio,
                "dividend_yield": self.dividend_yield,
                "beta": self.beta,
            }
        }


@dataclass
class MarketSummaryDTO:
    """市場サマリーDTO"""
    date: str
    indices: Dict[str, Dict[str, float]]  # index_name -> {value, change, change_percent}
    most_active: List[str]
    gainers: List[str]
    losers: List[str]
    market_breadth: Dict[str, int]  # advances, declines, unchanged
    total_volume: int
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "date": self.date,
            "indices": self.indices,
            "movers": {
                "most_active": self.most_active,
                "gainers": self.gainers,
                "losers": self.losers,
            },
            "breadth": self.market_breadth,
            "total_volume": self.total_volume,
        }