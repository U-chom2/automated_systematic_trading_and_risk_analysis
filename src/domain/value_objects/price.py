"""価格値オブジェクト"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Optional, Union


@dataclass(frozen=True)
class Price:
    """価格を表す値オブジェクト"""
    
    value: Decimal
    currency: str = "JPY"
    timestamp: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        # Decimalに変換
        object.__setattr__(self, 'value', Decimal(str(self.value)))
        
        if self.value <= 0:
            raise ValueError("Price must be positive")
    
    def __add__(self, other: Union[Price, Decimal, int, float]) -> Price:
        """加算"""
        if isinstance(other, Price):
            if self.currency != other.currency:
                raise ValueError(f"Cannot add different currencies: {self.currency} and {other.currency}")
            return Price(self.value + other.value, self.currency)
        else:
            return Price(self.value + Decimal(str(other)), self.currency)
    
    def __sub__(self, other: Union[Price, Decimal, int, float]) -> Price:
        """減算"""
        if isinstance(other, Price):
            if self.currency != other.currency:
                raise ValueError(f"Cannot subtract different currencies: {self.currency} and {other.currency}")
            result = self.value - other.value
        else:
            result = self.value - Decimal(str(other))
        
        if result <= 0:
            raise ValueError("Price cannot be zero or negative after subtraction")
        
        return Price(result, self.currency)
    
    def __mul__(self, other: Union[Decimal, int, float]) -> Price:
        """乗算"""
        result = self.value * Decimal(str(other))
        if result <= 0:
            raise ValueError("Price cannot be zero or negative after multiplication")
        return Price(result, self.currency)
    
    def __truediv__(self, other: Union[Price, Decimal, int, float]) -> Union[Price, Decimal]:
        """除算"""
        if isinstance(other, Price):
            if self.currency != other.currency:
                raise ValueError(f"Cannot divide different currencies: {self.currency} and {other.currency}")
            return self.value / other.value
        else:
            if other == 0:
                raise ValueError("Cannot divide by zero")
            return Price(self.value / Decimal(str(other)), self.currency)
    
    def __eq__(self, other: object) -> bool:
        """等価比較"""
        if not isinstance(other, Price):
            return False
        return self.value == other.value and self.currency == other.currency
    
    def __lt__(self, other: Price) -> bool:
        """小なり比較"""
        if not isinstance(other, Price):
            raise TypeError("Cannot compare Price with non-Price type")
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare different currencies: {self.currency} and {other.currency}")
        return self.value < other.value
    
    def __le__(self, other: Price) -> bool:
        """小なりイコール比較"""
        return self < other or self == other
    
    def __gt__(self, other: Price) -> bool:
        """大なり比較"""
        if not isinstance(other, Price):
            raise TypeError("Cannot compare Price with non-Price type")
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare different currencies: {self.currency} and {other.currency}")
        return self.value > other.value
    
    def __ge__(self, other: Price) -> bool:
        """大なりイコール比較"""
        return self > other or self == other
    
    def to_money(self, quantity: int) -> 'Money':
        """金額に変換"""
        from .money import Money
        return Money(self.value * Decimal(quantity), self.currency)
    
    def __str__(self) -> str:
        """文字列表現"""
        if self.currency == "JPY":
            return f"¥{self.value:,.0f}"
        else:
            return f"{self.currency} {self.value:,.2f}"
    
    def __repr__(self) -> str:
        """詳細表現"""
        return f"Price(value={self.value}, currency={self.currency!r})"


@dataclass(frozen=True)
class PriceRange:
    """価格範囲を表す値オブジェクト"""
    
    low: Price
    high: Price
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        if self.low.currency != self.high.currency:
            raise ValueError("Price range must have same currency")
        
        if self.low > self.high:
            raise ValueError("Low price must be less than or equal to high price")
    
    @property
    def spread(self) -> Decimal:
        """価格差"""
        return self.high.value - self.low.value
    
    @property
    def spread_percent(self) -> Decimal:
        """価格差率"""
        if self.low.value == 0:
            return Decimal("0")
        return self.spread / self.low.value
    
    @property
    def midpoint(self) -> Price:
        """中間価格"""
        return Price((self.low.value + self.high.value) / 2, self.low.currency)
    
    def contains(self, price: Price) -> bool:
        """価格が範囲内かどうか"""
        if price.currency != self.low.currency:
            raise ValueError("Cannot compare different currencies")
        return self.low <= price <= self.high
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"{self.low} - {self.high}"
    
    def __repr__(self) -> str:
        """詳細表現"""
        return f"PriceRange(low={self.low!r}, high={self.high!r})"


@dataclass(frozen=True)
class OHLCV:
    """OHLCV (Open-High-Low-Close-Volume) データ"""
    
    open: Price
    high: Price
    low: Price
    close: Price
    volume: int
    timestamp: datetime
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        # 通貨の一致を確認
        currencies = {self.open.currency, self.high.currency, 
                     self.low.currency, self.close.currency}
        if len(currencies) > 1:
            raise ValueError("All prices must have the same currency")
        
        # 価格の整合性を確認
        if self.low > self.high:
            raise ValueError("Low price cannot be greater than high price")
        
        if self.open < self.low or self.open > self.high:
            raise ValueError("Open price must be between low and high")
        
        if self.close < self.low or self.close > self.high:
            raise ValueError("Close price must be between low and high")
        
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def price_range(self) -> PriceRange:
        """価格範囲"""
        return PriceRange(self.low, self.high)
    
    @property
    def body_range(self) -> PriceRange:
        """実体の範囲"""
        if self.open <= self.close:
            return PriceRange(self.open, self.close)
        else:
            return PriceRange(self.close, self.open)
    
    @property
    def is_bullish(self) -> bool:
        """陽線かどうか"""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """陰線かどうか"""
        return self.close < self.open
    
    @property
    def is_doji(self) -> bool:
        """同時線かどうか"""
        return self.close == self.open
    
    @property
    def body_size(self) -> Decimal:
        """実体の大きさ"""
        return abs(self.close.value - self.open.value)
    
    @property
    def upper_shadow(self) -> Decimal:
        """上ヒゲの長さ"""
        return self.high.value - max(self.open.value, self.close.value)
    
    @property
    def lower_shadow(self) -> Decimal:
        """下ヒゲの長さ"""
        return min(self.open.value, self.close.value) - self.low.value
    
    @property
    def typical_price(self) -> Price:
        """代表価格 (HLC平均)"""
        avg = (self.high.value + self.low.value + self.close.value) / 3
        return Price(avg, self.close.currency)
    
    @property
    def weighted_close(self) -> Price:
        """加重終値 (HLCC平均)"""
        avg = (self.high.value + self.low.value + self.close.value * 2) / 4
        return Price(avg, self.close.currency)
    
    def __str__(self) -> str:
        """文字列表現"""
        return (f"OHLCV(O:{self.open.value}, H:{self.high.value}, "
                f"L:{self.low.value}, C:{self.close.value}, V:{self.volume})")
    
    def __repr__(self) -> str:
        """詳細表現"""
        return (f"OHLCV(open={self.open!r}, high={self.high!r}, "
                f"low={self.low!r}, close={self.close!r}, "
                f"volume={self.volume}, timestamp={self.timestamp!r})")