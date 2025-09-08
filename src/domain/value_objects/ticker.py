"""ティッカーシンボル値オブジェクト"""
from __future__ import annotations
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Ticker:
    """ティッカーシンボルを表す値オブジェクト"""
    
    symbol: str
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        self._validate()
    
    def _validate(self) -> None:
        """バリデーション"""
        if not self.symbol:
            raise ValueError("Ticker symbol cannot be empty")
        
        # 日本株の場合（4桁の数字）
        if not re.match(r'^[0-9]{4}$', self.symbol):
            # 米国株の場合（1-5文字の英字）
            if not re.match(r'^[A-Z]{1,5}$', self.symbol.upper()):
                raise ValueError(
                    f"Invalid ticker symbol: {self.symbol}. "
                    "Must be 4-digit number (JP) or 1-5 letters (US)"
                )
    
    @property
    def is_japanese(self) -> bool:
        """日本株かどうか"""
        return self.symbol.isdigit() and len(self.symbol) == 4
    
    @property
    def is_us(self) -> bool:
        """米国株かどうか"""
        return self.symbol.isalpha() and 1 <= len(self.symbol) <= 5
    
    @property
    def exchange(self) -> str:
        """取引所を推定"""
        if self.is_japanese:
            return "TSE"
        elif self.is_us:
            return "NYSE/NASDAQ"
        else:
            return "UNKNOWN"
    
    def __str__(self) -> str:
        """文字列表現"""
        return self.symbol
    
    def __repr__(self) -> str:
        """詳細表現"""
        return f"Ticker({self.symbol!r})"
    
    def __eq__(self, other: object) -> bool:
        """等価比較"""
        if not isinstance(other, Ticker):
            return False
        return self.symbol == other.symbol
    
    def __hash__(self) -> int:
        """ハッシュ値"""
        return hash(self.symbol)
    
    def __lt__(self, other: Ticker) -> bool:
        """小なり比較"""
        if not isinstance(other, Ticker):
            raise TypeError("Cannot compare Ticker with non-Ticker type")
        return self.symbol < other.symbol