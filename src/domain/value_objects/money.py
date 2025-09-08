"""金額値オブジェクト"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Union


@dataclass(frozen=True)
class Money:
    """金額を表す値オブジェクト"""
    
    amount: Decimal
    currency: str = "JPY"
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        # Decimalに変換
        object.__setattr__(self, 'amount', Decimal(str(self.amount)))
        
        if self.amount < 0:
            raise ValueError("Money amount cannot be negative")
    
    def __add__(self, other: Union[Money, Decimal, int, float]) -> Money:
        """加算"""
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError(f"Cannot add different currencies: {self.currency} and {other.currency}")
            return Money(self.amount + other.amount, self.currency)
        else:
            return Money(self.amount + Decimal(str(other)), self.currency)
    
    def __sub__(self, other: Union[Money, Decimal, int, float]) -> Money:
        """減算"""
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError(f"Cannot subtract different currencies: {self.currency} and {other.currency}")
            result = self.amount - other.amount
        else:
            result = self.amount - Decimal(str(other))
        
        if result < 0:
            raise ValueError("Money amount cannot be negative after subtraction")
        
        return Money(result, self.currency)
    
    def __mul__(self, other: Union[Decimal, int, float]) -> Money:
        """乗算"""
        return Money(self.amount * Decimal(str(other)), self.currency)
    
    def __truediv__(self, other: Union[Decimal, int, float]) -> Money:
        """除算"""
        if other == 0:
            raise ValueError("Cannot divide by zero")
        return Money(self.amount / Decimal(str(other)), self.currency)
    
    def __eq__(self, other: object) -> bool:
        """等価比較"""
        if not isinstance(other, Money):
            return False
        return self.amount == other.amount and self.currency == other.currency
    
    def __lt__(self, other: Money) -> bool:
        """小なり比較"""
        if not isinstance(other, Money):
            raise TypeError("Cannot compare Money with non-Money type")
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare different currencies: {self.currency} and {other.currency}")
        return self.amount < other.amount
    
    def __le__(self, other: Money) -> bool:
        """小なりイコール比較"""
        return self < other or self == other
    
    def __gt__(self, other: Money) -> bool:
        """大なり比較"""
        if not isinstance(other, Money):
            raise TypeError("Cannot compare Money with non-Money type")
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare different currencies: {self.currency} and {other.currency}")
        return self.amount > other.amount
    
    def __ge__(self, other: Money) -> bool:
        """大なりイコール比較"""
        return self > other or self == other
    
    def __str__(self) -> str:
        """文字列表現"""
        if self.currency == "JPY":
            return f"¥{self.amount:,.0f}"
        else:
            return f"{self.currency} {self.amount:,.2f}"
    
    def __repr__(self) -> str:
        """詳細表現"""
        return f"Money(amount={self.amount}, currency={self.currency!r})"


@dataclass(frozen=True)
class Yen(Money):
    """日本円を表す値オブジェクト"""
    
    def __init__(self, amount: Union[Decimal, int, float]) -> None:
        """初期化"""
        object.__setattr__(self, 'amount', Decimal(str(amount)))
        object.__setattr__(self, 'currency', "JPY")
        
        if self.amount < 0:
            raise ValueError("Yen amount cannot be negative")