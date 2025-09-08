"""パーセンテージ値オブジェクト"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Union


@dataclass(frozen=True)
class Percentage:
    """パーセンテージを表す値オブジェクト"""
    
    value: Decimal
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        # Decimalに変換
        object.__setattr__(self, 'value', Decimal(str(self.value)))
        
        # -100% から +∞ までを許可（-100%は全損を表す）
        if self.value < -100:
            raise ValueError("Percentage cannot be less than -100%")
    
    @classmethod
    def from_decimal(cls, decimal_value: Union[Decimal, float]) -> Percentage:
        """小数値から作成（0.05 -> 5%）"""
        return cls(Decimal(str(decimal_value)) * 100)
    
    @classmethod
    def from_basis_points(cls, bps: Union[int, float]) -> Percentage:
        """ベーシスポイントから作成（50bps -> 0.5%）"""
        return cls(Decimal(str(bps)) / 100)
    
    @property
    def decimal(self) -> Decimal:
        """小数値として取得（5% -> 0.05）"""
        return self.value / 100
    
    @property
    def basis_points(self) -> Decimal:
        """ベーシスポイントとして取得（0.5% -> 50bps）"""
        return self.value * 100
    
    @property
    def is_positive(self) -> bool:
        """正の値かどうか"""
        return self.value > 0
    
    @property
    def is_negative(self) -> bool:
        """負の値かどうか"""
        return self.value < 0
    
    @property
    def is_zero(self) -> bool:
        """ゼロかどうか"""
        return self.value == 0
    
    def __add__(self, other: Union[Percentage, Decimal, int, float]) -> Percentage:
        """加算"""
        if isinstance(other, Percentage):
            return Percentage(self.value + other.value)
        else:
            return Percentage(self.value + Decimal(str(other)))
    
    def __sub__(self, other: Union[Percentage, Decimal, int, float]) -> Percentage:
        """減算"""
        if isinstance(other, Percentage):
            result = self.value - other.value
        else:
            result = self.value - Decimal(str(other))
        
        if result < -100:
            raise ValueError("Percentage cannot be less than -100% after subtraction")
        
        return Percentage(result)
    
    def __mul__(self, other: Union[Decimal, int, float]) -> Percentage:
        """乗算"""
        return Percentage(self.value * Decimal(str(other)))
    
    def __truediv__(self, other: Union[Decimal, int, float]) -> Percentage:
        """除算"""
        if other == 0:
            raise ValueError("Cannot divide by zero")
        return Percentage(self.value / Decimal(str(other)))
    
    def __neg__(self) -> Percentage:
        """符号反転"""
        return Percentage(-self.value)
    
    def __abs__(self) -> Percentage:
        """絶対値"""
        return Percentage(abs(self.value))
    
    def __eq__(self, other: object) -> bool:
        """等価比較"""
        if isinstance(other, Percentage):
            return self.value == other.value
        elif isinstance(other, (int, float, Decimal)):
            return self.value == Decimal(str(other))
        return False
    
    def __lt__(self, other: Union[Percentage, Decimal, int, float]) -> bool:
        """小なり比較"""
        if isinstance(other, Percentage):
            return self.value < other.value
        else:
            return self.value < Decimal(str(other))
    
    def __le__(self, other: Union[Percentage, Decimal, int, float]) -> bool:
        """小なりイコール比較"""
        return self < other or self == other
    
    def __gt__(self, other: Union[Percentage, Decimal, int, float]) -> bool:
        """大なり比較"""
        if isinstance(other, Percentage):
            return self.value > other.value
        else:
            return self.value > Decimal(str(other))
    
    def __ge__(self, other: Union[Percentage, Decimal, int, float]) -> bool:
        """大なりイコール比較"""
        return self > other or self == other
    
    def __hash__(self) -> int:
        """ハッシュ値"""
        return hash(self.value)
    
    def __str__(self) -> str:
        """文字列表現"""
        if self.value == self.value.to_integral_value():
            return f"{self.value:.0f}%"
        else:
            return f"{self.value:.2f}%"
    
    def __repr__(self) -> str:
        """詳細表現"""
        return f"Percentage({self.value})"
    
    def apply_to(self, amount: Union[Decimal, int, float]) -> Decimal:
        """金額に適用"""
        return Decimal(str(amount)) * self.decimal
    
    def compound(self, other: Percentage) -> Percentage:
        """複利計算（1 + r1) * (1 + r2) - 1"""
        r1 = self.decimal
        r2 = other.decimal
        result = (1 + r1) * (1 + r2) - 1
        return Percentage.from_decimal(result)


@dataclass(frozen=True)
class Rate(Percentage):
    """レート（利率・手数料率など）を表す値オブジェクト
    
    Percentageの特殊化で、0以上の値のみを許可する。
    """
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        # 親クラスの初期化
        super().__post_init__()
        
        # レートは0以上
        if self.value < 0:
            raise ValueError("Rate cannot be negative")
    
    @classmethod
    def from_annual_to_daily(cls, annual_rate: Rate) -> Rate:
        """年率から日率を計算"""
        # (1 + annual_rate)^(1/365) - 1
        daily = (1 + annual_rate.decimal) ** (Decimal("1") / Decimal("365")) - 1
        return cls.from_decimal(daily)
    
    @classmethod
    def from_annual_to_monthly(cls, annual_rate: Rate) -> Rate:
        """年率から月率を計算"""
        # (1 + annual_rate)^(1/12) - 1
        monthly = (1 + annual_rate.decimal) ** (Decimal("1") / Decimal("12")) - 1
        return cls.from_decimal(monthly)
    
    def to_annual_from_daily(self) -> Rate:
        """日率から年率を計算"""
        # (1 + daily_rate)^365 - 1
        annual = (1 + self.decimal) ** Decimal("365") - 1
        return Rate.from_decimal(annual)
    
    def to_annual_from_monthly(self) -> Rate:
        """月率から年率を計算"""
        # (1 + monthly_rate)^12 - 1
        annual = (1 + self.decimal) ** Decimal("12") - 1
        return Rate.from_decimal(annual)
    
    def compound_periods(self, periods: int) -> Rate:
        """複数期間の複利計算"""
        if periods < 0:
            raise ValueError("Periods must be non-negative")
        
        # (1 + rate)^periods - 1
        result = (1 + self.decimal) ** Decimal(periods) - 1
        return Rate.from_decimal(result)
    
    def __str__(self) -> str:
        """文字列表現"""
        if self.value == self.value.to_integral_value():
            return f"{self.value:.0f}%"
        else:
            # 小数点以下4桁まで表示（bps単位の精度）
            return f"{self.value:.4f}%"