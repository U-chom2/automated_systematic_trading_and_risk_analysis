"""数量値オブジェクト"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Quantity:
    """株式数量を表す値オブジェクト"""
    
    value: int
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        if not isinstance(self.value, int):
            raise TypeError("Quantity must be an integer")
        
        if self.value < 0:
            raise ValueError("Quantity cannot be negative")
        
        # 日本株の単元株チェック（100株単位）
        if self.value > 0 and self.value % 100 != 0:
            # 単元未満株の場合は警告（エラーにはしない）
            pass
    
    @classmethod
    def from_units(cls, units: int, unit_size: int = 100) -> Quantity:
        """単元数から数量を作成"""
        return cls(units * unit_size)
    
    @property
    def is_zero(self) -> bool:
        """ゼロかどうか"""
        return self.value == 0
    
    @property
    def is_unit_share(self) -> bool:
        """単元株かどうか（100株単位）"""
        return self.value % 100 == 0
    
    @property
    def units(self) -> int:
        """単元数（100株単位）"""
        return self.value // 100
    
    @property
    def odd_lot(self) -> int:
        """単元未満株数"""
        return self.value % 100
    
    def __add__(self, other: Union[Quantity, int]) -> Quantity:
        """加算"""
        if isinstance(other, Quantity):
            return Quantity(self.value + other.value)
        else:
            return Quantity(self.value + other)
    
    def __sub__(self, other: Union[Quantity, int]) -> Quantity:
        """減算"""
        if isinstance(other, Quantity):
            result = self.value - other.value
        else:
            result = self.value - other
        
        if result < 0:
            raise ValueError("Quantity cannot be negative after subtraction")
        
        return Quantity(result)
    
    def __mul__(self, other: int) -> Quantity:
        """乗算"""
        if not isinstance(other, int):
            raise TypeError("Can only multiply quantity by integer")
        
        if other < 0:
            raise ValueError("Cannot multiply quantity by negative number")
        
        return Quantity(self.value * other)
    
    def __truediv__(self, other: Union[Quantity, int]) -> Union[float, Quantity]:
        """除算"""
        if isinstance(other, Quantity):
            if other.value == 0:
                raise ValueError("Cannot divide by zero quantity")
            return self.value / other.value
        else:
            if other == 0:
                raise ValueError("Cannot divide by zero")
            
            # 整数除算の場合
            if self.value % other == 0:
                return Quantity(self.value // other)
            else:
                # 端数が出る場合は float を返す
                return self.value / other
    
    def __floordiv__(self, other: int) -> Quantity:
        """整数除算"""
        if other == 0:
            raise ValueError("Cannot divide by zero")
        
        if other < 0:
            raise ValueError("Cannot divide quantity by negative number")
        
        return Quantity(self.value // other)
    
    def __mod__(self, other: int) -> Quantity:
        """剰余"""
        if other <= 0:
            raise ValueError("Modulo must be positive")
        
        return Quantity(self.value % other)
    
    def __eq__(self, other: object) -> bool:
        """等価比較"""
        if isinstance(other, Quantity):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return False
    
    def __lt__(self, other: Union[Quantity, int]) -> bool:
        """小なり比較"""
        if isinstance(other, Quantity):
            return self.value < other.value
        elif isinstance(other, int):
            return self.value < other
        else:
            raise TypeError("Cannot compare Quantity with non-numeric type")
    
    def __le__(self, other: Union[Quantity, int]) -> bool:
        """小なりイコール比較"""
        return self < other or self == other
    
    def __gt__(self, other: Union[Quantity, int]) -> bool:
        """大なり比較"""
        if isinstance(other, Quantity):
            return self.value > other.value
        elif isinstance(other, int):
            return self.value > other
        else:
            raise TypeError("Cannot compare Quantity with non-numeric type")
    
    def __ge__(self, other: Union[Quantity, int]) -> bool:
        """大なりイコール比較"""
        return self > other or self == other
    
    def __hash__(self) -> int:
        """ハッシュ値"""
        return hash(self.value)
    
    def __str__(self) -> str:
        """文字列表現"""
        if self.is_unit_share:
            return f"{self.value:,}株"
        else:
            units = self.units
            odd = self.odd_lot
            if units > 0:
                return f"{self.value:,}株 ({units}単元 + {odd}株)"
            else:
                return f"{self.value:,}株 (単元未満)"
    
    def __repr__(self) -> str:
        """詳細表現"""
        return f"Quantity({self.value})"
    
    def __bool__(self) -> bool:
        """bool変換"""
        return self.value > 0
    
    def split(self, ratio: int) -> Quantity:
        """株式分割"""
        if ratio <= 0:
            raise ValueError("Split ratio must be positive")
        
        return Quantity(self.value * ratio)
    
    def consolidate(self, ratio: int) -> Quantity:
        """株式併合"""
        if ratio <= 0:
            raise ValueError("Consolidation ratio must be positive")
        
        if self.value % ratio != 0:
            raise ValueError(f"Cannot consolidate {self.value} shares by ratio {ratio} without fractional shares")
        
        return Quantity(self.value // ratio)