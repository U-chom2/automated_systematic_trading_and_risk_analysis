"""バリデーションヘルパー"""
from typing import Any, Optional, List, Dict, TypeVar, Type
from datetime import datetime, date, timedelta
from decimal import Decimal
from uuid import UUID
import re

from .exceptions import ValidationException


T = TypeVar("T")


class Validator:
    """バリデーター基底クラス"""
    
    @staticmethod
    def required(value: Any, field_name: str) -> Any:
        """必須チェック"""
        if value is None:
            raise ValidationException(field_name, value, "Field is required")
        if isinstance(value, str) and not value.strip():
            raise ValidationException(field_name, value, "Field cannot be empty")
        return value
    
    @staticmethod
    def min_value(value: float, min_val: float, field_name: str) -> float:
        """最小値チェック"""
        if value < min_val:
            raise ValidationException(
                field_name, value, f"Value must be at least {min_val}"
            )
        return value
    
    @staticmethod
    def max_value(value: float, max_val: float, field_name: str) -> float:
        """最大値チェック"""
        if value > max_val:
            raise ValidationException(
                field_name, value, f"Value must be at most {max_val}"
            )
        return value
    
    @staticmethod
    def range_value(
        value: float, min_val: float, max_val: float, field_name: str
    ) -> float:
        """範囲チェック"""
        if not min_val <= value <= max_val:
            raise ValidationException(
                field_name, value, f"Value must be between {min_val} and {max_val}"
            )
        return value
    
    @staticmethod
    def positive(value: float, field_name: str) -> float:
        """正数チェック"""
        if value <= 0:
            raise ValidationException(field_name, value, "Value must be positive")
        return value
    
    @staticmethod
    def non_negative(value: float, field_name: str) -> float:
        """非負数チェック"""
        if value < 0:
            raise ValidationException(field_name, value, "Value cannot be negative")
        return value
    
    @staticmethod
    def percentage(value: float, field_name: str) -> float:
        """パーセンテージチェック（0-100）"""
        if not 0 <= value <= 100:
            raise ValidationException(
                field_name, value, "Percentage must be between 0 and 100"
            )
        return value
    
    @staticmethod
    def ratio(value: float, field_name: str) -> float:
        """比率チェック（0-1）"""
        if not 0 <= value <= 1:
            raise ValidationException(
                field_name, value, "Ratio must be between 0 and 1"
            )
        return value


class StringValidator(Validator):
    """文字列バリデーター"""
    
    @staticmethod
    def min_length(value: str, min_len: int, field_name: str) -> str:
        """最小長チェック"""
        if len(value) < min_len:
            raise ValidationException(
                field_name, value, f"Length must be at least {min_len}"
            )
        return value
    
    @staticmethod
    def max_length(value: str, max_len: int, field_name: str) -> str:
        """最大長チェック"""
        if len(value) > max_len:
            raise ValidationException(
                field_name, value, f"Length must be at most {max_len}"
            )
        return value
    
    @staticmethod
    def pattern(value: str, pattern: str, field_name: str) -> str:
        """正規表現パターンチェック"""
        if not re.match(pattern, value):
            raise ValidationException(
                field_name, value, f"Value does not match pattern {pattern}"
            )
        return value
    
    @staticmethod
    def ticker(value: str, field_name: str = "ticker") -> str:
        """ティッカーシンボルチェック"""
        # 日本株の場合（4桁数字+.T）
        if re.match(r"^\d{4}\.T$", value):
            return value
        # 米国株の場合（1-5文字の大文字英字）
        if re.match(r"^[A-Z]{1,5}$", value):
            return value
        raise ValidationException(
            field_name, value, "Invalid ticker symbol format"
        )
    
    @staticmethod
    def email(value: str, field_name: str = "email") -> str:
        """メールアドレスチェック"""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, value):
            raise ValidationException(field_name, value, "Invalid email format")
        return value
    
    @staticmethod
    def url(value: str, field_name: str = "url") -> str:
        """URLチェック"""
        pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        if not re.match(pattern, value):
            raise ValidationException(field_name, value, "Invalid URL format")
        return value


class DateTimeValidator(Validator):
    """日時バリデーター"""
    
    @staticmethod
    def future_date(value: date, field_name: str) -> date:
        """未来日付チェック"""
        if value <= date.today():
            raise ValidationException(field_name, value, "Date must be in the future")
        return value
    
    @staticmethod
    def past_date(value: date, field_name: str) -> date:
        """過去日付チェック"""
        if value >= date.today():
            raise ValidationException(field_name, value, "Date must be in the past")
        return value
    
    @staticmethod
    def date_range(
        start_date: date, end_date: date, field_name: str = "date_range"
    ) -> tuple[date, date]:
        """日付範囲チェック"""
        if start_date >= end_date:
            raise ValidationException(
                field_name,
                (start_date, end_date),
                "Start date must be before end date"
            )
        return start_date, end_date
    
    @staticmethod
    def max_duration(
        start_date: date,
        end_date: date,
        max_days: int,
        field_name: str = "duration"
    ) -> tuple[date, date]:
        """最大期間チェック"""
        duration = (end_date - start_date).days
        if duration > max_days:
            raise ValidationException(
                field_name,
                duration,
                f"Duration cannot exceed {max_days} days"
            )
        return start_date, end_date
    
    @staticmethod
    def business_day(value: date, field_name: str) -> date:
        """営業日チェック（簡易版）"""
        if value.weekday() >= 5:  # 土日
            raise ValidationException(
                field_name, value, "Date must be a business day"
            )
        return value


class CollectionValidator(Validator):
    """コレクションバリデーター"""
    
    @staticmethod
    def not_empty(value: List[Any], field_name: str) -> List[Any]:
        """空でないことをチェック"""
        if not value:
            raise ValidationException(field_name, value, "List cannot be empty")
        return value
    
    @staticmethod
    def min_items(value: List[Any], min_count: int, field_name: str) -> List[Any]:
        """最小要素数チェック"""
        if len(value) < min_count:
            raise ValidationException(
                field_name, value, f"List must have at least {min_count} items"
            )
        return value
    
    @staticmethod
    def max_items(value: List[Any], max_count: int, field_name: str) -> List[Any]:
        """最大要素数チェック"""
        if len(value) > max_count:
            raise ValidationException(
                field_name, value, f"List cannot have more than {max_count} items"
            )
        return value
    
    @staticmethod
    def unique_items(value: List[Any], field_name: str) -> List[Any]:
        """重複チェック"""
        if len(value) != len(set(value)):
            raise ValidationException(field_name, value, "List must have unique items")
        return value
    
    @staticmethod
    def all_match(
        value: List[Any],
        validator_func: callable,
        field_name: str
    ) -> List[Any]:
        """全要素が条件を満たすことをチェック"""
        for i, item in enumerate(value):
            try:
                validator_func(item)
            except Exception as e:
                raise ValidationException(
                    f"{field_name}[{i}]", item, str(e)
                )
        return value


class TradingValidator(Validator):
    """取引関連バリデーター"""
    
    @staticmethod
    def quantity(value: int, field_name: str = "quantity") -> int:
        """数量チェック"""
        if value <= 0:
            raise ValidationException(field_name, value, "Quantity must be positive")
        if value % 100 != 0:  # 日本株の単元株（100株単位）
            raise ValidationException(
                field_name, value, "Quantity must be in units of 100"
            )
        return value
    
    @staticmethod
    def price(value: Decimal, field_name: str = "price") -> Decimal:
        """価格チェック"""
        if value <= 0:
            raise ValidationException(field_name, value, "Price must be positive")
        # 呼値単位のチェックも可能
        return value
    
    @staticmethod
    def commission_rate(value: float, field_name: str = "commission_rate") -> float:
        """手数料率チェック"""
        if not 0 <= value <= 0.01:  # 0-1%
            raise ValidationException(
                field_name, value, "Commission rate must be between 0 and 1%"
            )
        return value
    
    @staticmethod
    def slippage_rate(value: float, field_name: str = "slippage_rate") -> float:
        """スリッページ率チェック"""
        if not 0 <= value <= 0.01:  # 0-1%
            raise ValidationException(
                field_name, value, "Slippage rate must be between 0 and 1%"
            )
        return value
    
    @staticmethod
    def stop_loss(
        stop_loss: float,
        entry_price: float,
        field_name: str = "stop_loss"
    ) -> float:
        """ストップロスチェック"""
        if stop_loss >= entry_price:
            raise ValidationException(
                field_name, stop_loss, "Stop loss must be below entry price"
            )
        max_loss_pct = 0.2  # 最大20%の損失
        min_stop = entry_price * (1 - max_loss_pct)
        if stop_loss < min_stop:
            raise ValidationException(
                field_name, stop_loss, f"Stop loss too far ({max_loss_pct*100}% max)"
            )
        return stop_loss
    
    @staticmethod
    def take_profit(
        take_profit: float,
        entry_price: float,
        field_name: str = "take_profit"
    ) -> float:
        """利益確定チェック"""
        if take_profit <= entry_price:
            raise ValidationException(
                field_name, take_profit, "Take profit must be above entry price"
            )
        return take_profit


def validate_dto(dto_class: Type[T], data: Dict[str, Any]) -> T:
    """DTOバリデーション
    
    Args:
        dto_class: DTOクラス
        data: バリデーション対象データ
    
    Returns:
        バリデーション済みDTOインスタンス
    """
    try:
        # DTOクラスに validate メソッドがある場合は呼び出す
        instance = dto_class(**data)
        if hasattr(instance, "validate"):
            instance.validate()
        return instance
    except Exception as e:
        raise ValidationException("dto", data, str(e))