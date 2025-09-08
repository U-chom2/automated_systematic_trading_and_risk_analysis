"""株式エンティティ"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from decimal import Decimal


@dataclass(frozen=True)
class Stock:
    """株式エンティティ
    
    株式銘柄を表すドメインエンティティ。
    不変オブジェクトとして実装。
    """
    
    ticker: str
    company_name: str
    id: UUID = field(default_factory=uuid4)
    exchange: str = field(default="TSE")
    sector: Optional[str] = field(default=None)
    industry: Optional[str] = field(default=None)
    market_cap: Optional[Decimal] = field(default=None)
    is_active: bool = field(default=True)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """初期化後の検証"""
        self._validate()
    
    def _validate(self) -> None:
        """バリデーション"""
        if not self.ticker:
            raise ValueError("Ticker symbol is required")
        
        # 日本株の場合のバリデーション（XXXX.T形式）
        if self.ticker.endswith('.T'):
            ticker_code = self.ticker[:-2]
            if not ticker_code.isdigit() or len(ticker_code) != 4:
                raise ValueError("Japanese ticker must be 4-digit number followed by .T")
        
        if not self.company_name:
            raise ValueError("Company name is required")
        
        if self.market_cap is not None and self.market_cap <= 0:
            raise ValueError("Market cap must be positive")
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"{self.ticker} - {self.company_name}"
    
    def __repr__(self) -> str:
        """詳細表現"""
        return (
            f"Stock(ticker={self.ticker!r}, "
            f"company_name={self.company_name!r}, "
            f"exchange={self.exchange!r})"
        )