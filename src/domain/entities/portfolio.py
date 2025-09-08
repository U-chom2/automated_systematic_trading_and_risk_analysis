"""ポートフォリオエンティティ"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
from decimal import Decimal
from enum import Enum


class PortfolioStatus(Enum):
    """ポートフォリオステータス"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"
    CLOSED = "CLOSED"


@dataclass
class Portfolio:
    """ポートフォリオエンティティ
    
    投資ポートフォリオを表すドメインエンティティ。
    """
    
    name: str
    initial_capital: Decimal
    id: UUID = field(default_factory=uuid4)
    current_value: Decimal = field(default=Decimal("0"))
    cash_balance: Decimal = field(default=Decimal("0"))
    positions: List['Position'] = field(default_factory=list)
    status: PortfolioStatus = field(default=PortfolioStatus.ACTIVE)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        self._validate()
        if self.cash_balance == Decimal("0"):
            self.cash_balance = self.initial_capital
    
    def _validate(self) -> None:
        """バリデーション"""
        if not self.name:
            raise ValueError("Portfolio name is required")
        
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if self.cash_balance < 0:
            raise ValueError("Cash balance cannot be negative")
    
    @property
    def total_return(self) -> Decimal:
        """総収益を計算"""
        return self.current_value - self.initial_capital
    
    @property
    def return_rate(self) -> Decimal:
        """収益率を計算"""
        if self.initial_capital == 0:
            return Decimal("0")
        return (self.current_value - self.initial_capital) / self.initial_capital
    
    @property
    def invested_amount(self) -> Decimal:
        """投資額を計算"""
        return sum(
            pos.market_value for pos in self.positions 
            if not pos.is_closed
        )
    
    def calculate_total_value(self) -> Decimal:
        """ポートフォリオの総価値を計算"""
        positions_value = sum(
            pos.market_value for pos in self.positions 
            if not pos.is_closed
        )
        self.current_value = self.cash_balance + positions_value
        return self.current_value
    
    def add_position(self, position: 'Position') -> None:
        """ポジションを追加"""
        if position.portfolio_id != self.id:
            raise ValueError("Position portfolio_id does not match")
        
        # 購入金額を現金残高から引く
        cost = position.quantity * position.average_cost
        if cost > self.cash_balance:
            raise ValueError(f"Insufficient cash balance. Required: {cost}, Available: {self.cash_balance}")
        
        self.cash_balance -= cost
        self.positions.append(position)
        self.updated_at = datetime.now()
    
    def close_position(self, position_id: UUID, closing_price: Decimal) -> Decimal:
        """ポジションをクローズ"""
        position = next(
            (p for p in self.positions if p.id == position_id), 
            None
        )
        
        if not position:
            raise ValueError(f"Position {position_id} not found")
        
        if position.is_closed:
            raise ValueError(f"Position {position_id} is already closed")
        
        # 売却金額を現金残高に追加
        proceeds = position.quantity * closing_price
        self.cash_balance += proceeds
        
        # ポジションをクローズ
        realized_pnl = position.close(closing_price)
        self.updated_at = datetime.now()
        
        return realized_pnl
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"{self.name} (Value: ¥{self.current_value:,.0f})"
    
    def __repr__(self) -> str:
        """詳細表現"""
        return (
            f"Portfolio(id={self.id}, name={self.name!r}, "
            f"current_value={self.current_value}, status={self.status})"
        )