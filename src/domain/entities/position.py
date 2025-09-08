"""ポジションエンティティ"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from decimal import Decimal


@dataclass
class Position:
    """ポジションエンティティ
    
    株式保有ポジションを表すドメインエンティティ。
    """
    
    portfolio_id: UUID
    stock_id: UUID
    ticker: str  # 参照用
    quantity: int
    average_cost: Decimal
    id: UUID = field(default_factory=uuid4)
    current_price: Decimal = field(default=Decimal("0"))
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = field(default=None)
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        self._validate()
        if self.current_price == Decimal("0"):
            self.current_price = self.average_cost
    
    def _validate(self) -> None:
        """バリデーション"""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if self.average_cost <= 0:
            raise ValueError("Average cost must be positive")
        
        if self.current_price < 0:
            raise ValueError("Current price cannot be negative")
    
    @property
    def is_closed(self) -> bool:
        """ポジションがクローズされているか"""
        return self.closed_at is not None
    
    @property
    def market_value(self) -> Decimal:
        """現在の市場価値"""
        if self.is_closed:
            return Decimal("0")
        return Decimal(self.quantity) * self.current_price
    
    @property
    def cost_basis(self) -> Decimal:
        """取得原価"""
        return Decimal(self.quantity) * self.average_cost
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """未実現損益"""
        if self.is_closed:
            return Decimal("0")
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """未実現損益率"""
        if self.cost_basis == 0:
            return Decimal("0")
        return self.unrealized_pnl / self.cost_basis
    
    def update_price(self, new_price: Decimal) -> None:
        """現在価格を更新"""
        if new_price <= 0:
            raise ValueError("Price must be positive")
        
        if self.is_closed:
            raise ValueError("Cannot update price of closed position")
        
        self.current_price = new_price
    
    def close(self, closing_price: Decimal) -> Decimal:
        """ポジションをクローズ"""
        if self.is_closed:
            raise ValueError("Position is already closed")
        
        if closing_price <= 0:
            raise ValueError("Closing price must be positive")
        
        self.current_price = closing_price
        self.closed_at = datetime.now()
        
        # 実現損益を計算
        realized_pnl = (closing_price - self.average_cost) * Decimal(self.quantity)
        return realized_pnl
    
    def split_position(self, split_quantity: int) -> 'Position':
        """ポジションを分割"""
        if self.is_closed:
            raise ValueError("Cannot split closed position")
        
        if split_quantity <= 0 or split_quantity >= self.quantity:
            raise ValueError(f"Split quantity must be between 1 and {self.quantity - 1}")
        
        # 新しいポジションを作成
        new_position = Position(
            portfolio_id=self.portfolio_id,
            stock_id=self.stock_id,
            ticker=self.ticker,
            quantity=split_quantity,
            average_cost=self.average_cost,
            current_price=self.current_price,
            opened_at=self.opened_at
        )
        
        # 元のポジションの数量を減らす
        self.quantity -= split_quantity
        
        return new_position
    
    def __str__(self) -> str:
        """文字列表現"""
        status = "CLOSED" if self.is_closed else "OPEN"
        return f"{self.ticker}: {self.quantity}株 @ ¥{self.average_cost:,.0f} ({status})"
    
    def __repr__(self) -> str:
        """詳細表現"""
        return (
            f"Position(ticker={self.ticker!r}, quantity={self.quantity}, "
            f"average_cost={self.average_cost}, is_closed={self.is_closed})"
        )