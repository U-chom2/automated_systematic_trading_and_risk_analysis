"""取引エンティティ"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from decimal import Decimal
from enum import Enum


class TradeType(Enum):
    """取引種別"""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class OrderType(Enum):
    """注文種別"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """注文ステータス"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Trade:
    """取引エンティティ
    
    株式取引を表すドメインエンティティ。
    """
    
    portfolio_id: UUID
    stock_id: UUID
    ticker: str  # 参照用
    trade_type: TradeType
    quantity: int
    id: UUID = field(default_factory=uuid4)
    order_type: OrderType = field(default=OrderType.MARKET)
    order_price: Optional[Decimal] = field(default=None)
    executed_price: Optional[Decimal] = field(default=None)
    commission: Decimal = field(default=Decimal("0"))
    status: OrderStatus = field(default=OrderStatus.PENDING)
    signal_id: Optional[UUID] = field(default=None)  # AIシグナルとの関連
    execution_id: Optional[str] = field(default=None)  # 証券会社の約定ID
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = field(default=None)
    executed_at: Optional[datetime] = field(default=None)
    cancelled_at: Optional[datetime] = field(default=None)
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        self._validate()
    
    def _validate(self) -> None:
        """バリデーション"""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if self.order_type == OrderType.LIMIT and self.order_price is None:
            raise ValueError("Limit order requires order price")
        
        if self.order_price is not None and self.order_price <= 0:
            raise ValueError("Order price must be positive")
        
        if self.executed_price is not None and self.executed_price <= 0:
            raise ValueError("Executed price must be positive")
        
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
    
    @property
    def is_buy(self) -> bool:
        """買い注文かどうか"""
        return self.trade_type in (TradeType.BUY, TradeType.COVER)
    
    @property
    def is_sell(self) -> bool:
        """売り注文かどうか"""
        return self.trade_type in (TradeType.SELL, TradeType.SHORT)
    
    @property
    def is_executed(self) -> bool:
        """約定済みかどうか"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_pending(self) -> bool:
        """待機中かどうか"""
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)
    
    @property
    def is_cancelled(self) -> bool:
        """キャンセル済みかどうか"""
        return self.status == OrderStatus.CANCELLED
    
    @property
    def total_cost(self) -> Decimal:
        """総コスト（約定価格 × 数量 + 手数料）"""
        if not self.is_executed or self.executed_price is None:
            return Decimal("0")
        
        base_cost = self.executed_price * Decimal(self.quantity)
        if self.is_buy:
            return base_cost + self.commission
        else:
            return base_cost - self.commission
    
    @property
    def net_proceeds(self) -> Decimal:
        """純収入（売却の場合）"""
        if not self.is_sell or not self.is_executed or self.executed_price is None:
            return Decimal("0")
        
        return self.executed_price * Decimal(self.quantity) - self.commission
    
    def submit(self) -> None:
        """注文を送信"""
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot submit trade with status {self.status}")
        
        self.status = OrderStatus.SUBMITTED
        self.submitted_at = datetime.now()
    
    def execute(self, executed_price: Decimal, commission: Optional[Decimal] = None) -> None:
        """注文を約定"""
        if not self.is_pending:
            raise ValueError(f"Cannot execute trade with status {self.status}")
        
        if executed_price <= 0:
            raise ValueError("Executed price must be positive")
        
        self.executed_price = executed_price
        if commission is not None:
            self.commission = commission
        
        self.status = OrderStatus.FILLED
        self.executed_at = datetime.now()
    
    def cancel(self) -> None:
        """注文をキャンセル"""
        if not self.is_pending:
            raise ValueError(f"Cannot cancel trade with status {self.status}")
        
        self.status = OrderStatus.CANCELLED
        self.cancelled_at = datetime.now()
    
    def reject(self, reason: Optional[str] = None) -> None:
        """注文を拒否"""
        if self.status != OrderStatus.SUBMITTED:
            raise ValueError(f"Cannot reject trade with status {self.status}")
        
        self.status = OrderStatus.REJECTED
    
    def calculate_commission(self, rate: Decimal = Decimal("0.001")) -> Decimal:
        """手数料を計算（デフォルト: 0.1%）"""
        if self.executed_price is None:
            price = self.order_price or Decimal("0")
        else:
            price = self.executed_price
        
        return price * Decimal(self.quantity) * rate
    
    def __str__(self) -> str:
        """文字列表現"""
        price = self.executed_price or self.order_price or Decimal("0")
        return (
            f"{self.trade_type.value} {self.quantity} {self.ticker} "
            f"@ ¥{price:,.0f} ({self.status.value})"
        )
    
    def __repr__(self) -> str:
        """詳細表現"""
        return (
            f"Trade(id={self.id}, ticker={self.ticker!r}, "
            f"trade_type={self.trade_type}, quantity={self.quantity}, "
            f"status={self.status})"
        )