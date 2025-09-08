"""取引関連DTO"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from uuid import UUID

from ...domain.entities.trade import Trade, TradeType, OrderType, OrderStatus


@dataclass
class TradeDTO:
    """取引DTO"""
    id: str
    portfolio_id: str
    stock_id: str
    ticker: str
    trade_type: str
    order_type: str
    quantity: int
    order_price: Optional[float]
    executed_price: Optional[float]
    commission: float
    status: str
    signal_id: Optional[str]
    execution_id: Optional[str]
    created_at: str
    submitted_at: Optional[str]
    executed_at: Optional[str]
    cancelled_at: Optional[str]
    total_cost: float
    net_proceeds: float
    
    @classmethod
    def from_entity(cls, trade: Trade) -> "TradeDTO":
        """エンティティからDTOを作成"""
        return cls(
            id=str(trade.id),
            portfolio_id=str(trade.portfolio_id),
            stock_id=str(trade.stock_id),
            ticker=trade.ticker,
            trade_type=trade.trade_type.value,
            order_type=trade.order_type.value,
            quantity=trade.quantity,
            order_price=float(trade.order_price) if trade.order_price else None,
            executed_price=float(trade.executed_price) if trade.executed_price else None,
            commission=float(trade.commission),
            status=trade.status.value,
            signal_id=str(trade.signal_id) if trade.signal_id else None,
            execution_id=trade.execution_id,
            created_at=trade.created_at.isoformat(),
            submitted_at=trade.submitted_at.isoformat() if trade.submitted_at else None,
            executed_at=trade.executed_at.isoformat() if trade.executed_at else None,
            cancelled_at=trade.cancelled_at.isoformat() if trade.cancelled_at else None,
            total_cost=float(trade.total_cost),
            net_proceeds=float(trade.net_proceeds),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "stock_id": self.stock_id,
            "ticker": self.ticker,
            "trade_type": self.trade_type,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "order_price": self.order_price,
            "executed_price": self.executed_price,
            "commission": self.commission,
            "status": self.status,
            "signal_id": self.signal_id,
            "execution_id": self.execution_id,
            "timestamps": {
                "created_at": self.created_at,
                "submitted_at": self.submitted_at,
                "executed_at": self.executed_at,
                "cancelled_at": self.cancelled_at,
            },
            "financials": {
                "total_cost": self.total_cost,
                "net_proceeds": self.net_proceeds,
            }
        }


@dataclass
class CreateTradeDTO:
    """取引作成DTO"""
    portfolio_id: UUID
    stock_id: UUID
    ticker: str
    trade_type: str
    quantity: int
    order_type: str = "MARKET"
    order_price: Optional[Decimal] = None
    signal_id: Optional[UUID] = None
    
    def validate(self) -> None:
        """バリデーション"""
        # 取引タイプの検証
        valid_trade_types = ["BUY", "SELL", "SHORT", "COVER"]
        if self.trade_type not in valid_trade_types:
            raise ValueError(f"Trade type must be one of {valid_trade_types}")
        
        # 注文タイプの検証
        valid_order_types = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
        if self.order_type not in valid_order_types:
            raise ValueError(f"Order type must be one of {valid_order_types}")
        
        # 数量の検証
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # 指値注文の場合は価格が必要
        if self.order_type in ["LIMIT", "STOP", "STOP_LIMIT"] and not self.order_price:
            raise ValueError(f"{self.order_type} order requires order price")
        
        # 価格の検証
        if self.order_price is not None and self.order_price <= 0:
            raise ValueError("Order price must be positive")
    
    def to_entity(self) -> Trade:
        """エンティティに変換"""
        return Trade(
            portfolio_id=self.portfolio_id,
            stock_id=self.stock_id,
            ticker=self.ticker,
            trade_type=TradeType[self.trade_type],
            order_type=OrderType[self.order_type],
            quantity=self.quantity,
            order_price=self.order_price,
            signal_id=self.signal_id,
        )


@dataclass
class UpdateTradeDTO:
    """取引更新DTO"""
    status: Optional[str] = None
    executed_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    execution_id: Optional[str] = None
    
    def validate(self) -> None:
        """バリデーション"""
        if self.status:
            valid_statuses = ["PENDING", "SUBMITTED", "PARTIAL", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"]
            if self.status not in valid_statuses:
                raise ValueError(f"Status must be one of {valid_statuses}")
        
        if self.executed_price is not None and self.executed_price <= 0:
            raise ValueError("Executed price must be positive")
        
        if self.commission is not None and self.commission < 0:
            raise ValueError("Commission cannot be negative")


@dataclass
class TradeExecutionDTO:
    """取引約定DTO"""
    trade_id: UUID
    executed_price: Decimal
    commission: Decimal
    execution_id: str
    executed_at: datetime
    
    def validate(self) -> None:
        """バリデーション"""
        if self.executed_price <= 0:
            raise ValueError("Executed price must be positive")
        
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
        
        if not self.execution_id:
            raise ValueError("Execution ID is required")


@dataclass
class TradeSummaryDTO:
    """取引サマリーDTO"""
    total_trades: int
    pending_trades: int
    executed_trades: int
    cancelled_trades: int
    total_buy_volume: float
    total_sell_volume: float
    total_commission: float
    net_profit_loss: float
    win_rate: float
    average_return: float
    
    @classmethod
    def from_trades(cls, trades: list[Trade]) -> "TradeSummaryDTO":
        """取引リストからサマリーを作成"""
        total_trades = len(trades)
        pending_trades = sum(1 for t in trades if t.is_pending)
        executed_trades = sum(1 for t in trades if t.is_executed)
        cancelled_trades = sum(1 for t in trades if t.is_cancelled)
        
        total_buy_volume = sum(
            float(t.total_cost) for t in trades
            if t.is_buy and t.is_executed
        )
        total_sell_volume = sum(
            float(t.net_proceeds) for t in trades
            if t.is_sell and t.is_executed
        )
        total_commission = sum(float(t.commission) for t in trades if t.is_executed)
        
        net_profit_loss = total_sell_volume - total_buy_volume - total_commission
        
        # 勝率計算（簡易版）
        profitable_trades = [t for t in trades if t.is_executed and t.is_sell]
        win_rate = len(profitable_trades) / executed_trades if executed_trades > 0 else 0.0
        
        # 平均リターン計算（簡易版）
        average_return = net_profit_loss / total_buy_volume if total_buy_volume > 0 else 0.0
        
        return cls(
            total_trades=total_trades,
            pending_trades=pending_trades,
            executed_trades=executed_trades,
            cancelled_trades=cancelled_trades,
            total_buy_volume=total_buy_volume,
            total_sell_volume=total_sell_volume,
            total_commission=total_commission,
            net_profit_loss=net_profit_loss,
            win_rate=win_rate,
            average_return=average_return,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "counts": {
                "total": self.total_trades,
                "pending": self.pending_trades,
                "executed": self.executed_trades,
                "cancelled": self.cancelled_trades,
            },
            "volumes": {
                "buy": self.total_buy_volume,
                "sell": self.total_sell_volume,
                "commission": self.total_commission,
            },
            "performance": {
                "net_profit_loss": self.net_profit_loss,
                "win_rate": self.win_rate,
                "average_return": self.average_return,
            }
        }