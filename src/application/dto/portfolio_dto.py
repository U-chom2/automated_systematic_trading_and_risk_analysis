"""ポートフォリオ関連DTO"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict
from uuid import UUID

from ...domain.entities.portfolio import Portfolio
from ...domain.entities.position import Position


@dataclass
class PositionDTO:
    """ポジションDTO"""
    id: str
    portfolio_id: str
    stock_id: str
    ticker: str
    quantity: int
    average_cost: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    opened_at: str
    closed_at: Optional[str] = None
    is_closed: bool = False
    
    @classmethod
    def from_entity(cls, position: Position) -> "PositionDTO":
        """エンティティからDTOを作成"""
        return cls(
            id=str(position.id),
            portfolio_id=str(position.portfolio_id),
            stock_id=str(position.stock_id),
            ticker=position.ticker,
            quantity=position.quantity,
            average_cost=float(position.average_cost),
            current_price=float(position.current_price),
            market_value=float(position.market_value),
            cost_basis=float(position.cost_basis),
            unrealized_pnl=float(position.unrealized_pnl),
            unrealized_pnl_percent=float(position.unrealized_pnl_percent),
            opened_at=position.opened_at.isoformat(),
            closed_at=position.closed_at.isoformat() if position.closed_at else None,
            is_closed=position.is_closed,
        )


@dataclass
class PortfolioDTO:
    """ポートフォリオDTO"""
    id: str
    name: str
    description: str
    initial_capital: float
    current_capital: float
    total_value: float
    available_cash: float
    total_invested: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    return_percentage: float
    currency: str
    strategy_type: str
    is_active: bool
    created_at: str
    updated_at: str
    positions: List[PositionDTO]
    position_count: int
    
    @classmethod
    def from_entity(cls, portfolio: Portfolio) -> "PortfolioDTO":
        """エンティティからDTOを作成"""
        positions = [
            PositionDTO.from_entity(pos)
            for pos in portfolio.positions.values()
            if not pos.is_closed
        ]
        
        return cls(
            id=str(portfolio.id),
            name=portfolio.name,
            description=portfolio.description or "",
            initial_capital=float(portfolio.initial_capital),
            current_capital=float(portfolio.current_capital),
            total_value=float(portfolio.total_value),
            available_cash=float(portfolio.available_cash),
            total_invested=float(portfolio.total_invested),
            unrealized_pnl=float(portfolio.unrealized_pnl),
            realized_pnl=float(portfolio.realized_pnl),
            total_pnl=float(portfolio.total_pnl),
            return_percentage=float(portfolio.return_percentage),
            currency=portfolio.currency,
            strategy_type=portfolio.strategy_type,
            is_active=portfolio.is_active,
            created_at=portfolio.created_at.isoformat(),
            updated_at=portfolio.updated_at.isoformat(),
            positions=positions,
            position_count=len(positions),
        )


@dataclass
class CreatePortfolioDTO:
    """ポートフォリオ作成DTO"""
    name: str
    initial_capital: Decimal
    description: Optional[str] = None
    currency: str = "JPY"
    strategy_type: str = "CORE_SATELLITE"
    
    def validate(self) -> None:
        """バリデーション"""
        if not self.name:
            raise ValueError("Portfolio name is required")
        
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        valid_currencies = ["JPY", "USD", "EUR"]
        if self.currency not in valid_currencies:
            raise ValueError(f"Currency must be one of {valid_currencies}")
        
        valid_strategies = ["CORE_SATELLITE", "MOMENTUM", "MEAN_REVERSION", "AI_DRIVEN", "HYBRID"]
        if self.strategy_type not in valid_strategies:
            raise ValueError(f"Strategy type must be one of {valid_strategies}")


@dataclass
class UpdatePortfolioDTO:
    """ポートフォリオ更新DTO"""
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    
    def validate(self) -> None:
        """バリデーション"""
        if self.name is not None and not self.name:
            raise ValueError("Portfolio name cannot be empty")


@dataclass
class PortfolioSummaryDTO:
    """ポートフォリオサマリーDTO"""
    id: str
    name: str
    total_value: float
    return_percentage: float
    unrealized_pnl: float
    position_count: int
    is_active: bool
    strategy_type: str
    
    @classmethod
    def from_portfolio_dto(cls, portfolio: PortfolioDTO) -> "PortfolioSummaryDTO":
        """PortfolioDTOからサマリーDTOを作成"""
        return cls(
            id=portfolio.id,
            name=portfolio.name,
            total_value=portfolio.total_value,
            return_percentage=portfolio.return_percentage,
            unrealized_pnl=portfolio.unrealized_pnl,
            position_count=portfolio.position_count,
            is_active=portfolio.is_active,
            strategy_type=portfolio.strategy_type,
        )


@dataclass
class PortfolioPerformanceDTO:
    """ポートフォリオパフォーマンスDTO"""
    portfolio_id: str
    portfolio_name: str
    period: str  # daily, weekly, monthly, yearly
    start_date: str
    end_date: str
    initial_value: float
    final_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    profit_factor: float
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "portfolio_id": self.portfolio_id,
            "portfolio_name": self.portfolio_name,
            "period": self.period,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "metrics": {
                "initial_value": self.initial_value,
                "final_value": self.final_value,
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "volatility": self.volatility,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
            },
            "trading_stats": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate,
                "average_win": self.average_win,
                "average_loss": self.average_loss,
                "profit_factor": self.profit_factor,
            }
        }