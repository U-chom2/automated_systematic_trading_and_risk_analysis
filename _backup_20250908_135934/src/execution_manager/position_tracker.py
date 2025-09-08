"""Position tracking and OCO order state management."""

from typing import Dict, List, Any, Optional, Set
from decimal import Decimal
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


class TradeResult(Enum):
    """Trade result."""
    PROFIT = "profit"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    ONGOING = "ongoing"


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: int
    entry_price: Decimal
    entry_time: datetime
    position_type: str = "LONG"
    status: PositionStatus = PositionStatus.OPEN
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    buy_order_id: Optional[str] = None
    oco_order_id: Optional[str] = None
    position_id: str = field(default_factory=lambda: f"pos_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    current_price: Decimal = field(default_factory=lambda: Decimal("0.0"))
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0.0"))
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0.0"))
    commission_paid: Decimal = field(default_factory=lambda: Decimal("0.0"))
    trade_result: TradeResult = TradeResult.ONGOING


@dataclass
class TradeRecord:
    """Trade record data structure."""
    symbol: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: int
    entry_time: datetime
    exit_time: datetime
    profit_loss: Decimal
    profit_loss_percentage: float
    holding_time: timedelta = None

    def __post_init__(self):
        if self.holding_time is None:
            self.holding_time = self.exit_time - self.entry_time


@dataclass
class PortfolioSummary:
    """Portfolio summary statistics."""
    total_positions: int
    open_positions: int
    closed_positions: int
    total_realized_pnl: Decimal
    total_unrealized_pnl: Decimal
    total_commission: Decimal
    win_rate: float
    average_win: Decimal
    average_loss: Decimal
    profit_factor: float
    max_drawdown: Decimal
    current_exposure: Decimal


class PositionTracker:
    """Position and OCO order state management."""
    
    def __init__(self) -> None:
        """Initialize Position Tracker."""
        self.positions: Dict[str, Position] = {}  # symbol -> position
        self.closed_positions: List[Position] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.max_positions = 10  # Maximum concurrent positions
        logger.info("PositionTracker initialized")
    
    def add_position(self, symbol: str, quantity: int, entry_price: Decimal,
                    stop_loss_price: Decimal, take_profit_price: Decimal,
                    buy_order_id: str, oco_order_id: str) -> bool:
        """
        Add new position to tracking.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            take_profit_price: Take-profit price
            buy_order_id: Buy order ID
            oco_order_id: OCO order ID
            
        Returns:
            True if position added successfully
        """
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Maximum positions ({self.max_positions}) reached")
            return False
        
        if symbol in self.positions:
            logger.warning(f"Position for {symbol} already exists")
            return False
        
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            buy_order_id=buy_order_id,
            oco_order_id=oco_order_id,
            current_price=entry_price
        )
        
        self.positions[symbol] = position
        logger.info(f"Added position: {symbol} x {quantity} @ {entry_price}")
        return True
    
    def update_position_price(self, symbol: str, current_price: Decimal) -> None:
        """
        Update current price and calculate unrealized PnL.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.current_price = current_price
        
        # Calculate unrealized PnL
        price_change = current_price - position.entry_price
        position.unrealized_pnl = price_change * Decimal(str(position.quantity))
        
        logger.debug(f"Updated {symbol} price: {current_price}, "
                    f"unrealized PnL: {position.unrealized_pnl}")
    
    def close_position(self, symbol: str, exit_price: Decimal, 
                      exit_reason: str = "manual") -> bool:
        """
        Close position and calculate realized PnL.
        
        Args:
            symbol: Stock symbol
            exit_price: Exit price
            exit_reason: Reason for closing
            
        Returns:
            True if position closed successfully
        """
        if symbol not in self.positions:
            logger.warning(f"No open position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        # Calculate realized PnL
        price_change = exit_price - position.entry_price
        realized_pnl = price_change * Decimal(str(position.quantity))
        
        # Update position
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.realized_pnl = realized_pnl
        position.status = PositionStatus.CLOSED
        position.trade_result = self._classify_trade_result(realized_pnl)
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Record in trade history
        trade_record = {
            "symbol": symbol,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "entry_time": position.entry_time,
            "exit_time": position.exit_time,
            "realized_pnl": realized_pnl,
            "exit_reason": exit_reason,
            "trade_result": position.trade_result.value
        }
        self.trade_history.append(trade_record)
        
        logger.info(f"Closed position: {symbol} @ {exit_price}, "
                   f"PnL: {realized_pnl}, reason: {exit_reason}")
        
        return True
    
    def _classify_trade_result(self, pnl: Decimal) -> TradeResult:
        """
        Classify trade result based on PnL.
        
        Args:
            pnl: Realized profit/loss
            
        Returns:
            Trade result classification
        """
        if pnl > Decimal("0.01"):  # Consider commission threshold
            return TradeResult.PROFIT
        elif pnl < Decimal("-0.01"):
            return TradeResult.LOSS
        else:
            return TradeResult.BREAKEVEN
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position by symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position object or None
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all open positions.
        
        Returns:
            Dictionary of open positions
        """
        return self.positions.copy()
    
    def get_open_symbols(self) -> Set[str]:
        """
        Get set of symbols with open positions.
        
        Returns:
            Set of symbols
        """
        return set(self.positions.keys())
    
    def check_stop_loss_trigger(self, symbol: str, current_price: Decimal) -> bool:
        """
        Check if stop-loss should be triggered.
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            
        Returns:
            True if stop-loss should be triggered
        """
        position = self.get_position(symbol)
        if not position:
            return False
        
        if current_price <= position.stop_loss_price:
            logger.warning(f"Stop-loss triggered for {symbol}: "
                          f"{current_price} <= {position.stop_loss_price}")
            return True
        
        return False
    
    def check_take_profit_trigger(self, symbol: str, current_price: Decimal) -> bool:
        """
        Check if take-profit should be triggered.
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            
        Returns:
            True if take-profit should be triggered
        """
        position = self.get_position(symbol)
        if not position:
            return False
        
        if current_price >= position.take_profit_price:
            logger.info(f"Take-profit triggered for {symbol}: "
                       f"{current_price} >= {position.take_profit_price}")
            return True
        
        return False
    
    def calculate_portfolio_summary(self) -> PortfolioSummary:
        """
        Calculate comprehensive portfolio statistics.
        
        Returns:
            Portfolio summary statistics
        """
        # Count positions
        open_positions = len(self.positions)
        closed_positions = len(self.closed_positions)
        total_positions = open_positions + closed_positions
        
        # Calculate PnL
        total_realized_pnl = sum(pos.realized_pnl for pos in self.closed_positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_commission = sum(pos.commission_paid for pos in self.closed_positions) + \
                          sum(pos.commission_paid for pos in self.positions.values())
        
        # Calculate win rate and averages
        profitable_trades = [pos for pos in self.closed_positions 
                           if pos.trade_result == TradeResult.PROFIT]
        losing_trades = [pos for pos in self.closed_positions 
                        if pos.trade_result == TradeResult.LOSS]
        
        win_rate = len(profitable_trades) / closed_positions if closed_positions > 0 else 0.0
        average_win = (sum(pos.realized_pnl for pos in profitable_trades) / 
                      len(profitable_trades)) if profitable_trades else Decimal("0.0")
        average_loss = (sum(abs(pos.realized_pnl) for pos in losing_trades) / 
                       len(losing_trades)) if losing_trades else Decimal("0.0")
        
        # Calculate profit factor
        gross_profit = sum(pos.realized_pnl for pos in profitable_trades)
        gross_loss = sum(abs(pos.realized_pnl) for pos in losing_trades)
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0
        
        # Calculate current exposure
        current_exposure = sum(pos.current_price * Decimal(str(pos.quantity)) 
                             for pos in self.positions.values())
        
        # TODO: Implement max drawdown calculation
        max_drawdown = Decimal("0.0")
        
        return PortfolioSummary(
            total_positions=total_positions,
            open_positions=open_positions,
            closed_positions=closed_positions,
            total_realized_pnl=total_realized_pnl,
            total_unrealized_pnl=total_unrealized_pnl,
            total_commission=total_commission,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            current_exposure=current_exposure
        )
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get complete trade history.
        
        Returns:
            List of trade records
        """
        return self.trade_history.copy()
    
    def clear_old_positions(self, days: int = 30) -> int:
        """
        Clear old closed positions from memory.
        
        Args:
            days: Days to keep in memory
            
        Returns:
            Number of positions cleared
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_positions = [pos for pos in self.closed_positions 
                        if pos.exit_time and pos.exit_time < cutoff_date]
        
        for pos in old_positions:
            self.closed_positions.remove(pos)
        
        cleared_count = len(old_positions)
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old positions")
        
        return cleared_count
    
    def add_position(self, position: Position) -> None:
        """
        Add position to tracking.
        
        Args:
            position: Position object to add
        """
        self.positions[position.position_id] = position
        logger.info(f"Added position {position.position_id} for {position.symbol}")
    
    def get_active_positions(self) -> List[Position]:
        """
        Get all active positions.
        
        Returns:
            List of active positions
        """
        return [pos for pos in self.positions.values() if pos.status == PositionStatus.OPEN]
    
    def update_position(self, position_id: str, updates: Dict[str, Any]) -> None:
        """
        Update position with new values.
        
        Args:
            position_id: Position ID to update
            updates: Dictionary of field updates
        """
        if position_id in self.positions:
            position = self.positions[position_id]
            for field, value in updates.items():
                if hasattr(position, field):
                    setattr(position, field, value)
            logger.info(f"Updated position {position_id}: {updates}")
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position object or None
        """
        return self.positions.get(position_id)
    
    def close_position(self, position_id: str, exit_price: Decimal, 
                      exit_time: datetime, exit_reason: str = "") -> None:
        """
        Close position.
        
        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exit
        """
        if position_id in self.positions:
            position = self.positions[position_id]
            position.exit_price = exit_price
            position.exit_time = exit_time
            position.status = PositionStatus.CLOSED
            
            # Calculate P&L
            if position.position_type == "LONG":
                position.realized_pnl = (exit_price - position.entry_price) * Decimal(str(position.quantity))
            else:
                position.realized_pnl = (position.entry_price - exit_price) * Decimal(str(position.quantity))
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[position_id]
            
            logger.info(f"Closed position {position_id}: P&L={position.realized_pnl}")
    
    def get_trade_record(self, position_id: str) -> Optional[TradeRecord]:
        """
        Get trade record for closed position.
        
        Args:
            position_id: Position ID
            
        Returns:
            TradeRecord or None
        """
        # Look for closed position
        for pos in self.closed_positions:
            if pos.position_id == position_id and pos.exit_price and pos.exit_time:
                profit_loss = pos.realized_pnl
                profit_loss_percentage = float(profit_loss / (pos.entry_price * Decimal(str(pos.quantity))) * 100)
                
                return TradeRecord(
                    symbol=pos.symbol,
                    entry_price=pos.entry_price,
                    exit_price=pos.exit_price,
                    quantity=pos.quantity,
                    entry_time=pos.entry_time,
                    exit_time=pos.exit_time,
                    profit_loss=profit_loss,
                    profit_loss_percentage=profit_loss_percentage
                )
        
        return None
    
    def get_portfolio_statistics(self) -> Dict[str, Any]:
        """
        Get portfolio statistics.
        
        Returns:
            Portfolio statistics dictionary
        """
        total_trades = len(self.closed_positions)
        winning_trades = len([pos for pos in self.closed_positions if pos.realized_pnl > 0])
        losing_trades = len([pos for pos in self.closed_positions if pos.realized_pnl < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_profit_loss = sum(pos.realized_pnl for pos in self.closed_positions)
        
        # Calculate average trade duration
        if self.closed_positions:
            total_duration = sum(
                (pos.exit_time - pos.entry_time).total_seconds() 
                for pos in self.closed_positions 
                if pos.exit_time and pos.entry_time
            )
            avg_duration_seconds = total_duration / len(self.closed_positions)
            average_trade_duration = timedelta(seconds=avg_duration_seconds)
        else:
            average_trade_duration = timedelta(0)
        
        # Calculate max drawdown (simplified version)
        max_drawdown = Decimal("0")
        if self.closed_positions:
            running_total = Decimal("0")
            peak = Decimal("0")
            for pos in self.closed_positions:
                running_total += pos.realized_pnl
                if running_total > peak:
                    peak = running_total
                else:
                    drawdown = peak - running_total
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
        
        # Calculate basic Sharpe ratio (simplified)
        sharpe_ratio = 0.0
        if self.closed_positions and len(self.closed_positions) > 1:
            returns = [float(pos.realized_pnl) for pos in self.closed_positions]
            mean_return = sum(returns) / len(returns)
            std_return = (sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
            if std_return > 0:
                sharpe_ratio = mean_return / std_return
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit_loss": total_profit_loss,
            "average_trade_duration": average_trade_duration,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }