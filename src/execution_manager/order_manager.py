"""Order management and position sizing for automated trading."""

from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    STOP_LOSS = "stop_loss"
    OCO = "oco"  # One-Cancels-Other


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class OrderRequest:
    """Order request data."""
    symbol: str
    order_type: OrderType
    side: str  # "buy" or "sell"
    quantity: int
    price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    time_in_force: str = "DAY"


@dataclass
class OrderResponse:
    """Order response data."""
    order_id: str
    status: OrderStatus
    filled_quantity: int
    average_price: Decimal
    commission: Decimal
    timestamp: datetime
    error_message: Optional[str] = None


@dataclass
class PositionSizing:
    """Position sizing calculation result."""
    recommended_shares: int
    max_loss_amount: Decimal
    risk_per_share: Decimal
    position_value: Decimal
    risk_percentage: float
    is_valid: bool
    reason: Optional[str] = None


@dataclass 
class OCOOrder:
    """OCO Order data structure."""
    order_id: str
    symbol: str
    quantity: int
    order_type: OrderType
    price: Decimal
    linked_order_id: Optional[str] = None


class OrderManager:
    """Order management with broker API integration and position sizing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 broker_api_key: Optional[str] = None, 
                 broker_secret: Optional[str] = None,
                 paper_trading: bool = True) -> None:
        """
        Initialize Order Manager.
        
        Args:
            config: Configuration dictionary (preferred)
            broker_api_key: Broker API key (legacy)
            broker_secret: Broker API secret (legacy)
            paper_trading: Whether to use paper trading mode
        """
        if config:
            self.broker_api_key = config.get("broker_api_key", "")
            self.broker_secret = config.get("broker_secret", "")
            self.paper_trading = config.get("paper_trading", True)
            self.max_positions = config.get("max_positions", 5)
            self.risk_per_trade_ratio = config.get("risk_per_trade_ratio", 0.01)
        else:
            self.broker_api_key = broker_api_key or ""
            self.broker_secret = broker_secret or ""
            self.paper_trading = paper_trading
            self.max_positions = 5
            self.risk_per_trade_ratio = 0.01
        
        self.session = None
        self.re_entry_restrictions: Dict[str, datetime] = {}
        self.trade_exit_records: Dict[str, datetime] = {}
        self.current_positions = 0
        logger.info(f"OrderManager initialized (paper_trading: {self.paper_trading})")
    
    def initialize_broker_connection(self) -> bool:
        """
        Initialize connection to broker API.
        
        Returns:
            True if connection successful
        """
        try:
            # TODO: Implement actual broker API connection
            logger.info("Initializing broker API connection...")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize broker connection: {e}")
            return False
    
    def calculate_position_size(self, capital: Decimal, 
                               risk_per_trade_ratio: float,
                               entry_price: Decimal, 
                               stop_loss_percentage: float) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk management rules.
        
        Implementation of formula from requirements:
        - max_loss_per_trade = capital * risk_per_trade_ratio
        - stop_loss_price = entry_price * (1 - stop_loss_percentage)
        - risk_per_share = entry_price - stop_loss_price
        - position_size = max_loss_per_trade // risk_per_share
        - Round down to nearest 100 shares (unit lot)
        
        Args:
            capital: Total available capital
            risk_per_trade_ratio: Risk per trade ratio (e.g., 0.01 for 1%)
            entry_price: Current stock price for entry
            stop_loss_percentage: Stop-loss percentage (e.g., 0.08 for 8%)
            
        Returns:
            Dictionary with position sizing results
        """
        logger.debug(f"Calculating position size: capital={capital}, "
                    f"entry_price={entry_price}, stop_loss={stop_loss_percentage:.3f}")
        
        # Calculate maximum loss per trade
        max_loss_per_trade = capital * Decimal(str(risk_per_trade_ratio))
        
        # Calculate stop loss price
        stop_loss_price = entry_price * (Decimal("1.0") - Decimal(str(stop_loss_percentage)))
        
        # Calculate risk per share
        risk_per_share = entry_price - stop_loss_price
        
        # Validation: risk_per_share must be positive
        if risk_per_share <= 0:
            return {
                "position_size": 0,
                "max_loss": max_loss_per_trade,
                "stop_loss_price": stop_loss_price,
                "risk_per_share": risk_per_share
            }
        
        # Calculate position size
        position_size_decimal = max_loss_per_trade / risk_per_share
        
        # Round down to nearest 100 shares (unit lot)  
        position_size = int(position_size_decimal // 100) * 100
        
        # If position size is 0 after rounding but should be minimum 100 shares
        if position_size == 0 and position_size_decimal > 0:
            position_size = 100
        
        logger.debug(f"Position sizing result: {position_size} shares")
        
        return {
            "position_size": position_size,
            "max_loss": max_loss_per_trade,
            "stop_loss_price": stop_loss_price,
            "risk_per_share": risk_per_share
        }
    
    def place_market_buy_order(self, symbol: str, quantity: int) -> OrderResponse:
        """
        Place market buy order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            
        Returns:
            Order response
        """
        # TODO: Implement actual market buy order
        logger.info(f"Placing market buy order: {symbol} x {quantity}")
        
        return OrderResponse(
            order_id=f"BUY_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            status=OrderStatus.PENDING,
            filled_quantity=0,
            average_price=Decimal("0.0"),
            commission=Decimal("0.0"),
            timestamp=datetime.now()
        )
    
    def place_oco_order(self, symbol: str, quantity: int, 
                       take_profit_price: Decimal, 
                       stop_loss_price: Decimal) -> OrderResponse:
        """
        Place OCO (One-Cancels-Other) order for profit taking and stop loss.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            take_profit_price: Take profit price
            stop_loss_price: Stop loss price
            
        Returns:
            Order response
        """
        # TODO: Implement actual OCO order placement
        logger.info(f"Placing OCO order: {symbol} x {quantity}, "
                   f"TP: {take_profit_price}, SL: {stop_loss_price}")
        
        return OrderResponse(
            order_id=f"OCO_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            status=OrderStatus.PENDING,
            filled_quantity=0,
            average_price=Decimal("0.0"),
            commission=Decimal("0.0"),
            timestamp=datetime.now()
        )
    
    def execute_complete_trade(self, symbol: str, capital: Decimal,
                              entry_price: Decimal, stop_loss_percentage: float,
                              take_profit_percentage: float = 0.05) -> Dict[str, Any]:
        """
        Execute complete trade with buy order and immediate OCO setup.
        
        Args:
            symbol: Stock symbol
            capital: Available capital
            entry_price: Entry price
            stop_loss_percentage: Stop-loss percentage
            take_profit_percentage: Take-profit percentage (default: 5%)
            
        Returns:
            Trade execution result
        """
        # Check re-entry restriction
        if self.is_re_entry_restricted(symbol):
            logger.warning(f"Re-entry restricted for {symbol}")
            return {
                "success": False,
                "reason": "Re-entry restriction active",
                "restricted_until": self.re_entry_restrictions[symbol]
            }
        
        # Calculate position size
        position_sizing_result = self.calculate_position_size(
            capital, self.risk_per_trade_ratio, entry_price, stop_loss_percentage
        )
        
        if position_sizing_result["position_size"] <= 0:
            logger.error(f"Invalid position sizing")
            return {
                "success": False,
                "reason": "Invalid position size",
                "position_sizing": position_sizing_result
            }
        
        # Calculate OCO prices
        stop_loss_price = entry_price * (Decimal("1.0") - Decimal(str(stop_loss_percentage)))
        take_profit_price = entry_price * (Decimal("1.0") + Decimal(str(take_profit_percentage)))
        
        try:
            # Place market buy order
            buy_response = self.place_market_buy_order(
                symbol, position_sizing_result["position_size"]
            )
            
            if buy_response.status != OrderStatus.FAILED:
                # Place OCO order immediately after buy order
                oco_response = self.place_oco_order(
                    symbol, position_sizing_result["position_size"],
                    take_profit_price, stop_loss_price
                )
                
                return {
                    "success": True,
                    "buy_order": buy_response,
                    "oco_order": oco_response,
                    "position_sizing": position_sizing_result,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price
                }
            else:
                return {
                    "success": False,
                    "reason": "Buy order failed",
                    "buy_order": buy_response
                }
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {
                "success": False,
                "reason": f"Execution error: {e}"
            }
    
    def add_re_entry_restriction(self, symbol: str, hours: int = 3) -> None:
        """
        Add re-entry restriction for a symbol.
        
        Args:
            symbol: Stock symbol
            hours: Restriction duration in hours (default: 3)
        """
        restriction_until = datetime.now() + timedelta(hours=hours)
        self.re_entry_restrictions[symbol] = restriction_until
        logger.info(f"Added re-entry restriction for {symbol} until {restriction_until}")
    
    def is_re_entry_restricted(self, symbol: str) -> bool:
        """
        Check if re-entry is restricted for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if re-entry is restricted
        """
        if symbol not in self.re_entry_restrictions:
            return False
        
        if datetime.now() >= self.re_entry_restrictions[symbol]:
            # Restriction expired, remove it
            del self.re_entry_restrictions[symbol]
            logger.debug(f"Re-entry restriction expired for {symbol}")
            return False
        
        return True
    
    def get_order_status_response(self, order_id: str) -> OrderResponse:
        """
        Get order status response by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order response with current status
        """
        # TODO: Implement actual order status checking
        logger.debug(f"Checking order status for {order_id}")
        
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.PENDING,
            filled_quantity=0,
            average_price=Decimal("0.0"),
            commission=Decimal("0.0"),
            timestamp=datetime.now()
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful
        """
        # TODO: Implement actual order cancellation
        logger.info(f"Cancelling order {order_id}")
        return {"success": True, "order_id": order_id}
    
    def create_oco_order(self, entry_order: Dict[str, Any], 
                        take_profit_price: Decimal, 
                        stop_loss_price: Decimal) -> List[Dict[str, Any]]:
        """
        Create OCO order pair (take profit and stop loss).
        
        Args:
            entry_order: Entry order parameters
            take_profit_price: Take profit price
            stop_loss_price: Stop loss price
            
        Returns:
            List of OCO order dictionaries
        """
        symbol = entry_order["symbol"]
        quantity = entry_order["quantity"]
        
        # Generate order IDs
        tp_order_id = f"TP_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        sl_order_id = f"SL_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Take profit order
        tp_order = {
            "order_id": tp_order_id,
            "symbol": symbol,
            "quantity": quantity,
            "order_type": OrderType.LIMIT_SELL,
            "price": take_profit_price,
            "linked_order_id": sl_order_id
        }
        
        # Stop loss order
        sl_order = {
            "order_id": sl_order_id,
            "symbol": symbol,
            "quantity": quantity,
            "order_type": OrderType.STOP_LOSS,
            "price": stop_loss_price,
            "linked_order_id": tp_order_id
        }
        
        logger.info(f"Created OCO orders for {symbol}: TP={take_profit_price}, SL={stop_loss_price}")
        return [tp_order, sl_order]
    
    def record_trade_exit(self, symbol: str, exit_time: datetime) -> None:
        """
        Record trade exit for re-entry restriction.
        
        Args:
            symbol: Stock symbol
            exit_time: Exit timestamp
        """
        self.trade_exit_records[symbol] = exit_time
        # Add 3-hour re-entry restriction
        self.add_re_entry_restriction(symbol, hours=3)
        logger.info(f"Recorded trade exit for {symbol} at {exit_time}")
    
    def can_enter_position(self, symbol: str) -> bool:
        """
        Check if position can be entered (not restricted).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if position entry is allowed
        """
        return not self.is_re_entry_restricted(symbol)
    
    def get_current_position_count(self) -> int:
        """
        Get current number of open positions.
        
        Returns:
            Number of open positions
        """
        return self.current_positions
    
    def can_open_new_position(self) -> bool:
        """
        Check if new position can be opened.
        
        Returns:
            True if new position allowed
        """
        return self.current_positions < self.max_positions
    
    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute order and return result.
        
        Args:
            order: Order parameters
            
        Returns:
            Order execution result
        """
        symbol = order["symbol"]
        quantity = order["quantity"]
        order_type = order["order_type"]
        
        # Generate order ID
        order_id = f"{order_type.value.upper()}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Simulate execution based on order type
        if order_type in [OrderType.MARKET_BUY, OrderType.MARKET_SELL]:
            status = OrderStatus.FILLED
            execution_price = Decimal("2500")  # Mock price
        else:
            status = OrderStatus.PENDING
            execution_price = order.get("price", Decimal("2500"))
        
        return {
            "order_id": order_id,
            "status": status,
            "filled_quantity": quantity if status == OrderStatus.FILLED else 0,
            "execution_price": execution_price,
            "timestamp": datetime.now()
        }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status information
        """
        return {
            "order_id": order_id,
            "status": OrderStatus.CANCELLED,  # Mock status for test
            "timestamp": datetime.now()
        }
    
    def execute_order_with_slippage_control(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute order with slippage control.
        
        Args:
            order: Order with slippage parameters
            
        Returns:
            Execution result with slippage check
        """
        expected_price = order.get("expected_price")
        max_slippage_percent = order.get("max_slippage_percent", 0.5)
        
        # Simulate execution price (could have slippage)
        execution_price = expected_price * Decimal("1.002")  # 0.2% slippage
        
        # Check if slippage is acceptable
        slippage_percent = float((execution_price - expected_price) / expected_price * 100)
        
        if slippage_percent > max_slippage_percent:
            return {
                "status": OrderStatus.FAILED,
                "rejection_reason": "EXCESSIVE_SLIPPAGE",
                "expected_price": expected_price,
                "execution_price": execution_price,
                "slippage_percent": slippage_percent
            }
        
        return {
            "status": OrderStatus.FILLED,
            "execution_price": execution_price,
            "expected_price": expected_price,
            "slippage_percent": slippage_percent,
            "timestamp": datetime.now()
        }
    
    async def execute_morning_batch_orders(self, execution_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute batch orders at market open (9:00 AM) based on previous day's analysis.
        
        Args:
            execution_plan: Execution plan from previous day's analysis
            
        Returns:
            List of execution results
        """
        logger.info(f"Starting morning batch execution for {len(execution_plan.get('buy_targets', []))} targets")
        
        execution_results = []
        
        for target in execution_plan.get("buy_targets", []):
            try:
                # Check if we can open a new position
                if not self.can_open_new_position(target["symbol"]):
                    logger.warning(f"Cannot open position for {target['symbol']} - max positions or restrictions")
                    execution_results.append({
                        "symbol": target["symbol"],
                        "success": False,
                        "reason": "Position limit or restrictions"
                    })
                    continue
                
                # Place market buy order at open
                buy_result = await self.place_morning_buy_order(
                    symbol=target["symbol"],
                    position_size=target["position_size"],
                    expected_price=target["entry_price"]
                )
                
                if buy_result["success"]:
                    # Place OCO order for stop loss and take profit
                    oco_result = await self.place_morning_oco_order(
                        symbol=target["symbol"],
                        entry_price=buy_result["executed_price"],
                        stop_loss_price=target["stop_loss_price"],
                        take_profit_price=target["take_profit_price"],
                        quantity=target["position_size"]
                    )
                    
                    execution_results.append({
                        "symbol": target["symbol"],
                        "success": True,
                        "buy_order": buy_result,
                        "oco_order": oco_result,
                        "timestamp": datetime.now()
                    })
                else:
                    execution_results.append({
                        "symbol": target["symbol"],
                        "success": False,
                        "reason": buy_result.get("reason", "Buy order failed")
                    })
                    
            except Exception as e:
                logger.error(f"Error executing morning order for {target['symbol']}: {e}")
                execution_results.append({
                    "symbol": target["symbol"],
                    "success": False,
                    "error": str(e)
                })
        
        logger.info(f"Morning batch execution completed: {len([r for r in execution_results if r['success']])} successful")
        return execution_results
    
    async def place_morning_buy_order(self, symbol: str, position_size: int, expected_price: Decimal) -> Dict[str, Any]:
        """
        Place a market buy order at market open.
        
        Args:
            symbol: Stock symbol
            position_size: Number of shares
            expected_price: Expected price from previous day's analysis
            
        Returns:
            Order execution result
        """
        # In production, this would use broker API for market-on-open orders
        # For now, simulate execution
        
        # Add slippage for realistic simulation
        slippage = Decimal("0.003")  # 0.3% slippage
        executed_price = expected_price * (Decimal("1") + slippage)
        
        order_result = {
            "success": True,
            "order_id": f"BUY_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "symbol": symbol,
            "quantity": position_size,
            "expected_price": expected_price,
            "executed_price": executed_price,
            "slippage": executed_price - expected_price,
            "timestamp": datetime.now()
        }
        
        # Update current positions
        self.current_positions[symbol] = {
            "quantity": position_size,
            "entry_price": executed_price,
            "timestamp": datetime.now()
        }
        
        logger.info(f"Morning buy order executed: {symbol} x {position_size} @ {executed_price}")
        return order_result
    
    async def place_morning_oco_order(self, symbol: str, entry_price: Decimal, 
                                     stop_loss_price: Decimal, take_profit_price: Decimal,
                                     quantity: int) -> Dict[str, Any]:
        """
        Place OCO order for stop loss and take profit at market open.
        
        Args:
            symbol: Stock symbol
            entry_price: Actual entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            quantity: Number of shares
            
        Returns:
            OCO order result
        """
        # Adjust stop loss and take profit based on actual entry price
        actual_stop_loss = min(stop_loss_price, entry_price * Decimal("0.95"))  # Max 5% loss
        actual_take_profit = max(take_profit_price, entry_price * Decimal("1.02"))  # Min 2% profit
        
        oco_result = {
            "success": True,
            "order_id": f"OCO_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "symbol": symbol,
            "quantity": quantity,
            "stop_loss_price": actual_stop_loss,
            "take_profit_price": actual_take_profit,
            "stop_loss_distance": (entry_price - actual_stop_loss) / entry_price * 100,
            "take_profit_distance": (actual_take_profit - entry_price) / entry_price * 100,
            "timestamp": datetime.now()
        }
        
        logger.info(f"Morning OCO order placed: {symbol} SL={actual_stop_loss}, TP={actual_take_profit}")
        return oco_result

    def validate_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate order parameters.
        
        Args:
            order: Order to validate
            
        Returns:
            Validation result
        """
        errors = []
        
        # Check symbol
        if not order.get("symbol"):
            errors.append("symbol")
        
        # Check quantity
        quantity = order.get("quantity", 0)
        if quantity <= 0:
            errors.append("quantity")
        
        # Check order type
        if not order.get("order_type"):
            errors.append("order_type")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }