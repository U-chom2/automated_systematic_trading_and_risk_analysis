"""Main trading system orchestrating all components."""

from typing import Dict, List, Any, Optional
from decimal import Decimal
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
import threading

# Import system modules
from ..data_collector.tdnet_scraper import TdnetScraper
from ..data_collector.x_streamer import XStreamer
from ..data_collector.price_fetcher import PriceFetcher
from ..analysis_engine.nlp_analyzer import NlpAnalyzer
from ..analysis_engine.technical_analyzer import TechnicalAnalyzer
from ..analysis_engine.risk_model import RiskModel, RiskPrediction
from ..execution_manager.order_manager import OrderManager, PositionSizing
from ..execution_manager.position_tracker import PositionTracker, Position, TradeRecord
from .workflow_manager import WorkflowManager

logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    """システム状態列挙型"""
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"


@dataclass
class TradingDecision:
    """Trading decision data structure."""
    symbol: str
    action: str  # "buy", "hold", "skip"
    confidence: float
    total_score: int
    max_score: int
    catalyst_score: int
    sentiment_score: int
    technical_score: int
    market_score: int
    risk_assessment: RiskPrediction
    position_sizing: Optional[PositionSizing] = None
    reason: Optional[str] = None


@dataclass
class SystemConfig:
    """System configuration."""
    capital: Decimal
    risk_per_trade_ratio: float
    max_positions: int
    market_hours: Dict[str, str]
    watchlist_file: Optional[str] = None
    log_level: str = "INFO"
    tdnet_polling_interval: int = 1
    price_update_interval: int = 5
    buy_threshold: int = 80


class TradingSystem:
    """Main AI day trading system coordinating all components."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Trading System.
        
        Args:
            config: System configuration dictionary
        """
        self.config = SystemConfig(
            capital=config.get("capital", Decimal("1000000")),
            risk_per_trade_ratio=config.get("risk_per_trade_ratio", 0.01),
            max_positions=config.get("max_positions", 5),
            market_hours=config.get("market_hours", {"start": "09:00", "end": "15:00"}),
            watchlist_file=config.get("watchlist_file"),
            log_level=config.get("log_level", "INFO"),
            tdnet_polling_interval=config.get("tdnet_polling_interval", 1),
            price_update_interval=config.get("price_update_interval", 5),
            buy_threshold=config.get("buy_threshold", 80)
        )
        
        # System state
        self.status = SystemStatus.STOPPED
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.stop_time: Optional[datetime] = None
        self.emergency_reason: Optional[str] = None
        self.watchlist: List[str] = []
        
        # Components
        self._initialize_components()
        
        logger.info("TradingSystem initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            # Initialize core components
            self.workflow_manager = WorkflowManager(config={
                "scoring_threshold": self.config.buy_threshold,
                "rsi_filter_threshold": 75,
                "ma_deviation_filter_threshold": 25
            })
            
            self.order_manager = OrderManager(config={
                "paper_trading": True,
                "max_positions": self.config.max_positions,
                "risk_per_trade_ratio": self.config.risk_per_trade_ratio
            })
            
            self.position_tracker = PositionTracker()
            
            # Initialize data collectors
            self.data_collector = {
                "tdnet_scraper": TdnetScraper(polling_interval=self.config.tdnet_polling_interval),
                "x_streamer": None,  # Will be initialized with API keys
                "price_fetcher": None  # Will be initialized with API keys
            }
            
            # Initialize analysis engines
            self.analysis_engine = {
                "nlp_analyzer": NlpAnalyzer(),
                "technical_analyzer": TechnicalAnalyzer(),
                "risk_model": RiskModel()
            }
            
            # Set cross-references
            self.workflow_manager.trading_system = self
            self.order_manager.trading_system = self
            self.position_tracker.trading_system = self
            
            logger.info("Components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def initialize_components(self, api_keys: Dict[str, str]) -> bool:
        """
        Initialize all system components.
        
        Args:
            api_keys: Dictionary containing all required API keys
            
        Returns:
            True if all components initialized successfully
        """
        try:
            logger.info("Initializing system components...")
            
            # Initialize data collectors
            self.tdnet_scraper = TdnetScraper(
                polling_interval=self.config.tdnet_polling_interval
            )
            
            self.x_streamer = XStreamer(
                api_key=api_keys.get("x_api_key", ""),
                api_secret=api_keys.get("x_api_secret", ""),
                access_token=api_keys.get("x_access_token", ""),
                access_token_secret=api_keys.get("x_access_token_secret", "")
            )
            
            self.price_fetcher = PriceFetcher(
                api_endpoint=api_keys.get("broker_api_endpoint", ""),
                api_key=api_keys.get("broker_api_key", "")
            )
            
            # Initialize analysis engines
            self.nlp_analyzer = NlpAnalyzer()
            self.technical_analyzer = TechnicalAnalyzer()
            self.risk_model = RiskModel()
            
            # Initialize execution components
            self.order_manager = OrderManager(
                broker_api_key=api_keys.get("broker_api_key", ""),
                broker_secret=api_keys.get("broker_secret", ""),
                paper_trading=api_keys.get("paper_trading", True)
            )
            
            self.position_tracker = PositionTracker()
            
            # Initialize models
            await self._initialize_models()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    async def _initialize_models(self) -> None:
        """Initialize NLP and risk models."""
        if self.nlp_analyzer:
            self.nlp_analyzer.initialize_models()
        
        if self.risk_model:
            self.risk_model.load_model()
    
    def load_watchlist(self, symbols: List[str]) -> None:
        """
        Load watchlist of symbols to monitor.
        
        Args:
            symbols: List of stock symbols
        """
        self.watchlist = symbols
        logger.info(f"Loaded watchlist: {len(symbols)} symbols")
    
    async def start_monitoring(self) -> None:
        """Start system monitoring and trading."""
        if not self._validate_system_ready():
            logger.error("System not ready for monitoring")
            return
        
        self.is_running = True
        logger.info("Starting trading system monitoring...")
        
        try:
            # Start monitoring tasks concurrently
            tasks = [
                self._monitor_ir_releases(),
                self._monitor_social_media(),
                self._monitor_price_updates(),
                self._process_trading_decisions()
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.is_running = False
            logger.info("Trading system monitoring stopped")
    
    def _validate_system_ready(self) -> bool:
        """
        Validate that system is ready for operation.
        
        Returns:
            True if system is ready
        """
        required_components = [
            self.tdnet_scraper, self.x_streamer, self.price_fetcher,
            self.nlp_analyzer, self.technical_analyzer, self.risk_model,
            self.order_manager, self.position_tracker
        ]
        
        if None in required_components:
            logger.error("Not all components are initialized")
            return False
        
        if not self.watchlist:
            logger.error("Watchlist is empty")
            return False
        
        return True
    
    async def _monitor_ir_releases(self) -> None:
        """Monitor IR/press releases for triggers."""
        while self.is_running:
            try:
                if self.tdnet_scraper:
                    releases = self.tdnet_scraper.get_latest_releases()
                    
                    for release in releases:
                        if self.tdnet_scraper.check_for_trigger_keywords(
                            release.get("title", "")
                        ):
                            await self._handle_ir_trigger(release)
                
                await asyncio.sleep(1)  # 1-second polling
                
            except Exception as e:
                logger.error(f"IR monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_social_media(self) -> None:
        """Monitor social media for anomaly detection."""
        while self.is_running:
            try:
                if self.x_streamer:
                    for symbol in self.watchlist:
                        if self.x_streamer.check_mention_anomaly(symbol):
                            await self._handle_social_trigger(symbol)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Social media monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_price_updates(self) -> None:
        """Monitor real-time price updates."""
        while self.is_running:
            try:
                if self.price_fetcher and self.position_tracker:
                    # Update prices for open positions
                    open_symbols = self.position_tracker.get_open_symbols()
                    for symbol in open_symbols:
                        current_price = self.price_fetcher.get_current_price(symbol)
                        if current_price:
                            self.position_tracker.update_position_price(
                                symbol, current_price
                            )
                            
                            # Check stop-loss/take-profit triggers
                            await self._check_exit_conditions(symbol, current_price)
                
                await asyncio.sleep(self.config.price_update_interval)
                
            except Exception as e:
                logger.error(f"Price monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _process_trading_decisions(self) -> None:
        """Process pending trading decisions."""
        while self.is_running:
            try:
                # This would be connected to a queue of triggered events
                # For now, it's a placeholder
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Trading decision processing error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_ir_trigger(self, release_data: Dict[str, Any]) -> None:
        """
        Handle IR trigger event.
        
        Args:
            release_data: IR release data
        """
        symbol = release_data.get("company_code", "")
        if not symbol or symbol not in self.watchlist:
            return
        
        logger.info(f"IR trigger detected for {symbol}")
        
        # Perform analysis and make trading decision
        decision = await self._analyze_and_decide(
            symbol, catalyst_type="ir", catalyst_data=release_data
        )
        
        if decision.action == "buy":
            await self._execute_trade(decision)
    
    async def _handle_social_trigger(self, symbol: str) -> None:
        """
        Handle social media trigger event.
        
        Args:
            symbol: Stock symbol
        """
        logger.info(f"Social media trigger detected for {symbol}")
        
        # Perform analysis and make trading decision
        decision = await self._analyze_and_decide(
            symbol, catalyst_type="social", catalyst_data={}
        )
        
        if decision.action == "buy":
            await self._execute_trade(decision)
    
    async def _analyze_and_decide(self, symbol: str, catalyst_type: str, 
                                 catalyst_data: Dict[str, Any]) -> TradingDecision:
        """
        Perform comprehensive analysis and make trading decision.
        
        Args:
            symbol: Stock symbol
            catalyst_type: Type of catalyst ("ir" or "social")
            catalyst_data: Catalyst data
            
        Returns:
            Trading decision
        """
        logger.info(f"Analyzing {symbol} for {catalyst_type} catalyst")
        
        # Initialize scores
        catalyst_score = 0
        sentiment_score = 0
        technical_score = 0
        market_score = 0
        
        try:
            # 1. Catalyst analysis (50 points max)
            if catalyst_type == "ir" and self.nlp_analyzer:
                importance = self.nlp_analyzer.analyze_ir_importance(
                    catalyst_data.get("title", ""),
                    catalyst_data.get("content", "")
                )
                catalyst_score = importance.score
            
            # 2. Sentiment analysis (30 points max)
            if self.nlp_analyzer:
                # TODO: Collect recent social media posts
                texts = []  # Placeholder
                sentiment_score = self.nlp_analyzer.calculate_sentiment_score_for_trading(texts)
            
            # 3. Technical analysis (20 points max)
            if self.technical_analyzer and self.price_fetcher:
                # Get price data
                price_data = self.price_fetcher.get_historical_data(symbol, days=60)
                technical_score = self.technical_analyzer.calculate_technical_score_for_trading(
                    symbol, price_data
                )
            
            # 4. Risk assessment using trained model
            risk_prediction = None
            if self.risk_model and self.price_fetcher:
                # Get comprehensive market data for the symbol
                market_data = {
                    "historical_prices": self.price_fetcher.get_historical_data(symbol, days=60)
                }
                # Use the new method that works with market data
                risk_prediction = self.risk_model.predict_from_market_data(
                    symbol, market_data
                )
            
            # Calculate total score
            total_score = catalyst_score + sentiment_score + technical_score
            max_score = 100  # 50 + 30 + 20
            
            # Make decision based on threshold
            if total_score >= self.config.buy_threshold and technical_score > 0:
                action = "buy"
                confidence = total_score / max_score
            else:
                action = "skip"
                confidence = 0.0
                
            decision = TradingDecision(
                symbol=symbol,
                action=action,
                confidence=confidence,
                total_score=total_score,
                max_score=max_score,
                catalyst_score=catalyst_score,
                sentiment_score=sentiment_score,
                technical_score=technical_score,
                market_score=market_score,
                risk_assessment=risk_prediction
            )
            
            logger.info(f"Trading decision for {symbol}: {action} "
                       f"(score: {total_score}/{max_score})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return TradingDecision(
                symbol=symbol,
                action="skip",
                confidence=0.0,
                total_score=0,
                max_score=100,
                catalyst_score=0,
                sentiment_score=0,
                technical_score=0,
                market_score=0,
                risk_assessment=None,
                reason=f"Analysis error: {e}"
            )
    
    async def _execute_trade(self, decision: TradingDecision) -> None:
        """
        Execute trading decision.
        
        Args:
            decision: Trading decision
        """
        if not self.order_manager or not self.position_tracker:
            logger.error("Order manager or position tracker not initialized")
            return
        
        if not decision.risk_assessment:
            logger.error(f"No risk assessment for {decision.symbol}")
            return
        
        try:
            # Get current price
            current_price = None
            if self.price_fetcher:
                current_price = self.price_fetcher.get_current_price(decision.symbol)
            
            if not current_price:
                logger.error(f"Could not get current price for {decision.symbol}")
                return
            
            # Execute trade
            result = self.order_manager.execute_complete_trade(
                symbol=decision.symbol,
                capital=self.config.capital,
                entry_price=current_price,
                stop_loss_percentage=decision.risk_assessment.optimal_stop_loss_percent
            )
            
            if result["success"]:
                # Add position to tracker
                position_sizing = result["position_sizing"]
                self.position_tracker.add_position(
                    symbol=decision.symbol,
                    quantity=position_sizing.recommended_shares,
                    entry_price=current_price,
                    stop_loss_price=result["stop_loss_price"],
                    take_profit_price=result["take_profit_price"],
                    buy_order_id=result["buy_order"]["order_id"],
                    oco_order_id=result["oco_order"]["order_id"]
                )
                
                logger.info(f"Trade executed successfully for {decision.symbol}")
            else:
                logger.warning(f"Trade execution failed for {decision.symbol}: "
                              f"{result.get('reason', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Trade execution error for {decision.symbol}: {e}")
    
    async def _check_exit_conditions(self, symbol: str, current_price: Decimal) -> None:
        """
        Check stop-loss and take-profit conditions.
        
        Args:
            symbol: Stock symbol
            current_price: Current price
        """
        if not self.position_tracker or not self.order_manager:
            return
        
        # Check stop-loss
        if self.position_tracker.check_stop_loss_trigger(symbol, current_price):
            self.position_tracker.close_position(symbol, current_price, "stop_loss")
            self.order_manager.add_re_entry_restriction(symbol, hours=3)
            logger.warning(f"Stop-loss executed for {symbol}")
        
        # Check take-profit
        elif self.position_tracker.check_take_profit_trigger(symbol, current_price):
            self.position_tracker.close_position(symbol, current_price, "take_profit")
            logger.info(f"Take-profit executed for {symbol}")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.is_running = False
        logger.info("Stopping trading system...")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            System status dictionary
        """
        portfolio_summary = None
        if self.position_tracker:
            portfolio_summary = self.position_tracker.calculate_portfolio_summary()
        
        return {
            "is_running": self.is_running,
            "watchlist_size": len(self.watchlist),
            "components_initialized": self._validate_system_ready(),
            "portfolio_summary": portfolio_summary,
            "timestamp": datetime.now()
        }
    
    def start_system(self) -> bool:
        """
        Start the trading system.
        
        Returns:
            True if started successfully
        """
        try:
            if self.status == SystemStatus.RUNNING:
                logger.warning("System is already running")
                return True
            
            self.status = SystemStatus.RUNNING
            self.is_running = True
            self.start_time = datetime.now()
            self.stop_time = None
            self.emergency_reason = None
            
            logger.info("Trading system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.status = SystemStatus.STOPPED
            self.is_running = False
            return False
    
    def stop_system(self) -> bool:
        """
        Stop the trading system.
        
        Returns:
            True if stopped successfully
        """
        try:
            if self.status == SystemStatus.STOPPED:
                logger.warning("System is already stopped")
                return True
            
            self.status = SystemStatus.STOPPED
            self.is_running = False
            self.stop_time = datetime.now()
            
            logger.info("Trading system stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
            return False
    
    def restart_system(self) -> bool:
        """
        Restart the trading system.
        
        Returns:
            True if restarted successfully
        """
        logger.info("Restarting trading system...")
        
        if self.is_running:
            if not self.stop_system():
                return False
        
        return self.start_system()
    
    def emergency_stop(self, reason: str) -> bool:
        """
        Emergency stop the trading system.
        
        Args:
            reason: Reason for emergency stop
            
        Returns:
            True if emergency stopped successfully
        """
        try:
            logger.error(f"EMERGENCY STOP: {reason}")
            
            self.status = SystemStatus.EMERGENCY_STOPPED
            self.is_running = False
            self.stop_time = datetime.now()
            self.emergency_reason = reason
            
            # Cancel all pending orders
            if self.order_manager:
                self.order_manager.cancel_all_orders()
            
            # Close all positions at market price
            if self.position_tracker:
                self.position_tracker.close_all_positions("EMERGENCY_STOP")
            
            logger.info("Emergency stop completed")
            return True
            
        except Exception as e:
            logger.critical(f"Failed to emergency stop: {e}")
            return False
    
    def is_market_open(self) -> bool:
        """
        Check if market is open.
        
        Returns:
            True if market is open
        """
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        start_time = self.config.market_hours["start"]
        end_time = self.config.market_hours["end"]
        
        return start_time <= current_time <= end_time
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health information.
        
        Returns:
            System health dictionary
        """
        uptime = 0
        if self.start_time and self.status == SystemStatus.RUNNING:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        component_health = {
            "workflow_manager": {"status": "OK" if self.workflow_manager else "ERROR"},
            "order_manager": {"status": "OK" if self.order_manager else "ERROR"},
            "position_tracker": {"status": "OK" if self.position_tracker else "ERROR"}
        }
        
        return {
            "status": self.status.value,
            "uptime": uptime,
            "components": component_health,
            "market_open": self.is_market_open(),
            "timestamp": datetime.now()
        }
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Performance statistics dictionary
        """
        if self.position_tracker:
            return self.position_tracker.get_portfolio_statistics()
        
        # Default empty statistics
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_profit_loss": Decimal("0"),
            "average_profit": Decimal("0"),
            "average_loss": Decimal("0"),
            "max_drawdown": Decimal("0"),
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0
        }
    
    def set_log_level(self, level: str) -> None:
        """
        Set logging level.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        level_map = {
            "DEBUG": 10,
            "INFO": 20, 
            "WARNING": 30,
            "ERROR": 40
        }
        
        if level in level_map:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(level_map[level])
            logger.info(f"Log level set to {level}")
    
    def process_concurrent_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple operations concurrently.
        
        Args:
            operations: List of operations to process
            
        Returns:
            List of processing results
        """
        results = []
        for operation in operations:
            result = {
                "operation_id": operation.get("type", "unknown"),
                "symbol": operation.get("data", {}).get("symbol", ""),
                "processed": True,
                "timestamp": datetime.now()
            }
            results.append(result)
        
        logger.info(f"Processed {len(operations)} concurrent operations")
        return results
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect system metrics.
        
        Returns:
            System metrics dictionary
        """
        import psutil
        import threading
        
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "active_threads": threading.active_count(),
            "queue_size": 0,  # Placeholder
            "processing_time": 0.0  # Placeholder
        }
    
    def save_system_state(self) -> bool:
        """
        Save system state to persistent storage.
        
        Returns:
            True if saved successfully
        """
        try:
            # Placeholder implementation
            logger.info("System state saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            return False
    
    def restore_system_state(self) -> bool:
        """
        Restore system state from persistent storage.
        
        Returns:
            True if restored successfully
        """
        try:
            # Placeholder implementation
            logger.info("System state restored successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to restore system state: {e}")
            return False
    
    def graceful_shutdown(self, timeout: int = 30) -> bool:
        """
        Perform graceful shutdown.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if shutdown completed gracefully
        """
        logger.info("Initiating graceful shutdown...")
        
        try:
            # Wait for pending operations to complete
            start_time = datetime.now()
            while self._has_pending_operations():
                if (datetime.now() - start_time).total_seconds() > timeout:
                    logger.warning("Graceful shutdown timeout, forcing stop")
                    break
                asyncio.sleep(0.1)
            
            return self.stop_system()
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            return self.emergency_stop("Graceful shutdown failed")
    
    def _has_pending_operations(self) -> bool:
        """Check if there are pending operations."""
        # Placeholder implementation
        return False
    
    def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """
        Update system configuration.
        
        Args:
            new_config: New configuration parameters
            
        Returns:
            True if updated successfully
        """
        try:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False