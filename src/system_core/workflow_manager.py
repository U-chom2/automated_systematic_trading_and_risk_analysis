"""WorkflowManager for scheduled trading system management."""

import asyncio
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class TradingPhase(Enum):
    """Trading workflow phases."""
    IDLE = "IDLE"
    DATA_COLLECTION = "DATA_COLLECTION"  # 16:15 - データ収集フェーズ
    ANALYSIS = "ANALYSIS"  # 分析フェーズ
    DECISION = "DECISION"  # 投資判断フェーズ
    PENDING_EXECUTION = "PENDING_EXECUTION"  # 翌日執行待機
    EXECUTION = "EXECUTION"  # 翌日朝執行フェーズ
    COMPLETED = "COMPLETED"  # 処理完了


class WorkflowState(Enum):
    """Workflow execution states."""
    WAITING = "WAITING"
    PROCESSING = "PROCESSING"
    ERROR = "ERROR"
    SCHEDULED = "SCHEDULED"  # 定時実行予約済み


@dataclass
class DailyDataCollectionResult:
    """Daily data collection result."""
    collection_date: datetime
    symbols: List[str]
    ir_announcements: List[Dict[str, Any]]
    sns_mentions: Dict[str, int]
    market_data: Dict[str, Any]
    success: bool
    errors: List[str]


@dataclass
class DailyAnalysisResult:
    """Daily comprehensive analysis result."""
    analysis_date: datetime
    symbol: str
    catalyst_score: int  # Max 50
    sentiment_score: int  # Max 30
    market_env_score: int  # Max 20
    total_score: int  # Max 100
    buy_decision: bool
    filter_passed: bool
    filter_reasons: List[str]
    position_size: Optional[int] = None
    entry_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None


@dataclass
class NextDayExecutionPlan:
    """Next day execution plan."""
    execution_date: datetime
    buy_targets: List[DailyAnalysisResult]
    total_capital_allocation: Decimal
    risk_per_trade: Decimal


class WorkflowManager:
    """Manages scheduled trading workflow."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize WorkflowManager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.current_phase = TradingPhase.IDLE
        self.state = WorkflowState.WAITING
        self.error_message: Optional[str] = None
        
        # 定時実行設定 (16:15)
        self.scheduled_time = time(16, 15)
        self.execution_time = time(9, 0)  # 翌日朝9:00
        
        # 本日の分析結果
        self.today_results: List[DailyAnalysisResult] = []
        self.next_day_plan: Optional[NextDayExecutionPlan] = None
        
        self.metrics = {
            "total_days_processed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time_minutes": 0.0,
            "total_buy_decisions": 0
        }
        
        # Initialize dummy analyzers
        self.nlp_analyzer = DummyNlpAnalyzer()
        self.technical_analyzer = DummyTechnicalAnalyzer()
        self.risk_model = DummyRiskModel()
        
        logger.info("WorkflowManager initialized for scheduled execution at 16:15")

    async def run_daily_analysis(self) -> bool:
        """
        Run daily analysis at 16:15.
        
        Returns:
            True if analysis successful
        """
        start_time = datetime.now()
        logger.info(f"Starting daily analysis at {start_time}")
        
        try:
            # Phase 1: Data Collection
            self._transition_to_phase(TradingPhase.DATA_COLLECTION)
            collection_result = await self.collect_daily_data()
            
            if not collection_result.success:
                logger.error(f"Data collection failed: {collection_result.errors}")
                return False
            
            # Phase 2: Analysis
            self._transition_to_phase(TradingPhase.ANALYSIS)
            analysis_results = await self.analyze_collected_data(collection_result)
            
            # Phase 3: Decision
            self._transition_to_phase(TradingPhase.DECISION)
            execution_plan = await self.make_trading_decisions(analysis_results)
            
            # Save execution plan for next day
            self.next_day_plan = execution_plan
            self._transition_to_phase(TradingPhase.PENDING_EXECUTION)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() / 60
            self.metrics["average_processing_time_minutes"] = processing_time
            self.metrics["total_days_processed"] += 1
            
            logger.info(f"Daily analysis completed in {processing_time:.2f} minutes")
            logger.info(f"Buy decisions: {len(execution_plan.buy_targets)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Daily analysis failed: {e}")
            self._set_error_state(str(e))
            return False

    async def collect_daily_data(self) -> DailyDataCollectionResult:
        """
        Collect all data for the day.
        
        Returns:
            Daily data collection result
        """
        logger.info("Collecting daily data...")
        
        # Dummy implementation - in real system would collect from various sources
        await asyncio.sleep(0.5)  # Simulate data collection
        
        # Get watchlist symbols
        symbols = self.config.get("watchlist", ["7203", "9984", "6758"])
        
        # Collect IR announcements
        ir_announcements = await self._collect_ir_data(symbols)
        
        # Collect SNS mentions
        sns_mentions = await self._collect_sns_data(symbols)
        
        # Collect market data
        market_data = await self._collect_market_data(symbols)
        
        return DailyDataCollectionResult(
            collection_date=datetime.now(),
            symbols=symbols,
            ir_announcements=ir_announcements,
            sns_mentions=sns_mentions,
            market_data=market_data,
            success=True,
            errors=[]
        )

    async def _collect_ir_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Collect IR announcements for symbols."""
        # Dummy implementation
        await asyncio.sleep(0.1)
        
        ir_data = []
        for symbol in symbols[:1]:  # Simulate finding IR for first symbol
            ir_data.append({
                "symbol": symbol,
                "title": "業績予想の上方修正に関するお知らせ",
                "content": "売上高を上方修正いたします",
                "timestamp": datetime.now(),
                "source": "TDnet"
            })
        
        return ir_data

    async def _collect_sns_data(self, symbols: List[str]) -> Dict[str, int]:
        """Collect SNS mention counts."""
        # Dummy implementation
        await asyncio.sleep(0.1)
        
        mentions = {}
        for symbol in symbols:
            # Simulate mention counts
            mentions[symbol] = 150 + (int(symbol) % 100)
        
        return mentions

    async def _collect_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect market data for symbols."""
        # Dummy implementation
        await asyncio.sleep(0.1)
        
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = {
                "close_price": Decimal("1500.00"),
                "volume": 1000000,
                "rsi": 65.0,
                "ma_deviation": 12.0,
                "volatility": 0.25
            }
        
        return market_data

    async def analyze_collected_data(self, collection_result: DailyDataCollectionResult) -> List[DailyAnalysisResult]:
        """
        Analyze all collected data.
        
        Args:
            collection_result: Data collection result
            
        Returns:
            List of analysis results
        """
        logger.info("Analyzing collected data...")
        
        analysis_results = []
        
        for symbol in collection_result.symbols:
            # Check for IR announcements
            symbol_ir = [ir for ir in collection_result.ir_announcements if ir["symbol"] == symbol]
            
            # Calculate catalyst score (0-50)
            catalyst_score = 0
            if symbol_ir:
                ir_content = symbol_ir[0]["content"]
                ir_analysis = await self.nlp_analyzer.analyze_ir_importance(ir_content)
                catalyst_score = min(ir_analysis.get("score", 0), 50)
            
            # Calculate sentiment score (0-30)
            mention_count = collection_result.sns_mentions.get(symbol, 0)
            sentiment_analysis = await self._analyze_sns_sentiment(symbol, mention_count)
            sentiment_score = min(sentiment_analysis.get("score", 0), 30)
            
            # Calculate market environment score (0-20)
            market_data = collection_result.market_data.get(symbol, {})
            market_env_score = self._calculate_market_environment_score(market_data)
            
            # Total score
            total_score = catalyst_score + sentiment_score + market_env_score
            
            # Check filters
            filter_result = self.check_risk_filters(market_data)
            
            # Buy decision
            buy_threshold = self.config.get("buy_threshold", 80)
            buy_decision = total_score >= buy_threshold and filter_result["passed"]
            
            # Calculate position sizing if buy decision
            position_size = None
            entry_price = None
            stop_loss_price = None
            take_profit_price = None
            
            if buy_decision:
                sizing_result = await self._calculate_position_sizing(symbol, market_data)
                position_size = sizing_result["position_size"]
                entry_price = sizing_result["entry_price"]
                stop_loss_price = sizing_result["stop_loss_price"]
                take_profit_price = sizing_result["take_profit_price"]
            
            result = DailyAnalysisResult(
                analysis_date=datetime.now(),
                symbol=symbol,
                catalyst_score=catalyst_score,
                sentiment_score=sentiment_score,
                market_env_score=market_env_score,
                total_score=total_score,
                buy_decision=buy_decision,
                filter_passed=filter_result["passed"],
                filter_reasons=filter_result.get("reasons", []),
                position_size=position_size,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            analysis_results.append(result)
            
            if buy_decision:
                self.metrics["total_buy_decisions"] += 1
        
        self.metrics["successful_analyses"] += len(analysis_results)
        
        return analysis_results

    async def _analyze_sns_sentiment(self, symbol: str, mention_count: int) -> Dict[str, Any]:
        """Analyze SNS sentiment based on mention counts."""
        # Dummy implementation
        # In real system would compare with historical baseline
        baseline = 100  # Average daily mentions
        
        # Calculate deviation
        deviation_ratio = (mention_count - baseline) / baseline if baseline > 0 else 0
        
        # Score based on deviation (max 30)
        if deviation_ratio > 2.0:  # 200% increase
            score = 30
        elif deviation_ratio > 1.0:  # 100% increase
            score = 20
        elif deviation_ratio > 0.5:  # 50% increase
            score = 10
        else:
            score = 5
        
        return {
            "score": score,
            "mention_count": mention_count,
            "baseline": baseline,
            "deviation_ratio": deviation_ratio
        }

    def _calculate_market_environment_score(self, market_data: Dict[str, Any]) -> int:
        """Calculate market environment score (0-20 points)."""
        score = 10  # Base score
        
        # Add points for positive momentum
        if market_data.get("ma_deviation", 0) > 0 and market_data.get("ma_deviation", 0) < 15:
            score += 5
        
        # Add points for moderate RSI
        rsi = market_data.get("rsi", 50)
        if 40 <= rsi <= 60:
            score += 5
        
        return min(score, 20)

    def check_risk_filters(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check risk management filters.
        
        Args:
            market_data: Market data to check
            
        Returns:
            Filter check result
        """
        reasons = []
        
        # RSI filter (> 75% = overheated)
        rsi = market_data.get("rsi", 50)
        rsi_threshold = self.config.get("rsi_filter_threshold", 75)
        if rsi > rsi_threshold:
            reasons.append("rsi_overheated")
        
        # MA deviation filter (> 25% = overheated)
        ma_deviation = market_data.get("ma_deviation", 0)
        ma_threshold = self.config.get("ma_deviation_filter_threshold", 25)
        if ma_deviation > ma_threshold:
            reasons.append("ma_deviation_overheated")
        
        return {
            "passed": len(reasons) == 0,
            "reasons": reasons
        }

    async def _calculate_position_sizing(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position sizing based on risk management."""
        # Get current price
        entry_price = market_data.get("close_price", Decimal("1000"))
        
        # Get stop loss percentage from risk model
        risk_assessment = self.risk_model.assess_risk(symbol, market_data)
        stop_loss_percentage = risk_assessment.get("stop_loss_recommendation", 0.08)
        
        # Calculate prices
        stop_loss_price = entry_price * Decimal(1 - stop_loss_percentage)
        take_profit_price = entry_price * Decimal(1 + stop_loss_percentage * 2)  # 2:1 R/R ratio
        
        # Calculate position size (dummy - in real system would use capital and risk per trade)
        capital = self.config.get("capital", Decimal("1000000"))
        risk_per_trade = self.config.get("risk_per_trade_ratio", 0.01)
        
        max_loss = capital * Decimal(risk_per_trade)
        risk_per_share = entry_price - stop_loss_price
        
        if risk_per_share > 0:
            position_size = int(max_loss / risk_per_share)
            position_size = (position_size // 100) * 100  # Round to lot size
        else:
            position_size = 0
        
        return {
            "position_size": position_size,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price
        }

    async def make_trading_decisions(self, analysis_results: List[DailyAnalysisResult]) -> NextDayExecutionPlan:
        """
        Make final trading decisions for next day.
        
        Args:
            analysis_results: Analysis results
            
        Returns:
            Next day execution plan
        """
        # Filter buy targets
        buy_targets = [r for r in analysis_results if r.buy_decision]
        
        # Sort by total score (highest first)
        buy_targets.sort(key=lambda x: x.total_score, reverse=True)
        
        # Limit number of positions based on capital
        max_positions = self.config.get("max_concurrent_positions", 5)
        buy_targets = buy_targets[:max_positions]
        
        # Calculate total capital allocation
        capital = self.config.get("capital", Decimal("1000000"))
        risk_per_trade = self.config.get("risk_per_trade_ratio", 0.01)
        total_allocation = sum(
            t.entry_price * t.position_size 
            for t in buy_targets 
            if t.position_size and t.entry_price
        )
        
        execution_plan = NextDayExecutionPlan(
            execution_date=datetime.now() + timedelta(days=1),
            buy_targets=buy_targets,
            total_capital_allocation=total_allocation,
            risk_per_trade=Decimal(risk_per_trade)
        )
        
        # Save execution plan
        self._save_execution_plan(execution_plan)
        
        return execution_plan

    def _save_execution_plan(self, plan: NextDayExecutionPlan) -> None:
        """Save execution plan to file for next day execution."""
        try:
            plan_dict = {
                "execution_date": plan.execution_date.isoformat(),
                "buy_targets": [
                    {
                        "symbol": t.symbol,
                        "total_score": t.total_score,
                        "position_size": t.position_size,
                        "entry_price": str(t.entry_price) if t.entry_price else None,
                        "stop_loss_price": str(t.stop_loss_price) if t.stop_loss_price else None,
                        "take_profit_price": str(t.take_profit_price) if t.take_profit_price else None
                    }
                    for t in plan.buy_targets
                ],
                "total_capital_allocation": str(plan.total_capital_allocation),
                "risk_per_trade": str(plan.risk_per_trade)
            }
            
            # In real system, would save to database
            logger.info(f"Execution plan saved: {len(plan.buy_targets)} buy targets")
            
        except Exception as e:
            logger.error(f"Failed to save execution plan: {e}")

    async def execute_morning_trades(self) -> bool:
        """
        Execute trades at 9:00 AM based on previous day's analysis.
        
        Returns:
            True if execution successful
        """
        if self.current_phase != TradingPhase.PENDING_EXECUTION:
            logger.warning("No pending execution plan")
            return False
        
        if not self.next_day_plan:
            logger.warning("No execution plan available")
            return False
        
        self._transition_to_phase(TradingPhase.EXECUTION)
        
        try:
            logger.info(f"Executing morning trades for {len(self.next_day_plan.buy_targets)} targets")
            
            for target in self.next_day_plan.buy_targets:
                # In real system, would place actual orders through broker API
                logger.info(f"Placing buy order: {target.symbol} x {target.position_size} @ {target.entry_price}")
                logger.info(f"  Stop Loss: {target.stop_loss_price}")
                logger.info(f"  Take Profit: {target.take_profit_price}")
                
                await asyncio.sleep(0.1)  # Simulate order placement
            
            self._transition_to_phase(TradingPhase.COMPLETED)
            self.next_day_plan = None  # Clear executed plan
            
            return True
            
        except Exception as e:
            logger.error(f"Morning trade execution failed: {e}")
            self._set_error_state(str(e))
            return False

    def get_workflow_metrics(self) -> Dict[str, Any]:
        """
        Get workflow performance metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            **self.metrics,
            "current_phase": self.current_phase.value,
            "current_state": self.state.value,
            "scheduled_time": str(self.scheduled_time),
            "execution_time": str(self.execution_time),
            "has_pending_plan": self.next_day_plan is not None
        }

    def is_scheduled_time(self) -> bool:
        """Check if current time matches scheduled execution time."""
        current_time = datetime.now().time()
        
        # Check if within 1 minute of scheduled time
        scheduled_datetime = datetime.combine(datetime.today(), self.scheduled_time)
        current_datetime = datetime.now()
        
        time_diff = abs((current_datetime - scheduled_datetime).total_seconds())
        
        return time_diff <= 60  # Within 1 minute

    def is_execution_time(self) -> bool:
        """Check if current time matches morning execution time."""
        current_time = datetime.now().time()
        
        # Check if within 1 minute of execution time
        execution_datetime = datetime.combine(datetime.today(), self.execution_time)
        current_datetime = datetime.now()
        
        time_diff = abs((current_datetime - execution_datetime).total_seconds())
        
        return time_diff <= 60  # Within 1 minute

    def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """
        Update workflow configuration.
        
        Args:
            new_config: New configuration parameters
            
        Returns:
            True if update successful
        """
        try:
            self.config.update(new_config)
            logger.info(f"Configuration updated: {new_config}")
            return True
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return False

    def recover_from_error(self) -> bool:
        """
        Recover from error state.
        
        Returns:
            True if recovery successful
        """
        if self.state == WorkflowState.ERROR:
            self.state = WorkflowState.WAITING
            self.current_phase = TradingPhase.IDLE
            self.error_message = None
            logger.info("Recovered from error state")
            return True
        return False

    def _transition_to_phase(self, new_phase: TradingPhase) -> None:
        """Transition to new trading phase."""
        old_phase = self.current_phase
        self.current_phase = new_phase
        self.state = WorkflowState.PROCESSING
        logger.debug(f"Phase transition: {old_phase.value} -> {new_phase.value}")

    def _set_error_state(self, error_message: str) -> None:
        """Set workflow to error state."""
        self.state = WorkflowState.ERROR
        self.error_message = error_message
        logger.error(f"Workflow error: {error_message}")


# Dummy analyzer classes
class DummyNlpAnalyzer:
    """Dummy NLP analyzer for testing."""

    async def analyze_ir_importance(self, content: str) -> Dict[str, Any]:
        """Analyze IR importance (dummy implementation)."""
        # Look for high-importance keywords
        high_value_keywords = ["上方修正", "増配", "業績予想"]
        score = 45 if any(keyword in content for keyword in high_value_keywords) else 20
        
        return {
            "score": score,
            "keywords": ["上方修正"] if "上方修正" in content else []
        }

    async def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment (dummy implementation)."""
        return {
            "sentiment_score": 25,
            "confidence": 0.8
        }


class DummyTechnicalAnalyzer:
    """Dummy technical analyzer for testing."""

    def analyze_indicators(self, symbol: str) -> Dict[str, Any]:
        """Analyze technical indicators (dummy implementation)."""
        return {
            "rsi": 65.0,
            "ma_deviation": 15.0,
            "momentum": 1.2
        }


class DummyRiskModel:
    """Dummy risk model for testing."""

    def assess_risk(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess trading risk (dummy implementation)."""
        return {
            "risk_score": 0.3,
            "confidence": 0.85,
            "stop_loss_recommendation": 0.08
        }