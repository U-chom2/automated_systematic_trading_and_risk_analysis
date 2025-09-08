"""バックテストユースケース"""
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import List, Dict, Optional, Any
from uuid import UUID

from ...domain.entities.portfolio import Portfolio
from ...domain.entities.trade import Trade
from ...domain.services.trading_strategy import TradingStrategy, StrategyType
from ...domain.services.signal_generator import SignalGenerator
from ...domain.repositories.market_data_repository import MarketDataRepository
from ...domain.repositories.stock_repository import StockRepository


@dataclass
class BacktestConfig:
    """バックテスト設定"""
    start_date: date
    end_date: date
    initial_capital: Decimal
    strategy_type: StrategyType
    tickers: List[str]
    commission_rate: Decimal = Decimal("0.001")  # 0.1%
    slippage_rate: Decimal = Decimal("0.0005")  # 0.05%
    max_positions: int = 20
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    
    def validate(self) -> None:
        """バリデーション"""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if not self.tickers:
            raise ValueError("At least one ticker is required")
        
        valid_frequencies = ["daily", "weekly", "monthly"]
        if self.rebalance_frequency not in valid_frequencies:
            raise ValueError(f"Rebalance frequency must be one of {valid_frequencies}")


@dataclass
class BacktestResult:
    """バックテスト結果"""
    config: BacktestConfig
    final_value: Decimal
    total_return: Decimal
    annualized_return: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    total_trades: int
    profit_trades: int
    loss_trades: int
    average_profit: Decimal
    average_loss: Decimal
    profit_factor: Decimal
    portfolio_values: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "config": {
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "initial_capital": float(self.config.initial_capital),
                "strategy_type": self.config.strategy_type.value,
                "tickers": self.config.tickers,
            },
            "performance": {
                "final_value": float(self.final_value),
                "total_return": float(self.total_return),
                "annualized_return": float(self.annualized_return),
                "volatility": float(self.volatility),
                "sharpe_ratio": float(self.sharpe_ratio),
                "max_drawdown": float(self.max_drawdown),
            },
            "trading_stats": {
                "total_trades": self.total_trades,
                "profit_trades": self.profit_trades,
                "loss_trades": self.loss_trades,
                "win_rate": float(self.win_rate),
                "average_profit": float(self.average_profit),
                "average_loss": float(self.average_loss),
                "profit_factor": float(self.profit_factor),
            },
            "portfolio_values": self.portfolio_values,
            "trades": self.trades,
        }


@dataclass
class RunBacktestUseCase:
    """バックテスト実行ユースケース"""
    
    market_data_repository: MarketDataRepository
    stock_repository: StockRepository
    trading_strategy: TradingStrategy
    signal_generator: SignalGenerator
    
    async def execute(self, config: BacktestConfig) -> BacktestResult:
        """バックテストを実行
        
        Args:
            config: バックテスト設定
        
        Returns:
            バックテスト結果
        """
        # バリデーション
        config.validate()
        
        # ポートフォリオを初期化
        portfolio = Portfolio(
            name="Backtest Portfolio",
            initial_capital=config.initial_capital,
            strategy_type=config.strategy_type.value,
        )
        
        # 市場データを取得
        market_data = await self.market_data_repository.get_multiple_ohlcv(
            config.tickers,
            config.start_date,
            config.end_date,
            interval="1d"
        )
        
        # 銘柄情報を取得
        stocks = []
        for ticker in config.tickers:
            stock = await self.stock_repository.find_by_ticker(ticker)
            if stock:
                stocks.append(stock)
        
        # バックテストを実行
        portfolio_values = []
        trades_history = []
        current_date = config.start_date
        
        while current_date <= config.end_date:
            # 現在の市場データを取得
            current_market_data = self._get_market_data_for_date(
                market_data, current_date
            )
            
            if not current_market_data:
                current_date += timedelta(days=1)
                continue
            
            # ポートフォリオの価値を更新
            self._update_portfolio_values(portfolio, current_market_data)
            
            # リバランスのタイミングかチェック
            if self._should_rebalance(current_date, config.rebalance_frequency):
                # シグナルを生成
                signals = self.signal_generator.generate_signals(
                    stocks,
                    self._get_historical_data(market_data, current_date, 30),
                )
                
                # 取引を生成
                trades = self.trading_strategy.generate_trades(
                    portfolio,
                    self._convert_signals(signals),
                    current_market_data,
                )
                
                # 取引を実行
                for trade in trades:
                    executed_trade = self._execute_trade(
                        portfolio,
                        trade,
                        current_market_data,
                        config.commission_rate,
                        config.slippage_rate,
                    )
                    
                    if executed_trade:
                        trades_history.append({
                            "date": current_date.isoformat(),
                            "ticker": executed_trade.ticker,
                            "type": executed_trade.trade_type.value,
                            "quantity": executed_trade.quantity,
                            "price": float(executed_trade.executed_price),
                            "commission": float(executed_trade.commission),
                        })
            
            # ポートフォリオの価値を記録
            portfolio_values.append({
                "date": current_date.isoformat(),
                "total_value": float(portfolio.total_value),
                "cash": float(portfolio.available_cash),
                "positions_value": float(portfolio.total_invested),
                "unrealized_pnl": float(portfolio.unrealized_pnl),
                "realized_pnl": float(portfolio.realized_pnl),
            })
            
            current_date += timedelta(days=1)
        
        # 結果を計算
        result = self._calculate_results(
            config,
            portfolio,
            portfolio_values,
            trades_history,
        )
        
        return result
    
    def _get_market_data_for_date(
        self,
        market_data: Dict[str, List],
        target_date: date,
    ) -> Dict[str, Any]:
        """特定日の市場データを取得"""
        result = {}
        
        for ticker, ohlcv_list in market_data.items():
            for ohlcv in ohlcv_list:
                if ohlcv.timestamp.date() == target_date:
                    result[ticker] = ohlcv
                    break
        
        return result
    
    def _get_historical_data(
        self,
        market_data: Dict[str, List],
        end_date: date,
        lookback_days: int,
    ) -> Dict[str, List]:
        """過去データを取得"""
        start_date = end_date - timedelta(days=lookback_days)
        result = {}
        
        for ticker, ohlcv_list in market_data.items():
            historical = [
                ohlcv for ohlcv in ohlcv_list
                if start_date <= ohlcv.timestamp.date() <= end_date
            ]
            if historical:
                result[ticker] = historical
        
        return result
    
    def _should_rebalance(self, current_date: date, frequency: str) -> bool:
        """リバランスのタイミングかチェック"""
        if frequency == "daily":
            return True
        elif frequency == "weekly":
            return current_date.weekday() == 0  # 月曜日
        elif frequency == "monthly":
            return current_date.day == 1  # 月初
        return False
    
    def _update_portfolio_values(self, portfolio: Portfolio, market_data: Dict) -> None:
        """ポートフォリオの価値を更新"""
        for ticker, position in portfolio.positions.items():
            if ticker in market_data and not position.is_closed:
                position.update_price(market_data[ticker].close.value)
    
    def _convert_signals(self, signals: List) -> List:
        """シグナルを戦略用に変換"""
        # 簡易実装
        return signals
    
    def _execute_trade(
        self,
        portfolio: Portfolio,
        trade: Trade,
        market_data: Dict,
        commission_rate: Decimal,
        slippage_rate: Decimal,
    ) -> Optional[Trade]:
        """取引を実行"""
        if trade.ticker not in market_data:
            return None
        
        current_price = market_data[trade.ticker].close.value
        
        # スリッページを適用
        if trade.is_buy:
            executed_price = current_price * (1 + slippage_rate)
        else:
            executed_price = current_price * (1 - slippage_rate)
        
        # 手数料を計算
        commission = executed_price * Decimal(trade.quantity) * commission_rate
        
        # 取引を約定
        trade.execute(executed_price, commission)
        
        # ポートフォリオを更新
        if trade.is_buy:
            portfolio.add_position(
                stock_id=trade.stock_id,
                ticker=trade.ticker,
                quantity=trade.quantity,
                price=executed_price,
            )
        else:
            portfolio.reduce_position(
                ticker=trade.ticker,
                quantity=trade.quantity,
                price=executed_price,
            )
        
        return trade
    
    def _calculate_results(
        self,
        config: BacktestConfig,
        portfolio: Portfolio,
        portfolio_values: List[Dict],
        trades_history: List[Dict],
    ) -> BacktestResult:
        """バックテスト結果を計算"""
        # 基本指標
        final_value = portfolio.total_value
        total_return = (final_value - config.initial_capital) / config.initial_capital
        
        # 年率換算リターン
        days = (config.end_date - config.start_date).days
        years = days / 365.25
        annualized_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else total_return
        
        # ボラティリティとシャープレシオ
        returns = []
        for i in range(1, len(portfolio_values)):
            prev_value = Decimal(str(portfolio_values[i-1]["total_value"]))
            curr_value = Decimal(str(portfolio_values[i]["total_value"]))
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)
        
        if returns:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = (variance ** Decimal("0.5")) * (Decimal("252") ** Decimal("0.5"))
            
            risk_free_rate = Decimal("0.001")  # 0.1%
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else Decimal("0")
        else:
            volatility = Decimal("0")
            sharpe_ratio = Decimal("0")
        
        # 最大ドローダウン
        peak = config.initial_capital
        max_drawdown = Decimal("0")
        for record in portfolio_values:
            value = Decimal(str(record["total_value"]))
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak if peak > 0 else Decimal("0")
                max_drawdown = max(max_drawdown, drawdown)
        
        # 取引統計
        total_trades = len(trades_history)
        profit_trades = 0
        loss_trades = 0
        total_profit = Decimal("0")
        total_loss = Decimal("0")
        
        # 簡易的な損益計算
        for i, trade in enumerate(trades_history):
            if trade["type"] == "SELL":
                # 売却取引の損益を推定（簡易版）
                if i > 0:
                    profit_trades += 1
                    total_profit += Decimal("1000")  # 仮の値
        
        win_rate = Decimal(profit_trades) / Decimal(total_trades) if total_trades > 0 else Decimal("0")
        average_profit = total_profit / Decimal(profit_trades) if profit_trades > 0 else Decimal("0")
        average_loss = total_loss / Decimal(loss_trades) if loss_trades > 0 else Decimal("0")
        profit_factor = total_profit / total_loss if total_loss > 0 else Decimal("1")
        
        return BacktestResult(
            config=config,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profit_trades=profit_trades,
            loss_trades=loss_trades,
            average_profit=average_profit,
            average_loss=average_loss,
            profit_factor=profit_factor,
            portfolio_values=portfolio_values,
            trades=trades_history,
        )


@dataclass
class AnalyzeBacktestResultsUseCase:
    """バックテスト結果分析ユースケース"""
    
    async def execute(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """複数のバックテスト結果を分析
        
        Args:
            results: バックテスト結果のリスト
        
        Returns:
            分析結果
        """
        if not results:
            return {}
        
        # 最良・最悪のパフォーマンス
        best_result = max(results, key=lambda r: r.total_return)
        worst_result = min(results, key=lambda r: r.total_return)
        
        # 平均パフォーマンス
        avg_return = sum(r.total_return for r in results) / len(results)
        avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
        avg_drawdown = sum(r.max_drawdown for r in results) / len(results)
        
        # 戦略別の統計
        strategy_stats = {}
        for result in results:
            strategy = result.config.strategy_type.value
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "count": 0,
                    "total_return": Decimal("0"),
                    "sharpe_ratio": Decimal("0"),
                }
            
            strategy_stats[strategy]["count"] += 1
            strategy_stats[strategy]["total_return"] += result.total_return
            strategy_stats[strategy]["sharpe_ratio"] += result.sharpe_ratio
        
        # 平均を計算
        for strategy, stats in strategy_stats.items():
            count = stats["count"]
            stats["avg_return"] = float(stats["total_return"] / count)
            stats["avg_sharpe"] = float(stats["sharpe_ratio"] / count)
        
        return {
            "summary": {
                "total_backtests": len(results),
                "average_return": float(avg_return),
                "average_sharpe": float(avg_sharpe),
                "average_max_drawdown": float(avg_drawdown),
            },
            "best_performance": {
                "strategy": best_result.config.strategy_type.value,
                "total_return": float(best_result.total_return),
                "sharpe_ratio": float(best_result.sharpe_ratio),
            },
            "worst_performance": {
                "strategy": worst_result.config.strategy_type.value,
                "total_return": float(worst_result.total_return),
                "sharpe_ratio": float(worst_result.sharpe_ratio),
            },
            "strategy_comparison": strategy_stats,
        }