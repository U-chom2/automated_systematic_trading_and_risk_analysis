"""リスク計算ドメインサービス"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import math

from ..entities.portfolio import Portfolio
from ..entities.position import Position
from ..value_objects.price import Price, OHLCV
from ..value_objects.percentage import Percentage


@dataclass
class RiskMetrics:
    """リスク指標"""
    value_at_risk: Decimal  # VaR
    conditional_value_at_risk: Decimal  # CVaR
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    max_drawdown: Percentage
    volatility: Percentage
    beta: Decimal
    correlation: Decimal
    tracking_error: Optional[Percentage] = None
    
    def __str__(self) -> str:
        """文字列表現"""
        return (
            f"VaR: ¥{self.value_at_risk:,.0f}, "
            f"CVaR: ¥{self.conditional_value_at_risk:,.0f}, "
            f"Sharpe: {self.sharpe_ratio:.2f}, "
            f"Volatility: {self.volatility}"
        )


@dataclass
class PositionRisk:
    """個別ポジションのリスク"""
    position: Position
    weight: Percentage  # ポートフォリオ内の比重
    contribution_to_risk: Percentage  # リスク寄与度
    concentration_risk: Percentage  # 集中リスク
    liquidity_risk: Percentage  # 流動性リスク
    
    @property
    def total_risk_score(self) -> Decimal:
        """総合リスクスコア（0-100）"""
        return (
            self.contribution_to_risk.value * Decimal("0.4") +
            self.concentration_risk.value * Decimal("0.3") +
            self.liquidity_risk.value * Decimal("0.3")
        )


class RiskCalculator:
    """リスク計算ドメインサービス
    
    ポートフォリオのリスク分析と管理を行う。
    """
    
    def __init__(
        self,
        confidence_level: Decimal = Decimal("0.95"),  # VaR信頼水準
        risk_free_rate: Decimal = Decimal("0.001"),  # 無リスク金利（年率0.1%）
        lookback_days: int = 252,  # 過去データ参照期間（1年）
    ) -> None:
        """初期化"""
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days
    
    def calculate_portfolio_risk(
        self,
        portfolio: Portfolio,
        historical_prices: Dict[str, List[OHLCV]],
        benchmark_prices: Optional[List[OHLCV]] = None,
    ) -> RiskMetrics:
        """ポートフォリオ全体のリスクを計算
        
        Args:
            portfolio: ポートフォリオ
            historical_prices: 銘柄ごとの過去価格データ
            benchmark_prices: ベンチマーク価格データ（オプション）
        
        Returns:
            リスク指標
        """
        # ポートフォリオのリターンを計算
        portfolio_returns = self._calculate_portfolio_returns(portfolio, historical_prices)
        
        if not portfolio_returns:
            # データ不足の場合はデフォルト値を返す
            return RiskMetrics(
                value_at_risk=Decimal("0"),
                conditional_value_at_risk=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                sortino_ratio=Decimal("0"),
                max_drawdown=Percentage(0),
                volatility=Percentage(0),
                beta=Decimal("1"),
                correlation=Decimal("0"),
            )
        
        # VaRとCVaRを計算
        var = self._calculate_var(portfolio_returns, portfolio.total_value)
        cvar = self._calculate_cvar(portfolio_returns, portfolio.total_value)
        
        # ボラティリティを計算
        volatility = self._calculate_volatility(portfolio_returns)
        
        # シャープレシオを計算
        sharpe_ratio = self._calculate_sharpe_ratio(
            portfolio_returns,
            self.risk_free_rate,
            volatility
        )
        
        # ソルティノレシオを計算
        sortino_ratio = self._calculate_sortino_ratio(
            portfolio_returns,
            self.risk_free_rate
        )
        
        # 最大ドローダウンを計算
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        # ベータとコリレーションを計算
        beta = Decimal("1")
        correlation = Decimal("0")
        tracking_error = None
        
        if benchmark_prices:
            benchmark_returns = self._calculate_returns(benchmark_prices)
            beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            correlation = self._calculate_correlation(portfolio_returns, benchmark_returns)
            tracking_error = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
        
        return RiskMetrics(
            value_at_risk=var,
            conditional_value_at_risk=cvar,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            beta=beta,
            correlation=correlation,
            tracking_error=tracking_error,
        )
    
    def calculate_position_risks(
        self,
        portfolio: Portfolio,
        historical_prices: Dict[str, List[OHLCV]],
    ) -> List[PositionRisk]:
        """個別ポジションのリスクを計算
        
        Args:
            portfolio: ポートフォリオ
            historical_prices: 銘柄ごとの過去価格データ
        
        Returns:
            ポジションリスクのリスト
        """
        position_risks = []
        total_value = portfolio.total_value
        
        for ticker, position in portfolio.positions.items():
            if position.is_closed:
                continue
            
            # ポジションの比重を計算
            weight = Percentage.from_decimal(position.market_value / total_value)
            
            # リスク寄与度を計算
            contribution = self._calculate_risk_contribution(
                position,
                portfolio,
                historical_prices.get(ticker, [])
            )
            
            # 集中リスクを計算（単純化：比重が高いほどリスク大）
            concentration = self._calculate_concentration_risk(weight)
            
            # 流動性リスクを計算（単純化：取引量ベース）
            liquidity = self._calculate_liquidity_risk(
                position,
                historical_prices.get(ticker, [])
            )
            
            position_risk = PositionRisk(
                position=position,
                weight=weight,
                contribution_to_risk=contribution,
                concentration_risk=concentration,
                liquidity_risk=liquidity,
            )
            position_risks.append(position_risk)
        
        return position_risks
    
    def check_risk_limits(
        self,
        portfolio: Portfolio,
        risk_metrics: RiskMetrics,
        max_var: Optional[Decimal] = None,
        max_volatility: Optional[Percentage] = None,
        min_sharpe: Optional[Decimal] = None,
        max_drawdown: Optional[Percentage] = None,
    ) -> Dict[str, bool]:
        """リスク制限のチェック
        
        Returns:
            制限名 -> 適合状態のマッピング
        """
        checks = {}
        
        if max_var is not None:
            checks["max_var"] = risk_metrics.value_at_risk <= max_var
        
        if max_volatility is not None:
            checks["max_volatility"] = risk_metrics.volatility <= max_volatility
        
        if min_sharpe is not None:
            checks["min_sharpe"] = risk_metrics.sharpe_ratio >= min_sharpe
        
        if max_drawdown is not None:
            checks["max_drawdown"] = risk_metrics.max_drawdown <= max_drawdown
        
        return checks
    
    def _calculate_portfolio_returns(
        self,
        portfolio: Portfolio,
        historical_prices: Dict[str, List[OHLCV]],
    ) -> List[Decimal]:
        """ポートフォリオのリターンを計算"""
        if not historical_prices:
            return []
        
        # 各銘柄のリターンを重み付け平均
        portfolio_returns = []
        total_value = portfolio.total_value
        
        # 最小の履歴数を取得
        min_length = min(len(prices) for prices in historical_prices.values())
        
        for i in range(1, min_length):
            daily_return = Decimal("0")
            
            for ticker, position in portfolio.positions.items():
                if position.is_closed or ticker not in historical_prices:
                    continue
                
                prices = historical_prices[ticker]
                if i < len(prices):
                    prev_price = prices[i-1].close.value
                    curr_price = prices[i].close.value
                    
                    if prev_price > 0:
                        stock_return = (curr_price - prev_price) / prev_price
                        weight = position.market_value / total_value
                        daily_return += stock_return * weight
            
            portfolio_returns.append(daily_return)
        
        return portfolio_returns
    
    def _calculate_returns(self, prices: List[OHLCV]) -> List[Decimal]:
        """価格データからリターンを計算"""
        returns = []
        
        for i in range(1, len(prices)):
            prev_price = prices[i-1].close.value
            curr_price = prices[i].close.value
            
            if prev_price > 0:
                daily_return = (curr_price - prev_price) / prev_price
                returns.append(daily_return)
        
        return returns
    
    def _calculate_var(self, returns: List[Decimal], portfolio_value: Decimal) -> Decimal:
        """Value at Risk を計算"""
        if not returns:
            return Decimal("0")
        
        sorted_returns = sorted(returns)
        index = int(len(sorted_returns) * (1 - self.confidence_level))
        
        if index < len(sorted_returns):
            var_percentage = abs(sorted_returns[index])
            return portfolio_value * var_percentage
        
        return Decimal("0")
    
    def _calculate_cvar(self, returns: List[Decimal], portfolio_value: Decimal) -> Decimal:
        """Conditional Value at Risk を計算"""
        if not returns:
            return Decimal("0")
        
        sorted_returns = sorted(returns)
        index = int(len(sorted_returns) * (1 - self.confidence_level))
        
        if index > 0:
            tail_returns = sorted_returns[:index]
            avg_tail_return = sum(tail_returns) / len(tail_returns)
            return portfolio_value * abs(avg_tail_return)
        
        return Decimal("0")
    
    def _calculate_volatility(self, returns: List[Decimal]) -> Percentage:
        """ボラティリティを計算（年率換算）"""
        if len(returns) < 2:
            return Percentage(0)
        
        # 平均リターン
        mean_return = sum(returns) / len(returns)
        
        # 分散
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        
        # 標準偏差（日次）
        std_dev = Decimal(str(math.sqrt(float(variance))))
        
        # 年率換算（252営業日）
        annual_volatility = std_dev * Decimal(str(math.sqrt(252)))
        
        return Percentage.from_decimal(annual_volatility)
    
    def _calculate_sharpe_ratio(
        self,
        returns: List[Decimal],
        risk_free_rate: Decimal,
        volatility: Percentage,
    ) -> Decimal:
        """シャープレシオを計算"""
        if not returns or volatility.value == 0:
            return Decimal("0")
        
        # 平均リターン（年率換算）
        mean_return = sum(returns) / len(returns) * Decimal("252")
        
        # 超過リターン
        excess_return = mean_return - risk_free_rate
        
        # シャープレシオ
        return excess_return / volatility.decimal
    
    def _calculate_sortino_ratio(
        self,
        returns: List[Decimal],
        risk_free_rate: Decimal,
    ) -> Decimal:
        """ソルティノレシオを計算"""
        if not returns:
            return Decimal("0")
        
        # 平均リターン（年率換算）
        mean_return = sum(returns) / len(returns) * Decimal("252")
        
        # 下方偏差を計算
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return Decimal("10")  # 下方リスクがない場合は高い値
        
        downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_deviation = Decimal(str(math.sqrt(float(downside_variance) * 252)))
        
        if downside_deviation == 0:
            return Decimal("10")
        
        # ソルティノレシオ
        excess_return = mean_return - risk_free_rate
        return excess_return / downside_deviation
    
    def _calculate_max_drawdown(self, returns: List[Decimal]) -> Percentage:
        """最大ドローダウンを計算"""
        if not returns:
            return Percentage(0)
        
        # 累積リターンを計算
        cumulative = [Decimal("1")]
        for r in returns:
            cumulative.append(cumulative[-1] * (1 + r))
        
        # 最大ドローダウンを計算
        max_drawdown = Decimal("0")
        peak = cumulative[0]
        
        for value in cumulative[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return Percentage.from_decimal(max_drawdown)
    
    def _calculate_beta(
        self,
        portfolio_returns: List[Decimal],
        benchmark_returns: List[Decimal],
    ) -> Decimal:
        """ベータを計算"""
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return Decimal("1")
        
        # 共通の長さに合わせる
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        # 共分散を計算
        portfolio_mean = sum(portfolio_returns) / len(portfolio_returns)
        benchmark_mean = sum(benchmark_returns) / len(benchmark_returns)
        
        covariance = sum(
            (p - portfolio_mean) * (b - benchmark_mean)
            for p, b in zip(portfolio_returns, benchmark_returns)
        ) / (len(portfolio_returns) - 1)
        
        # ベンチマークの分散を計算
        benchmark_variance = sum(
            (b - benchmark_mean) ** 2 for b in benchmark_returns
        ) / (len(benchmark_returns) - 1)
        
        if benchmark_variance == 0:
            return Decimal("1")
        
        return covariance / benchmark_variance
    
    def _calculate_correlation(
        self,
        portfolio_returns: List[Decimal],
        benchmark_returns: List[Decimal],
    ) -> Decimal:
        """相関係数を計算"""
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return Decimal("0")
        
        # 共通の長さに合わせる
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        # 平均を計算
        portfolio_mean = sum(portfolio_returns) / len(portfolio_returns)
        benchmark_mean = sum(benchmark_returns) / len(benchmark_returns)
        
        # 共分散を計算
        covariance = sum(
            (p - portfolio_mean) * (b - benchmark_mean)
            for p, b in zip(portfolio_returns, benchmark_returns)
        ) / (len(portfolio_returns) - 1)
        
        # 標準偏差を計算
        portfolio_std = Decimal(str(math.sqrt(float(sum(
            (p - portfolio_mean) ** 2 for p in portfolio_returns
        ) / (len(portfolio_returns) - 1)))))
        
        benchmark_std = Decimal(str(math.sqrt(float(sum(
            (b - benchmark_mean) ** 2 for b in benchmark_returns
        ) / (len(benchmark_returns) - 1)))))
        
        if portfolio_std == 0 or benchmark_std == 0:
            return Decimal("0")
        
        return covariance / (portfolio_std * benchmark_std)
    
    def _calculate_tracking_error(
        self,
        portfolio_returns: List[Decimal],
        benchmark_returns: List[Decimal],
    ) -> Percentage:
        """トラッキングエラーを計算"""
        if not portfolio_returns or not benchmark_returns:
            return Percentage(0)
        
        # 共通の長さに合わせる
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        # 差分リターンを計算
        diff_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
        
        if len(diff_returns) < 2:
            return Percentage(0)
        
        # 標準偏差を計算
        mean_diff = sum(diff_returns) / len(diff_returns)
        variance = sum((d - mean_diff) ** 2 for d in diff_returns) / (len(diff_returns) - 1)
        std_dev = Decimal(str(math.sqrt(float(variance))))
        
        # 年率換算
        annual_tracking_error = std_dev * Decimal(str(math.sqrt(252)))
        
        return Percentage.from_decimal(annual_tracking_error)
    
    def _calculate_risk_contribution(
        self,
        position: Position,
        portfolio: Portfolio,
        historical_prices: List[OHLCV],
    ) -> Percentage:
        """ポジションのリスク寄与度を計算"""
        if not historical_prices:
            return Percentage(0)
        
        # 簡易的な実装：ボラティリティ × 比重
        returns = self._calculate_returns(historical_prices)
        if not returns:
            return Percentage(0)
        
        volatility = self._calculate_volatility(returns)
        weight = position.market_value / portfolio.total_value
        
        return Percentage.from_decimal(volatility.decimal * weight)
    
    def _calculate_concentration_risk(self, weight: Percentage) -> Percentage:
        """集中リスクを計算"""
        # 簡易的な実装：比重が高いほどリスク増大
        if weight.value > 20:
            return Percentage(80)
        elif weight.value > 15:
            return Percentage(60)
        elif weight.value > 10:
            return Percentage(40)
        elif weight.value > 5:
            return Percentage(20)
        else:
            return Percentage(10)
    
    def _calculate_liquidity_risk(
        self,
        position: Position,
        historical_prices: List[OHLCV],
    ) -> Percentage:
        """流動性リスクを計算"""
        if not historical_prices:
            return Percentage(50)  # データなしの場合は中程度のリスク
        
        # 平均取引量を計算
        avg_volume = sum(p.volume for p in historical_prices) / len(historical_prices)
        
        # ポジションサイズと平均取引量の比率でリスクを評価
        if avg_volume == 0:
            return Percentage(100)
        
        days_to_liquidate = Decimal(position.quantity) / Decimal(avg_volume)
        
        if days_to_liquidate > 10:
            return Percentage(90)
        elif days_to_liquidate > 5:
            return Percentage(70)
        elif days_to_liquidate > 2:
            return Percentage(50)
        elif days_to_liquidate > 1:
            return Percentage(30)
        else:
            return Percentage(10)