"""ポートフォリオ最適化ドメインサービス"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
import math

from ..entities.portfolio import Portfolio
from ..entities.stock import Stock
from ..value_objects.price import OHLCV
from ..value_objects.percentage import Percentage


@dataclass
class OptimizationConstraints:
    """最適化制約"""
    min_weight: Percentage = Percentage(0)  # 最小ウェイト
    max_weight: Percentage = Percentage(30)  # 最大ウェイト
    max_positions: int = 20  # 最大銘柄数
    min_positions: int = 5  # 最小銘柄数
    sector_limits: Dict[str, Percentage] = None  # セクター制限
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        if self.sector_limits is None:
            self.sector_limits = {}


@dataclass
class OptimizedPortfolio:
    """最適化されたポートフォリオ"""
    weights: Dict[str, Percentage]  # ticker -> weight
    expected_return: Percentage
    expected_risk: Percentage
    sharpe_ratio: Decimal
    diversification_ratio: Decimal
    
    def __str__(self) -> str:
        """文字列表現"""
        return (
            f"Expected Return: {self.expected_return}, "
            f"Risk: {self.expected_risk}, "
            f"Sharpe: {self.sharpe_ratio:.2f}"
        )


class PortfolioOptimizer:
    """ポートフォリオ最適化ドメインサービス
    
    効率的フロンティアの計算とポートフォリオの最適化を行う。
    """
    
    def __init__(
        self,
        risk_free_rate: Decimal = Decimal("0.001"),  # 無リスク金利
        target_return: Optional[Percentage] = None,  # 目標リターン
        optimization_method: str = "mean_variance",  # 最適化手法
    ) -> None:
        """初期化"""
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.optimization_method = optimization_method
    
    def optimize_portfolio(
        self,
        candidates: List[Stock],
        historical_prices: Dict[str, List[OHLCV]],
        constraints: OptimizationConstraints,
        current_portfolio: Optional[Portfolio] = None,
    ) -> OptimizedPortfolio:
        """ポートフォリオを最適化
        
        Args:
            candidates: 候補銘柄リスト
            historical_prices: 過去価格データ
            constraints: 最適化制約
            current_portfolio: 現在のポートフォリオ（リバランス用）
        
        Returns:
            最適化されたポートフォリオ
        """
        # リターンとリスクを計算
        returns_matrix = self._calculate_returns_matrix(candidates, historical_prices)
        
        if not returns_matrix:
            # データ不足の場合は均等配分
            return self._create_equal_weight_portfolio(candidates)
        
        # 共分散行列を計算
        covariance_matrix = self._calculate_covariance_matrix(returns_matrix)
        expected_returns = self._calculate_expected_returns(returns_matrix)
        
        # 最適化手法に応じて最適化
        if self.optimization_method == "mean_variance":
            weights = self._mean_variance_optimization(
                expected_returns,
                covariance_matrix,
                constraints
            )
        elif self.optimization_method == "min_variance":
            weights = self._min_variance_optimization(
                covariance_matrix,
                constraints
            )
        elif self.optimization_method == "max_sharpe":
            weights = self._max_sharpe_optimization(
                expected_returns,
                covariance_matrix,
                constraints
            )
        elif self.optimization_method == "risk_parity":
            weights = self._risk_parity_optimization(
                covariance_matrix,
                constraints
            )
        else:
            weights = self._equal_weight_optimization(candidates, constraints)
        
        # ポートフォリオのメトリクスを計算
        portfolio_return = self._calculate_portfolio_return(weights, expected_returns)
        portfolio_risk = self._calculate_portfolio_risk(weights, covariance_matrix)
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_return, portfolio_risk)
        diversification_ratio = self._calculate_diversification_ratio(weights, covariance_matrix)
        
        return OptimizedPortfolio(
            weights=weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
        )
    
    def rebalance_portfolio(
        self,
        current_portfolio: Portfolio,
        target_portfolio: OptimizedPortfolio,
        threshold: Percentage = Percentage(5),
    ) -> Dict[str, Tuple[str, int]]:
        """ポートフォリオをリバランス
        
        Args:
            current_portfolio: 現在のポートフォリオ
            target_portfolio: 目標ポートフォリオ
            threshold: リバランス閾値
        
        Returns:
            ticker -> (action, quantity) のマッピング
        """
        rebalance_actions = {}
        total_value = current_portfolio.total_value
        
        # 現在のウェイトを計算
        current_weights = {}
        for ticker, position in current_portfolio.positions.items():
            if not position.is_closed:
                weight = Percentage.from_decimal(position.market_value / total_value)
                current_weights[ticker] = weight
        
        # 目標ウェイトと比較
        for ticker, target_weight in target_portfolio.weights.items():
            current_weight = current_weights.get(ticker, Percentage(0))
            diff = abs(target_weight.value - current_weight.value)
            
            # 閾値を超える場合はリバランス
            if diff > threshold.value:
                target_value = total_value * target_weight.decimal
                current_value = total_value * current_weight.decimal
                value_diff = target_value - current_value
                
                if value_diff > 0:
                    # 買い増し
                    rebalance_actions[ticker] = ("BUY", abs(value_diff))
                else:
                    # 売却
                    rebalance_actions[ticker] = ("SELL", abs(value_diff))
        
        # 現在保有しているが目標にない銘柄は売却
        for ticker in current_weights:
            if ticker not in target_portfolio.weights:
                position = current_portfolio.positions[ticker]
                if not position.is_closed:
                    rebalance_actions[ticker] = ("SELL", position.market_value)
        
        return rebalance_actions
    
    def calculate_efficient_frontier(
        self,
        candidates: List[Stock],
        historical_prices: Dict[str, List[OHLCV]],
        n_portfolios: int = 100,
    ) -> List[OptimizedPortfolio]:
        """効率的フロンティアを計算
        
        Args:
            candidates: 候補銘柄リスト
            historical_prices: 過去価格データ
            n_portfolios: 計算するポートフォリオ数
        
        Returns:
            効率的フロンティア上のポートフォリオリスト
        """
        frontier_portfolios = []
        
        # リターンとリスクを計算
        returns_matrix = self._calculate_returns_matrix(candidates, historical_prices)
        if not returns_matrix:
            return []
        
        covariance_matrix = self._calculate_covariance_matrix(returns_matrix)
        expected_returns = self._calculate_expected_returns(returns_matrix)
        
        # 最小リターンと最大リターンを取得
        min_return = min(expected_returns.values())
        max_return = max(expected_returns.values())
        
        # 目標リターンを段階的に設定
        for i in range(n_portfolios):
            target_return = min_return + (max_return - min_return) * Decimal(i) / Decimal(n_portfolios - 1)
            target_return_pct = Percentage.from_decimal(target_return)
            
            # 各目標リターンで最適化
            self.target_return = target_return_pct
            optimized = self.optimize_portfolio(
                candidates,
                historical_prices,
                OptimizationConstraints(),
            )
            frontier_portfolios.append(optimized)
        
        return frontier_portfolios
    
    def _calculate_returns_matrix(
        self,
        candidates: List[Stock],
        historical_prices: Dict[str, List[OHLCV]],
    ) -> Dict[str, List[Decimal]]:
        """リターン行列を計算"""
        returns_matrix = {}
        
        for stock in candidates:
            ticker = stock.ticker
            if ticker not in historical_prices:
                continue
            
            prices = historical_prices[ticker]
            returns = []
            
            for i in range(1, len(prices)):
                prev_price = prices[i-1].close.value
                curr_price = prices[i].close.value
                
                if prev_price > 0:
                    daily_return = (curr_price - prev_price) / prev_price
                    returns.append(daily_return)
            
            if returns:
                returns_matrix[ticker] = returns
        
        return returns_matrix
    
    def _calculate_covariance_matrix(
        self,
        returns_matrix: Dict[str, List[Decimal]],
    ) -> Dict[Tuple[str, str], Decimal]:
        """共分散行列を計算"""
        covariance_matrix = {}
        tickers = list(returns_matrix.keys())
        
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                returns1 = returns_matrix[ticker1]
                returns2 = returns_matrix[ticker2]
                
                # 共通の長さに合わせる
                min_length = min(len(returns1), len(returns2))
                returns1 = returns1[:min_length]
                returns2 = returns2[:min_length]
                
                if len(returns1) < 2:
                    covariance_matrix[(ticker1, ticker2)] = Decimal("0")
                    continue
                
                # 平均を計算
                mean1 = sum(returns1) / len(returns1)
                mean2 = sum(returns2) / len(returns2)
                
                # 共分散を計算
                covariance = sum(
                    (r1 - mean1) * (r2 - mean2)
                    for r1, r2 in zip(returns1, returns2)
                ) / (len(returns1) - 1)
                
                # 年率換算
                covariance_annual = covariance * Decimal("252")
                covariance_matrix[(ticker1, ticker2)] = covariance_annual
        
        return covariance_matrix
    
    def _calculate_expected_returns(
        self,
        returns_matrix: Dict[str, List[Decimal]],
    ) -> Dict[str, Decimal]:
        """期待リターンを計算"""
        expected_returns = {}
        
        for ticker, returns in returns_matrix.items():
            if returns:
                # 平均リターン（年率換算）
                mean_return = sum(returns) / len(returns) * Decimal("252")
                expected_returns[ticker] = mean_return
        
        return expected_returns
    
    def _mean_variance_optimization(
        self,
        expected_returns: Dict[str, Decimal],
        covariance_matrix: Dict[Tuple[str, str], Decimal],
        constraints: OptimizationConstraints,
    ) -> Dict[str, Percentage]:
        """平均分散最適化"""
        # 簡易的な実装：目標リターンに応じて配分
        tickers = list(expected_returns.keys())
        
        if not tickers:
            return {}
        
        # 目標リターンがない場合は均等配分
        if self.target_return is None:
            weight = Percentage(100 / len(tickers))
            return {ticker: weight for ticker in tickers}
        
        # リターンでソート
        sorted_tickers = sorted(tickers, key=lambda t: expected_returns[t], reverse=True)
        
        # 上位銘柄に重点配分
        weights = {}
        remaining_weight = Decimal("100")
        n_positions = min(constraints.max_positions, len(sorted_tickers))
        
        for i, ticker in enumerate(sorted_tickers[:n_positions]):
            if i < n_positions // 3:
                # 上位1/3には多めに配分
                weight = min(constraints.max_weight.value, remaining_weight / Decimal(n_positions - i) * Decimal("1.5"))
            else:
                # 残りは均等配分
                weight = remaining_weight / Decimal(n_positions - i)
            
            weight = max(constraints.min_weight.value, min(constraints.max_weight.value, weight))
            weights[ticker] = Percentage(weight)
            remaining_weight -= weight
        
        return weights
    
    def _min_variance_optimization(
        self,
        covariance_matrix: Dict[Tuple[str, str], Decimal],
        constraints: OptimizationConstraints,
    ) -> Dict[str, Percentage]:
        """最小分散最適化"""
        tickers = list(set(t[0] for t in covariance_matrix.keys()))
        
        if not tickers:
            return {}
        
        # 簡易的な実装：分散の逆数で重み付け
        variances = {}
        for ticker in tickers:
            variance = covariance_matrix.get((ticker, ticker), Decimal("1"))
            variances[ticker] = variance if variance > 0 else Decimal("0.001")
        
        # 分散の逆数を計算
        inv_variances = {t: Decimal("1") / v for t, v in variances.items()}
        total_inv_var = sum(inv_variances.values())
        
        # ウェイトを正規化
        weights = {}
        for ticker in tickers[:constraints.max_positions]:
            weight = (inv_variances[ticker] / total_inv_var) * Decimal("100")
            weight = max(constraints.min_weight.value, min(constraints.max_weight.value, weight))
            weights[ticker] = Percentage(weight)
        
        # 合計が100%になるように調整
        total_weight = sum(w.value for w in weights.values())
        if total_weight > 0:
            for ticker in weights:
                weights[ticker] = Percentage(weights[ticker].value * Decimal("100") / total_weight)
        
        return weights
    
    def _max_sharpe_optimization(
        self,
        expected_returns: Dict[str, Decimal],
        covariance_matrix: Dict[Tuple[str, str], Decimal],
        constraints: OptimizationConstraints,
    ) -> Dict[str, Percentage]:
        """シャープレシオ最大化"""
        tickers = list(expected_returns.keys())
        
        if not tickers:
            return {}
        
        # 簡易的な実装：シャープレシオの近似値で重み付け
        sharpe_scores = {}
        
        for ticker in tickers:
            expected_return = expected_returns[ticker]
            variance = covariance_matrix.get((ticker, ticker), Decimal("1"))
            volatility = Decimal(str(math.sqrt(float(variance))))
            
            if volatility > 0:
                sharpe = (expected_return - self.risk_free_rate) / volatility
                sharpe_scores[ticker] = sharpe
            else:
                sharpe_scores[ticker] = Decimal("0")
        
        # シャープレシオでソート
        sorted_tickers = sorted(tickers, key=lambda t: sharpe_scores[t], reverse=True)
        
        # 上位銘柄に配分
        weights = {}
        n_positions = min(constraints.max_positions, len(sorted_tickers))
        
        for i, ticker in enumerate(sorted_tickers[:n_positions]):
            # シャープレシオに応じて配分
            if sharpe_scores[ticker] > 0:
                weight = Percentage(100 / n_positions)
            else:
                weight = Percentage(0)
            
            weights[ticker] = weight
        
        # 合計が100%になるように調整
        total_weight = sum(w.value for w in weights.values())
        if total_weight > 0:
            for ticker in weights:
                weights[ticker] = Percentage(weights[ticker].value * Decimal("100") / total_weight)
        
        return weights
    
    def _risk_parity_optimization(
        self,
        covariance_matrix: Dict[Tuple[str, str], Decimal],
        constraints: OptimizationConstraints,
    ) -> Dict[str, Percentage]:
        """リスクパリティ最適化"""
        tickers = list(set(t[0] for t in covariance_matrix.keys()))
        
        if not tickers:
            return {}
        
        # 簡易的な実装：ボラティリティの逆数で重み付け
        volatilities = {}
        for ticker in tickers:
            variance = covariance_matrix.get((ticker, ticker), Decimal("1"))
            volatility = Decimal(str(math.sqrt(float(variance))))
            volatilities[ticker] = volatility if volatility > 0 else Decimal("0.001")
        
        # ボラティリティの逆数を計算
        inv_vols = {t: Decimal("1") / v for t, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        # ウェイトを正規化
        weights = {}
        for ticker in tickers[:constraints.max_positions]:
            weight = (inv_vols[ticker] / total_inv_vol) * Decimal("100")
            weight = max(constraints.min_weight.value, min(constraints.max_weight.value, weight))
            weights[ticker] = Percentage(weight)
        
        return weights
    
    def _equal_weight_optimization(
        self,
        candidates: List[Stock],
        constraints: OptimizationConstraints,
    ) -> Dict[str, Percentage]:
        """均等配分最適化"""
        n_positions = min(constraints.max_positions, len(candidates))
        weight = Percentage(100 / n_positions)
        
        weights = {}
        for stock in candidates[:n_positions]:
            weights[stock.ticker] = weight
        
        return weights
    
    def _create_equal_weight_portfolio(
        self,
        candidates: List[Stock],
    ) -> OptimizedPortfolio:
        """均等配分ポートフォリオを作成"""
        if not candidates:
            return OptimizedPortfolio(
                weights={},
                expected_return=Percentage(0),
                expected_risk=Percentage(0),
                sharpe_ratio=Decimal("0"),
                diversification_ratio=Decimal("1"),
            )
        
        weight = Percentage(100 / len(candidates))
        weights = {stock.ticker: weight for stock in candidates}
        
        return OptimizedPortfolio(
            weights=weights,
            expected_return=Percentage(5),  # デフォルト値
            expected_risk=Percentage(15),  # デフォルト値
            sharpe_ratio=Decimal("0.3"),  # デフォルト値
            diversification_ratio=Decimal(str(math.sqrt(len(candidates)))),
        )
    
    def _calculate_portfolio_return(
        self,
        weights: Dict[str, Percentage],
        expected_returns: Dict[str, Decimal],
    ) -> Percentage:
        """ポートフォリオリターンを計算"""
        portfolio_return = Decimal("0")
        
        for ticker, weight in weights.items():
            if ticker in expected_returns:
                portfolio_return += weight.decimal * expected_returns[ticker]
        
        return Percentage.from_decimal(portfolio_return)
    
    def _calculate_portfolio_risk(
        self,
        weights: Dict[str, Percentage],
        covariance_matrix: Dict[Tuple[str, str], Decimal],
    ) -> Percentage:
        """ポートフォリオリスクを計算"""
        portfolio_variance = Decimal("0")
        
        for ticker1, weight1 in weights.items():
            for ticker2, weight2 in weights.items():
                covariance = covariance_matrix.get((ticker1, ticker2), Decimal("0"))
                portfolio_variance += weight1.decimal * weight2.decimal * covariance
        
        portfolio_volatility = Decimal(str(math.sqrt(float(portfolio_variance))))
        return Percentage.from_decimal(portfolio_volatility)
    
    def _calculate_sharpe_ratio(
        self,
        portfolio_return: Percentage,
        portfolio_risk: Percentage,
    ) -> Decimal:
        """シャープレシオを計算"""
        if portfolio_risk.value == 0:
            return Decimal("0")
        
        excess_return = portfolio_return.decimal - self.risk_free_rate
        return excess_return / portfolio_risk.decimal
    
    def _calculate_diversification_ratio(
        self,
        weights: Dict[str, Percentage],
        covariance_matrix: Dict[Tuple[str, str], Decimal],
    ) -> Decimal:
        """分散化比率を計算"""
        # 加重平均ボラティリティ
        weighted_vol = Decimal("0")
        for ticker, weight in weights.items():
            variance = covariance_matrix.get((ticker, ticker), Decimal("0"))
            volatility = Decimal(str(math.sqrt(float(variance))))
            weighted_vol += weight.decimal * volatility
        
        # ポートフォリオボラティリティ
        portfolio_risk = self._calculate_portfolio_risk(weights, covariance_matrix)
        
        if portfolio_risk.value == 0:
            return Decimal("1")
        
        return weighted_vol / portfolio_risk.decimal