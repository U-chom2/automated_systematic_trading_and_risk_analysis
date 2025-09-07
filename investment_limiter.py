"""
Investment Limiter Module
投資リミッターモジュール - 投資額の制限とリスク管理
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from config import config

logger = logging.getLogger(__name__)


@dataclass
class LimitedInvestment:
    """制限適用後の投資情報"""
    original_amount: float
    limited_amount: float
    shares: int
    is_limited: bool
    limit_reason: Optional[str] = None


@dataclass
class PortfolioLimits:
    """ポートフォリオ制限情報"""
    total_investment: float
    max_position_count: int
    max_single_position: float
    total_risk_limit: float


class InvestmentLimiter:
    """投資リミッタークラス"""
    
    def __init__(self):
        self.limits = config.investment_limits
        self.thresholds = config.investment_thresholds
    
    def apply_stock_limit(
        self, 
        stock_price: float, 
        recommended_amount: float
    ) -> LimitedInvestment:
        """1株当たりの投資制限を適用
        
        Args:
            stock_price: 株価
            recommended_amount: 推奨投資額
        
        Returns:
            制限適用後の投資情報
        """
        if stock_price <= 0:
            return LimitedInvestment(
                original_amount=recommended_amount,
                limited_amount=0.0,
                shares=0,
                is_limited=True,
                limit_reason="無効な株価"
            )
        
        # 最大購入可能株数を計算
        max_shares_by_limit = int(self.limits.max_investment_per_stock / stock_price)
        original_shares = int(recommended_amount / stock_price)
        
        # より制限の厳しい方を採用
        final_shares = min(max_shares_by_limit, original_shares)
        final_amount = final_shares * stock_price
        
        is_limited = final_shares < original_shares
        limit_reason = None
        if is_limited:
            limit_reason = f"1株当たり¥{self.limits.max_investment_per_stock:,.0f}制限"
        
        return LimitedInvestment(
            original_amount=recommended_amount,
            limited_amount=final_amount,
            shares=final_shares,
            is_limited=is_limited,
            limit_reason=limit_reason
        )
    
    def calculate_portfolio_allocation(
        self, 
        recommendations: List[Dict[str, any]]
    ) -> Dict[str, LimitedInvestment]:
        """ポートフォリオ全体の投資配分を計算
        
        Args:
            recommendations: 投資推奨のリスト
        
        Returns:
            シンボルをキーとした制限適用後の投資情報
        """
        allocations = {}
        total_limited_investment = 0.0
        
        for rec in recommendations:
            symbol = rec['symbol']
            stock_price = rec['current_price']
            position_size = abs(rec.get('position_size', 0))
            
            # 元の推奨投資額を計算
            original_amount = self.limits.base_investment_amount * position_size
            
            # 株式制限を適用
            limited_investment = self.apply_stock_limit(stock_price, original_amount)
            
            allocations[symbol] = limited_investment
            total_limited_investment += limited_investment.limited_amount
            
            logger.debug(f"{symbol}: {original_amount:,.0f} → {limited_investment.limited_amount:,.0f}")
        
        logger.info(f"Total limited investment: ¥{total_limited_investment:,.0f}")
        
        return allocations
    
    def apply_risk_limits(
        self, 
        allocations: Dict[str, LimitedInvestment],
        max_total_risk: float = 100000.0  # 最大リスク額（例：10万円）
    ) -> Dict[str, LimitedInvestment]:
        """リスク制限を適用
        
        Args:
            allocations: 投資配分
            max_total_risk: 最大許容リスク額
        
        Returns:
            リスク制限適用後の投資配分
        """
        total_risk = sum(alloc.limited_amount for alloc in allocations.values())
        
        if total_risk <= max_total_risk:
            return allocations
        
        # リスク制限が必要な場合はプロポーショナルに減額
        risk_ratio = max_total_risk / total_risk
        
        adjusted_allocations = {}
        for symbol, allocation in allocations.items():
            adjusted_amount = allocation.limited_amount * risk_ratio
            
            # 株数を再計算（株価から逆算）
            if allocation.shares > 0:
                stock_price = allocation.limited_amount / allocation.shares
                adjusted_shares = max(1, int(adjusted_amount / stock_price))
                adjusted_amount = adjusted_shares * stock_price
            else:
                adjusted_shares = 0
            
            adjusted_allocations[symbol] = LimitedInvestment(
                original_amount=allocation.original_amount,
                limited_amount=adjusted_amount,
                shares=adjusted_shares,
                is_limited=True,
                limit_reason=f"リスク制限適用 (¥{max_total_risk:,.0f})"
            )
        
        logger.info(f"Risk limit applied: {total_risk:,.0f} → {max_total_risk:,.0f}")
        
        return adjusted_allocations
    
    def validate_investment_safety(
        self, 
        allocations: Dict[str, LimitedInvestment]
    ) -> Dict[str, any]:
        """投資の安全性を検証
        
        Args:
            allocations: 投資配分
        
        Returns:
            安全性検証結果
        """
        total_investment = sum(alloc.limited_amount for alloc in allocations.values())
        position_count = len([alloc for alloc in allocations.values() if alloc.shares > 0])
        
        max_single_position = max(
            (alloc.limited_amount for alloc in allocations.values()), 
            default=0
        )
        
        # 安全性指標
        diversification_score = min(100, position_count * 5)  # 分散度スコア
        concentration_risk = (max_single_position / total_investment * 100) if total_investment > 0 else 0
        
        safety_level = "高"
        if concentration_risk > 30:
            safety_level = "低"
        elif concentration_risk > 15:
            safety_level = "中"
        
        return {
            "total_investment": total_investment,
            "position_count": position_count,
            "max_single_position": max_single_position,
            "concentration_risk_pct": concentration_risk,
            "diversification_score": diversification_score,
            "safety_level": safety_level,
            "max_loss_estimate": total_investment * 0.1  # 想定最大損失（10%）
        }
    
    def generate_risk_warning(
        self, 
        safety_validation: Dict[str, any]
    ) -> List[str]:
        """リスク警告メッセージを生成"""
        warnings = []
        
        if safety_validation["concentration_risk_pct"] > 20:
            warnings.append(
                f"⚠️ 集中リスク: 単一銘柄が{safety_validation['concentration_risk_pct']:.1f}%を占めています"
            )
        
        if safety_validation["position_count"] < 5:
            warnings.append("⚠️ 分散不足: より多くの銘柄への分散投資を検討してください")
        
        if safety_validation["total_investment"] > 50000:
            warnings.append("⚠️ 高額投資: 余裕資金での投資を推奨します")
        
        return warnings
    
    def format_investment_summary(
        self, 
        allocations: Dict[str, LimitedInvestment],
        safety_validation: Dict[str, any]
    ) -> str:
        """投資サマリーをフォーマット"""
        lines = []
        lines.append("💰 【投資制限後サマリー】")
        lines.append("-" * 50)
        lines.append(f"総投資額: ¥{safety_validation['total_investment']:,.0f}")
        lines.append(f"投資銘柄数: {safety_validation['position_count']}銘柄")
        lines.append(f"最大単一投資: ¥{safety_validation['max_single_position']:,.0f}")
        lines.append(f"想定最大損失: ¥{safety_validation['max_loss_estimate']:,.0f}")
        lines.append(f"安全性レベル: {safety_validation['safety_level']}")
        
        # 制限適用銘柄の詳細
        limited_count = sum(1 for alloc in allocations.values() if alloc.is_limited)
        if limited_count > 0:
            lines.append(f"制限適用銘柄: {limited_count}銘柄")
        
        return "\n".join(lines)