"""
Investment Limiter Module
投資リミッターモジュール - 投資額の制限とリスク管理
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, date

from config import config, Config, TradingMode

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


@dataclass
class DayTradingRisk:
    """デイトレード用リスク管理情報"""
    daily_loss: float
    current_positions: int
    max_single_loss: float
    remaining_daily_budget: float
    is_daily_limit_reached: bool
    risk_warnings: List[str]


@dataclass
class StopLossOrder:
    """損切り注文情報"""
    symbol: str
    entry_price: float
    current_price: float
    stop_loss_price: float
    loss_amount: float
    loss_percentage: float
    should_execute: bool
    urgency_level: str  # "低", "中", "高", "緊急"


class InvestmentLimiter:
    """投資リミッタークラス（取引モード対応）"""
    
    def __init__(self, config_instance: Optional[Config] = None):
        """
        Args:
            config_instance: 設定インスタンス。Noneの場合はデフォルトconfig使用
        """
        self.config = config_instance or config
        self.limits = self.config.investment_limits
        self.thresholds = self.config.investment_thresholds
        self.trading_mode = getattr(self.config, 'trading_mode', TradingMode.LONG_TERM)
        
        # デイトレード用の追加状態管理
        self.daily_losses: Dict[date, float] = {}  # 日別損失記録
        self.current_positions: Dict[str, Dict] = {}  # 現在のポジション
    
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
        max_total_risk: Optional[float] = None
    ) -> Dict[str, LimitedInvestment]:
        """リスク制限を適用（取引モード対応）
        
        Args:
            allocations: 投資配分
            max_total_risk: 最大許容リスク額（Noneの場合は設定から取得）
        
        Returns:
            リスク制限適用後の投資配分
        """
        if max_total_risk is None:
            if self.trading_mode == TradingMode.DAY_TRADING:
                # デイトレード: より厳格な制限
                max_total_risk = getattr(self.limits, 'max_daily_loss', 5000.0)
            else:
                # 中長期: 従来の制限
                max_total_risk = 100000.0
        
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
            
            limit_type = "デイトレード日次" if self.trading_mode == TradingMode.DAY_TRADING else "リスク"
            adjusted_allocations[symbol] = LimitedInvestment(
                original_amount=allocation.original_amount,
                limited_amount=adjusted_amount,
                shares=adjusted_shares,
                is_limited=True,
                limit_reason=f"{limit_type}制限適用 (¥{max_total_risk:,.0f})"
            )
        
        logger.info(f"Risk limit applied: {total_risk:,.0f} → {max_total_risk:,.0f}")
        
        return adjusted_allocations
    
    def update_daily_loss(self, loss_amount: float, trading_date: Optional[date] = None) -> None:
        """日次損失を更新
        
        Args:
            loss_amount: 損失額
            trading_date: 取引日（Noneの場合は今日）
        """
        if trading_date is None:
            trading_date = date.today()
        
        if trading_date not in self.daily_losses:
            self.daily_losses[trading_date] = 0.0
        
        self.daily_losses[trading_date] += loss_amount
        logger.info(f"Daily loss updated: {trading_date} +¥{loss_amount:,.0f} = ¥{self.daily_losses[trading_date]:,.0f}")
    
    def add_position(self, symbol: str, entry_price: float, shares: int) -> None:
        """ポジションを追加
        
        Args:
            symbol: シンボル
            entry_price: エントリー価格
            shares: 株数
        """
        self.current_positions[symbol] = {
            'entry_price': entry_price,
            'shares': shares,
            'entry_date': date.today()
        }
        logger.info(f"Position added: {symbol} {shares} shares @ ¥{entry_price}")
    
    def remove_position(self, symbol: str) -> None:
        """ポジションを削除
        
        Args:
            symbol: シンボル
        """
        if symbol in self.current_positions:
            del self.current_positions[symbol]
            logger.info(f"Position removed: {symbol}")
    
    def check_daytrading_limits(self, allocations: Dict[str, LimitedInvestment]) -> DayTradingRisk:
        """デイトレード用制限チェック
        
        Args:
            allocations: 投資配分
        
        Returns:
            デイトレードリスク情報
        """
        today = date.today()
        daily_loss = self.daily_losses.get(today, 0.0)
        
        if not hasattr(self.limits, 'max_daily_positions'):
            # デイトレード設定がない場合のデフォルト値
            max_daily_positions = 5
            max_daily_loss = 5000.0
            max_single_loss = 1000.0
        else:
            max_daily_positions = self.limits.max_daily_positions
            max_daily_loss = self.limits.max_daily_loss
            max_single_loss = self.limits.max_single_loss
        
        current_positions = len([alloc for alloc in allocations.values() if alloc.shares > 0])
        remaining_budget = max(0, max_daily_loss - daily_loss)
        is_limit_reached = daily_loss >= max_daily_loss or current_positions >= max_daily_positions
        
        warnings = []
        if daily_loss > max_daily_loss * 0.8:
            warnings.append(f"⚠️ 日次損失が上限の80%に達しています：¥{daily_loss:,.0f}/¥{max_daily_loss:,.0f}")
        
        if current_positions >= max_daily_positions:
            warnings.append(f"⚠️ 同時保有上限に達しています：{current_positions}/{max_daily_positions}銀柄")
        
        return DayTradingRisk(
            daily_loss=daily_loss,
            current_positions=current_positions,
            max_single_loss=max_single_loss,
            remaining_daily_budget=remaining_budget,
            is_daily_limit_reached=is_limit_reached,
            risk_warnings=warnings
        )
    
    def calculate_stop_loss_orders(self, current_positions: Dict[str, Dict]) -> List[StopLossOrder]:
        """損切り注文を計算
        
        Args:
            current_positions: 現在のポジション情報
                {'symbol': {'entry_price': float, 'current_price': float, 'shares': int}}
        
        Returns:
            損切り注文リスト
        """
        stop_loss_orders = []
        
        for symbol, position in current_positions.items():
            entry_price = position['entry_price']
            current_price = position['current_price']
            shares = position.get('shares', 0)
            
            if shares <= 0:
                continue
            
            # 取引モードに応じた損切りラインを計算
            if self.trading_mode == TradingMode.DAY_TRADING:
                # デイトレード: 早めの損切り
                stop_loss_percentage = abs(self.thresholds.stop_loss_strong)  # -1.5% → 1.5%
            else:
                # 中長期: 従来の損切り
                stop_loss_percentage = abs(self.thresholds.stop_loss_strong)  # -8.0% → 8.0%
            
            stop_loss_price = entry_price * (1 - stop_loss_percentage / 100)
            loss_amount = (entry_price - current_price) * shares
            loss_percentage = ((current_price - entry_price) / entry_price) * 100
            
            # 損切り実行判定
            should_execute = current_price <= stop_loss_price
            
            # 緊急度レベル判定
            urgency_level = self._calculate_urgency_level(loss_percentage, stop_loss_percentage)
            
            stop_loss_orders.append(StopLossOrder(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                stop_loss_price=stop_loss_price,
                loss_amount=loss_amount,
                loss_percentage=loss_percentage,
                should_execute=should_execute,
                urgency_level=urgency_level
            ))
        
        return stop_loss_orders
    
    def _calculate_urgency_level(self, current_loss_pct: float, stop_loss_pct: float) -> str:
        """緊急度レベルを計算"""
        if current_loss_pct <= -stop_loss_pct:
            return "緊急"  # 損切りライン達成
        elif current_loss_pct <= -stop_loss_pct * 0.8:
            return "高"     # 80%達成
        elif current_loss_pct <= -stop_loss_pct * 0.5:
            return "中"     # 50%達成
        else:
            return "低"     # 軽微な損失
    
    def validate_investment_safety(
        self, 
        allocations: Dict[str, LimitedInvestment]
    ) -> Dict[str, any]:
        """投資の安全性を検証（取引モード対応）
        
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
        
        # 取引モード別の安全性基準
        if self.trading_mode == TradingMode.DAY_TRADING:
            # デイトレード: より厳格な基準
            diversification_score = min(100, position_count * 10)  # 分散重視
            max_loss_pct = 0.05  # 5%想定損失
            concentration_threshold_high = 25  # 25%以上で高リスク
            concentration_threshold_mid = 15   # 15%以上で中リスク
        else:
            # 中長期: 従来基準
            diversification_score = min(100, position_count * 5)
            max_loss_pct = 0.1  # 10%想定損失
            concentration_threshold_high = 30
            concentration_threshold_mid = 15
        
        concentration_risk = (max_single_position / total_investment * 100) if total_investment > 0 else 0
        
        safety_level = "高"
        if concentration_risk > concentration_threshold_high:
            safety_level = "低"
        elif concentration_risk > concentration_threshold_mid:
            safety_level = "中"
        
        result = {
            "trading_mode": self.trading_mode.value,
            "total_investment": total_investment,
            "position_count": position_count,
            "max_single_position": max_single_position,
            "concentration_risk_pct": concentration_risk,
            "diversification_score": diversification_score,
            "safety_level": safety_level,
            "max_loss_estimate": total_investment * max_loss_pct
        }
        
        # デイトレード特有情報を追加
        if self.trading_mode == TradingMode.DAY_TRADING:
            daytrading_risk = self.check_daytrading_limits(allocations)
            result.update({
                "daily_loss": daytrading_risk.daily_loss,
                "remaining_daily_budget": daytrading_risk.remaining_daily_budget,
                "is_daily_limit_reached": daytrading_risk.is_daily_limit_reached
            })
        
        return result
    
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
    
    def format_daytrading_summary(
        self, 
        allocations: Dict[str, LimitedInvestment],
        safety_validation: Dict[str, any]
    ) -> str:
        """デイトレード用サマリーをフォーマット"""
        lines = []
        lines.append("🏃 【デイトレード制限後サマリー】")
        lines.append("-" * 50)
        lines.append(f"総投資額: ¥{safety_validation['total_investment']:,.0f}")
        lines.append(f"投資銀柄数: {safety_validation['position_count']}銀柄")
        lines.append(f"最大単一投資: ¥{safety_validation['max_single_position']:,.0f}")
        lines.append(f"想定最大損失: ¥{safety_validation['max_loss_estimate']:,.0f} (5%)")
        lines.append(f"安全性レベル: {safety_validation['safety_level']}")
        
        # デイトレード特有情報
        lines.append(f"日次損失: ¥{safety_validation.get('daily_loss', 0):,.0f}")
        lines.append(f"日次予算残高: ¥{safety_validation.get('remaining_daily_budget', 0):,.0f}")
        
        # 制限適用銀柄の詳細
        limited_count = sum(1 for alloc in allocations.values() if alloc.is_limited)
        if limited_count > 0:
            lines.append(f"制限適用銀柄: {limited_count}銀柄")
        
        return "\n".join(lines)
    
    def format_stop_loss_summary(self, stop_loss_orders: List[StopLossOrder]) -> str:
        """損切りサマリーをフォーマット"""
        if not stop_loss_orders:
            return "🔒 損切り対象なし"
        
        lines = []
        lines.append("⚡ 【損切り状況】")
        lines.append("-" * 40)
        
        # 緊急度別にソート
        urgency_order = {"緊急": 0, "高": 1, "中": 2, "低": 3}
        sorted_orders = sorted(stop_loss_orders, key=lambda x: urgency_order.get(x.urgency_level, 4))
        
        for order in sorted_orders:
            urgency_icon = {
                "緊急": "🆘", 
                "高": "⚠️", 
                "中": "🟡", 
                "低": "🟢"
            }.get(order.urgency_level, "❔")
            
            execute_text = "実行推奨" if order.should_execute else "監視中"
            
            lines.append(
                f"{urgency_icon} {order.symbol}: ¥{order.current_price:.0f} → ¥{order.stop_loss_price:.0f} "
                f"({order.loss_percentage:+.1f}%) [{execute_text}]"
            )
        
        return "\n".join(lines)
    
    def get_risk_management_summary(self) -> Dict[str, any]:
        """リスク管理サマリーを取得"""
        today = date.today()
        
        summary = {
            "trading_mode": self.trading_mode.value,
            "current_positions_count": len(self.current_positions),
            "today_loss": self.daily_losses.get(today, 0.0)
        }
        
        if self.trading_mode == TradingMode.DAY_TRADING:
            max_daily_loss = getattr(self.limits, 'max_daily_loss', 5000.0)
            max_daily_positions = getattr(self.limits, 'max_daily_positions', 5)
            
            summary.update({
                "max_daily_loss": max_daily_loss,
                "max_daily_positions": max_daily_positions,
                "remaining_daily_budget": max(0, max_daily_loss - summary["today_loss"]),
                "remaining_position_slots": max(0, max_daily_positions - summary["current_positions_count"]),
                "daily_limit_utilization": (summary["today_loss"] / max_daily_loss) * 100 if max_daily_loss > 0 else 0
            })
        
        return summary