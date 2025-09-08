"""リスク関連DTO"""
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID

from ...domain.services.risk_calculator import RiskMetrics, PositionRisk


@dataclass
class RiskMetricsDTO:
    """リスク指標DTO"""
    portfolio_id: str
    value_at_risk: float
    conditional_value_at_risk: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation: float
    tracking_error: Optional[float]
    calculated_at: str
    
    @classmethod
    def from_entity(cls, portfolio_id: UUID, metrics: RiskMetrics, calculated_at: str) -> "RiskMetricsDTO":
        """エンティティからDTOを作成"""
        return cls(
            portfolio_id=str(portfolio_id),
            value_at_risk=float(metrics.value_at_risk),
            conditional_value_at_risk=float(metrics.conditional_value_at_risk),
            sharpe_ratio=float(metrics.sharpe_ratio),
            sortino_ratio=float(metrics.sortino_ratio),
            max_drawdown=float(metrics.max_drawdown.value),
            volatility=float(metrics.volatility.value),
            beta=float(metrics.beta),
            correlation=float(metrics.correlation),
            tracking_error=float(metrics.tracking_error.value) if metrics.tracking_error else None,
            calculated_at=calculated_at,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "portfolio_id": self.portfolio_id,
            "metrics": {
                "var": self.value_at_risk,
                "cvar": self.conditional_value_at_risk,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "volatility": self.volatility,
                "beta": self.beta,
                "correlation": self.correlation,
                "tracking_error": self.tracking_error,
            },
            "calculated_at": self.calculated_at,
        }


@dataclass
class PositionRiskDTO:
    """ポジションリスクDTO"""
    position_id: str
    ticker: str
    weight: float
    contribution_to_risk: float
    concentration_risk: float
    liquidity_risk: float
    total_risk_score: float
    
    @classmethod
    def from_entity(cls, position_risk: PositionRisk) -> "PositionRiskDTO":
        """エンティティからDTOを作成"""
        return cls(
            position_id=str(position_risk.position.id),
            ticker=position_risk.position.ticker,
            weight=float(position_risk.weight.value),
            contribution_to_risk=float(position_risk.contribution_to_risk.value),
            concentration_risk=float(position_risk.concentration_risk.value),
            liquidity_risk=float(position_risk.liquidity_risk.value),
            total_risk_score=float(position_risk.total_risk_score),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "position": {
                "id": self.position_id,
                "ticker": self.ticker,
                "weight": self.weight,
            },
            "risks": {
                "contribution": self.contribution_to_risk,
                "concentration": self.concentration_risk,
                "liquidity": self.liquidity_risk,
                "total_score": self.total_risk_score,
            }
        }


@dataclass
class RiskLimitDTO:
    """リスク制限DTO"""
    max_var: Optional[float] = None
    max_volatility: Optional[float] = None
    min_sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_position_size: Optional[float] = None
    max_sector_exposure: Optional[float] = None
    max_correlation: Optional[float] = None
    
    def validate(self) -> None:
        """バリデーション"""
        if self.max_var is not None and self.max_var <= 0:
            raise ValueError("Max VaR must be positive")
        
        if self.max_volatility is not None and (self.max_volatility <= 0 or self.max_volatility > 100):
            raise ValueError("Max volatility must be between 0 and 100")
        
        if self.min_sharpe_ratio is not None and self.min_sharpe_ratio < -10:
            raise ValueError("Min Sharpe ratio seems unrealistic")
        
        if self.max_drawdown is not None and (self.max_drawdown <= 0 or self.max_drawdown > 100):
            raise ValueError("Max drawdown must be between 0 and 100")
        
        if self.max_position_size is not None and (self.max_position_size <= 0 or self.max_position_size > 100):
            raise ValueError("Max position size must be between 0 and 100")
        
        if self.max_sector_exposure is not None and (self.max_sector_exposure <= 0 or self.max_sector_exposure > 100):
            raise ValueError("Max sector exposure must be between 0 and 100")
        
        if self.max_correlation is not None and (self.max_correlation < -1 or self.max_correlation > 1):
            raise ValueError("Max correlation must be between -1 and 1")
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """辞書形式に変換"""
        return {
            "max_var": self.max_var,
            "max_volatility": self.max_volatility,
            "min_sharpe_ratio": self.min_sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_position_size": self.max_position_size,
            "max_sector_exposure": self.max_sector_exposure,
            "max_correlation": self.max_correlation,
        }


@dataclass
class RiskCheckResultDTO:
    """リスクチェック結果DTO"""
    portfolio_id: str
    checks_passed: Dict[str, bool]
    all_passed: bool
    failed_checks: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    @classmethod
    def create(
        cls,
        portfolio_id: UUID,
        checks: Dict[str, bool],
        current_metrics: RiskMetricsDTO,
    ) -> "RiskCheckResultDTO":
        """チェック結果からDTOを作成"""
        failed_checks = [name for name, passed in checks.items() if not passed]
        all_passed = len(failed_checks) == 0
        
        # 警告とレコメンデーションを生成
        warnings = []
        recommendations = []
        
        if "max_var" in failed_checks:
            warnings.append("Value at Risk exceeds limit")
            recommendations.append("Consider reducing position sizes or diversifying portfolio")
        
        if "max_volatility" in failed_checks:
            warnings.append("Portfolio volatility is too high")
            recommendations.append("Add more stable assets or reduce high-volatility positions")
        
        if "min_sharpe" in failed_checks:
            warnings.append("Sharpe ratio is below minimum threshold")
            recommendations.append("Improve risk-adjusted returns by optimizing asset allocation")
        
        if "max_drawdown" in failed_checks:
            warnings.append("Maximum drawdown exceeds limit")
            recommendations.append("Implement stop-loss strategies or reduce exposure")
        
        return cls(
            portfolio_id=str(portfolio_id),
            checks_passed=checks,
            all_passed=all_passed,
            failed_checks=failed_checks,
            warnings=warnings,
            recommendations=recommendations,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "portfolio_id": self.portfolio_id,
            "result": {
                "all_passed": self.all_passed,
                "checks": self.checks_passed,
                "failed": self.failed_checks,
            },
            "alerts": {
                "warnings": self.warnings,
                "recommendations": self.recommendations,
            }
        }


@dataclass
class RiskReportDTO:
    """リスクレポートDTO"""
    portfolio_id: str
    report_date: str
    metrics: RiskMetricsDTO
    position_risks: List[PositionRiskDTO]
    risk_checks: RiskCheckResultDTO
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "portfolio_id": self.portfolio_id,
            "report_date": self.report_date,
            "risk_metrics": self.metrics.to_dict(),
            "position_risks": [pr.to_dict() for pr in self.position_risks],
            "risk_checks": self.risk_checks.to_dict(),
            "summary": self.summary,
        }