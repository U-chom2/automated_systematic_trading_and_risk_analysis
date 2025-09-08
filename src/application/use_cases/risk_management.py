"""リスク管理ユースケース"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional
from uuid import UUID

from ...domain.entities.portfolio import Portfolio
from ...domain.services.risk_calculator import RiskCalculator, RiskMetrics, PositionRisk
from ...domain.repositories.portfolio_repository import PortfolioRepository
from ...domain.repositories.market_data_repository import MarketDataRepository
from ...domain.repositories.trade_repository import TradeRepository
from ..dto.risk_dto import RiskMetricsDTO, PositionRiskDTO, RiskLimitDTO, RiskCheckResultDTO, RiskReportDTO


@dataclass
class CalculateRiskMetricsUseCase:
    """リスク指標計算ユースケース"""
    
    risk_calculator: RiskCalculator
    portfolio_repository: PortfolioRepository
    market_data_repository: MarketDataRepository
    
    async def execute(
        self,
        portfolio_id: UUID,
        lookback_days: int = 252,
        benchmark_ticker: Optional[str] = None,
    ) -> RiskMetricsDTO:
        """ポートフォリオのリスク指標を計算
        
        Args:
            portfolio_id: ポートフォリオID
            lookback_days: 過去データ参照期間
            benchmark_ticker: ベンチマークティッカー
        
        Returns:
            リスク指標DTO
        """
        # ポートフォリオを取得
        portfolio = await self.portfolio_repository.find_by_id(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
        
        # ポジションを取得
        positions = await self.portfolio_repository.find_positions(portfolio_id)
        for position in positions:
            portfolio.positions[position.ticker] = position
        
        # 過去価格データを取得
        end_date = datetime.now().date()
        start_date = (datetime.now() - timedelta(days=lookback_days)).date()
        
        tickers = [pos.ticker for pos in positions if not pos.is_closed]
        if not tickers:
            # ポジションがない場合はデフォルト値を返す
            return RiskMetricsDTO(
                portfolio_id=str(portfolio_id),
                value_at_risk=0.0,
                conditional_value_at_risk=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                beta=1.0,
                correlation=0.0,
                tracking_error=None,
                calculated_at=datetime.now().isoformat(),
            )
        
        historical_prices = await self.market_data_repository.get_multiple_ohlcv(
            tickers, start_date, end_date
        )
        
        # ベンチマークデータを取得
        benchmark_prices = None
        if benchmark_ticker:
            benchmark_data = await self.market_data_repository.get_ohlcv(
                benchmark_ticker, start_date, end_date
            )
            benchmark_prices = benchmark_data
        
        # リスク指標を計算
        risk_metrics = self.risk_calculator.calculate_portfolio_risk(
            portfolio, historical_prices, benchmark_prices
        )
        
        # DTOに変換して返却
        return RiskMetricsDTO.from_entity(
            portfolio_id,
            risk_metrics,
            datetime.now().isoformat()
        )


@dataclass
class CheckRiskLimitsUseCase:
    """リスク制限チェックユースケース"""
    
    risk_calculator: RiskCalculator
    portfolio_repository: PortfolioRepository
    market_data_repository: MarketDataRepository
    
    async def execute(
        self,
        portfolio_id: UUID,
        risk_limits: RiskLimitDTO,
    ) -> RiskCheckResultDTO:
        """リスク制限をチェック
        
        Args:
            portfolio_id: ポートフォリオID
            risk_limits: リスク制限
        
        Returns:
            リスクチェック結果DTO
        """
        # バリデーション
        risk_limits.validate()
        
        # ポートフォリオを取得
        portfolio = await self.portfolio_repository.find_by_id(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
        
        # 現在のリスク指標を計算
        metrics_use_case = CalculateRiskMetricsUseCase(
            self.risk_calculator,
            self.portfolio_repository,
            self.market_data_repository
        )
        current_metrics = await metrics_use_case.execute(portfolio_id)
        
        # リスク制限をチェック
        checks = {}
        
        if risk_limits.max_var is not None:
            checks["max_var"] = current_metrics.value_at_risk <= risk_limits.max_var
        
        if risk_limits.max_volatility is not None:
            checks["max_volatility"] = current_metrics.volatility <= risk_limits.max_volatility
        
        if risk_limits.min_sharpe_ratio is not None:
            checks["min_sharpe"] = current_metrics.sharpe_ratio >= risk_limits.min_sharpe_ratio
        
        if risk_limits.max_drawdown is not None:
            checks["max_drawdown"] = current_metrics.max_drawdown <= risk_limits.max_drawdown
        
        # チェック結果DTOを作成
        return RiskCheckResultDTO.create(portfolio_id, checks, current_metrics)


@dataclass
class GetPositionRisksUseCase:
    """ポジションリスク取得ユースケース"""
    
    risk_calculator: RiskCalculator
    portfolio_repository: PortfolioRepository
    market_data_repository: MarketDataRepository
    
    async def execute(
        self,
        portfolio_id: UUID,
        lookback_days: int = 30,
    ) -> List[PositionRiskDTO]:
        """個別ポジションのリスクを取得
        
        Args:
            portfolio_id: ポートフォリオID
            lookback_days: 過去データ参照期間
        
        Returns:
            ポジションリスクDTOのリスト
        """
        # ポートフォリオを取得
        portfolio = await self.portfolio_repository.find_by_id(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
        
        # ポジションを取得
        positions = await self.portfolio_repository.find_positions(portfolio_id)
        for position in positions:
            portfolio.positions[position.ticker] = position
        
        if not positions:
            return []
        
        # 過去価格データを取得
        end_date = datetime.now().date()
        start_date = (datetime.now() - timedelta(days=lookback_days)).date()
        
        tickers = [pos.ticker for pos in positions if not pos.is_closed]
        historical_prices = await self.market_data_repository.get_multiple_ohlcv(
            tickers, start_date, end_date
        )
        
        # ポジションリスクを計算
        position_risks = self.risk_calculator.calculate_position_risks(
            portfolio, historical_prices
        )
        
        # DTOに変換して返却
        return [PositionRiskDTO.from_entity(pr) for pr in position_risks]


@dataclass
class GenerateRiskReportUseCase:
    """リスクレポート生成ユースケース"""
    
    risk_calculator: RiskCalculator
    portfolio_repository: PortfolioRepository
    market_data_repository: MarketDataRepository
    trade_repository: TradeRepository
    
    async def execute(
        self,
        portfolio_id: UUID,
        risk_limits: Optional[RiskLimitDTO] = None,
    ) -> RiskReportDTO:
        """リスクレポートを生成
        
        Args:
            portfolio_id: ポートフォリオID
            risk_limits: リスク制限（オプション）
        
        Returns:
            リスクレポートDTO
        """
        # ポートフォリオを取得
        portfolio = await self.portfolio_repository.find_by_id(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
        
        # リスク指標を計算
        metrics_use_case = CalculateRiskMetricsUseCase(
            self.risk_calculator,
            self.portfolio_repository,
            self.market_data_repository
        )
        risk_metrics = await metrics_use_case.execute(portfolio_id)
        
        # ポジションリスクを取得
        position_use_case = GetPositionRisksUseCase(
            self.risk_calculator,
            self.portfolio_repository,
            self.market_data_repository
        )
        position_risks = await position_use_case.execute(portfolio_id)
        
        # リスクチェック
        risk_checks = None
        if risk_limits:
            check_use_case = CheckRiskLimitsUseCase(
                self.risk_calculator,
                self.portfolio_repository,
                self.market_data_repository
            )
            risk_checks = await check_use_case.execute(portfolio_id, risk_limits)
        else:
            # デフォルトのリスクチェック結果
            risk_checks = RiskCheckResultDTO(
                portfolio_id=str(portfolio_id),
                checks_passed={},
                all_passed=True,
                failed_checks=[],
                warnings=[],
                recommendations=[],
            )
        
        # サマリーを生成
        summary = self._generate_risk_summary(
            portfolio,
            risk_metrics,
            position_risks,
            risk_checks
        )
        
        return RiskReportDTO(
            portfolio_id=str(portfolio_id),
            report_date=datetime.now().isoformat(),
            metrics=risk_metrics,
            position_risks=position_risks,
            risk_checks=risk_checks,
            summary=summary,
        )
    
    def _generate_risk_summary(
        self,
        portfolio: Portfolio,
        metrics: RiskMetricsDTO,
        position_risks: List[PositionRiskDTO],
        risk_checks: RiskCheckResultDTO,
    ) -> str:
        """リスクサマリーを生成"""
        summary_parts = []
        
        # 全体的なリスク評価
        if metrics.volatility < 10:
            risk_level = "Low"
        elif metrics.volatility < 20:
            risk_level = "Moderate"
        elif metrics.volatility < 30:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        summary_parts.append(f"Overall Risk Level: {risk_level}")
        
        # VaR情報
        summary_parts.append(
            f"Value at Risk (95%): ¥{metrics.value_at_risk:,.0f} "
            f"({metrics.value_at_risk / float(portfolio.total_value) * 100:.1f}% of portfolio)"
        )
        
        # パフォーマンス指標
        if metrics.sharpe_ratio > 1:
            performance = "Excellent"
        elif metrics.sharpe_ratio > 0.5:
            performance = "Good"
        elif metrics.sharpe_ratio > 0:
            performance = "Positive"
        else:
            performance = "Poor"
        
        summary_parts.append(
            f"Risk-Adjusted Performance: {performance} (Sharpe Ratio: {metrics.sharpe_ratio:.2f})"
        )
        
        # 集中リスク
        if position_risks:
            high_concentration = [
                pr for pr in position_risks
                if pr.concentration_risk > 50
            ]
            if high_concentration:
                summary_parts.append(
                    f"Concentration Warning: {len(high_concentration)} position(s) "
                    "have high concentration risk"
                )
        
        # リスクチェック結果
        if not risk_checks.all_passed:
            summary_parts.append(
                f"Risk Limits Violated: {', '.join(risk_checks.failed_checks)}"
            )
        
        # 推奨事項
        if risk_checks.recommendations:
            summary_parts.append(
                f"Recommendations: {'; '.join(risk_checks.recommendations[:2])}"
            )
        
        return " | ".join(summary_parts)


@dataclass
class MonitorRiskAlertsUseCase:
    """リスクアラート監視ユースケース"""
    
    risk_calculator: RiskCalculator
    portfolio_repository: PortfolioRepository
    market_data_repository: MarketDataRepository
    
    async def execute(self) -> List[Dict]:
        """全ポートフォリオのリスクアラートを監視
        
        Returns:
            アラートのリスト
        """
        alerts = []
        
        # アクティブなポートフォリオを取得
        portfolios = await self.portfolio_repository.find_active()
        
        for portfolio in portfolios:
            try:
                # リスク指標を計算
                metrics_use_case = CalculateRiskMetricsUseCase(
                    self.risk_calculator,
                    self.portfolio_repository,
                    self.market_data_repository
                )
                metrics = await metrics_use_case.execute(portfolio.id)
                
                # アラート条件をチェック
                portfolio_alerts = []
                
                # 高ボラティリティ
                if metrics.volatility > 30:
                    portfolio_alerts.append({
                        "type": "HIGH_VOLATILITY",
                        "severity": "WARNING",
                        "message": f"Volatility is {metrics.volatility:.1f}%",
                    })
                
                # 低シャープレシオ
                if metrics.sharpe_ratio < 0:
                    portfolio_alerts.append({
                        "type": "NEGATIVE_SHARPE",
                        "severity": "WARNING",
                        "message": f"Negative Sharpe ratio: {metrics.sharpe_ratio:.2f}",
                    })
                
                # 大きなドローダウン
                if metrics.max_drawdown > 20:
                    portfolio_alerts.append({
                        "type": "LARGE_DRAWDOWN",
                        "severity": "CRITICAL",
                        "message": f"Max drawdown is {metrics.max_drawdown:.1f}%",
                    })
                
                # ポートフォリオのアラートを追加
                if portfolio_alerts:
                    alerts.append({
                        "portfolio_id": str(portfolio.id),
                        "portfolio_name": portfolio.name,
                        "timestamp": datetime.now().isoformat(),
                        "alerts": portfolio_alerts,
                    })
                    
            except Exception as e:
                # エラーの場合はスキップ
                continue
        
        return alerts