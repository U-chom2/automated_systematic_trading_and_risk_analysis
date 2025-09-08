"""ポートフォリオ管理ユースケース"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from ...domain.entities.portfolio import Portfolio
from ...domain.entities.position import Position
from ...domain.repositories.portfolio_repository import PortfolioRepository
from ...domain.repositories.market_data_repository import MarketDataRepository
from ..dto.portfolio_dto import PortfolioDTO, PositionDTO, CreatePortfolioDTO, UpdatePortfolioDTO


@dataclass
class CreatePortfolioUseCase:
    """ポートフォリオ作成ユースケース"""
    
    portfolio_repository: PortfolioRepository
    
    async def execute(self, dto: CreatePortfolioDTO) -> PortfolioDTO:
        """ポートフォリオを作成
        
        Args:
            dto: ポートフォリオ作成DTO
        
        Returns:
            作成されたポートフォリオDTO
        """
        # ドメインエンティティを作成
        portfolio = Portfolio(
            name=dto.name,
            description=dto.description,
            initial_capital=dto.initial_capital,
            currency=dto.currency,
            strategy_type=dto.strategy_type,
        )
        
        # リポジトリで保存
        saved_portfolio = await self.portfolio_repository.save(portfolio)
        
        # DTOに変換して返却
        return PortfolioDTO.from_entity(saved_portfolio)


@dataclass
class UpdatePortfolioUseCase:
    """ポートフォリオ更新ユースケース"""
    
    portfolio_repository: PortfolioRepository
    
    async def execute(self, portfolio_id: UUID, dto: UpdatePortfolioDTO) -> PortfolioDTO:
        """ポートフォリオを更新
        
        Args:
            portfolio_id: ポートフォリオID
            dto: ポートフォリオ更新DTO
        
        Returns:
            更新されたポートフォリオDTO
        """
        # 既存のポートフォリオを取得
        portfolio = await self.portfolio_repository.find_by_id(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
        
        # 更新可能なフィールドを更新
        if dto.name is not None:
            portfolio.name = dto.name
        if dto.description is not None:
            portfolio.description = dto.description
        if dto.is_active is not None:
            if dto.is_active:
                portfolio.activate()
            else:
                portfolio.deactivate()
        
        # リポジトリで更新
        updated_portfolio = await self.portfolio_repository.update(portfolio)
        
        # DTOに変換して返却
        return PortfolioDTO.from_entity(updated_portfolio)


@dataclass
class GetPortfolioUseCase:
    """ポートフォリオ取得ユースケース"""
    
    portfolio_repository: PortfolioRepository
    market_data_repository: MarketDataRepository
    
    async def execute(
        self,
        portfolio_id: UUID,
        include_positions: bool = True,
        update_prices: bool = False,
    ) -> Optional[PortfolioDTO]:
        """ポートフォリオを取得
        
        Args:
            portfolio_id: ポートフォリオID
            include_positions: ポジション情報を含めるか
            update_prices: 最新価格で更新するか
        
        Returns:
            ポートフォリオDTO（見つからない場合はNone）
        """
        # ポートフォリオを取得
        portfolio = await self.portfolio_repository.find_by_id(portfolio_id)
        if not portfolio:
            return None
        
        # ポジション情報を取得
        if include_positions:
            positions = await self.portfolio_repository.find_positions(portfolio_id)
            for position in positions:
                portfolio.positions[position.ticker] = position
        
        # 最新価格で更新
        if update_prices and portfolio.positions:
            tickers = list(portfolio.positions.keys())
            latest_prices = await self.market_data_repository.get_latest_prices(tickers)
            
            for ticker, price in latest_prices.items():
                if ticker in portfolio.positions:
                    portfolio.positions[ticker].update_price(price.value)
        
        # DTOに変換して返却
        return PortfolioDTO.from_entity(portfolio)


@dataclass
class ListPortfoliosUseCase:
    """ポートフォリオ一覧取得ユースケース"""
    
    portfolio_repository: PortfolioRepository
    
    async def execute(
        self,
        active_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[PortfolioDTO]:
        """ポートフォリオ一覧を取得
        
        Args:
            active_only: アクティブなもののみ
            limit: 取得件数上限
            offset: オフセット
        
        Returns:
            ポートフォリオDTOのリスト
        """
        if active_only:
            portfolios = await self.portfolio_repository.find_active()
        else:
            portfolios = await self.portfolio_repository.find_all(limit, offset)
        
        # DTOに変換して返却
        return [PortfolioDTO.from_entity(p) for p in portfolios]


@dataclass
class DeletePortfolioUseCase:
    """ポートフォリオ削除ユースケース"""
    
    portfolio_repository: PortfolioRepository
    
    async def execute(self, portfolio_id: UUID, archive: bool = True) -> bool:
        """ポートフォリオを削除
        
        Args:
            portfolio_id: ポートフォリオID
            archive: アーカイブするか（Falseの場合は完全削除）
        
        Returns:
            削除成功の可否
        """
        # ポートフォリオの存在確認
        portfolio = await self.portfolio_repository.find_by_id(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
        
        # オープンポジションの確認
        positions = await self.portfolio_repository.find_positions(portfolio_id)
        open_positions = [p for p in positions if not p.is_closed]
        
        if open_positions:
            raise ValueError(
                f"Cannot delete portfolio with {len(open_positions)} open positions"
            )
        
        # 削除またはアーカイブ
        if archive:
            return await self.portfolio_repository.archive(portfolio_id)
        else:
            return await self.portfolio_repository.delete(portfolio_id)


@dataclass
class GetPortfolioPerformanceUseCase:
    """ポートフォリオパフォーマンス取得ユースケース"""
    
    portfolio_repository: PortfolioRepository
    market_data_repository: MarketDataRepository
    
    async def execute(
        self,
        portfolio_id: UUID,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """ポートフォリオのパフォーマンスを取得
        
        Args:
            portfolio_id: ポートフォリオID
            start_date: 開始日
            end_date: 終了日
        
        Returns:
            パフォーマンスデータ
        """
        # ポートフォリオを取得
        portfolio = await self.portfolio_repository.find_by_id(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
        
        # パフォーマンス履歴を取得
        performance_history = await self.portfolio_repository.get_performance_history(
            portfolio_id, start_date, end_date
        )
        
        if not performance_history:
            return {
                "portfolio_id": str(portfolio_id),
                "portfolio_name": portfolio.name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }
        
        # パフォーマンス指標を計算
        initial_value = performance_history[0].get("total_value", portfolio.initial_capital)
        final_value = performance_history[-1].get("total_value", portfolio.total_value)
        
        # トータルリターン
        total_return = float((final_value - initial_value) / initial_value) if initial_value > 0 else 0.0
        
        # 年率換算リターン
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (((1 + total_return) ** (1 / years)) - 1) if years > 0 else total_return
        
        # 日次リターンを計算
        daily_returns = []
        for i in range(1, len(performance_history)):
            prev_value = performance_history[i-1].get("total_value", 0)
            curr_value = performance_history[i].get("total_value", 0)
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
        
        # ボラティリティ（年率換算）
        if daily_returns:
            variance = sum((r - sum(daily_returns)/len(daily_returns))**2 for r in daily_returns) / len(daily_returns)
            volatility = (variance ** 0.5) * (252 ** 0.5)  # 年率換算
        else:
            volatility = 0.0
        
        # シャープレシオ（リスクフリーレート0.1%と仮定）
        risk_free_rate = 0.001
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # 最大ドローダウン
        peak = initial_value
        max_drawdown = 0.0
        for record in performance_history:
            value = record.get("total_value", 0)
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        # 勝率（簡易計算）
        winning_days = sum(1 for r in daily_returns if r > 0)
        win_rate = winning_days / len(daily_returns) if daily_returns else 0.0
        
        return {
            "portfolio_id": str(portfolio_id),
            "portfolio_name": portfolio.name,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_value": float(initial_value),
            "final_value": float(final_value),
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": len(performance_history),
        }