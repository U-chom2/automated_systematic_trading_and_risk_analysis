"""ポートフォリオリポジトリ実装"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.entities.portfolio import Portfolio
from ...domain.entities.position import Position
from ...domain.repositories.portfolio_repository import PortfolioRepository
from ..database.models import PortfolioModel, PositionModel, PerformanceModel
from ..database.session import get_session_manager


class PortfolioRepositoryImpl(PortfolioRepository):
    """ポートフォリオリポジトリの実装"""
    
    def __init__(self):
        """初期化"""
        self._session_manager = get_session_manager()
    
    def _to_entity(self, model: PortfolioModel) -> Portfolio:
        """モデルをエンティティに変換"""
        portfolio = Portfolio(
            name=model.name,
            initial_capital=model.initial_capital,
            description=model.description,
            currency=model.currency,
            strategy_type=model.strategy_type,
        )
        portfolio.id = model.id
        portfolio.current_capital = model.current_capital
        portfolio.is_active = model.is_active
        portfolio.created_at = model.created_at
        portfolio.updated_at = model.updated_at
        return portfolio
    
    def _position_to_entity(self, model: PositionModel) -> Position:
        """ポジションモデルをエンティティに変換"""
        return Position(
            id=model.id,
            portfolio_id=model.portfolio_id,
            stock_id=model.stock_id,
            ticker=model.ticker,
            quantity=model.quantity,
            average_cost=model.average_cost,
            current_price=model.current_price or model.average_cost,
            opened_at=model.opened_at,
            closed_at=model.closed_at,
        )
    
    async def find_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]:
        """IDでポートフォリオを検索"""
        async with self._session_manager.session() as session:
            result = await session.get(PortfolioModel, portfolio_id)
            return self._to_entity(result) if result else None
    
    async def find_by_name(self, name: str) -> Optional[Portfolio]:
        """名前でポートフォリオを検索"""
        async with self._session_manager.session() as session:
            stmt = select(PortfolioModel).where(PortfolioModel.name == name)
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            return self._to_entity(model) if model else None
    
    async def find_active(self) -> List[Portfolio]:
        """アクティブなポートフォリオを取得"""
        async with self._session_manager.session() as session:
            stmt = select(PortfolioModel).where(
                PortfolioModel.is_active == True
            ).order_by(PortfolioModel.name)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Portfolio]:
        """すべてのポートフォリオを取得"""
        async with self._session_manager.session() as session:
            stmt = select(PortfolioModel).order_by(
                PortfolioModel.created_at.desc()
            ).limit(limit).offset(offset)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def save(self, portfolio: Portfolio) -> Portfolio:
        """ポートフォリオを保存"""
        async with self._session_manager.session() as session:
            model = PortfolioModel(
                id=portfolio.id,
                name=portfolio.name,
                description=portfolio.description,
                initial_capital=portfolio.initial_capital,
                current_capital=portfolio.current_capital,
                currency=portfolio.currency,
                strategy_type=portfolio.strategy_type,
                is_active=portfolio.is_active,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_entity(model)
    
    async def update(self, portfolio: Portfolio) -> Portfolio:
        """ポートフォリオを更新"""
        async with self._session_manager.session() as session:
            model = await session.get(PortfolioModel, portfolio.id)
            if not model:
                raise ValueError(f"Portfolio not found: {portfolio.id}")
            
            model.name = portfolio.name
            model.description = portfolio.description
            model.current_capital = portfolio.current_capital
            model.is_active = portfolio.is_active
            model.updated_at = datetime.now()
            
            await session.commit()
            await session.refresh(model)
            return self._to_entity(model)
    
    async def delete(self, portfolio_id: UUID) -> bool:
        """ポートフォリオを削除"""
        async with self._session_manager.session() as session:
            model = await session.get(PortfolioModel, portfolio_id)
            if not model:
                return False
            
            await session.delete(model)
            await session.commit()
            return True
    
    async def archive(self, portfolio_id: UUID) -> bool:
        """ポートフォリオをアーカイブ"""
        async with self._session_manager.session() as session:
            model = await session.get(PortfolioModel, portfolio_id)
            if not model:
                return False
            
            model.is_active = False
            model.updated_at = datetime.now()
            await session.commit()
            return True
    
    async def find_positions(self, portfolio_id: UUID) -> List[Position]:
        """ポートフォリオのポジションを取得"""
        async with self._session_manager.session() as session:
            stmt = select(PositionModel).where(
                and_(
                    PositionModel.portfolio_id == portfolio_id,
                    PositionModel.closed_at.is_(None)
                )
            ).order_by(PositionModel.ticker)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._position_to_entity(model) for model in models]
    
    async def find_position(
        self, portfolio_id: UUID, ticker: str
    ) -> Optional[Position]:
        """特定銘柄のポジションを取得"""
        async with self._session_manager.session() as session:
            stmt = select(PositionModel).where(
                and_(
                    PositionModel.portfolio_id == portfolio_id,
                    PositionModel.ticker == ticker,
                    PositionModel.closed_at.is_(None)
                )
            ).limit(1)
            
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            return self._position_to_entity(model) if model else None
    
    async def save_position(
        self, portfolio_id: UUID, position: Position
    ) -> Position:
        """ポジションを保存"""
        async with self._session_manager.session() as session:
            model = PositionModel(
                id=position.id,
                portfolio_id=portfolio_id,
                stock_id=position.stock_id,
                ticker=position.ticker,
                quantity=position.quantity,
                average_cost=position.average_cost,
                current_price=position.current_price,
                opened_at=position.opened_at,
                closed_at=position.closed_at,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._position_to_entity(model)
    
    async def update_position(
        self, portfolio_id: UUID, position: Position
    ) -> Position:
        """ポジションを更新"""
        async with self._session_manager.session() as session:
            model = await session.get(PositionModel, position.id)
            if not model or model.portfolio_id != portfolio_id:
                raise ValueError(f"Position not found: {position.id}")
            
            model.quantity = position.quantity
            model.average_cost = position.average_cost
            model.current_price = position.current_price
            model.closed_at = position.closed_at
            
            await session.commit()
            await session.refresh(model)
            return self._position_to_entity(model)
    
    async def close_position(
        self, portfolio_id: UUID, position_id: UUID
    ) -> bool:
        """ポジションをクローズ"""
        async with self._session_manager.session() as session:
            model = await session.get(PositionModel, position_id)
            if not model or model.portfolio_id != portfolio_id:
                return False
            
            model.closed_at = datetime.now()
            await session.commit()
            return True
    
    async def get_performance_history(
        self,
        portfolio_id: UUID,
        start_date: datetime,
        end_date: datetime,
    ) -> List[dict]:
        """パフォーマンス履歴を取得"""
        async with self._session_manager.session() as session:
            stmt = select(PerformanceModel).where(
                and_(
                    PerformanceModel.portfolio_id == portfolio_id,
                    PerformanceModel.date >= start_date.date(),
                    PerformanceModel.date <= end_date.date(),
                )
            ).order_by(PerformanceModel.date)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            
            return [
                {
                    "date": model.date.isoformat(),
                    "total_value": float(model.total_value),
                    "cash_balance": float(model.cash_balance),
                    "positions_value": float(model.positions_value),
                    "daily_return": float(model.daily_return) if model.daily_return else None,
                    "cumulative_return": float(model.cumulative_return) if model.cumulative_return else None,
                    "realized_pnl": float(model.realized_pnl) if model.realized_pnl else None,
                    "unrealized_pnl": float(model.unrealized_pnl) if model.unrealized_pnl else None,
                    "sharpe_ratio": float(model.sharpe_ratio) if model.sharpe_ratio else None,
                    "max_drawdown": float(model.max_drawdown) if model.max_drawdown else None,
                }
                for model in models
            ]
    
    async def save_performance_snapshot(
        self,
        portfolio_id: UUID,
        snapshot: dict,
    ) -> bool:
        """パフォーマンススナップショットを保存"""
        async with self._session_manager.session() as session:
            model = PerformanceModel(
                portfolio_id=portfolio_id,
                date=snapshot["date"],
                total_value=snapshot["total_value"],
                cash_balance=snapshot["cash_balance"],
                positions_value=snapshot["positions_value"],
                daily_return=snapshot.get("daily_return"),
                cumulative_return=snapshot.get("cumulative_return"),
                realized_pnl=snapshot.get("realized_pnl"),
                unrealized_pnl=snapshot.get("unrealized_pnl"),
                sharpe_ratio=snapshot.get("sharpe_ratio"),
                max_drawdown=snapshot.get("max_drawdown"),
            )
            session.add(model)
            await session.commit()
            return True