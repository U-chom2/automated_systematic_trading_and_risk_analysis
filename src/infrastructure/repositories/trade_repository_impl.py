"""取引リポジトリ実装"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.entities.trade import Trade, TradeType, OrderType, OrderStatus
from ...domain.repositories.trade_repository import TradeRepository
from ..database.models import TradeModel
from ..database.session import get_session_manager


class TradeRepositoryImpl(TradeRepository):
    """取引リポジトリの実装"""
    
    def __init__(self):
        """初期化"""
        self._session_manager = get_session_manager()
    
    def _to_entity(self, model: TradeModel) -> Trade:
        """モデルをエンティティに変換"""
        trade = Trade(
            portfolio_id=model.portfolio_id,
            stock_id=model.stock_id,
            ticker=model.ticker,
            trade_type=TradeType[model.trade_type],
            order_type=OrderType[model.order_type],
            quantity=model.quantity,
            order_price=model.order_price,
            signal_id=model.signal_id,
        )
        trade.id = model.id
        trade.executed_price = model.executed_price
        trade.commission = model.commission
        trade.status = OrderStatus[model.status]
        trade.execution_id = model.execution_id
        trade.created_at = model.created_at
        trade.submitted_at = model.submitted_at
        trade.executed_at = model.executed_at
        trade.cancelled_at = model.cancelled_at
        return trade
    
    def _to_model(self, entity: Trade) -> TradeModel:
        """エンティティをモデルに変換"""
        return TradeModel(
            id=entity.id,
            portfolio_id=entity.portfolio_id,
            stock_id=entity.stock_id,
            ticker=entity.ticker,
            trade_type=entity.trade_type.value,
            order_type=entity.order_type.value,
            quantity=entity.quantity,
            order_price=entity.order_price,
            executed_price=entity.executed_price,
            commission=entity.commission,
            status=entity.status.value,
            signal_id=entity.signal_id,
            execution_id=entity.execution_id,
            created_at=entity.created_at,
            submitted_at=entity.submitted_at,
            executed_at=entity.executed_at,
            cancelled_at=entity.cancelled_at,
        )
    
    async def find_by_id(self, trade_id: UUID) -> Optional[Trade]:
        """IDで取引を検索"""
        async with self._session_manager.session() as session:
            result = await session.get(TradeModel, trade_id)
            return self._to_entity(result) if result else None
    
    async def find_by_portfolio(
        self,
        portfolio_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trade]:
        """ポートフォリオの取引を取得"""
        async with self._session_manager.session() as session:
            stmt = select(TradeModel).where(
                TradeModel.portfolio_id == portfolio_id
            ).order_by(TradeModel.created_at.desc()).limit(limit).offset(offset)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_by_ticker(
        self,
        ticker: str,
        portfolio_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trade]:
        """ティッカーで取引を検索"""
        async with self._session_manager.session() as session:
            conditions = [TradeModel.ticker == ticker]
            if portfolio_id:
                conditions.append(TradeModel.portfolio_id == portfolio_id)
            
            stmt = select(TradeModel).where(
                and_(*conditions)
            ).order_by(TradeModel.created_at.desc()).limit(limit).offset(offset)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_by_status(
        self,
        status: OrderStatus,
        portfolio_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trade]:
        """ステータスで取引を検索"""
        async with self._session_manager.session() as session:
            conditions = [TradeModel.status == status.value]
            if portfolio_id:
                conditions.append(TradeModel.portfolio_id == portfolio_id)
            
            stmt = select(TradeModel).where(
                and_(*conditions)
            ).order_by(TradeModel.created_at.desc()).limit(limit).offset(offset)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_pending_trades(
        self,
        portfolio_id: Optional[UUID] = None,
    ) -> List[Trade]:
        """待機中の取引を取得"""
        async with self._session_manager.session() as session:
            pending_statuses = [
                OrderStatus.PENDING.value,
                OrderStatus.SUBMITTED.value,
                OrderStatus.PARTIAL.value,
            ]
            
            conditions = [TradeModel.status.in_(pending_statuses)]
            if portfolio_id:
                conditions.append(TradeModel.portfolio_id == portfolio_id)
            
            stmt = select(TradeModel).where(
                and_(*conditions)
            ).order_by(TradeModel.created_at)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        portfolio_id: Optional[UUID] = None,
        trade_type: Optional[TradeType] = None,
    ) -> List[Trade]:
        """日付範囲で取引を検索"""
        async with self._session_manager.session() as session:
            conditions = [
                TradeModel.created_at >= start_date,
                TradeModel.created_at <= end_date,
            ]
            
            if portfolio_id:
                conditions.append(TradeModel.portfolio_id == portfolio_id)
            
            if trade_type:
                conditions.append(TradeModel.trade_type == trade_type.value)
            
            stmt = select(TradeModel).where(
                and_(*conditions)
            ).order_by(TradeModel.created_at)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_by_signal(self, signal_id: UUID) -> List[Trade]:
        """シグナルIDで取引を検索"""
        async with self._session_manager.session() as session:
            stmt = select(TradeModel).where(
                TradeModel.signal_id == signal_id
            ).order_by(TradeModel.created_at)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def save(self, trade: Trade) -> Trade:
        """取引を保存"""
        async with self._session_manager.session() as session:
            model = self._to_model(trade)
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_entity(model)
    
    async def save_all(self, trades: List[Trade]) -> List[Trade]:
        """複数の取引を一括保存"""
        async with self._session_manager.session() as session:
            models = [self._to_model(trade) for trade in trades]
            session.add_all(models)
            await session.commit()
            
            for model in models:
                await session.refresh(model)
            
            return [self._to_entity(model) for model in models]
    
    async def update(self, trade: Trade) -> Trade:
        """取引を更新"""
        async with self._session_manager.session() as session:
            model = await session.get(TradeModel, trade.id)
            if not model:
                raise ValueError(f"Trade not found: {trade.id}")
            
            model.status = trade.status.value
            model.executed_price = trade.executed_price
            model.commission = trade.commission
            model.execution_id = trade.execution_id
            model.submitted_at = trade.submitted_at
            model.executed_at = trade.executed_at
            model.cancelled_at = trade.cancelled_at
            
            await session.commit()
            await session.refresh(model)
            return self._to_entity(model)
    
    async def update_status(
        self,
        trade_id: UUID,
        status: OrderStatus,
    ) -> bool:
        """取引ステータスを更新"""
        async with self._session_manager.session() as session:
            model = await session.get(TradeModel, trade_id)
            if not model:
                return False
            
            model.status = status.value
            
            # ステータスに応じてタイムスタンプを更新
            if status == OrderStatus.SUBMITTED:
                model.submitted_at = datetime.now()
            elif status == OrderStatus.FILLED:
                model.executed_at = datetime.now()
            elif status == OrderStatus.CANCELLED:
                model.cancelled_at = datetime.now()
            
            await session.commit()
            return True
    
    async def cancel(self, trade_id: UUID) -> bool:
        """取引をキャンセル"""
        return await self.update_status(trade_id, OrderStatus.CANCELLED)
    
    async def delete(self, trade_id: UUID) -> bool:
        """取引を削除"""
        async with self._session_manager.session() as session:
            model = await session.get(TradeModel, trade_id)
            if not model:
                return False
            
            await session.delete(model)
            await session.commit()
            return True
    
    async def get_trade_statistics(
        self,
        portfolio_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """取引統計を取得"""
        async with self._session_manager.session() as session:
            conditions = [TradeModel.portfolio_id == portfolio_id]
            
            if start_date:
                conditions.append(TradeModel.executed_at >= start_date)
            if end_date:
                conditions.append(TradeModel.executed_at <= end_date)
            
            # 統計クエリ
            stmt = select(
                func.count(TradeModel.id).label("total_trades"),
                func.count(TradeModel.id).filter(
                    TradeModel.status == OrderStatus.FILLED.value
                ).label("executed_trades"),
                func.sum(TradeModel.commission).label("total_commission"),
            ).where(and_(*conditions))
            
            result = await session.execute(stmt)
            stats = result.one()
            
            return {
                "total_trades": stats.total_trades or 0,
                "executed_trades": stats.executed_trades or 0,
                "total_commission": float(stats.total_commission or 0),
            }
    
    async def get_win_rate(
        self,
        portfolio_id: UUID,
        ticker: Optional[str] = None,
    ) -> float:
        """勝率を取得"""
        # 簡易実装（実際には損益計算が必要）
        trades = await self.find_by_portfolio(portfolio_id, limit=1000)
        
        if ticker:
            trades = [t for t in trades if t.ticker == ticker]
        
        if not trades:
            return 0.0
        
        # 売却取引のみを対象（簡易版）
        sell_trades = [t for t in trades if t.is_sell and t.is_executed]
        if not sell_trades:
            return 0.0
        
        # 仮の勝率計算
        return 0.55  # 55%の勝率（仮）
    
    async def get_average_return(
        self,
        portfolio_id: UUID,
        ticker: Optional[str] = None,
    ) -> float:
        """平均リターンを取得"""
        # 簡易実装
        return 0.05  # 5%の平均リターン（仮）