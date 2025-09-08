"""シグナルリポジトリ実装"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.services.signal_generator import Signal, SignalType, SignalDirection
from ...domain.entities.stock import Stock
from ...domain.repositories.signal_repository import SignalRepository
from ...domain.value_objects.price import Price
from ...domain.value_objects.percentage import Percentage
from ..database.models import SignalModel, StockModel
from ..database.session import get_session_manager


class SignalRepositoryImpl(SignalRepository):
    """シグナルリポジトリの実装"""
    
    def __init__(self):
        """初期化"""
        self._session_manager = get_session_manager()
    
    def _to_entity(self, model: SignalModel, stock: Optional[Stock] = None) -> Signal:
        """モデルをエンティティに変換"""
        if not stock:
            # 簡易的なStock作成（実際には別途取得が必要）
            stock = Stock(
                id=model.stock_id,
                ticker=model.ticker,
                company_name="Unknown",
                exchange="TSE",
            )
        
        signal = Signal(
            stock=stock,
            signal_type=SignalType[model.signal_type],
            direction=SignalDirection[model.direction],
            strength=model.strength,
            confidence=Percentage(model.confidence),
            target_price=Price(model.target_price) if model.target_price else None,
            stop_loss=Price(model.stop_loss) if model.stop_loss else None,
            time_horizon=model.time_horizon,
            created_at=model.created_at,
            expires_at=model.expires_at,
            metadata=model.metadata or {},
        )
        signal.id = model.id
        return signal
    
    def _to_model(self, entity: Signal) -> SignalModel:
        """エンティティをモデルに変換"""
        return SignalModel(
            id=entity.id,
            stock_id=entity.stock.id,
            ticker=entity.stock.ticker,
            signal_type=entity.signal_type.value,
            direction=entity.direction.value,
            strength=entity.strength,
            confidence=entity.confidence.value,
            target_price=entity.target_price.value if entity.target_price else None,
            stop_loss=entity.stop_loss.value if entity.stop_loss else None,
            time_horizon=entity.time_horizon,
            metadata=entity.metadata,
            created_at=entity.created_at,
            expires_at=entity.expires_at,
        )
    
    async def find_by_id(self, signal_id: UUID) -> Optional[Signal]:
        """IDでシグナルを検索"""
        async with self._session_manager.session() as session:
            result = await session.get(SignalModel, signal_id)
            return self._to_entity(result) if result else None
    
    async def find_active(
        self,
        ticker: Optional[str] = None,
        signal_type: Optional[SignalType] = None,
    ) -> List[Signal]:
        """アクティブなシグナルを取得"""
        async with self._session_manager.session() as session:
            conditions = [
                or_(
                    SignalModel.expires_at.is_(None),
                    SignalModel.expires_at > datetime.now()
                )
            ]
            
            if ticker:
                conditions.append(SignalModel.ticker == ticker)
            
            if signal_type:
                conditions.append(SignalModel.signal_type == signal_type.value)
            
            stmt = select(SignalModel).where(
                and_(*conditions)
            ).order_by(SignalModel.created_at.desc())
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_by_ticker(
        self,
        ticker: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Signal]:
        """ティッカーでシグナルを検索"""
        async with self._session_manager.session() as session:
            stmt = select(SignalModel).where(
                SignalModel.ticker == ticker
            ).order_by(SignalModel.created_at.desc()).limit(limit).offset(offset)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_by_type(
        self,
        signal_type: SignalType,
        active_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Signal]:
        """タイプでシグナルを検索"""
        async with self._session_manager.session() as session:
            conditions = [SignalModel.signal_type == signal_type.value]
            
            if active_only:
                conditions.append(
                    or_(
                        SignalModel.expires_at.is_(None),
                        SignalModel.expires_at > datetime.now()
                    )
                )
            
            stmt = select(SignalModel).where(
                and_(*conditions)
            ).order_by(SignalModel.created_at.desc()).limit(limit).offset(offset)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_by_direction(
        self,
        direction: SignalDirection,
        active_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Signal]:
        """方向でシグナルを検索"""
        async with self._session_manager.session() as session:
            conditions = [SignalModel.direction == direction.value]
            
            if active_only:
                conditions.append(
                    or_(
                        SignalModel.expires_at.is_(None),
                        SignalModel.expires_at > datetime.now()
                    )
                )
            
            stmt = select(SignalModel).where(
                and_(*conditions)
            ).order_by(SignalModel.created_at.desc()).limit(limit).offset(offset)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        ticker: Optional[str] = None,
        signal_type: Optional[SignalType] = None,
    ) -> List[Signal]:
        """日付範囲でシグナルを検索"""
        async with self._session_manager.session() as session:
            conditions = [
                SignalModel.created_at >= start_date,
                SignalModel.created_at <= end_date,
            ]
            
            if ticker:
                conditions.append(SignalModel.ticker == ticker)
            
            if signal_type:
                conditions.append(SignalModel.signal_type == signal_type.value)
            
            stmt = select(SignalModel).where(
                and_(*conditions)
            ).order_by(SignalModel.created_at)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_high_confidence(
        self,
        min_confidence: float = 0.7,
        active_only: bool = True,
    ) -> List[Signal]:
        """高信頼度のシグナルを取得"""
        async with self._session_manager.session() as session:
            conditions = [SignalModel.confidence >= min_confidence * 100]
            
            if active_only:
                conditions.append(
                    or_(
                        SignalModel.expires_at.is_(None),
                        SignalModel.expires_at > datetime.now()
                    )
                )
            
            stmt = select(SignalModel).where(
                and_(*conditions)
            ).order_by(SignalModel.confidence.desc(), SignalModel.created_at.desc())
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def save(self, signal: Signal) -> Signal:
        """シグナルを保存"""
        async with self._session_manager.session() as session:
            model = self._to_model(signal)
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_entity(model, signal.stock)
    
    async def save_all(self, signals: List[Signal]) -> List[Signal]:
        """複数のシグナルを一括保存"""
        async with self._session_manager.session() as session:
            models = [self._to_model(signal) for signal in signals]
            session.add_all(models)
            await session.commit()
            
            for model in models:
                await session.refresh(model)
            
            return [
                self._to_entity(model, signal.stock)
                for model, signal in zip(models, signals)
            ]
    
    async def update(self, signal: Signal) -> Signal:
        """シグナルを更新"""
        async with self._session_manager.session() as session:
            model = await session.get(SignalModel, signal.id)
            if not model:
                raise ValueError(f"Signal not found: {signal.id}")
            
            model.strength = signal.strength
            model.confidence = signal.confidence.value
            model.target_price = signal.target_price.value if signal.target_price else None
            model.stop_loss = signal.stop_loss.value if signal.stop_loss else None
            model.expires_at = signal.expires_at
            model.metadata = signal.metadata
            
            await session.commit()
            await session.refresh(model)
            return self._to_entity(model, signal.stock)
    
    async def expire(self, signal_id: UUID) -> bool:
        """シグナルを期限切れにする"""
        async with self._session_manager.session() as session:
            model = await session.get(SignalModel, signal_id)
            if not model:
                return False
            
            model.expires_at = datetime.now()
            await session.commit()
            return True
    
    async def delete(self, signal_id: UUID) -> bool:
        """シグナルを削除"""
        async with self._session_manager.session() as session:
            model = await session.get(SignalModel, signal_id)
            if not model:
                return False
            
            await session.delete(model)
            await session.commit()
            return True
    
    async def delete_expired(self) -> int:
        """期限切れシグナルを削除"""
        async with self._session_manager.session() as session:
            stmt = select(SignalModel).where(
                and_(
                    SignalModel.expires_at.is_not(None),
                    SignalModel.expires_at < datetime.now()
                )
            )
            
            result = await session.execute(stmt)
            expired_models = result.scalars().all()
            
            count = len(expired_models)
            for model in expired_models:
                await session.delete(model)
            
            await session.commit()
            return count
    
    async def get_signal_statistics(
        self,
        ticker: Optional[str] = None,
        signal_type: Optional[SignalType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """シグナル統計を取得"""
        async with self._session_manager.session() as session:
            conditions = []
            
            if ticker:
                conditions.append(SignalModel.ticker == ticker)
            if signal_type:
                conditions.append(SignalModel.signal_type == signal_type.value)
            if start_date:
                conditions.append(SignalModel.created_at >= start_date)
            if end_date:
                conditions.append(SignalModel.created_at <= end_date)
            
            # 統計クエリ
            if conditions:
                stmt = select(
                    func.count(SignalModel.id).label("total_signals"),
                    func.avg(SignalModel.strength).label("avg_strength"),
                    func.avg(SignalModel.confidence).label("avg_confidence"),
                ).where(and_(*conditions))
            else:
                stmt = select(
                    func.count(SignalModel.id).label("total_signals"),
                    func.avg(SignalModel.strength).label("avg_strength"),
                    func.avg(SignalModel.confidence).label("avg_confidence"),
                )
            
            result = await session.execute(stmt)
            stats = result.one()
            
            return {
                "total_signals": stats.total_signals or 0,
                "average_strength": float(stats.avg_strength or 0),
                "average_confidence": float(stats.avg_confidence or 0),
            }
    
    async def get_accuracy(
        self,
        signal_type: Optional[SignalType] = None,
        ticker: Optional[str] = None,
    ) -> float:
        """シグナル精度を取得"""
        # 簡易実装（実際には結果との照合が必要）
        return 0.65  # 65%の精度（仮）