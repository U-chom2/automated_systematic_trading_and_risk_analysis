"""株式リポジトリ実装"""
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.entities.stock import Stock
from ...domain.repositories.stock_repository import StockRepository
from ..database.models import StockModel
from ..database.session import get_session_manager


class StockRepositoryImpl(StockRepository):
    """株式リポジトリの実装"""
    
    def __init__(self):
        """初期化"""
        self._session_manager = get_session_manager()
    
    def _to_entity(self, model: StockModel) -> Stock:
        """モデルをエンティティに変換"""
        return Stock(
            id=model.id,
            ticker=model.ticker,
            company_name=model.company_name,
            company_name_jp=model.company_name_jp,
            exchange=model.exchange,
            sector=model.sector,
            industry=model.industry,
            market_cap=model.market_cap,
            currency=model.currency,
            is_active=model.is_active,
        )
    
    def _to_model(self, entity: Stock) -> StockModel:
        """エンティティをモデルに変換"""
        return StockModel(
            id=entity.id,
            ticker=entity.ticker,
            company_name=entity.company_name,
            company_name_jp=entity.company_name_jp,
            exchange=entity.exchange,
            sector=entity.sector,
            industry=entity.industry,
            market_cap=entity.market_cap,
            currency=entity.currency,
            is_active=entity.is_active,
        )
    
    async def find_by_id(self, stock_id: UUID) -> Optional[Stock]:
        """IDで株式を検索"""
        async with self._session_manager.session() as session:
            result = await session.get(StockModel, stock_id)
            return self._to_entity(result) if result else None
    
    async def find_by_ticker(self, ticker: str) -> Optional[Stock]:
        """ティッカーシンボルで株式を検索"""
        async with self._session_manager.session() as session:
            stmt = select(StockModel).where(StockModel.ticker == ticker)
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            return self._to_entity(model) if model else None
    
    async def find_by_exchange(self, exchange: str) -> List[Stock]:
        """取引所で株式を検索"""
        async with self._session_manager.session() as session:
            stmt = select(StockModel).where(
                and_(
                    StockModel.exchange == exchange,
                    StockModel.is_active == True
                )
            ).order_by(StockModel.ticker)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_by_sector(self, sector: str) -> List[Stock]:
        """セクターで株式を検索"""
        async with self._session_manager.session() as session:
            stmt = select(StockModel).where(
                and_(
                    StockModel.sector == sector,
                    StockModel.is_active == True
                )
            ).order_by(StockModel.ticker)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Stock]:
        """すべての株式を取得"""
        async with self._session_manager.session() as session:
            stmt = select(StockModel).where(
                StockModel.is_active == True
            ).order_by(StockModel.ticker).limit(limit).offset(offset)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def search(self, query: str) -> List[Stock]:
        """株式を検索"""
        async with self._session_manager.session() as session:
            search_term = f"%{query}%"
            stmt = select(StockModel).where(
                and_(
                    or_(
                        StockModel.ticker.ilike(search_term),
                        StockModel.company_name.ilike(search_term),
                        StockModel.company_name_jp.ilike(search_term),
                    ),
                    StockModel.is_active == True
                )
            ).order_by(StockModel.ticker).limit(50)
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._to_entity(model) for model in models]
    
    async def save(self, stock: Stock) -> Stock:
        """株式を保存"""
        async with self._session_manager.session() as session:
            model = self._to_model(stock)
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_entity(model)
    
    async def save_all(self, stocks: List[Stock]) -> List[Stock]:
        """複数の株式を一括保存"""
        async with self._session_manager.session() as session:
            models = [self._to_model(stock) for stock in stocks]
            session.add_all(models)
            await session.commit()
            
            # リフレッシュ
            for model in models:
                await session.refresh(model)
            
            return [self._to_entity(model) for model in models]
    
    async def update(self, stock: Stock) -> Stock:
        """株式を更新"""
        async with self._session_manager.session() as session:
            # 既存のモデルを取得
            existing = await session.get(StockModel, stock.id)
            if not existing:
                raise ValueError(f"Stock not found: {stock.id}")
            
            # フィールドを更新
            existing.ticker = stock.ticker
            existing.company_name = stock.company_name
            existing.company_name_jp = stock.company_name_jp
            existing.exchange = stock.exchange
            existing.sector = stock.sector
            existing.industry = stock.industry
            existing.market_cap = stock.market_cap
            existing.currency = stock.currency
            existing.is_active = stock.is_active
            
            await session.commit()
            await session.refresh(existing)
            return self._to_entity(existing)
    
    async def delete(self, stock_id: UUID) -> bool:
        """株式を削除（論理削除）"""
        async with self._session_manager.session() as session:
            model = await session.get(StockModel, stock_id)
            if not model:
                return False
            
            model.is_active = False
            await session.commit()
            return True
    
    async def exists(self, ticker: str) -> bool:
        """株式が存在するか確認"""
        async with self._session_manager.session() as session:
            stmt = select(StockModel.id).where(
                and_(
                    StockModel.ticker == ticker,
                    StockModel.is_active == True
                )
            ).limit(1)
            
            result = await session.execute(stmt)
            return result.scalar_one_or_none() is not None