"""株式リポジトリインターフェース"""
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.stock import Stock


class StockRepository(ABC):
    """株式リポジトリの抽象インターフェース"""
    
    @abstractmethod
    async def find_by_id(self, stock_id: UUID) -> Optional[Stock]:
        """IDで株式を検索
        
        Args:
            stock_id: 株式ID
        
        Returns:
            株式エンティティ（見つからない場合はNone）
        """
        pass
    
    @abstractmethod
    async def find_by_ticker(self, ticker: str) -> Optional[Stock]:
        """ティッカーシンボルで株式を検索
        
        Args:
            ticker: ティッカーシンボル
        
        Returns:
            株式エンティティ（見つからない場合はNone）
        """
        pass
    
    @abstractmethod
    async def find_by_exchange(self, exchange: str) -> List[Stock]:
        """取引所で株式を検索
        
        Args:
            exchange: 取引所コード（TSE, NYSE, NASDAQ等）
        
        Returns:
            株式エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_by_sector(self, sector: str) -> List[Stock]:
        """セクターで株式を検索
        
        Args:
            sector: セクター名
        
        Returns:
            株式エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Stock]:
        """すべての株式を取得
        
        Args:
            limit: 取得件数上限
            offset: オフセット
        
        Returns:
            株式エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def search(self, query: str) -> List[Stock]:
        """株式を検索
        
        Args:
            query: 検索クエリ（企業名、ティッカー等）
        
        Returns:
            株式エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def save(self, stock: Stock) -> Stock:
        """株式を保存
        
        Args:
            stock: 株式エンティティ
        
        Returns:
            保存された株式エンティティ
        """
        pass
    
    @abstractmethod
    async def save_all(self, stocks: List[Stock]) -> List[Stock]:
        """複数の株式を一括保存
        
        Args:
            stocks: 株式エンティティのリスト
        
        Returns:
            保存された株式エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def update(self, stock: Stock) -> Stock:
        """株式を更新
        
        Args:
            stock: 株式エンティティ
        
        Returns:
            更新された株式エンティティ
        """
        pass
    
    @abstractmethod
    async def delete(self, stock_id: UUID) -> bool:
        """株式を削除
        
        Args:
            stock_id: 株式ID
        
        Returns:
            削除成功の可否
        """
        pass
    
    @abstractmethod
    async def exists(self, ticker: str) -> bool:
        """株式が存在するか確認
        
        Args:
            ticker: ティッカーシンボル
        
        Returns:
            存在する場合True
        """
        pass