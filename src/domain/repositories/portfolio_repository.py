"""ポートフォリオリポジトリインターフェース"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from ..entities.portfolio import Portfolio
from ..entities.position import Position


class PortfolioRepository(ABC):
    """ポートフォリオリポジトリの抽象インターフェース"""
    
    @abstractmethod
    async def find_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]:
        """IDでポートフォリオを検索
        
        Args:
            portfolio_id: ポートフォリオID
        
        Returns:
            ポートフォリオエンティティ（見つからない場合はNone）
        """
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Portfolio]:
        """名前でポートフォリオを検索
        
        Args:
            name: ポートフォリオ名
        
        Returns:
            ポートフォリオエンティティ（見つからない場合はNone）
        """
        pass
    
    @abstractmethod
    async def find_active(self) -> List[Portfolio]:
        """アクティブなポートフォリオを取得
        
        Returns:
            アクティブなポートフォリオのリスト
        """
        pass
    
    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Portfolio]:
        """すべてのポートフォリオを取得
        
        Args:
            limit: 取得件数上限
            offset: オフセット
        
        Returns:
            ポートフォリオエンティティのリスト
        """
        pass
    
    @abstractmethod
    async def save(self, portfolio: Portfolio) -> Portfolio:
        """ポートフォリオを保存
        
        Args:
            portfolio: ポートフォリオエンティティ
        
        Returns:
            保存されたポートフォリオエンティティ
        """
        pass
    
    @abstractmethod
    async def update(self, portfolio: Portfolio) -> Portfolio:
        """ポートフォリオを更新
        
        Args:
            portfolio: ポートフォリオエンティティ
        
        Returns:
            更新されたポートフォリオエンティティ
        """
        pass
    
    @abstractmethod
    async def delete(self, portfolio_id: UUID) -> bool:
        """ポートフォリオを削除
        
        Args:
            portfolio_id: ポートフォリオID
        
        Returns:
            削除成功の可否
        """
        pass
    
    @abstractmethod
    async def archive(self, portfolio_id: UUID) -> bool:
        """ポートフォリオをアーカイブ
        
        Args:
            portfolio_id: ポートフォリオID
        
        Returns:
            アーカイブ成功の可否
        """
        pass
    
    # ポジション関連のメソッド
    
    @abstractmethod
    async def find_positions(self, portfolio_id: UUID) -> List[Position]:
        """ポートフォリオのポジションを取得
        
        Args:
            portfolio_id: ポートフォリオID
        
        Returns:
            ポジションのリスト
        """
        pass
    
    @abstractmethod
    async def find_position(
        self, portfolio_id: UUID, ticker: str
    ) -> Optional[Position]:
        """特定銘柄のポジションを取得
        
        Args:
            portfolio_id: ポートフォリオID
            ticker: ティッカーシンボル
        
        Returns:
            ポジションエンティティ（見つからない場合はNone）
        """
        pass
    
    @abstractmethod
    async def save_position(
        self, portfolio_id: UUID, position: Position
    ) -> Position:
        """ポジションを保存
        
        Args:
            portfolio_id: ポートフォリオID
            position: ポジションエンティティ
        
        Returns:
            保存されたポジションエンティティ
        """
        pass
    
    @abstractmethod
    async def update_position(
        self, portfolio_id: UUID, position: Position
    ) -> Position:
        """ポジションを更新
        
        Args:
            portfolio_id: ポートフォリオID
            position: ポジションエンティティ
        
        Returns:
            更新されたポジションエンティティ
        """
        pass
    
    @abstractmethod
    async def close_position(
        self, portfolio_id: UUID, position_id: UUID
    ) -> bool:
        """ポジションをクローズ
        
        Args:
            portfolio_id: ポートフォリオID
            position_id: ポジションID
        
        Returns:
            クローズ成功の可否
        """
        pass
    
    # パフォーマンス関連のメソッド
    
    @abstractmethod
    async def get_performance_history(
        self,
        portfolio_id: UUID,
        start_date: datetime,
        end_date: datetime,
    ) -> List[dict]:
        """パフォーマンス履歴を取得
        
        Args:
            portfolio_id: ポートフォリオID
            start_date: 開始日
            end_date: 終了日
        
        Returns:
            パフォーマンスデータのリスト
        """
        pass
    
    @abstractmethod
    async def save_performance_snapshot(
        self,
        portfolio_id: UUID,
        snapshot: dict,
    ) -> bool:
        """パフォーマンススナップショットを保存
        
        Args:
            portfolio_id: ポートフォリオID
            snapshot: スナップショットデータ
        
        Returns:
            保存成功の可否
        """
        pass