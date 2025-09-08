"""取引リポジトリインターフェース"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from ..entities.trade import Trade, TradeType, OrderStatus


class TradeRepository(ABC):
    """取引リポジトリの抽象インターフェース"""
    
    @abstractmethod
    async def find_by_id(self, trade_id: UUID) -> Optional[Trade]:
        """IDで取引を検索
        
        Args:
            trade_id: 取引ID
        
        Returns:
            取引エンティティ（見つからない場合はNone）
        """
        pass
    
    @abstractmethod
    async def find_by_portfolio(
        self,
        portfolio_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trade]:
        """ポートフォリオの取引を取得
        
        Args:
            portfolio_id: ポートフォリオID
            limit: 取得件数上限
            offset: オフセット
        
        Returns:
            取引エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_by_ticker(
        self,
        ticker: str,
        portfolio_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trade]:
        """ティッカーで取引を検索
        
        Args:
            ticker: ティッカーシンボル
            portfolio_id: ポートフォリオID（オプション）
            limit: 取得件数上限
            offset: オフセット
        
        Returns:
            取引エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_by_status(
        self,
        status: OrderStatus,
        portfolio_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trade]:
        """ステータスで取引を検索
        
        Args:
            status: 注文ステータス
            portfolio_id: ポートフォリオID（オプション）
            limit: 取得件数上限
            offset: オフセット
        
        Returns:
            取引エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_pending_trades(
        self,
        portfolio_id: Optional[UUID] = None,
    ) -> List[Trade]:
        """待機中の取引を取得
        
        Args:
            portfolio_id: ポートフォリオID（オプション）
        
        Returns:
            待機中の取引エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        portfolio_id: Optional[UUID] = None,
        trade_type: Optional[TradeType] = None,
    ) -> List[Trade]:
        """日付範囲で取引を検索
        
        Args:
            start_date: 開始日
            end_date: 終了日
            portfolio_id: ポートフォリオID（オプション）
            trade_type: 取引タイプ（オプション）
        
        Returns:
            取引エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_by_signal(self, signal_id: UUID) -> List[Trade]:
        """シグナルIDで取引を検索
        
        Args:
            signal_id: シグナルID
        
        Returns:
            取引エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def save(self, trade: Trade) -> Trade:
        """取引を保存
        
        Args:
            trade: 取引エンティティ
        
        Returns:
            保存された取引エンティティ
        """
        pass
    
    @abstractmethod
    async def save_all(self, trades: List[Trade]) -> List[Trade]:
        """複数の取引を一括保存
        
        Args:
            trades: 取引エンティティのリスト
        
        Returns:
            保存された取引エンティティのリスト
        """
        pass
    
    @abstractmethod
    async def update(self, trade: Trade) -> Trade:
        """取引を更新
        
        Args:
            trade: 取引エンティティ
        
        Returns:
            更新された取引エンティティ
        """
        pass
    
    @abstractmethod
    async def update_status(
        self,
        trade_id: UUID,
        status: OrderStatus,
    ) -> bool:
        """取引ステータスを更新
        
        Args:
            trade_id: 取引ID
            status: 新しいステータス
        
        Returns:
            更新成功の可否
        """
        pass
    
    @abstractmethod
    async def cancel(self, trade_id: UUID) -> bool:
        """取引をキャンセル
        
        Args:
            trade_id: 取引ID
        
        Returns:
            キャンセル成功の可否
        """
        pass
    
    @abstractmethod
    async def delete(self, trade_id: UUID) -> bool:
        """取引を削除
        
        Args:
            trade_id: 取引ID
        
        Returns:
            削除成功の可否
        """
        pass
    
    # 統計関連のメソッド
    
    @abstractmethod
    async def get_trade_statistics(
        self,
        portfolio_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """取引統計を取得
        
        Args:
            portfolio_id: ポートフォリオID
            start_date: 開始日（オプション）
            end_date: 終了日（オプション）
        
        Returns:
            統計データ
        """
        pass
    
    @abstractmethod
    async def get_win_rate(
        self,
        portfolio_id: UUID,
        ticker: Optional[str] = None,
    ) -> float:
        """勝率を取得
        
        Args:
            portfolio_id: ポートフォリオID
            ticker: ティッカーシンボル（オプション）
        
        Returns:
            勝率（0.0-1.0）
        """
        pass
    
    @abstractmethod
    async def get_average_return(
        self,
        portfolio_id: UUID,
        ticker: Optional[str] = None,
    ) -> float:
        """平均リターンを取得
        
        Args:
            portfolio_id: ポートフォリオID
            ticker: ティッカーシンボル（オプション）
        
        Returns:
            平均リターン
        """
        pass