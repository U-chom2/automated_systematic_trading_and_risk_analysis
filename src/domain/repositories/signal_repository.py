"""シグナルリポジトリインターフェース"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from ..services.signal_generator import Signal, SignalType, SignalDirection


class SignalRepository(ABC):
    """シグナルリポジトリの抽象インターフェース"""
    
    @abstractmethod
    async def find_by_id(self, signal_id: UUID) -> Optional[Signal]:
        """IDでシグナルを検索
        
        Args:
            signal_id: シグナルID
        
        Returns:
            シグナルエンティティ（見つからない場合はNone）
        """
        pass
    
    @abstractmethod
    async def find_active(
        self,
        ticker: Optional[str] = None,
        signal_type: Optional[SignalType] = None,
    ) -> List[Signal]:
        """アクティブなシグナルを取得
        
        Args:
            ticker: ティッカーシンボル（オプション）
            signal_type: シグナルタイプ（オプション）
        
        Returns:
            アクティブなシグナルのリスト
        """
        pass
    
    @abstractmethod
    async def find_by_ticker(
        self,
        ticker: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Signal]:
        """ティッカーでシグナルを検索
        
        Args:
            ticker: ティッカーシンボル
            limit: 取得件数上限
            offset: オフセット
        
        Returns:
            シグナルエンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_by_type(
        self,
        signal_type: SignalType,
        active_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Signal]:
        """タイプでシグナルを検索
        
        Args:
            signal_type: シグナルタイプ
            active_only: アクティブなもののみ
            limit: 取得件数上限
            offset: オフセット
        
        Returns:
            シグナルエンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_by_direction(
        self,
        direction: SignalDirection,
        active_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Signal]:
        """方向でシグナルを検索
        
        Args:
            direction: シグナル方向
            active_only: アクティブなもののみ
            limit: 取得件数上限
            offset: オフセット
        
        Returns:
            シグナルエンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        ticker: Optional[str] = None,
        signal_type: Optional[SignalType] = None,
    ) -> List[Signal]:
        """日付範囲でシグナルを検索
        
        Args:
            start_date: 開始日
            end_date: 終了日
            ticker: ティッカーシンボル（オプション）
            signal_type: シグナルタイプ（オプション）
        
        Returns:
            シグナルエンティティのリスト
        """
        pass
    
    @abstractmethod
    async def find_high_confidence(
        self,
        min_confidence: float = 0.7,
        active_only: bool = True,
    ) -> List[Signal]:
        """高信頼度のシグナルを取得
        
        Args:
            min_confidence: 最小信頼度
            active_only: アクティブなもののみ
        
        Returns:
            シグナルエンティティのリスト
        """
        pass
    
    @abstractmethod
    async def save(self, signal: Signal) -> Signal:
        """シグナルを保存
        
        Args:
            signal: シグナルエンティティ
        
        Returns:
            保存されたシグナルエンティティ
        """
        pass
    
    @abstractmethod
    async def save_all(self, signals: List[Signal]) -> List[Signal]:
        """複数のシグナルを一括保存
        
        Args:
            signals: シグナルエンティティのリスト
        
        Returns:
            保存されたシグナルエンティティのリスト
        """
        pass
    
    @abstractmethod
    async def update(self, signal: Signal) -> Signal:
        """シグナルを更新
        
        Args:
            signal: シグナルエンティティ
        
        Returns:
            更新されたシグナルエンティティ
        """
        pass
    
    @abstractmethod
    async def expire(self, signal_id: UUID) -> bool:
        """シグナルを期限切れにする
        
        Args:
            signal_id: シグナルID
        
        Returns:
            更新成功の可否
        """
        pass
    
    @abstractmethod
    async def delete(self, signal_id: UUID) -> bool:
        """シグナルを削除
        
        Args:
            signal_id: シグナルID
        
        Returns:
            削除成功の可否
        """
        pass
    
    @abstractmethod
    async def delete_expired(self) -> int:
        """期限切れシグナルを削除
        
        Returns:
            削除されたシグナル数
        """
        pass
    
    # 統計関連のメソッド
    
    @abstractmethod
    async def get_signal_statistics(
        self,
        ticker: Optional[str] = None,
        signal_type: Optional[SignalType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """シグナル統計を取得
        
        Args:
            ticker: ティッカーシンボル（オプション）
            signal_type: シグナルタイプ（オプション）
            start_date: 開始日（オプション）
            end_date: 終了日（オプション）
        
        Returns:
            統計データ
        """
        pass
    
    @abstractmethod
    async def get_accuracy(
        self,
        signal_type: Optional[SignalType] = None,
        ticker: Optional[str] = None,
    ) -> float:
        """シグナル精度を取得
        
        Args:
            signal_type: シグナルタイプ（オプション）
            ticker: ティッカーシンボル（オプション）
        
        Returns:
            精度（0.0-1.0）
        """
        pass