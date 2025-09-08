"""シグナル関連DTO"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from uuid import UUID

from ...domain.services.signal_generator import Signal, SignalType, SignalDirection


@dataclass
class SignalDTO:
    """シグナルDTO"""
    id: str
    stock_id: str
    ticker: str
    company_name: str
    signal_type: str
    direction: str
    strength: float
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    time_horizon: str
    created_at: str
    expires_at: Optional[str]
    is_active: bool
    risk_reward_ratio: Optional[float]
    metadata: Dict[str, Any]
    
    @classmethod
    def from_entity(cls, signal: Signal) -> "SignalDTO":
        """エンティティからDTOを作成"""
        return cls(
            id=str(signal.id),
            stock_id=str(signal.stock.id),
            ticker=signal.stock.ticker,
            company_name=signal.stock.company_name,
            signal_type=signal.signal_type.value,
            direction=signal.direction.value,
            strength=float(signal.strength),
            confidence=float(signal.confidence.value),
            target_price=float(signal.target_price.value) if signal.target_price else None,
            stop_loss=float(signal.stop_loss.value) if signal.stop_loss else None,
            time_horizon=signal.time_horizon,
            created_at=signal.created_at.isoformat(),
            expires_at=signal.expires_at.isoformat() if signal.expires_at else None,
            is_active=signal.is_active,
            risk_reward_ratio=float(signal.risk_reward_ratio) if signal.risk_reward_ratio else None,
            metadata=signal.metadata,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "id": self.id,
            "stock": {
                "id": self.stock_id,
                "ticker": self.ticker,
                "name": self.company_name,
            },
            "signal": {
                "type": self.signal_type,
                "direction": self.direction,
                "strength": self.strength,
                "confidence": self.confidence,
            },
            "prices": {
                "target": self.target_price,
                "stop_loss": self.stop_loss,
                "risk_reward_ratio": self.risk_reward_ratio,
            },
            "timing": {
                "time_horizon": self.time_horizon,
                "created_at": self.created_at,
                "expires_at": self.expires_at,
                "is_active": self.is_active,
            },
            "metadata": self.metadata,
        }


@dataclass
class CreateSignalDTO:
    """シグナル作成DTO"""
    stock_id: UUID
    ticker: str
    signal_type: str
    direction: str
    strength: float
    confidence: float
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    time_horizon: str = "medium"
    expires_in_hours: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def validate(self) -> None:
        """バリデーション"""
        # シグナルタイプの検証
        valid_signal_types = ["TECHNICAL", "FUNDAMENTAL", "SENTIMENT", "AI_PREDICTION", "HYBRID"]
        if self.signal_type not in valid_signal_types:
            raise ValueError(f"Signal type must be one of {valid_signal_types}")
        
        # 方向の検証
        valid_directions = ["LONG", "SHORT", "NEUTRAL"]
        if self.direction not in valid_directions:
            raise ValueError(f"Direction must be one of {valid_directions}")
        
        # 強度の検証
        if not 0 <= self.strength <= 100:
            raise ValueError("Strength must be between 0 and 100")
        
        # 信頼度の検証
        if not 0 <= self.confidence <= 100:
            raise ValueError("Confidence must be between 0 and 100")
        
        # 時間軸の検証
        valid_horizons = ["short", "medium", "long"]
        if self.time_horizon not in valid_horizons:
            raise ValueError(f"Time horizon must be one of {valid_horizons}")
        
        # 価格の検証
        if self.target_price is not None and self.target_price <= 0:
            raise ValueError("Target price must be positive")
        
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("Stop loss must be positive")
        
        # ロング/ショートの価格整合性チェック
        if self.target_price and self.stop_loss:
            if self.direction == "LONG" and self.stop_loss >= self.target_price:
                raise ValueError("Stop loss must be below target price for long positions")
            elif self.direction == "SHORT" and self.stop_loss <= self.target_price:
                raise ValueError("Stop loss must be above target price for short positions")


@dataclass
class SignalResponseDTO:
    """シグナルレスポンスDTO"""
    signal: SignalDTO
    recommended_action: str
    position_size: Optional[float]
    risk_amount: Optional[float]
    expected_return: Optional[float]
    execution_notes: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "signal": self.signal.to_dict(),
            "recommendation": {
                "action": self.recommended_action,
                "position_size": self.position_size,
                "risk_amount": self.risk_amount,
                "expected_return": self.expected_return,
            },
            "execution_notes": self.execution_notes,
        }


@dataclass
class SignalStatisticsDTO:
    """シグナル統計DTO"""
    total_signals: int
    active_signals: int
    signal_by_type: Dict[str, int]
    signal_by_direction: Dict[str, int]
    average_strength: float
    average_confidence: float
    accuracy_rate: float
    profitable_rate: float
    average_return: float
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "counts": {
                "total": self.total_signals,
                "active": self.active_signals,
            },
            "distribution": {
                "by_type": self.signal_by_type,
                "by_direction": self.signal_by_direction,
            },
            "metrics": {
                "average_strength": self.average_strength,
                "average_confidence": self.average_confidence,
            },
            "performance": {
                "accuracy_rate": self.accuracy_rate,
                "profitable_rate": self.profitable_rate,
                "average_return": self.average_return,
            }
        }


@dataclass
class SignalBatchDTO:
    """シグナルバッチDTO（複数シグナルの一括処理用）"""
    signals: List[SignalDTO]
    batch_id: str
    created_at: str
    processing_time_ms: int
    total_stocks_analyzed: int
    signals_generated: int
    high_confidence_count: int
    
    @classmethod
    def create(
        cls,
        signals: List[Signal],
        batch_id: str,
        processing_time_ms: int,
        total_stocks_analyzed: int,
    ) -> "SignalBatchDTO":
        """シグナルリストからバッチDTOを作成"""
        signal_dtos = [SignalDTO.from_entity(s) for s in signals]
        high_confidence = sum(1 for s in signal_dtos if s.confidence >= 70)
        
        return cls(
            signals=signal_dtos,
            batch_id=batch_id,
            created_at=datetime.now().isoformat(),
            processing_time_ms=processing_time_ms,
            total_stocks_analyzed=total_stocks_analyzed,
            signals_generated=len(signal_dtos),
            high_confidence_count=high_confidence,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "batch_info": {
                "id": self.batch_id,
                "created_at": self.created_at,
                "processing_time_ms": self.processing_time_ms,
            },
            "statistics": {
                "total_stocks_analyzed": self.total_stocks_analyzed,
                "signals_generated": self.signals_generated,
                "high_confidence_count": self.high_confidence_count,
            },
            "signals": [s.to_dict() for s in self.signals],
        }