"""
分析モジュールのデータモデル
"""
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Literal, Dict, Any
from enum import Enum


class ActionType(Enum):
    """取引アクション種別"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(Enum):
    """リスクレベル"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class TradingRecommendation:
    """取引推奨"""
    ticker: str
    company_name: str
    action: ActionType
    quantity: int
    hold_until: Optional[date] = None
    expected_return: float = 0.0
    confidence: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    reasoning: str = ""
    analysis_details: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def should_execute(self) -> bool:
        """実行すべきかの判定"""
        return self.action in [ActionType.BUY, ActionType.SELL] and self.confidence >= 0.6
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "action": self.action.value,
            "quantity": self.quantity,
            "hold_until": self.hold_until.isoformat() if self.hold_until else None,
            "expected_return": self.expected_return,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "reasoning": self.reasoning,
            "analysis_details": self.analysis_details,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class TodoItem:
    """取引TODOアイテム"""
    stock_id: str
    ticker: str
    company_name: str
    action_type: ActionType
    quantity: int
    target_date: date
    hold_until: Optional[date] = None
    status: Literal["PENDING", "EXECUTED", "CANCELLED"] = "PENDING"
    recommendation_details: Optional[TradingRecommendation] = None
    created_at: datetime = None
    executed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def mark_as_executed(self):
        """実行済みにマーク"""
        self.status = "EXECUTED"
        self.executed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "stock_id": self.stock_id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "action_type": self.action_type.value,
            "quantity": self.quantity,
            "target_date": self.target_date.isoformat(),
            "hold_until": self.hold_until.isoformat() if self.hold_until else None,
            "status": self.status,
            "recommendation_details": self.recommendation_details.to_dict() if self.recommendation_details else None,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None
        }


@dataclass
class AnalysisResult:
    """分析結果"""
    ticker: str
    company_name: str
    technical_score: float  # テクニカル分析スコア（0-100）
    fundamental_score: float  # ファンダメンタル分析スコア（0-100）
    sentiment_score: float  # センチメント分析スコア（-1 to 1）
    ai_prediction: Optional[Dict[str, Any]] = None
    indicators: Optional[Dict[str, float]] = None
    
    @property
    def total_score(self) -> float:
        """総合スコア計算"""
        # 重み付け平均
        tech_weight = 0.3
        fund_weight = 0.4
        sent_weight = 0.3
        
        # センチメントスコアを0-100に正規化
        normalized_sentiment = (self.sentiment_score + 1) * 50
        
        return (
            self.technical_score * tech_weight +
            self.fundamental_score * fund_weight +
            normalized_sentiment * sent_weight
        )
    
    def to_recommendation(self, quantity: int = 100) -> TradingRecommendation:
        """分析結果から推奨を生成"""
        # スコアに基づいてアクションを決定
        if self.total_score >= 70:
            action = ActionType.BUY
            confidence = min(self.total_score / 100, 0.95)
        elif self.total_score <= 30:
            action = ActionType.SELL
            confidence = min((100 - self.total_score) / 100, 0.95)
        else:
            action = ActionType.HOLD
            confidence = 0.5
        
        # リスクレベル判定
        if abs(self.sentiment_score) > 0.5:
            risk_level = RiskLevel.HIGH
        elif abs(self.sentiment_score) > 0.2:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # 期待リターン計算（簡易版）
        expected_return = (self.total_score - 50) / 100 * 0.3  # 最大±15%
        
        # 保有期限設定（1ヶ月後）
        from datetime import timedelta
        hold_until = date.today() + timedelta(days=30)
        
        reasoning = self._generate_reasoning()
        
        return TradingRecommendation(
            ticker=self.ticker,
            company_name=self.company_name,
            action=action,
            quantity=quantity,
            hold_until=hold_until,
            expected_return=expected_return,
            confidence=confidence,
            risk_level=risk_level,
            reasoning=reasoning,
            analysis_details={
                "technical_score": self.technical_score,
                "fundamental_score": self.fundamental_score,
                "sentiment_score": self.sentiment_score,
                "total_score": self.total_score,
                "indicators": self.indicators,
                "ai_prediction": self.ai_prediction
            }
        )
    
    def _generate_reasoning(self) -> str:
        """推奨理由を生成"""
        reasons = []
        
        if self.technical_score >= 70:
            reasons.append("テクニカル指標が買いシグナル")
        elif self.technical_score <= 30:
            reasons.append("テクニカル指標が売りシグナル")
        
        if self.fundamental_score >= 70:
            reasons.append("ファンダメンタルズが良好")
        elif self.fundamental_score <= 30:
            reasons.append("ファンダメンタルズに懸念")
        
        if self.sentiment_score >= 0.3:
            reasons.append("ポジティブなセンチメント")
        elif self.sentiment_score <= -0.3:
            reasons.append("ネガティブなセンチメント")
        
        return "、".join(reasons) if reasons else "総合的な判断"