"""シグナル生成ドメインサービス"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4

from ..entities.stock import Stock
from ..value_objects.price import Price, OHLCV
from ..value_objects.percentage import Percentage


class SignalType(Enum):
    """シグナルタイプ"""
    TECHNICAL = "TECHNICAL"  # テクニカル分析
    FUNDAMENTAL = "FUNDAMENTAL"  # ファンダメンタル分析
    SENTIMENT = "SENTIMENT"  # センチメント分析
    AI_PREDICTION = "AI_PREDICTION"  # AI予測
    HYBRID = "HYBRID"  # ハイブリッド


class SignalDirection(Enum):
    """シグナル方向"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class Signal:
    """取引シグナル"""
    stock: Stock
    signal_type: SignalType
    direction: SignalDirection
    strength: Decimal  # 0-100のシグナル強度
    confidence: Percentage  # 信頼度
    id: UUID = field(default_factory=uuid4)
    target_price: Optional[Price] = field(default=None)
    stop_loss: Optional[Price] = field(default=None)
    time_horizon: str = field(default="medium")  # short/medium/long
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        if self.strength < 0 or self.strength > 100:
            raise ValueError("Signal strength must be between 0 and 100")
        
        if self.target_price and self.stop_loss:
            if self.direction == SignalDirection.LONG:
                if self.stop_loss >= self.target_price:
                    raise ValueError("Stop loss must be below target price for long positions")
            elif self.direction == SignalDirection.SHORT:
                if self.stop_loss <= self.target_price:
                    raise ValueError("Stop loss must be above target price for short positions")
    
    @property
    def is_active(self) -> bool:
        """シグナルが有効かどうか"""
        if self.expires_at:
            return datetime.now() < self.expires_at
        return True
    
    @property
    def risk_reward_ratio(self) -> Optional[Decimal]:
        """リスクリワード比率"""
        if not self.target_price or not self.stop_loss:
            return None
        
        # 現在価格を基準にリスクリワードを計算
        # 簡易的な実装（実際には現在価格が必要）
        if self.direction == SignalDirection.LONG:
            risk = abs(self.stop_loss.value - self.target_price.value)
            reward = abs(self.target_price.value - self.stop_loss.value)
        else:
            risk = abs(self.target_price.value - self.stop_loss.value)
            reward = abs(self.stop_loss.value - self.target_price.value)
        
        if risk == 0:
            return None
        
        return reward / risk


class SignalGenerator:
    """シグナル生成ドメインサービス
    
    各種分析手法を組み合わせて取引シグナルを生成する。
    """
    
    def __init__(
        self,
        min_strength: Decimal = Decimal("30"),  # 最小シグナル強度
        min_confidence: Percentage = Percentage(50),  # 最小信頼度
        enable_technical: bool = True,
        enable_fundamental: bool = False,
        enable_sentiment: bool = True,
        enable_ai: bool = True,
    ) -> None:
        """初期化"""
        self.min_strength = min_strength
        self.min_confidence = min_confidence
        self.enable_technical = enable_technical
        self.enable_fundamental = enable_fundamental
        self.enable_sentiment = enable_sentiment
        self.enable_ai = enable_ai
    
    def generate_signals(
        self,
        stocks: List[Stock],
        market_data: Dict[str, List[OHLCV]],
        sentiment_data: Optional[Dict[str, Decimal]] = None,
        ai_predictions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Signal]:
        """統合シグナルを生成
        
        Args:
            stocks: 対象銘柄リスト
            market_data: 市場データ
            sentiment_data: センチメントスコア（ticker -> score）
            ai_predictions: AI予測結果
        
        Returns:
            生成されたシグナルリスト
        """
        all_signals = []
        
        for stock in stocks:
            ticker = stock.ticker
            
            if ticker not in market_data:
                continue
            
            ohlcv_data = market_data[ticker]
            
            # テクニカル分析シグナル
            if self.enable_technical:
                technical_signals = self._generate_technical_signals(stock, ohlcv_data)
                all_signals.extend(technical_signals)
            
            # センチメント分析シグナル
            if self.enable_sentiment and sentiment_data:
                sentiment_score = sentiment_data.get(ticker)
                if sentiment_score is not None:
                    sentiment_signal = self._generate_sentiment_signal(stock, sentiment_score)
                    if sentiment_signal:
                        all_signals.append(sentiment_signal)
            
            # AI予測シグナル
            if self.enable_ai and ai_predictions:
                ai_pred = ai_predictions.get(ticker)
                if ai_pred:
                    ai_signal = self._generate_ai_signal(stock, ai_pred, ohlcv_data)
                    if ai_signal:
                        all_signals.append(ai_signal)
        
        # 統合シグナルを生成
        consolidated_signals = self._consolidate_signals(all_signals)
        
        # フィルタリング
        filtered_signals = [
            s for s in consolidated_signals
            if s.strength >= self.min_strength and s.confidence >= self.min_confidence
        ]
        
        return filtered_signals
    
    def _generate_technical_signals(
        self,
        stock: Stock,
        ohlcv_data: List[OHLCV],
    ) -> List[Signal]:
        """テクニカル分析によるシグナル生成"""
        signals = []
        
        if len(ohlcv_data) < 20:
            return signals
        
        # 移動平均を計算
        sma_20 = self._calculate_sma(ohlcv_data, 20)
        sma_50 = self._calculate_sma(ohlcv_data, 50) if len(ohlcv_data) >= 50 else None
        
        current_price = ohlcv_data[-1].close
        
        # ゴールデンクロス/デッドクロス
        if sma_50:
            if sma_20 > sma_50 and ohlcv_data[-2] and self._calculate_sma(ohlcv_data[:-1], 20) <= sma_50:
                # ゴールデンクロス
                signal = Signal(
                    stock=stock,
                    signal_type=SignalType.TECHNICAL,
                    direction=SignalDirection.LONG,
                    strength=Decimal("70"),
                    confidence=Percentage(60),
                    target_price=Price(current_price.value * Decimal("1.05")),
                    stop_loss=Price(current_price.value * Decimal("0.97")),
                    time_horizon="medium",
                    metadata={"pattern": "golden_cross"}
                )
                signals.append(signal)
            elif sma_20 < sma_50 and ohlcv_data[-2] and self._calculate_sma(ohlcv_data[:-1], 20) >= sma_50:
                # デッドクロス
                signal = Signal(
                    stock=stock,
                    signal_type=SignalType.TECHNICAL,
                    direction=SignalDirection.SHORT,
                    strength=Decimal("70"),
                    confidence=Percentage(60),
                    target_price=Price(current_price.value * Decimal("0.95")),
                    stop_loss=Price(current_price.value * Decimal("1.03")),
                    time_horizon="medium",
                    metadata={"pattern": "death_cross"}
                )
                signals.append(signal)
        
        # RSI
        rsi = self._calculate_rsi(ohlcv_data, 14)
        if rsi is not None:
            if rsi < 30:
                # 売られ過ぎ
                signal = Signal(
                    stock=stock,
                    signal_type=SignalType.TECHNICAL,
                    direction=SignalDirection.LONG,
                    strength=Decimal("60"),
                    confidence=Percentage(55),
                    target_price=Price(current_price.value * Decimal("1.03")),
                    stop_loss=Price(current_price.value * Decimal("0.98")),
                    time_horizon="short",
                    metadata={"indicator": "rsi_oversold", "rsi": float(rsi)}
                )
                signals.append(signal)
            elif rsi > 70:
                # 買われ過ぎ
                signal = Signal(
                    stock=stock,
                    signal_type=SignalType.TECHNICAL,
                    direction=SignalDirection.SHORT,
                    strength=Decimal("60"),
                    confidence=Percentage(55),
                    target_price=Price(current_price.value * Decimal("0.97")),
                    stop_loss=Price(current_price.value * Decimal("1.02")),
                    time_horizon="short",
                    metadata={"indicator": "rsi_overbought", "rsi": float(rsi)}
                )
                signals.append(signal)
        
        # ボリンジャーバンド
        bb_upper, bb_lower = self._calculate_bollinger_bands(ohlcv_data, 20)
        if bb_upper and bb_lower:
            if current_price.value < bb_lower:
                # 下限バンドタッチ
                signal = Signal(
                    stock=stock,
                    signal_type=SignalType.TECHNICAL,
                    direction=SignalDirection.LONG,
                    strength=Decimal("50"),
                    confidence=Percentage(50),
                    target_price=Price(sma_20),
                    stop_loss=Price(current_price.value * Decimal("0.98")),
                    time_horizon="short",
                    metadata={"pattern": "bollinger_lower_touch"}
                )
                signals.append(signal)
            elif current_price.value > bb_upper:
                # 上限バンドタッチ
                signal = Signal(
                    stock=stock,
                    signal_type=SignalType.TECHNICAL,
                    direction=SignalDirection.SHORT,
                    strength=Decimal("50"),
                    confidence=Percentage(50),
                    target_price=Price(sma_20),
                    stop_loss=Price(current_price.value * Decimal("1.02")),
                    time_horizon="short",
                    metadata={"pattern": "bollinger_upper_touch"}
                )
                signals.append(signal)
        
        return signals
    
    def _generate_sentiment_signal(
        self,
        stock: Stock,
        sentiment_score: Decimal,
    ) -> Optional[Signal]:
        """センチメント分析によるシグナル生成"""
        # センチメントスコア: -1.0（非常にネガティブ）〜 +1.0（非常にポジティブ）
        
        if sentiment_score > Decimal("0.5"):
            # ポジティブセンチメント
            return Signal(
                stock=stock,
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.LONG,
                strength=Decimal(str(float(sentiment_score) * 100)),
                confidence=Percentage(65),
                time_horizon="medium",
                metadata={"sentiment_score": float(sentiment_score)}
            )
        elif sentiment_score < Decimal("-0.5"):
            # ネガティブセンチメント
            return Signal(
                stock=stock,
                signal_type=SignalType.SENTIMENT,
                direction=SignalDirection.SHORT,
                strength=Decimal(str(abs(float(sentiment_score)) * 100)),
                confidence=Percentage(65),
                time_horizon="medium",
                metadata={"sentiment_score": float(sentiment_score)}
            )
        
        return None
    
    def _generate_ai_signal(
        self,
        stock: Stock,
        ai_prediction: Dict[str, Any],
        ohlcv_data: List[OHLCV],
    ) -> Optional[Signal]:
        """AI予測によるシグナル生成"""
        # AI予測フォーマット:
        # {
        #     "action": "buy/sell/hold",
        #     "confidence": 0.0-1.0,
        #     "predicted_return": 予測リターン,
        #     "time_horizon": "short/medium/long"
        # }
        
        action = ai_prediction.get("action", "hold")
        confidence = Decimal(str(ai_prediction.get("confidence", 0.5)))
        predicted_return = Decimal(str(ai_prediction.get("predicted_return", 0)))
        time_horizon = ai_prediction.get("time_horizon", "medium")
        
        if action == "hold":
            return None
        
        current_price = ohlcv_data[-1].close if ohlcv_data else Price(Decimal("100"))
        
        if action == "buy":
            target_price = Price(current_price.value * (1 + predicted_return))
            stop_loss = Price(current_price.value * Decimal("0.95"))
            direction = SignalDirection.LONG
        else:  # sell
            target_price = Price(current_price.value * (1 - abs(predicted_return)))
            stop_loss = Price(current_price.value * Decimal("1.05"))
            direction = SignalDirection.SHORT
        
        return Signal(
            stock=stock,
            signal_type=SignalType.AI_PREDICTION,
            direction=direction,
            strength=confidence * 100,
            confidence=Percentage.from_decimal(confidence),
            target_price=target_price,
            stop_loss=stop_loss,
            time_horizon=time_horizon,
            metadata={
                "ai_action": action,
                "predicted_return": float(predicted_return),
                "model_confidence": float(confidence)
            }
        )
    
    def _consolidate_signals(self, signals: List[Signal]) -> List[Signal]:
        """複数のシグナルを統合"""
        # 銘柄ごとにグループ化
        signals_by_stock = {}
        for signal in signals:
            ticker = signal.stock.ticker
            if ticker not in signals_by_stock:
                signals_by_stock[ticker] = []
            signals_by_stock[ticker].append(signal)
        
        consolidated = []
        
        for ticker, stock_signals in signals_by_stock.items():
            if len(stock_signals) == 1:
                consolidated.append(stock_signals[0])
                continue
            
            # 複数シグナルの統合
            # 方向性の集計
            long_signals = [s for s in stock_signals if s.direction == SignalDirection.LONG]
            short_signals = [s for s in stock_signals if s.direction == SignalDirection.SHORT]
            
            if len(long_signals) > len(short_signals):
                # ロング優勢
                avg_strength = sum(s.strength for s in long_signals) / len(long_signals)
                avg_confidence = sum(s.confidence.value for s in long_signals) / len(long_signals)
                
                # 目標価格とストップロスの平均を計算
                target_prices = [s.target_price for s in long_signals if s.target_price]
                stop_losses = [s.stop_loss for s in long_signals if s.stop_loss]
                
                hybrid_signal = Signal(
                    stock=stock_signals[0].stock,
                    signal_type=SignalType.HYBRID,
                    direction=SignalDirection.LONG,
                    strength=avg_strength,
                    confidence=Percentage(avg_confidence),
                    target_price=target_prices[0] if target_prices else None,
                    stop_loss=stop_losses[0] if stop_losses else None,
                    time_horizon="medium",
                    metadata={
                        "signal_count": len(long_signals),
                        "signal_types": [s.signal_type.value for s in long_signals]
                    }
                )
                consolidated.append(hybrid_signal)
            
            elif len(short_signals) > len(long_signals):
                # ショート優勢
                avg_strength = sum(s.strength for s in short_signals) / len(short_signals)
                avg_confidence = sum(s.confidence.value for s in short_signals) / len(short_signals)
                
                target_prices = [s.target_price for s in short_signals if s.target_price]
                stop_losses = [s.stop_loss for s in short_signals if s.stop_loss]
                
                hybrid_signal = Signal(
                    stock=stock_signals[0].stock,
                    signal_type=SignalType.HYBRID,
                    direction=SignalDirection.SHORT,
                    strength=avg_strength,
                    confidence=Percentage(avg_confidence),
                    target_price=target_prices[0] if target_prices else None,
                    stop_loss=stop_losses[0] if stop_losses else None,
                    time_horizon="medium",
                    metadata={
                        "signal_count": len(short_signals),
                        "signal_types": [s.signal_type.value for s in short_signals]
                    }
                )
                consolidated.append(hybrid_signal)
        
        return consolidated
    
    def _calculate_sma(self, ohlcv_data: List[OHLCV], period: int) -> Decimal:
        """単純移動平均を計算"""
        if len(ohlcv_data) < period:
            return Decimal("0")
        
        prices = [d.close.value for d in ohlcv_data[-period:]]
        return sum(prices) / len(prices)
    
    def _calculate_rsi(self, ohlcv_data: List[OHLCV], period: int = 14) -> Optional[Decimal]:
        """RSIを計算"""
        if len(ohlcv_data) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(ohlcv_data)):
            change = ohlcv_data[i].close.value - ohlcv_data[i-1].close.value
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return Decimal("100")
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return Decimal(str(rsi))
    
    def _calculate_bollinger_bands(
        self,
        ohlcv_data: List[OHLCV],
        period: int = 20,
        std_dev: int = 2,
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """ボリンジャーバンドを計算"""
        if len(ohlcv_data) < period:
            return None, None
        
        # SMAを計算
        sma = self._calculate_sma(ohlcv_data, period)
        
        # 標準偏差を計算
        prices = [d.close.value for d in ohlcv_data[-period:]]
        variance = sum((p - sma) ** 2 for p in prices) / period
        std = Decimal(str(math.sqrt(float(variance))))
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, lower_band