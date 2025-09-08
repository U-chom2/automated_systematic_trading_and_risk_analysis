"""取引戦略ドメインサービス"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID

from ..entities.portfolio import Portfolio
from ..entities.position import Position
from ..entities.trade import Trade, TradeType, OrderType
from ..entities.stock import Stock
from ..value_objects.price import Price, OHLCV
from ..value_objects.quantity import Quantity
from ..value_objects.percentage import Percentage, Rate


class StrategyType(Enum):
    """戦略タイプ"""
    CORE_SATELLITE = "CORE_SATELLITE"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    ARBITRAGE = "ARBITRAGE"
    AI_DRIVEN = "AI_DRIVEN"
    HYBRID = "HYBRID"


class SignalStrength(Enum):
    """シグナル強度"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingSignal:
    """取引シグナル"""
    stock: Stock
    signal_strength: SignalStrength
    target_price: Optional[Price] = None
    stop_loss_price: Optional[Price] = None
    confidence: Percentage = Percentage(50)
    reason: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """初期化後の処理"""
        if self.metadata is None:
            self.metadata = {}


class TradingStrategy:
    """取引戦略ドメインサービス
    
    ポートフォリオの取引戦略を管理し、売買判断を行う。
    """
    
    def __init__(
        self,
        strategy_type: StrategyType,
        max_position_size: Percentage = Percentage(10),  # 最大ポジションサイズ（ポートフォリオの10%）
        max_positions: int = 20,  # 最大保有銘柄数
        min_trade_amount: Decimal = Decimal("10000"),  # 最小取引金額（1万円）
        target_return: Percentage = Percentage(5),  # 目標月次リターン（5%）
    ) -> None:
        """初期化"""
        self.strategy_type = strategy_type
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.min_trade_amount = min_trade_amount
        self.target_return = target_return
    
    def generate_trades(
        self,
        portfolio: Portfolio,
        signals: List[TradingSignal],
        market_data: Dict[str, OHLCV],
    ) -> List[Trade]:
        """取引シグナルから実際の取引を生成
        
        Args:
            portfolio: ポートフォリオ
            signals: 取引シグナルリスト
            market_data: 市場データ（ticker -> OHLCV）
        
        Returns:
            生成された取引リスト
        """
        trades = []
        
        # コア・サテライト戦略の場合
        if self.strategy_type == StrategyType.CORE_SATELLITE:
            trades = self._generate_core_satellite_trades(portfolio, signals, market_data)
        # AI駆動戦略の場合
        elif self.strategy_type == StrategyType.AI_DRIVEN:
            trades = self._generate_ai_driven_trades(portfolio, signals, market_data)
        # その他の戦略
        else:
            trades = self._generate_basic_trades(portfolio, signals, market_data)
        
        return trades
    
    def _generate_core_satellite_trades(
        self,
        portfolio: Portfolio,
        signals: List[TradingSignal],
        market_data: Dict[str, OHLCV],
    ) -> List[Trade]:
        """コア・サテライト戦略の取引生成
        
        - コア資産（70%）: 安定的な大型株・ETF
        - サテライト資産（30%）: 高リターン狙いの中小型株
        """
        trades = []
        
        # ポートフォリオの現在価値を計算
        total_value = portfolio.total_value
        available_cash = portfolio.available_cash
        
        # コア・サテライトの配分を計算
        core_allocation = total_value * Decimal("0.7")
        satellite_allocation = total_value * Decimal("0.3")
        
        # 現在のポジションをコア・サテライトに分類
        core_positions = []
        satellite_positions = []
        
        for position in portfolio.positions.values():
            if not position.is_closed:
                # 大型株をコアとみなす（簡易的な判定）
                if self._is_core_asset(position.ticker):
                    core_positions.append(position)
                else:
                    satellite_positions.append(position)
        
        # シグナルを強度順にソート
        sorted_signals = sorted(
            signals,
            key=lambda s: self._signal_priority(s.signal_strength),
            reverse=True
        )
        
        for signal in sorted_signals:
            ticker = signal.stock.ticker
            
            # 市場データが存在しない場合はスキップ
            if ticker not in market_data:
                continue
            
            ohlcv = market_data[ticker]
            current_price = ohlcv.close
            
            # 買いシグナルの処理
            if signal.signal_strength in (SignalStrength.BUY, SignalStrength.STRONG_BUY):
                # ポジションサイズを計算
                is_core = self._is_core_asset(ticker)
                
                if is_core:
                    max_position_value = core_allocation * (self.max_position_size.decimal * 2)  # コアは大きめ
                else:
                    max_position_value = satellite_allocation * self.max_position_size.decimal
                
                # 既存ポジションがある場合は追加購入を検討
                existing_position = portfolio.positions.get(ticker)
                if existing_position and not existing_position.is_closed:
                    current_position_value = existing_position.market_value
                    additional_value = max_position_value - current_position_value
                    
                    if additional_value > self.min_trade_amount:
                        quantity = int(additional_value / current_price.value)
                        if quantity > 0:
                            trade = Trade(
                                portfolio_id=portfolio.id,
                                stock_id=signal.stock.id,
                                ticker=ticker,
                                trade_type=TradeType.BUY,
                                order_type=OrderType.LIMIT if signal.target_price else OrderType.MARKET,
                                quantity=quantity,
                                order_price=signal.target_price.value if signal.target_price else None,
                            )
                            trades.append(trade)
                
                # 新規ポジションの場合
                elif available_cash > self.min_trade_amount:
                    # ポジション数制限チェック
                    if len([p for p in portfolio.positions.values() if not p.is_closed]) < self.max_positions:
                        position_value = min(max_position_value, available_cash)
                        quantity = int(position_value / current_price.value)
                        
                        if quantity > 0:
                            trade = Trade(
                                portfolio_id=portfolio.id,
                                stock_id=signal.stock.id,
                                ticker=ticker,
                                trade_type=TradeType.BUY,
                                order_type=OrderType.LIMIT if signal.target_price else OrderType.MARKET,
                                quantity=quantity,
                                order_price=signal.target_price.value if signal.target_price else None,
                            )
                            trades.append(trade)
                            available_cash -= position_value
            
            # 売りシグナルの処理
            elif signal.signal_strength in (SignalStrength.SELL, SignalStrength.STRONG_SELL):
                existing_position = portfolio.positions.get(ticker)
                if existing_position and not existing_position.is_closed:
                    # 強い売りシグナルの場合は全売却
                    if signal.signal_strength == SignalStrength.STRONG_SELL:
                        trade = Trade(
                            portfolio_id=portfolio.id,
                            stock_id=signal.stock.id,
                            ticker=ticker,
                            trade_type=TradeType.SELL,
                            order_type=OrderType.LIMIT if signal.target_price else OrderType.MARKET,
                            quantity=existing_position.quantity,
                            order_price=signal.target_price.value if signal.target_price else None,
                        )
                        trades.append(trade)
                    # 通常の売りシグナルの場合は部分売却
                    else:
                        sell_quantity = existing_position.quantity // 2
                        if sell_quantity > 0:
                            trade = Trade(
                                portfolio_id=portfolio.id,
                                stock_id=signal.stock.id,
                                ticker=ticker,
                                trade_type=TradeType.SELL,
                                order_type=OrderType.LIMIT if signal.target_price else OrderType.MARKET,
                                quantity=sell_quantity,
                                order_price=signal.target_price.value if signal.target_price else None,
                            )
                            trades.append(trade)
        
        return trades
    
    def _generate_ai_driven_trades(
        self,
        portfolio: Portfolio,
        signals: List[TradingSignal],
        market_data: Dict[str, OHLCV],
    ) -> List[Trade]:
        """AI駆動戦略の取引生成
        
        AIモデルの予測に基づいて取引を生成。
        信頼度の高いシグナルを優先。
        """
        trades = []
        available_cash = portfolio.available_cash
        
        # 信頼度でシグナルをソート
        sorted_signals = sorted(
            signals,
            key=lambda s: s.confidence.value,
            reverse=True
        )
        
        for signal in sorted_signals:
            # 信頼度が60%未満のシグナルは無視
            if signal.confidence < Percentage(60):
                continue
            
            ticker = signal.stock.ticker
            if ticker not in market_data:
                continue
            
            ohlcv = market_data[ticker]
            current_price = ohlcv.close
            
            # ポジションサイズを信頼度に応じて調整
            confidence_factor = signal.confidence.decimal
            max_position_value = portfolio.total_value * self.max_position_size.decimal * confidence_factor
            
            if signal.signal_strength in (SignalStrength.BUY, SignalStrength.STRONG_BUY):
                if available_cash > self.min_trade_amount:
                    position_value = min(max_position_value, available_cash)
                    quantity = int(position_value / current_price.value)
                    
                    if quantity > 0:
                        trade = Trade(
                            portfolio_id=portfolio.id,
                            stock_id=signal.stock.id,
                            ticker=ticker,
                            trade_type=TradeType.BUY,
                            order_type=OrderType.LIMIT,
                            quantity=quantity,
                            order_price=signal.target_price.value if signal.target_price else current_price.value,
                        )
                        trades.append(trade)
                        available_cash -= position_value
            
            elif signal.signal_strength in (SignalStrength.SELL, SignalStrength.STRONG_SELL):
                existing_position = portfolio.positions.get(ticker)
                if existing_position and not existing_position.is_closed:
                    # 信頼度に応じて売却量を調整
                    sell_ratio = confidence_factor if signal.signal_strength == SignalStrength.SELL else 1.0
                    sell_quantity = int(existing_position.quantity * Decimal(str(sell_ratio)))
                    
                    if sell_quantity > 0:
                        trade = Trade(
                            portfolio_id=portfolio.id,
                            stock_id=signal.stock.id,
                            ticker=ticker,
                            trade_type=TradeType.SELL,
                            order_type=OrderType.LIMIT,
                            quantity=sell_quantity,
                            order_price=signal.target_price.value if signal.target_price else current_price.value,
                        )
                        trades.append(trade)
        
        return trades
    
    def _generate_basic_trades(
        self,
        portfolio: Portfolio,
        signals: List[TradingSignal],
        market_data: Dict[str, OHLCV],
    ) -> List[Trade]:
        """基本的な取引生成"""
        trades = []
        available_cash = portfolio.available_cash
        
        for signal in signals:
            ticker = signal.stock.ticker
            if ticker not in market_data:
                continue
            
            ohlcv = market_data[ticker]
            current_price = ohlcv.close
            
            if signal.signal_strength in (SignalStrength.BUY, SignalStrength.STRONG_BUY):
                if available_cash > self.min_trade_amount:
                    max_position_value = portfolio.total_value * self.max_position_size.decimal
                    position_value = min(max_position_value, available_cash)
                    quantity = int(position_value / current_price.value)
                    
                    if quantity > 0:
                        trade = Trade(
                            portfolio_id=portfolio.id,
                            stock_id=signal.stock.id,
                            ticker=ticker,
                            trade_type=TradeType.BUY,
                            order_type=OrderType.MARKET,
                            quantity=quantity,
                        )
                        trades.append(trade)
                        available_cash -= position_value
            
            elif signal.signal_strength in (SignalStrength.SELL, SignalStrength.STRONG_SELL):
                existing_position = portfolio.positions.get(ticker)
                if existing_position and not existing_position.is_closed:
                    trade = Trade(
                        portfolio_id=portfolio.id,
                        stock_id=signal.stock.id,
                        ticker=ticker,
                        trade_type=TradeType.SELL,
                        order_type=OrderType.MARKET,
                        quantity=existing_position.quantity,
                    )
                    trades.append(trade)
        
        return trades
    
    def _is_core_asset(self, ticker: str) -> bool:
        """コア資産かどうかを判定
        
        簡易的な実装。実際には時価総額や流動性で判定。
        """
        # 日本の大型株の例（日経225構成銘柄の一部）
        core_tickers = {
            "7203",  # トヨタ自動車
            "6758",  # ソニー
            "6861",  # キーエンス
            "8306",  # 三菱UFJ
            "9432",  # NTT
            "7267",  # ホンダ
            "4063",  # 信越化学
            "6098",  # リクルート
            "8035",  # 東京エレクトロン
            "9984",  # ソフトバンクグループ
        }
        
        return ticker in core_tickers
    
    def _signal_priority(self, signal_strength: SignalStrength) -> int:
        """シグナル強度の優先度を取得"""
        priorities = {
            SignalStrength.STRONG_BUY: 5,
            SignalStrength.BUY: 4,
            SignalStrength.HOLD: 3,
            SignalStrength.SELL: 2,
            SignalStrength.STRONG_SELL: 1,
        }
        return priorities.get(signal_strength, 0)