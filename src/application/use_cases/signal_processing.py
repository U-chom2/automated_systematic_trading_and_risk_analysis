"""シグナル処理ユースケース"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from ...domain.entities.stock import Stock
from ...domain.services.signal_generator import SignalGenerator, Signal, SignalType
from ...domain.services.trading_strategy import TradingStrategy, TradingSignal, SignalStrength
from ...domain.repositories.signal_repository import SignalRepository
from ...domain.repositories.stock_repository import StockRepository
from ...domain.repositories.market_data_repository import MarketDataRepository
from ...domain.repositories.portfolio_repository import PortfolioRepository
from ..dto.signal_dto import SignalDTO, CreateSignalDTO, SignalResponseDTO, SignalBatchDTO


@dataclass
class GenerateSignalsUseCase:
    """シグナル生成ユースケース"""
    
    signal_generator: SignalGenerator
    signal_repository: SignalRepository
    stock_repository: StockRepository
    market_data_repository: MarketDataRepository
    
    async def execute(
        self,
        tickers: Optional[List[str]] = None,
        signal_types: Optional[List[str]] = None,
        lookback_days: int = 30,
    ) -> SignalBatchDTO:
        """シグナルを生成
        
        Args:
            tickers: 対象ティッカーリスト（Noneの場合は全銘柄）
            signal_types: 生成するシグナルタイプ
            lookback_days: 過去データ参照期間
        
        Returns:
            生成されたシグナルのバッチDTO
        """
        start_time = datetime.now()
        
        # 対象銘柄を取得
        if tickers:
            stocks = []
            for ticker in tickers:
                stock = await self.stock_repository.find_by_ticker(ticker)
                if stock:
                    stocks.append(stock)
        else:
            stocks = await self.stock_repository.find_all(limit=500)
        
        if not stocks:
            return SignalBatchDTO.create([], str(uuid4()), 0, 0)
        
        # 市場データを取得
        end_date = datetime.now().date()
        start_date = (datetime.now() - timedelta(days=lookback_days)).date()
        
        ticker_list = [s.ticker for s in stocks]
        market_data = await self.market_data_repository.get_multiple_ohlcv(
            ticker_list, start_date, end_date
        )
        
        # センチメントデータ（仮実装）
        sentiment_data = {}  # TODO: センチメント分析サービスから取得
        
        # AI予測（仮実装）
        ai_predictions = {}  # TODO: AIモデルから予測を取得
        
        # シグナルを生成
        signals = self.signal_generator.generate_signals(
            stocks, market_data, sentiment_data, ai_predictions
        )
        
        # シグナルを保存
        if signals:
            saved_signals = await self.signal_repository.save_all(signals)
        else:
            saved_signals = []
        
        # 処理時間を計算
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # バッチDTOを作成
        batch_id = str(uuid4())
        return SignalBatchDTO.create(
            saved_signals,
            batch_id,
            processing_time_ms,
            len(stocks)
        )


@dataclass
class ProcessSignalUseCase:
    """シグナル処理ユースケース"""
    
    signal_repository: SignalRepository
    portfolio_repository: PortfolioRepository
    trading_strategy: TradingStrategy
    
    async def execute(
        self,
        signal_id: UUID,
        portfolio_id: UUID,
    ) -> SignalResponseDTO:
        """シグナルを処理して取引推奨を生成
        
        Args:
            signal_id: シグナルID
            portfolio_id: ポートフォリオID
        
        Returns:
            シグナルレスポンスDTO
        """
        # シグナルを取得
        signal = await self.signal_repository.find_by_id(signal_id)
        if not signal:
            raise ValueError(f"Signal not found: {signal_id}")
        
        # ポートフォリオを取得
        portfolio = await self.portfolio_repository.find_by_id(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
        
        # シグナルをDTOに変換
        signal_dto = SignalDTO.from_entity(signal)
        
        # 推奨アクションを決定
        recommended_action = self._determine_action(signal, portfolio)
        
        # ポジションサイズを計算
        position_size = self._calculate_position_size(signal, portfolio)
        
        # リスク額を計算
        risk_amount = self._calculate_risk_amount(signal, position_size)
        
        # 期待リターンを計算
        expected_return = self._calculate_expected_return(signal, position_size)
        
        # 実行ノートを生成
        execution_notes = self._generate_execution_notes(signal, portfolio)
        
        return SignalResponseDTO(
            signal=signal_dto,
            recommended_action=recommended_action,
            position_size=position_size,
            risk_amount=risk_amount,
            expected_return=expected_return,
            execution_notes=execution_notes,
        )
    
    def _determine_action(self, signal: Signal, portfolio) -> str:
        """推奨アクションを決定"""
        if not signal.is_active:
            return "IGNORE"
        
        # 既存ポジションの確認
        existing_position = portfolio.positions.get(signal.stock.ticker)
        
        if signal.direction.value == "LONG":
            if existing_position and not existing_position.is_closed:
                if signal.strength > 70:
                    return "ADD_TO_POSITION"
                else:
                    return "HOLD"
            else:
                if signal.strength > 50 and signal.confidence.value > 60:
                    return "BUY"
                else:
                    return "WATCH"
        
        elif signal.direction.value == "SHORT":
            if existing_position and not existing_position.is_closed:
                if signal.strength > 70:
                    return "SELL_ALL"
                elif signal.strength > 50:
                    return "SELL_PARTIAL"
                else:
                    return "HOLD"
            else:
                return "AVOID"
        
        else:  # NEUTRAL
            if existing_position and not existing_position.is_closed:
                return "HOLD"
            else:
                return "WATCH"
    
    def _calculate_position_size(self, signal: Signal, portfolio) -> float:
        """ポジションサイズを計算"""
        # Kelly基準の簡易版
        confidence = float(signal.confidence.value) / 100
        strength = float(signal.strength) / 100
        
        # 基本ポジションサイズ（ポートフォリオの5-15%）
        base_size = float(portfolio.total_value) * 0.1
        
        # 信頼度と強度で調整
        adjusted_size = base_size * confidence * strength
        
        # 最大サイズ制限（ポートフォリオの20%）
        max_size = float(portfolio.total_value) * 0.2
        
        return min(adjusted_size, max_size)
    
    def _calculate_risk_amount(self, signal: Signal, position_size: float) -> float:
        """リスク額を計算"""
        if signal.stop_loss and signal.target_price:
            # ストップロスまでの距離から計算
            if signal.direction.value == "LONG":
                risk_per_share = float(signal.target_price.value - signal.stop_loss.value)
            else:
                risk_per_share = float(signal.stop_loss.value - signal.target_price.value)
            
            shares = position_size / float(signal.target_price.value)
            return abs(risk_per_share * shares)
        else:
            # デフォルトは3%のリスク
            return position_size * 0.03
    
    def _calculate_expected_return(self, signal: Signal, position_size: float) -> float:
        """期待リターンを計算"""
        if signal.risk_reward_ratio:
            risk_amount = self._calculate_risk_amount(signal, position_size)
            return risk_amount * float(signal.risk_reward_ratio)
        else:
            # デフォルトは10%のリターン
            return position_size * 0.1
    
    def _generate_execution_notes(self, signal: Signal, portfolio) -> List[str]:
        """実行ノートを生成"""
        notes = []
        
        # シグナルタイプ別のノート
        if signal.signal_type == SignalType.TECHNICAL:
            notes.append("Technical signal based on price patterns")
        elif signal.signal_type == SignalType.SENTIMENT:
            notes.append("Sentiment-driven signal from news analysis")
        elif signal.signal_type == SignalType.AI_PREDICTION:
            notes.append("AI model prediction with high confidence")
        
        # 時間軸のノート
        if signal.time_horizon == "short":
            notes.append("Short-term trade (1-5 days)")
        elif signal.time_horizon == "medium":
            notes.append("Medium-term position (1-4 weeks)")
        else:
            notes.append("Long-term investment (1+ months)")
        
        # リスク管理のノート
        if signal.stop_loss:
            notes.append(f"Set stop loss at {signal.stop_loss.value}")
        if signal.target_price:
            notes.append(f"Target price: {signal.target_price.value}")
        
        # ポートフォリオ状態のノート
        cash_ratio = float(portfolio.available_cash / portfolio.total_value)
        if cash_ratio < 0.1:
            notes.append("Warning: Low cash position")
        
        return notes


@dataclass
class GetActiveSignalsUseCase:
    """アクティブシグナル取得ユースケース"""
    
    signal_repository: SignalRepository
    
    async def execute(
        self,
        ticker: Optional[str] = None,
        signal_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[SignalDTO]:
        """アクティブなシグナルを取得
        
        Args:
            ticker: ティッカーシンボル（オプション）
            signal_type: シグナルタイプ（オプション）
            min_confidence: 最小信頼度
        
        Returns:
            シグナルDTOのリスト
        """
        # シグナルタイプの変換
        signal_type_enum = None
        if signal_type:
            try:
                signal_type_enum = SignalType[signal_type]
            except KeyError:
                raise ValueError(f"Invalid signal type: {signal_type}")
        
        # アクティブシグナルを取得
        signals = await self.signal_repository.find_active(ticker, signal_type_enum)
        
        # 信頼度でフィルタリング
        filtered_signals = [
            s for s in signals
            if s.confidence.value >= min_confidence
        ]
        
        # DTOに変換して返却
        return [SignalDTO.from_entity(signal) for signal in filtered_signals]


@dataclass
class ExpireOldSignalsUseCase:
    """古いシグナル期限切れ処理ユースケース"""
    
    signal_repository: SignalRepository
    
    async def execute(self, days_old: int = 7) -> int:
        """古いシグナルを期限切れにする
        
        Args:
            days_old: 何日前のシグナルを期限切れにするか
        
        Returns:
            期限切れにしたシグナル数
        """
        # 期限切れ日時を計算
        expire_before = datetime.now() - timedelta(days=days_old)
        
        # 古いシグナルを取得
        old_signals = await self.signal_repository.find_by_date_range(
            datetime.min, expire_before
        )
        
        # 期限切れにする
        expired_count = 0
        for signal in old_signals:
            if signal.is_active:
                success = await self.signal_repository.expire(signal.id)
                if success:
                    expired_count += 1
        
        return expired_count