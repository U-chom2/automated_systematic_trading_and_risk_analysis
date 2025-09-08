"""AI分析ユースケース"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import yfinance as yf

from ..dto.signal_dto import SignalDTO, SignalBatchDTO
from ...domain.entities.signal import Signal
from ...domain.services.signal_generator import SignalGenerator
from ...infrastructure.ai_models.ppo_integration import PPOModelIntegration
from ...infrastructure.external_apis.yahoo_finance import YahooFinanceClient
from ...common.logging import get_logger


logger = get_logger(__name__)


class AIAnalysisUseCase:
    """AI分析ユースケース"""
    
    def __init__(
        self,
        signal_generator: SignalGenerator,
        ppo_model: Optional[PPOModelIntegration] = None,
        yahoo_client: Optional[YahooFinanceClient] = None
    ):
        """初期化
        
        Args:
            signal_generator: シグナルジェネレーター
            ppo_model: PPOモデル統合
            yahoo_client: Yahoo Financeクライアント
        """
        self.signal_generator = signal_generator
        self.ppo_model = ppo_model or PPOModelIntegration()
        self.yahoo_client = yahoo_client or YahooFinanceClient()
    
    async def analyze_stocks(
        self,
        tickers: List[str],
        lookback_days: int = 30,
        use_ai: bool = True,
        use_technical: bool = True
    ) -> SignalBatchDTO:
        """複数銘柄を分析
        
        Args:
            tickers: ティッカーリスト
            lookback_days: 過去データ参照期間
            use_ai: AIモデルを使用するか
            use_technical: テクニカル分析を使用するか
        
        Returns:
            シグナルバッチDTO
        """
        signals = []
        
        # 日経225データを取得
        nikkei_data = None
        if use_ai:
            try:
                nikkei = yf.Ticker("^N225")
                nikkei_data = nikkei.history(period=f"{lookback_days}d")
                logger.info("Nikkei 225 data fetched for AI analysis")
            except Exception as e:
                logger.warning(f"Failed to fetch Nikkei data: {e}")
        
        for ticker in tickers:
            try:
                # 市場データを取得
                market_data = await self._fetch_market_data(ticker, lookback_days)
                
                if market_data is None or market_data.empty:
                    logger.warning(f"No data available for {ticker}")
                    continue
                
                # AIシグナル生成
                ai_signal = None
                if use_ai and self.ppo_model:
                    ai_signal = self.ppo_model.predict(
                        market_data,
                        ticker,
                        nikkei_data
                    )
                    signals.append(SignalDTO.from_entity(ai_signal))
                
                # テクニカルシグナル生成
                if use_technical:
                    technical_signals = self.signal_generator.generate_technical_signals(
                        ticker, market_data
                    )
                    for signal in technical_signals:
                        signals.append(SignalDTO.from_entity(signal))
                
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
        
        return SignalBatchDTO(
            signals=signals,
            generated_at=datetime.now(),
            metadata={
                "tickers": tickers,
                "lookback_days": lookback_days,
                "use_ai": use_ai,
                "use_technical": use_technical
            }
        )
    
    async def get_recommendations(
        self,
        num_stocks: int = 5,
        min_confidence: float = 0.6
    ) -> List[Dict]:
        """推奨銘柄を取得
        
        Args:
            num_stocks: 推奨銘柄数
            min_confidence: 最小信頼度
        
        Returns:
            推奨銘柄リスト
        """
        # 主要銘柄リスト
        major_stocks = [
            "7203.T",  # トヨタ
            "9984.T",  # ソフトバンクG
            "6758.T",  # ソニー
            "8306.T",  # 三菱UFJ
            "9432.T",  # NTT
            "6861.T",  # キーエンス
            "4063.T",  # 信越化学
            "7267.T",  # ホンダ
            "8058.T",  # 三菱商事
            "7974.T",  # 任天堂
            "4661.T",  # オリエンタルランド
            "6098.T",  # リクルート
            "3382.T",  # セブン&アイ
            "6501.T",  # 日立
        ]
        
        # 各銘柄を分析
        batch_result = await self.analyze_stocks(
            major_stocks,
            lookback_days=30,
            use_ai=True,
            use_technical=True
        )
        
        # スコアリング
        stock_scores = {}
        for signal in batch_result.signals:
            if signal.signal_type == "BUY" and signal.confidence >= min_confidence:
                ticker = signal.ticker
                if ticker not in stock_scores:
                    stock_scores[ticker] = {
                        "ticker": ticker,
                        "total_score": 0,
                        "signals": [],
                        "max_confidence": 0
                    }
                
                # スコアを加算
                score = signal.strength * signal.confidence * 100
                stock_scores[ticker]["total_score"] += score
                stock_scores[ticker]["signals"].append({
                    "source": signal.source,
                    "confidence": signal.confidence,
                    "strength": signal.strength
                })
                stock_scores[ticker]["max_confidence"] = max(
                    stock_scores[ticker]["max_confidence"],
                    signal.confidence
                )
        
        # スコアでソート
        recommendations = sorted(
            stock_scores.values(),
            key=lambda x: x["total_score"],
            reverse=True
        )[:num_stocks]
        
        # 追加情報を付与
        for rec in recommendations:
            try:
                stock = yf.Ticker(rec["ticker"])
                info = stock.info
                rec["company_name"] = info.get("longName", rec["ticker"])
                rec["current_price"] = info.get("currentPrice", 0)
                rec["market_cap"] = info.get("marketCap", 0)
                rec["per"] = info.get("trailingPE", 0)
                
                # 直近のパフォーマンス
                hist = stock.history(period="5d")
                if not hist.empty:
                    rec["5d_return"] = (
                        (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / 
                        hist['Close'].iloc[0] * 100
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch info for {rec['ticker']}: {e}")
        
        return recommendations
    
    async def _fetch_market_data(
        self,
        ticker: str,
        lookback_days: int
    ) -> Optional[pd.DataFrame]:
        """市場データを取得
        
        Args:
            ticker: ティッカーシンボル
            lookback_days: 過去データ参照期間
        
        Returns:
            市場データ
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 10)  # 余裕を持って取得
            
            # Yahoo Financeから取得
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            # テクニカル指標を追加
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch market data for {ticker}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標を追加
        
        Args:
            df: 市場データ
        
        Returns:
            指標追加済みデータ
        """
        # 移動平均
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_60'] = df['Close'].rolling(window=60).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # ボリンジャーバンド
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        return df


class RealtimeAnalysisUseCase:
    """リアルタイム分析ユースケース"""
    
    def __init__(
        self,
        ai_analysis: AIAnalysisUseCase,
        update_interval: int = 60  # seconds
    ):
        """初期化
        
        Args:
            ai_analysis: AI分析ユースケース
            update_interval: 更新間隔（秒）
        """
        self.ai_analysis = ai_analysis
        self.update_interval = update_interval
        self.logger = get_logger(__name__)
    
    async def start_monitoring(
        self,
        tickers: List[str],
        callback: Optional[callable] = None
    ):
        """リアルタイム監視を開始
        
        Args:
            tickers: 監視対象ティッカー
            callback: シグナル発生時のコールバック
        """
        import asyncio
        
        self.logger.info(f"Starting real-time monitoring for {tickers}")
        
        while True:
            try:
                # 分析実行
                signals = await self.ai_analysis.analyze_stocks(
                    tickers,
                    lookback_days=30,
                    use_ai=True,
                    use_technical=True
                )
                
                # 強いシグナルを検出
                strong_signals = [
                    s for s in signals.signals
                    if s.signal_type in ["BUY", "SELL"] and s.confidence > 0.7
                ]
                
                if strong_signals and callback:
                    await callback(strong_signals)
                
                # 次の更新まで待機
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)  # エラー時は短い待機