"""PPO強化学習モデル統合"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from decimal import Decimal

# 既存のtrainディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "train"))

try:
    from stable_baselines3 import PPO
    from models.trading_model import TradingDecisionModel, MarketData
except ImportError:
    PPO = None
    TradingDecisionModel = None
    MarketData = None

from ...domain.entities.signal import Signal
from ...domain.value_objects.signal_strength import SignalStrength
from ...common.logging import get_logger


logger = get_logger(__name__)


class PPOModelIntegration:
    """PPOモデル統合クラス"""
    
    def __init__(self, model_path: Optional[str] = None):
        """初期化
        
        Args:
            model_path: モデルファイルのパス
        """
        self.model_path = model_path
        self.model = None
        self.trading_model = None
        self._load_model()
    
    def _load_model(self):
        """モデルを読み込む"""
        if not self.model_path:
            # デフォルトのモデルパスを探す
            model_dir = Path(__file__).parent.parent.parent.parent / "train" / "models" / "rl"
            models = list(model_dir.glob("ppo_nikkei_model_*.zip"))
            if models:
                # 最新のモデルを使用
                self.model_path = str(sorted(models)[-1])
                logger.info(f"Using latest model: {self.model_path}")
        
        if self.model_path and Path(self.model_path).exists():
            try:
                if PPO:
                    self.model = PPO.load(self.model_path)
                    logger.info(f"PPO model loaded from {self.model_path}")
                
                if TradingDecisionModel:
                    self.trading_model = TradingDecisionModel()
                    logger.info("Trading decision model initialized")
                    
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
    
    def predict(
        self,
        market_data: pd.DataFrame,
        ticker: str,
        nikkei_data: Optional[pd.DataFrame] = None,
        ir_news: Optional[List[str]] = None
    ) -> Signal:
        """予測を実行してシグナルを生成
        
        Args:
            market_data: 市場データ（OHLCV）
            ticker: ティッカーシンボル
            nikkei_data: 日経225データ
            ir_news: IRニュース
        
        Returns:
            生成されたシグナル
        """
        try:
            # データ準備
            if len(market_data) < 30:
                logger.warning("Insufficient data for prediction")
                return self._create_neutral_signal(ticker)
            
            # 特徴量を抽出
            features = self._extract_features(market_data, nikkei_data)
            
            # PPOモデルで予測
            if self.model:
                action, _ = self.model.predict(features, deterministic=True)
                confidence = self._calculate_confidence(features, action)
            else:
                # モデルがない場合はニュートラル
                action = 0
                confidence = 0.5
            
            # TradingDecisionModelでも予測（利用可能な場合）
            if self.trading_model and MarketData:
                try:
                    trading_data = self._prepare_trading_data(
                        market_data, nikkei_data, ir_news
                    )
                    trading_decision = self.trading_model.forward(trading_data)
                    
                    # 両モデルの結果を統合
                    action = (action + trading_decision['action']) / 2
                    confidence = (confidence + trading_decision['confidence']) / 2
                except Exception as e:
                    logger.warning(f"Trading model prediction failed: {e}")
            
            # シグナルを生成
            return self._create_signal(ticker, action, confidence)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._create_neutral_signal(ticker)
    
    def _extract_features(
        self,
        market_data: pd.DataFrame,
        nikkei_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """特徴量を抽出
        
        Args:
            market_data: 市場データ
            nikkei_data: 日経225データ
        
        Returns:
            特徴量ベクトル
        """
        features = []
        
        # 価格データの正規化
        latest_30d = market_data.tail(30)
        
        # リターンを計算
        returns = latest_30d['Close'].pct_change().fillna(0)
        features.extend(returns.values)
        
        # ボラティリティ
        volatility = returns.rolling(window=5).std().fillna(0)
        features.extend(volatility.values)
        
        # 出来高の変化率
        volume_change = latest_30d['Volume'].pct_change().fillna(0)
        features.extend(volume_change.values)
        
        # 日経225との相関（データがある場合）
        if nikkei_data is not None and len(nikkei_data) >= 30:
            nikkei_returns = nikkei_data.tail(30)['Close'].pct_change().fillna(0)
            correlation = returns.rolling(window=10).corr(nikkei_returns).fillna(0)
            features.extend(correlation.values)
        
        return np.array(features, dtype=np.float32)
    
    def _prepare_trading_data(
        self,
        market_data: pd.DataFrame,
        nikkei_data: Optional[pd.DataFrame],
        ir_news: Optional[List[str]]
    ) -> 'MarketData':
        """TradingDecisionModel用のデータを準備"""
        latest_30d = market_data.tail(30)
        
        target_high = latest_30d['High'].values
        target_low = latest_30d['Low'].values
        target_close = latest_30d['Close'].values
        
        if nikkei_data is not None and len(nikkei_data) >= 30:
            nikkei_30d = nikkei_data.tail(30)
            nikkei_high = nikkei_30d['High'].values
            nikkei_low = nikkei_30d['Low'].values
            nikkei_close = nikkei_30d['Close'].values
        else:
            # ダミーデータ
            nikkei_high = np.ones(30) * 40000
            nikkei_low = np.ones(30) * 39000
            nikkei_close = np.ones(30) * 39500
        
        if not ir_news:
            ir_news = ["市場は安定しています"]
        
        return MarketData(
            nikkei_high=nikkei_high,
            nikkei_low=nikkei_low,
            nikkei_close=nikkei_close,
            target_high=target_high,
            target_low=target_low,
            target_close=target_close,
            ir_news=ir_news
        )
    
    def _calculate_confidence(
        self,
        features: np.ndarray,
        action: float
    ) -> float:
        """信頼度を計算
        
        Args:
            features: 特徴量
            action: アクション値
        
        Returns:
            信頼度（0-1）
        """
        # 簡易的な信頼度計算
        # アクションの絶対値が大きいほど信頼度が高い
        base_confidence = min(abs(action), 1.0)
        
        # 特徴量のボラティリティを考慮
        feature_volatility = np.std(features)
        if feature_volatility > 0.5:
            # ボラティリティが高い場合は信頼度を下げる
            base_confidence *= 0.8
        
        return max(0.3, min(base_confidence, 0.95))
    
    def _create_signal(
        self,
        ticker: str,
        action: float,
        confidence: float
    ) -> Signal:
        """シグナルを作成
        
        Args:
            ticker: ティッカーシンボル
            action: アクション値（-1 to 1）
            confidence: 信頼度（0 to 1）
        
        Returns:
            シグナルエンティティ
        """
        # アクションから売買タイプを決定
        if action > 0.3:
            signal_type = "BUY"
            strength = min(action, 1.0)
        elif action < -0.3:
            signal_type = "SELL"
            strength = min(abs(action), 1.0)
        else:
            signal_type = "HOLD"
            strength = 0.5
        
        return Signal(
            ticker=ticker,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            source="PPO_AI_MODEL",
            metadata={
                "model_path": self.model_path,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _create_neutral_signal(self, ticker: str) -> Signal:
        """ニュートラルシグナルを作成"""
        return Signal(
            ticker=ticker,
            signal_type="HOLD",
            strength=0.5,
            confidence=0.3,
            source="PPO_AI_MODEL",
            metadata={
                "status": "neutral",
                "reason": "insufficient_data_or_model_unavailable"
            }
        )


class PPOBacktestIntegration:
    """PPOモデルのバックテスト統合"""
    
    def __init__(self, model_integration: PPOModelIntegration):
        """初期化
        
        Args:
            model_integration: PPOモデル統合インスタンス
        """
        self.model = model_integration
        self.logger = get_logger(__name__)
    
    def backtest(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal
    ) -> Dict:
        """バックテストを実行
        
        Args:
            tickers: ティッカーリスト
            start_date: 開始日
            end_date: 終了日
            initial_capital: 初期資金
        
        Returns:
            バックテスト結果
        """
        results = {
            "initial_capital": float(initial_capital),
            "final_capital": float(initial_capital),
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "trades": [],
            "daily_returns": []
        }
        
        try:
            # ここでバックテストロジックを実装
            # 現在は簡易版
            
            current_capital = initial_capital
            positions = {}
            
            # 各銘柄でシグナルを生成してトレード
            for ticker in tickers:
                # ダミーの市場データ（実際はyfinanceなどから取得）
                market_data = pd.DataFrame({
                    'Close': np.random.randn(60) * 100 + 1000,
                    'High': np.random.randn(60) * 100 + 1100,
                    'Low': np.random.randn(60) * 100 + 900,
                    'Volume': np.random.randint(100000, 1000000, 60)
                })
                
                # シグナル生成
                signal = self.model.predict(market_data, ticker)
                
                # トレード実行（簡易版）
                if signal.signal_type == "BUY" and ticker not in positions:
                    trade_amount = current_capital * Decimal(str(signal.strength * 0.1))
                    positions[ticker] = {
                        "quantity": 100,
                        "entry_price": market_data['Close'].iloc[-1],
                        "amount": trade_amount
                    }
                    current_capital -= trade_amount
                    
                    results["trades"].append({
                        "ticker": ticker,
                        "type": "BUY",
                        "price": float(market_data['Close'].iloc[-1]),
                        "quantity": 100,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif signal.signal_type == "SELL" and ticker in positions:
                    position = positions[ticker]
                    exit_price = market_data['Close'].iloc[-1]
                    profit = (exit_price - position["entry_price"]) * position["quantity"]
                    current_capital += position["amount"] + Decimal(str(profit))
                    
                    results["trades"].append({
                        "ticker": ticker,
                        "type": "SELL",
                        "price": float(exit_price),
                        "quantity": position["quantity"],
                        "profit": float(profit),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    del positions[ticker]
            
            # 最終資本を計算
            results["final_capital"] = float(current_capital)
            results["total_return"] = (
                (float(current_capital) - float(initial_capital)) / 
                float(initial_capital) * 100
            )
            
            self.logger.info(f"Backtest completed: {results['total_return']:.2f}% return")
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
        
        return results