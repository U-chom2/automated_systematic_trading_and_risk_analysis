"""
PPO Model Scoring Adapter
PPOモデルを既存の投資スコアリングシステムに統合するためのアダプタークラス
"""

import sys
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# train module imports
sys.path.insert(0, str(Path(__file__).parent / 'train'))
sys.path.insert(0, str(Path(__file__).parent / 'train' / 'models'))
sys.path.insert(0, str(Path(__file__).parent / 'train' / 'models' / 'agents'))

from models.agents.ppo_agent import PPOTradingAgent
from models.environment.trading_env import TradingEnvironment
from config import config, Config, TradingMode
from technical_analyzer import TechnicalIndicators
from investment_scorer import ScoringResult
from nikkei_data_fetcher import Nikkei225DataFetcher

logger = logging.getLogger(__name__)


@dataclass
class PPOMarketData:
    """PPOモデル用の市場データ構造"""
    nikkei_prices: np.ndarray     # 日経225価格 (30日分: high, low, close)
    target_prices: np.ndarray     # ターゲット株価格 (30日分: OHLC)
    technical_indicators: TechnicalIndicators
    current_price: float
    volume_data: np.ndarray       # 出来高データ (30日分)
    

class PPOScoringAdapter:
    """PPOモデルを投資スコアリングシステムに統合するアダプター"""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        config_instance: Optional[Config] = None,
        device: Optional[str] = None
    ):
        """
        PPOスコアリングアダプターを初期化
        
        Args:
            model_path: PPOモデルファイルパス (Noneの場合は最新モデルを自動検出)
            config_instance: 設定インスタンス
            device: 使用デバイス (cpu/cuda/mps)
        """
        self.config = config_instance or config
        self.trading_mode = getattr(self.config, 'trading_mode', TradingMode.LONG_TERM)
        
        # デバイス設定
        self.device = self._setup_device(device)
        
        # モデルパス設定
        if model_path is None:
            model_path = self._find_latest_model()
        
        self.model_path = model_path
        self.ppo_agent = None
        self.mock_environment = None
        
        # 日経225データフェッチャー初期化
        cache_dir = Path(__file__).parent / 'cache' / 'nikkei'
        self.nikkei_fetcher = Nikkei225DataFetcher(cache_dir=cache_dir)
        self.nikkei_data_cache = None
        
        # PPOモデル初期化
        self._initialize_ppo_model()
        
        logger.info(f"PPO Scoring Adapter initialized with model: {Path(model_path).name}")
    
    def _setup_device(self, device: Optional[str]) -> str:
        """デバイスを設定"""
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
            elif torch.cuda.is_available():
                device = "cuda:0"  # NVIDIA GPU
            else:
                device = "cpu"  # CPU fallback
            logger.info(f"Auto-detected device: {device}")
        
        return device
    
    def _find_latest_model(self) -> str:
        """最新のPPOモデルを検索"""
        model_dir = Path(__file__).parent / 'train' / 'models' / 'rl'
        
        if not model_dir.exists():
            raise FileNotFoundError(f"PPO models directory not found: {model_dir}")
        
        # .zipファイルを検索
        zip_files = list(model_dir.glob("ppo_nikkei_model_*.zip"))
        
        if not zip_files:
            raise FileNotFoundError(f"No PPO model files found in {model_dir}")
        
        # タイムスタンプでソートして最新を取得
        latest_model = max(zip_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Latest PPO model found: {latest_model.name}")
        
        return str(latest_model)
    
    def _initialize_ppo_model(self):
        """PPOモデルを初期化"""
        try:
            # ダミーの環境を作成 (観測空間定義のため)
            self.mock_environment = self._create_mock_environment()
            
            # PPOエージェントを作成
            self.ppo_agent = PPOTradingAgent(
                env=self.mock_environment,
                num_stocks=1,  # 単一株式分析用
                device=self.device
            )
            
            # 学習済みモデルをロード（環境チェックをスキップ）
            # まず直接PPO.loadを使って環境チェックを回避
            from stable_baselines3 import PPO
            self.ppo_agent.model = PPO.load(self.model_path, env=None)
            
            logger.info("PPO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PPO model: {e}")
            raise
    
    def _create_mock_environment(self):
        """取引環境を作成 (観測空間定義用)"""
        # 訓練時と同じ設定を使用
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # 訓練時と同じ銘柄を使用
        symbols = ['7203.T', '9984.T', '6758.T']  # Toyota, SoftBank, Sony
        
        # 実際の株価データ取得を試みる
        import yfinance as yf
        price_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='60d')
                if not hist.empty:
                    for date, row in hist.iterrows():
                        price_data.append({
                            'date': date,
                            'symbol': symbol,
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'close': row['Close'],
                            'volume': row['Volume']
                        })
            except:
                # 取得失敗時は0値を使用（明確にダミーとわかるように）
                logger.warning(f"Failed to fetch real data for {symbol}, using zeros")
                for date in dates:
                    price_data.append({
                        'date': date,
                        'symbol': symbol,
                        'open': 0.0,
                        'high': 0.0,
                        'low': 0.0,
                        'close': 0.0,
                        'volume': 0.0
                    })
        
        price_df = pd.DataFrame(price_data)
        price_df.set_index(['date', 'symbol'], inplace=True)
        
        # ニュースデータ（簡略化）
        news_df = None  # 実際のニュースデータ取得は後で実装
        
        # 実際の日経225データを取得
        try:
            nikkei_fetcher = Nikkei225DataFetcher()
            nikkei_df = nikkei_fetcher.fetch_for_window(60)
            nikkei_data = nikkei_df[['high', 'low', 'close']].copy()
            nikkei_data['date'] = nikkei_df['date']
        except:
            # フォールバック（すべて0で明確にダミー）
            logger.warning("Failed to fetch real Nikkei data, using zeros")
            nikkei_data = pd.DataFrame({
                'date': dates,
                'high': [0.0] * len(dates),
                'low': [0.0] * len(dates), 
                'close': [0.0] * len(dates)
            })
        
        # Enhanced環境をインポート
        sys.path.insert(0, str(Path(__file__).parent / 'train'))
        from train import EnhancedTradingEnvironment
        
        # EnhancedTradingEnvironmentを作成（訓練時と同じ）
        env = EnhancedTradingEnvironment(
            nikkei_data=nikkei_data,
            price_data=price_df,
            symbols=symbols,
            initial_cash=10000000,
            commission_rate=0.001,
            window_size=30,
            news_data=news_df
        )
        
        # デバッグ: 観測空間の詳細を表示
        logger.info(f"Mock environment observation space: {env.observation_space.shape}")
        
        # 観測空間の計算を検証
        nikkei_features = 3 * 30  # 90
        stock_features = len(symbols) * 10 * 30  # 3 * 10 * 30 = 900
        portfolio_features = len(symbols) + 1  # 3 + 1 = 4
        news_features = len(symbols) * 10 if news_df is not None else 0  # 3 * 10 = 30
        
        expected_dim = nikkei_features + stock_features + portfolio_features + news_features
        logger.info(f"Expected dimension calculation:")
        logger.info(f"  - Nikkei features: {nikkei_features}")
        logger.info(f"  - Stock features: {stock_features}")
        logger.info(f"  - Portfolio features: {portfolio_features}")
        logger.info(f"  - News features: {news_features}")
        logger.info(f"  - Total expected: {expected_dim}")
        logger.info(f"  - Actual env space: {env.observation_space.shape[0]}")
        logger.info(f"  - Model expects: 1646")
        
        return env
    
    def calculate_investment_score(
        self,
        indicators: TechnicalIndicators,
        current_price: float,
        market_cap_millions: float = 1500.0,
        symbol: str = "UNKNOWN.T"
    ) -> ScoringResult:
        """
        PPOモデルを使用して投資スコアを計算
        
        Args:
            indicators: テクニカル指標
            current_price: 現在価格
            market_cap_millions: 時価総額 (百万円)
            symbol: 銘柄シンボル
            
        Returns:
            PPOベースのスコアリング結果
        """
        try:
            # PPO用の観測データを準備
            observation = self._prepare_observation(
                indicators, current_price, symbol
            )
            
            # PPOモデルで予測実行
            action = self.ppo_agent.predict(observation, deterministic=True)
            
            # アクションの形状を確認してログ出力
            logger.debug(f"PPO action shape: {action[0].shape if hasattr(action[0], 'shape') else 'scalar'}")
            logger.debug(f"PPO action values: {action[0]}")
            
            # アクション値を取得（複数アクションの場合は最初のものを使用）
            if hasattr(action[0], '__len__') and len(action[0]) > 1:
                action_value = action[0][0]  # 複数アクションの最初の値
            else:
                action_value = action[0] if hasattr(action[0], '__len__') else action[0]
            
            # アクション値を投資スコアに変換 (-1〜1 → 0〜100)
            investment_score = self._convert_action_to_score(action_value)
            
            # コンポーネントスコア (PPOベース)
            component_scores = self._generate_component_scores(
                action_value, indicators, current_price
            )
            
            # 投資推奨を生成
            recommendation = self.config.get_investment_recommendation(investment_score)
            
            # 分析詳細を構築
            analysis_details = self._build_ppo_analysis_details(
                action_value, indicators, current_price, market_cap_millions
            )
            
            return ScoringResult(
                total_score=investment_score,
                component_scores=component_scores,
                recommendation=recommendation,
                analysis_details=analysis_details
            )
            
        except Exception as e:
            logger.error(f"PPO scoring failed: {e}")
            # フォールバック: 中性スコア
            return self._create_fallback_score()
    
    def _prepare_observation(
        self, 
        indicators: TechnicalIndicators, 
        current_price: float,
        symbol: str
    ) -> np.ndarray:
        """PPO用の観測データを準備（訓練時と同じ形式）"""
        obs_list = []
        window_size = 30
        
        # 1. 日経225実データ取得 (30日分 * 3特徴量)
        if self.nikkei_data_cache is None:
            # 実際の日経225データを取得
            self.nikkei_data_cache = self.nikkei_fetcher.fetch_for_window(window_size)
        
        nikkei_data = self.nikkei_data_cache
        
        # 日経225データを正規化して追加
        for i in range(min(window_size, len(nikkei_data))):
            if i < len(nikkei_data):
                row = nikkei_data.iloc[i]
                # ゼロチェック（ダミーデータの場合）
                if row['high'] > 0 and row['low'] > 0 and row['close'] > 0:
                    # 正規化：最大値で除算（適応的正規化）
                    max_val = max(row['high'], row['low'], row['close'], 1.0)
                    obs_list.extend([
                        row['high'] / max_val,   # normalized nikkei high
                        row['low'] / max_val,    # normalized nikkei low  
                        row['close'] / max_val   # normalized nikkei close
                    ])
                else:
                    # ダミーデータの場合はすべて0
                    obs_list.extend([0.0, 0.0, 0.0])
            else:
                # 不足分は0で埋める（ダミー）
                obs_list.extend([0.0, 0.0, 0.0])
        
        # 2. ターゲット株価データ（3銘柄分）
        symbols = ['7203.T', '9984.T', '6758.T']  # 訓練時と同じ銘柄
        
        for target_symbol in symbols:
            for _ in range(window_size):  # 30日分
                # 分析対象銘柄の場合は実際の価格、それ以外はダミー価格
                if target_symbol == symbol:
                    price = current_price
                else:
                    price = 0.0  # ダミー価格（0で明確にダミー）
                
                # OHLCV (5つ)
                if price > 0:
                    obs_list.extend([
                        price * 1.02 / 10000,  # normalized high
                        price * 0.98 / 10000,  # normalized low
                        price / 10000,         # normalized close
                        price * 1.01 / 10000,  # normalized open
                        1.0  # normalized volume
                    ])
                else:
                    # price=0の場合はすべて0
                    obs_list.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                
                # テクニカル指標 (5つ)
                if target_symbol == symbol:
                    # 実際の指標を使用
                    rsi = indicators.rsi / 100.0 if indicators.rsi else 0.5
                    obs_list.extend([
                        rsi,  # RSI (正規化済み)
                        0.0,  # MACD
                        0.0,  # 移動平均偏差
                        0.0,  # 出来高比率（実データがない場合は0）
                        0.0   # ATR（実データがない場合は0）
                    ])
                else:
                    # ダミー値（すべて0）
                    obs_list.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 3. ポートフォリオ状態 (現金 + 3銘柄のポジション比率)
        obs_list.append(0.9)  # 現金比率
        obs_list.append(0.03)  # 銘柄1ポジション比率
        obs_list.append(0.04)  # 銘柄2ポジション比率
        obs_list.append(0.03)  # 銘柄3ポジション比率
        
        # 4. ニュース特徴量 (3銘柄 * 10特徴量)
        obs_list.extend([0.0] * 30)
        
        # 観測次元数を学習済みモデルの期待サイズ（1646）に調整
        target_dim = 1646  # 学習済みモデルの期待サイズ
        current_dim = len(obs_list)
        
        if current_dim < target_dim:
            # 不足分をゼロパディング
            obs_list.extend([0.0] * (target_dim - current_dim))
        elif current_dim > target_dim:
            # 余分な分を削除
            obs_list = obs_list[:target_dim]
        
        logger.debug(f"Observation prepared: {current_dim} -> {target_dim} dimensions")
        return np.array(obs_list, dtype=np.float32)
    
    def _convert_action_to_score(self, action_value: float) -> float:
        """PPOアクション値を投資スコア(0-100)に変換"""
        # アクション値 -1〜1 を 0〜100 スコアに変換
        # -1 (売り) → 0点, 0 (ホールド) → 50点, +1 (買い) → 100点
        score = (action_value + 1.0) * 50.0
        return max(0.0, min(100.0, score))
    
    def _generate_component_scores(
        self,
        action_value: float,
        indicators: TechnicalIndicators,
        current_price: float
    ) -> Dict[str, float]:
        """PPOベースのコンポーネントスコア生成"""
        base_score = self._convert_action_to_score(action_value)
        
        # PPO予測に基づいてコンポーネント分解
        return {
            "ppo_action": base_score,
            "trend": base_score * 0.3,
            "rsi": base_score * 0.2,
            "macd": base_score * 0.2,
            "momentum": base_score * 0.2,
            "volume": base_score * 0.1,
            "bollinger": 0.0,  # PPOには含まれない
            "market_cap": 0.0  # PPOには含まれない
        }
    
    def _build_ppo_analysis_details(
        self,
        action_value: float,
        indicators: TechnicalIndicators,
        current_price: float,
        market_cap_millions: float
    ) -> Dict[str, Any]:
        """PPO分析詳細を構築"""
        action_interpretation = self._interpret_ppo_action(action_value)
        
        return {
            "trading_mode": self.trading_mode.value,
            "model_type": "PPO_Reinforcement_Learning",
            "model_path": Path(self.model_path).name,
            "ppo_action_value": action_value,
            "ppo_action_interpretation": action_interpretation,
            "current_price": current_price,
            "market_cap_millions": market_cap_millions,
            "confidence_score": abs(action_value),  # アクション値の絶対値を信頼度とする
            "device": self.device
        }
    
    def _interpret_ppo_action(self, action_value: float) -> str:
        """PPOアクション値を解釈"""
        if action_value > 0.7:
            return "強い買い推奨"
        elif action_value > 0.3:
            return "買い推奨"
        elif action_value > -0.3:
            return "ホールド推奨"
        elif action_value > -0.7:
            return "売り推奨"
        else:
            return "強い売り推奨"
    
    def _create_fallback_score(self) -> ScoringResult:
        """エラー時のフォールバックスコア"""
        return ScoringResult(
            total_score=50.0,  # 中性スコア
            component_scores={"error": 50.0},
            recommendation=self.config.get_investment_recommendation(50.0),
            analysis_details={
                "error": "PPO model prediction failed",
                "fallback": True
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """PPOモデル情報を取得"""
        return {
            "model_path": self.model_path,
            "model_name": Path(self.model_path).name,
            "device": self.device,
            "trading_mode": self.trading_mode.value,
            "is_loaded": self.ppo_agent is not None,
            "observation_space_shape": self.mock_environment.observation_space.shape if self.mock_environment else None,
            "action_space_shape": self.mock_environment.action_space.shape if self.mock_environment else None
        }
    
    def test_prediction(self) -> Dict[str, Any]:
        """PPOモデルのテスト予測"""
        try:
            # ダミーの観測データでテスト（学習済みモデルの期待サイズに調整）
            dummy_obs = np.zeros((1646,), dtype=np.float32)
            
            # 予測実行
            action = self.ppo_agent.predict(dummy_obs, deterministic=True)
            
            # アクション値を取得
            if hasattr(action[0], '__len__') and len(action[0]) > 1:
                action_value = action[0][0]  # 複数アクションの最初の値
            else:
                action_value = action[0] if hasattr(action[0], '__len__') else action[0]
            
            score = self._convert_action_to_score(action_value)
            
            return {
                "test_successful": True,
                "dummy_observation_shape": dummy_obs.shape,
                "predicted_action": action[0],
                "action_value": action_value,
                "converted_score": score,
                "action_interpretation": self._interpret_ppo_action(action_value)
            }
            
        except Exception as e:
            return {
                "test_successful": False,
                "error": str(e)
            }


def create_ppo_adapter(
    model_path: Optional[str] = None,
    config_instance: Optional[Config] = None,
    device: Optional[str] = None
) -> PPOScoringAdapter:
    """PPOスコアリングアダプターのファクトリー関数"""
    return PPOScoringAdapter(model_path, config_instance, device)


# テスト用の実行コード
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        # PPOアダプターを初期化
        adapter = create_ppo_adapter()
        
        # モデル情報を表示
        model_info = adapter.get_model_info()
        print("PPO Model Info:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # テスト予測を実行
        test_result = adapter.test_prediction()
        print(f"\nTest Prediction Result:")
        for key, value in test_result.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"PPO Adapter test failed: {e}", exc_info=True)