"""PPOモデルローダー - 学習済みAIモデルの読み込みと予測実行"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import logging
from datetime import datetime

# train ディレクトリのパスを追加
train_dir = Path(__file__).parent.parent.parent / "train"
sys.path.append(str(train_dir))

from src.utils.logger_utils import create_dual_logger

try:
    from models.agents.ppo_agent import PPOTradingAgent
    from models.environment.trading_env import TradingEnvironment
    from stable_baselines3 import PPO
    HAS_PPO_MODULES = True
except ImportError as e:
    HAS_PPO_MODULES = False
    IMPORT_ERROR = e


logger = create_dual_logger(__name__, console_output=True)


class PPOModelLoader:
    """PPO学習済みモデルのローダーとプレディクター"""
    
    def __init__(self, model_path: str = "train/models/rl/ppo_nikkei_model_20250909_232115.zip"):
        """初期化
        
        Args:
            model_path: 学習済みモデルのパス
        """
        self.model_path = Path(model_path)
        self.agent: Optional[PPOTradingAgent] = None
        self.model: Optional[PPO] = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        
        # モデルの自動ロード
        self._load_model()
    
    def _load_model(self) -> bool:
        """学習済みモデルを読み込み
        
        Returns:
            読み込み成功フラグ
        """
        if not HAS_PPO_MODULES:
            self.load_error = f"PPOモジュールが利用できません: {IMPORT_ERROR}"
            logger.error(self.load_error)
            return False
        
        if not self.model_path.exists():
            self.load_error = f"モデルファイルが見つかりません: {self.model_path}"
            logger.error(self.load_error)
            return False
        
        try:
            # PPOモデルを直接読み込み
            logger.info(f"PPOモデル読み込み開始: {self.model_path}")
            self.model = PPO.load(str(self.model_path))
            logger.info("PPOモデル読み込み完了")
            
            # PPOTradingAgentは使用せず、直接PPOモデルを使用
            self.agent = None  # 不要
            
            self.is_loaded = True
            logger.info("PPOモデル統合完了")
            return True
            
        except Exception as e:
            self.load_error = f"PPOモデル読み込みエラー: {e}"
            logger.error(self.load_error)
            return False
    
    def predict_action(self, ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """株式の売買アクションを予測
        
        Args:
            ticker: ティッカーシンボル  
            market_data: 市場データ
            
        Returns:
            予測結果 {action: buy/sell/hold, confidence: float, raw_action: array}
        """
        if not self.is_loaded:
            return {
                "action": "hold",
                "confidence": 0.0,
                "error": self.load_error,
                "raw_action": None
            }
        
        try:
            # 市場データを観測値に変換
            observation = self._convert_to_observation(ticker, market_data)
            
            # PPOモデルで予測
            raw_action, _ = self.model.predict(observation, deterministic=True)
            
            # アクションを解釈
            interpreted_action = self._interpret_action(raw_action)
            
            logger.debug(f"AI予測完了 {ticker}: {interpreted_action['action']} (信頼度: {interpreted_action['confidence']:.2f})")
            
            return {
                "action": interpreted_action["action"],
                "confidence": interpreted_action["confidence"],
                "raw_action": raw_action.tolist() if hasattr(raw_action, 'tolist') else raw_action,
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"AI予測エラー {ticker}: {e}"
            logger.warning(error_msg)
            return {
                "action": "hold",
                "confidence": 0.0,
                "error": error_msg,
                "raw_action": None
            }
    
    def _convert_to_observation(self, ticker: str, market_data: Dict[str, Any]) -> np.ndarray:
        """市場データをPPOモデルの観測値形式に変換
        
        Args:
            ticker: ティッカーシンボル
            market_data: 市場データ
            
        Returns:
            PPOモデル用観測値
        """
        try:
            # PPOモデルが期待する観測値サイズ
            observation_size = 7956  # PPOモデルの期待する観測値サイズ
            
            # ダミー観測値（実装を段階的に進める）
            observation = np.zeros(observation_size, dtype=np.float32)
            
            # 可能な限り実際のデータを使用
            if 'price' in market_data:
                observation[0] = float(market_data['price'])
            if 'volume' in market_data:
                observation[1] = float(market_data['volume'])
            if 'change_pct' in market_data:
                observation[2] = float(market_data['change_pct'])
            
            # 正規化（簡易版）
            observation = np.clip(observation, -10.0, 10.0)
            
            return observation
            
        except Exception as e:
            logger.warning(f"観測値変換エラー {ticker}: {e}")
            # フォールバック: ゼロ観測値
            return np.zeros(7956, dtype=np.float32)
    
    def _interpret_action(self, raw_action: np.ndarray) -> Dict[str, Any]:
        """PPOの生アクションを売買シグナルに解釈
        
        Args:
            raw_action: PPOモデルの出力アクション
            
        Returns:
            解釈されたアクション
        """
        try:
            # PPOアクションが配列の場合
            if hasattr(raw_action, '__len__') and len(raw_action) > 0:
                action_value = float(raw_action[0])
            else:
                action_value = float(raw_action)
            
            # アクション値に基づく解釈（調整が必要）
            if action_value > 0.5:
                action = "buy"
                confidence = min(action_value, 1.0)
            elif action_value < -0.5:
                action = "sell" 
                confidence = min(abs(action_value), 1.0)
            else:
                action = "hold"
                confidence = 1.0 - abs(action_value)
            
            return {
                "action": action,
                "confidence": float(confidence),
                "raw_value": float(action_value)
            }
            
        except Exception as e:
            logger.warning(f"アクション解釈エラー: {e}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "raw_value": 0.0
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得
        
        Returns:
            モデル情報
        """
        return {
            "model_path": str(self.model_path),
            "is_loaded": self.is_loaded,
            "load_error": self.load_error,
            "has_ppo_modules": HAS_PPO_MODULES,
            "model_exists": self.model_path.exists() if self.model_path else False
        }
    
    def reload_model(self, model_path: Optional[str] = None) -> bool:
        """モデルを再読み込み
        
        Args:
            model_path: 新しいモデルパス（Noneの場合は現在のパスを使用）
            
        Returns:
            再読み込み成功フラグ
        """
        if model_path:
            self.model_path = Path(model_path)
        
        self.agent = None
        self.model = None
        self.is_loaded = False
        self.load_error = None
        
        return self._load_model()