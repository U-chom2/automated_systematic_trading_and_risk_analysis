"""
統合型株式売買AIモデル

日経225を市場指標として使用し、ターゲット企業の売買判断を行う
方針.mdに基づいた実装（LSTM/Transformer + ModernBERT + PPO）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """市場データ構造"""
    nikkei_high: np.ndarray      # 日経225高値 (30日分)
    nikkei_low: np.ndarray       # 日経225安値 (30日分)
    nikkei_close: np.ndarray     # 日経225終値 (30日分)
    target_high: np.ndarray      # ターゲット株高値 (30日分)
    target_low: np.ndarray       # ターゲット株安値 (30日分)
    target_close: np.ndarray     # ターゲット株終値 (30日分)
    ir_news: List[str]           # IR情報テキスト


class MarketEncoder(nn.Module):
    """市場データエンコーダー（LSTM）"""
    
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LSTMエンコーディング"""
        output, (hidden, cell) = self.lstm(x)
        return output[:, -1, :]  # 最後の時刻の特徴


# ModernBERTエンコーダーをインポート（利用可能な場合）
try:
    from modernbert_encoder import ModernBERTNewsEncoder
    MODERNBERT_AVAILABLE = True
except ImportError:
    MODERNBERT_AVAILABLE = False
    logger.warning("ModernBERT encoder not available, using simple encoder")


class SimpleIREncoder(nn.Module):
    """シンプルなIRニュースエンコーダー（フォールバック用）"""
    
    def __init__(self, output_size: int = 64):
        super().__init__()
        self.output_size = output_size
        self.projection = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )
    
    def forward(self, news_texts: List[str]) -> torch.Tensor:
        """ニュースをエンコード（キーワードベース）"""
        device = next(self.parameters()).device
        feature_vector = torch.zeros(1, 100, device=device)
        
        positive_words = ['増収', '増益', '上方修正', '好調', '拡大', '成長', '黒字', '最高益']
        negative_words = ['減収', '減益', '下方修正', '低迷', '縮小', '赤字', '損失', '悪化']
        
        for text in news_texts:
            for i, word in enumerate(positive_words[:50]):
                if word in text:
                    feature_vector[0, i] += 1
            for i, word in enumerate(negative_words[:50]):
                if word in text:
                    feature_vector[0, 50 + i] -= 1
        
        return self.projection(torch.tanh(feature_vector))


class TradingDecisionModel(nn.Module):
    """統合型売買判断モデル（方針.md準拠）
    
    アーキテクチャ:
    - 時系列エンコーダー: LSTM（株価データ処理）
    - ニュースエンコーダー: ModernBERT-ja（IRニュース処理）
    - 意思決定: Actor-Critic構造（PPO互換）
    """
    
    def __init__(self, window_size: int = 30, use_modernbert: bool = True, device: str = None):
        super().__init__()
        self.window_size = window_size
        self.use_modernbert = use_modernbert and MODERNBERT_AVAILABLE
        
        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon GPU
            elif torch.cuda.is_available():
                device = 'cuda:0'  # NVIDIA GPU
            else:
                device = 'cpu'  # CPU fallback
            logger.info(f"Auto-detected device: {device}")
        
        # Convert device string to torch device
        if device == 'mps':
            self.device = torch.device('mps')
        elif device.startswith('cuda'):
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')
        
        # 時系列エンコーダー（LSTM）
        self.nikkei_encoder = MarketEncoder(input_size=3, hidden_size=64)
        self.target_encoder = MarketEncoder(input_size=3, hidden_size=64)
        
        # IRニュースエンコーダー（ModernBERT or フォールバック）
        if self.use_modernbert:
            self.ir_encoder = ModernBERTNewsEncoder(output_size=64, device=device)
            logger.info("Using ModernBERT-ja for IR news encoding")
        else:
            # フォールバック用シンプルエンコーダー
            self.ir_encoder = SimpleIREncoder(output_size=64)
            logger.info("Using simple encoder for IR news")
        
        # 決定ネットワーク（Actor-Critic構造）
        feature_size = 64 + 64 + 64  # nikkei + target + IR
        self.decision_network = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)  # 4つのアクション
        )
        
        # Move model to device
        self.to(self.device)
        logger.info(f"TradingDecisionModel initialized on {device}")
    
    def forward(self, market_data: MarketData) -> Dict[str, float]:
        """売買判断を実行"""
        # データ準備と正規化
        nikkei_data = self._prepare_market_data(
            market_data.nikkei_high,
            market_data.nikkei_low,
            market_data.nikkei_close,
            normalize_factor=30000
        )
        
        target_data = self._prepare_market_data(
            market_data.target_high,
            market_data.target_low,
            market_data.target_close,
            normalize_factor=10000
        )
        
        # エンコード
        nikkei_features = self.nikkei_encoder(nikkei_data)
        target_features = self.target_encoder(target_data)
        
        # IRニュースエンコード（ModernBERT or シンプル）
        if isinstance(self.ir_encoder, ModernBERTNewsEncoder):
            ir_features = self.ir_encoder(market_data.ir_news)
        else:
            ir_features = self.ir_encoder(market_data.ir_news)
        
        # 特徴量結合
        combined = torch.cat([nikkei_features, target_features, ir_features], dim=-1)
        
        # 売買判断
        logits = self.decision_network(combined)
        probs = F.softmax(logits, dim=-1).squeeze()
        
        return self._interpret_decision(probs)
    
    def _prepare_market_data(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, normalize_factor: float) -> torch.Tensor:
        """市場データの準備"""
        data = np.stack([high, low, close], axis=-1)
        data = data / normalize_factor
        return torch.FloatTensor(data).unsqueeze(0).to(self.device)
    
    def _interpret_decision(self, probs: torch.Tensor) -> Dict[str, float]:
        """判断結果の解釈"""
        actions = ['強売り', 'ホールド', '少量買い', '強買い']
        decision_idx = torch.argmax(probs).item()
        
        return {
            'action': actions[decision_idx],
            'confidence': probs[decision_idx].item(),
            'sell_prob': probs[0].item(),
            'hold_prob': probs[1].item(),
            'buy_small_prob': probs[2].item(),
            'buy_large_prob': probs[3].item(),
            'recommended_position': (decision_idx - 1) * 0.33
        }
    
    def save(self, path: str):
        """モデル保存"""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str, device: str = None):
        """モデル読み込み"""
        if device:
            # Convert device string to torch device if needed
            if device == 'mps':
                map_location = torch.device('mps')
            elif device.startswith('cuda'):
                map_location = torch.device(device)
            else:
                map_location = torch.device('cpu')
        else:
            map_location = self.device
            
        self.load_state_dict(torch.load(path, map_location=map_location))
        logger.info(f"Model loaded from {path} to {map_location}")
        
    def train_step(self, batch_data: List[MarketData], targets: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> float:
        """訓練ステップ"""
        self.train()
        optimizer.zero_grad()
        
        total_loss = 0
        for data, target in zip(batch_data, targets):
            output = self.forward(data)
            # 簡略化: 確率分布のクロスエントロピー
            logits = self.decision_network(
                torch.cat([
                    self.nikkei_encoder(self._prepare_market_data(
                        data.nikkei_high, data.nikkei_low, data.nikkei_close, 30000)),
                    self.target_encoder(self._prepare_market_data(
                        data.target_high, data.target_low, data.target_close, 10000)),
                    self.ir_encoder.forward(data.ir_news) if hasattr(self.ir_encoder, 'forward') else torch.zeros(1, 64)
                ], dim=-1)
            )
            loss = F.cross_entropy(logits, target.unsqueeze(0))
            total_loss += loss.item()
            loss.backward()
        
        optimizer.step()
        return total_loss / len(batch_data)