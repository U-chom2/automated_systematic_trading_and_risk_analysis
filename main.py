"""
株式売買AI統合システム - メインエントリーポイント

強化学習（PPO）+ LSTM/Transformer + ModernBERTを使用した
高度な株価予測・売買AIシステム

方針.mdに基づいた実装：
- 時系列パターン抽出（LSTM/Transformer）
- IRニュース理解（ModernBERT-ja）
- 強化学習による意思決定（PPO）
"""

import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch
import pandas as pd
import numpy as np

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "train"))

# 刷新されたAIモデルをインポート
from train import Nikkei225TradingPipeline
from models.trading_model import TradingDecisionModel, MarketData
from models.agents.ppo_agent import PPOTradingAgent

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """ログ設定"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("trading_system.log")
        ]
    )


class AITradingSystem:
    """統合型AI株式売買システム"""
    
    def __init__(self, device: str = None):
        """
        システム初期化
        
        Args:
            device: 使用デバイス ('mps', 'cuda:0', 'cpu', None for auto-detect)
        """
        # デバイス自動検出
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon GPU
            elif torch.cuda.is_available():
                device = 'cuda:0'  # NVIDIA GPU
            else:
                device = 'cpu'  # CPU fallback
                
        self.device = device
        logger.info(f"AI Trading System initialized on {device}")
        
        # モデル格納
        self.model: Optional[TradingDecisionModel] = None
        self.trained_agent: Optional[PPOTradingAgent] = None
    
    def train_model(
        self,
        target_symbols: List[str],
        start_date: str,
        end_date: str,
        total_timesteps: int = 50000,
        initial_cash: float = 10000000,  # 10 million yen
        save_model: bool = True
    ) -> PPOTradingAgent:
        """
        強化学習モデルの訓練
        
        Args:
            target_symbols: 対象銘柄リスト (例: ['7203.T', '6758.T'])
            start_date: 訓練開始日 (YYYY-MM-DD)
            end_date: 訓練終了日 (YYYY-MM-DD)
            total_timesteps: 訓練ステップ数
            initial_cash: 初期資金
            save_model: モデル保存フラグ
            
        Returns:
            訓練済みPPOエージェント
        """
        logger.info("=" * 60)
        logger.info("強化学習モデル訓練開始")
        logger.info("=" * 60)
        logger.info(f"対象銘柄: {target_symbols}")
        logger.info(f"期間: {start_date} - {end_date}")
        logger.info(f"訓練ステップ数: {total_timesteps}")
        logger.info(f"初期資金: ¥{initial_cash:,.0f}")
        logger.info(f"使用デバイス: {self.device}")
        
        # 訓練パイプライン作成
        pipeline = Nikkei225TradingPipeline(
            target_symbols=target_symbols,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            commission_rate=0.001,  # 0.1% commission
            window_size=30  # 30日の履歴データ
        )
        
        # PPO強化学習で訓練
        trained_agent = pipeline.train(
            total_timesteps=total_timesteps,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,
            n_epochs=10,
            device=self.device
        )
        
        self.trained_agent = trained_agent
        
        logger.info("=" * 60)
        logger.info("✅ 強化学習モデル訓練完了")
        logger.info("=" * 60)
        
        return trained_agent
    
    def load_inference_model(self, model_path: str = None) -> TradingDecisionModel:
        """
        推論用モデルの読み込み
        
        Args:
            model_path: モデルファイルパス
            
        Returns:
            推論用モデル
        """
        logger.info(f"推論モデル読み込み: {model_path or 'デモモデル'}")
        
        # 推論用モデル作成
        model = TradingDecisionModel(device=self.device)
        
        # 保存されたモデルがあれば読み込み
        if model_path and Path(model_path).exists():
            model.load(model_path, device=self.device)
            logger.info(f"✅ モデル読み込み完了: {model_path}")
        else:
            logger.info("🆕 新規デモモデルを使用")
        
        model.eval()
        self.model = model
        
        return model
    
    def predict(
        self,
        nikkei_data: np.ndarray,  # [日数, 3] (high, low, close)
        target_data: np.ndarray,  # [日数, 3] (high, low, close)  
        ir_news: List[str]
    ) -> Dict[str, float]:
        """
        売買判断実行
        
        Args:
            nikkei_data: 日経225データ (30日分)
            target_data: ターゲット株価データ (30日分)
            ir_news: IRニュースリスト
            
        Returns:
            売買判断結果
        """
        if self.model is None:
            raise RuntimeError("モデルが読み込まれていません。load_inference_model()を先に実行してください。")
        
        # MarketDataオブジェクト作成
        market_data = MarketData(
            nikkei_high=nikkei_data[:, 0],
            nikkei_low=nikkei_data[:, 1], 
            nikkei_close=nikkei_data[:, 2],
            target_high=target_data[:, 0],
            target_low=target_data[:, 1],
            target_close=target_data[:, 2],
            ir_news=ir_news
        )
        
        # 推論実行
        with torch.no_grad():
            decision = self.model(market_data)
        
        return decision
    
    def demo_mode(self) -> Dict[str, float]:
        """
        デモモード実行（サンプルデータで推論テスト）
        
        Returns:
            売買判断結果
        """
        logger.info("=" * 60)
        logger.info("🚀 AI Trading System - デモモード")
        logger.info("=" * 60)
        
        # デモモデル読み込み
        self.load_inference_model()
        
        # サンプルデータ生成
        logger.info("📊 サンプルデータ生成中...")
        np.random.seed(42)
        
        # 日経225データ（30日分）
        nikkei_base = 28000
        nikkei_close = nikkei_base + np.cumsum(np.random.randn(30) * 100)
        nikkei_high = nikkei_close + np.abs(np.random.randn(30) * 50)
        nikkei_low = nikkei_close - np.abs(np.random.randn(30) * 50)
        nikkei_data = np.stack([nikkei_high, nikkei_low, nikkei_close], axis=1)
        
        # ターゲット株データ（30日分）
        target_base = 3000
        target_close = target_base + np.cumsum(np.random.randn(30) * 20)
        target_high = target_close + np.abs(np.random.randn(30) * 10)
        target_low = target_close - np.abs(np.random.randn(30) * 10)
        target_data = np.stack([target_high, target_low, target_close], axis=1)
        
        # IRニュース（サンプル）
        ir_news = [
            "2024年第3四半期決算：売上高は前年同期比15%増収を達成",
            "新製品の販売が計画を上回る好調な推移を見せています",
            "通期業績予想を上方修正、増収増益を見込む"
        ]
        
        # データ情報表示
        logger.info("📈 入力データ情報:")
        logger.info(f"  日経225最新値: {nikkei_close[-1]:.0f} (30日変化率: {(nikkei_close[-1]/nikkei_close[0]-1)*100:.2f}%)")
        logger.info(f"  ターゲット株最新値: {target_close[-1]:.0f} (30日変化率: {(target_close[-1]/target_close[0]-1)*100:.2f}%)")
        logger.info(f"  IRニュース件数: {len(ir_news)}件")
        
        # 推論実行
        logger.info("🤖 AI推論実行中...")
        decision = self.predict(nikkei_data, target_data, ir_news)
        
        # 結果表示
        self._print_decision_results(decision)
        
        return decision
    
    def _print_decision_results(self, decision: Dict[str, float]) -> None:
        """売買判断結果の表示"""
        logger.info("=" * 60)
        logger.info("🎯 AI売買判断結果")
        logger.info("=" * 60)
        logger.info(f"推奨アクション: {decision['action']}")
        logger.info(f"信頼度: {decision['confidence']*100:.1f}%")
        logger.info("")
        logger.info("📊 詳細確率分布:")
        logger.info(f"  強売り:   {decision['sell_prob']*100:5.1f}%")
        logger.info(f"  ホールド: {decision['hold_prob']*100:5.1f}%")
        logger.info(f"  少量買い: {decision['buy_small_prob']*100:5.1f}%")
        logger.info(f"  強買い:   {decision['buy_large_prob']*100:5.1f}%")
        logger.info("")
        logger.info(f"💰 推奨ポジション: {decision['recommended_position']:.2f}")
        logger.info("   (-0.33=全売却, 0=ホールド, 1.0=全力買い)")
        logger.info("=" * 60)
    
    def evaluate_model(self, model_path: str = None, n_episodes: int = 10) -> Dict[str, float]:
        """
        モデル性能評価
        
        Args:
            model_path: 評価対象モデルパス
            n_episodes: 評価エピソード数
            
        Returns:
            評価メトリクス
        """
        logger.info("=" * 60)
        logger.info("📊 モデル性能評価")
        logger.info("=" * 60)
        
        if self.trained_agent is None:
            logger.error("評価対象のエージェントがありません。先に訓練を実行してください。")
            return {}
        
        # 評価実行
        metrics = self.trained_agent.evaluate(n_episodes=n_episodes)
        
        logger.info("評価結果:")
        logger.info(f"  平均報酬: {metrics['mean_reward']:.4f}")
        logger.info(f"  報酬標準偏差: {metrics['std_reward']:.4f}")
        logger.info(f"  平均エピソード長: {metrics['mean_length']:.1f}")
        logger.info(f"  最小報酬: {metrics['min_reward']:.4f}")
        logger.info(f"  最大報酬: {metrics['max_reward']:.4f}")
        
        return metrics


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="株式売買AI統合システム（強化学習+LSTM/Transformer+ModernBERT）"
    )
    
    # モード選択
    parser.add_argument(
        '--mode', 
        choices=['train', 'inference', 'demo', 'evaluate'],
        default='demo',
        help='実行モード'
    )
    
    # 訓練パラメータ
    parser.add_argument('--symbols', nargs='+', default=['7203.T', '6758.T', '9984.T'],
                       help='対象銘柄（例: 7203.T 6758.T）')
    parser.add_argument('--start-date', default='2022-01-01', help='開始日（YYYY-MM-DD）')
    parser.add_argument('--end-date', default='2024-01-01', help='終了日（YYYY-MM-DD）')
    parser.add_argument('--timesteps', type=int, default=50000, help='訓練ステップ数')
    parser.add_argument('--initial-cash', type=float, default=10000000, help='初期資金')
    
    # システムパラメータ
    parser.add_argument('--device', choices=['mps', 'cuda', 'cuda:0', 'cpu'],
                       help='使用デバイス（自動検出: 指定なし）')
    parser.add_argument('--model-path', help='モデルファイルパス')
    parser.add_argument('--log-level', default='INFO', help='ログレベル')
    
    # 評価パラメータ
    parser.add_argument('--eval-episodes', type=int, default=10, help='評価エピソード数')
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(args.log_level)
    
    try:
        # AIシステム初期化
        ai_system = AITradingSystem(device=args.device)
        
        if args.mode == 'train':
            # 訓練モード
            logger.info("🎓 訓練モード開始")
            ai_system.train_model(
                target_symbols=args.symbols,
                start_date=args.start_date,
                end_date=args.end_date,
                total_timesteps=args.timesteps,
                initial_cash=args.initial_cash
            )
            
            # 訓練後評価
            if ai_system.trained_agent:
                ai_system.evaluate_model(n_episodes=args.eval_episodes)
                
        elif args.mode == 'inference':
            # 推論モード
            logger.info("🔮 推論モード開始")
            model = ai_system.load_inference_model(args.model_path)
            
            # TODO: 実データ取得と推論実行
            logger.info("実データでの推論は今後実装予定")
            logger.info("現在はデモモードをお試しください: --mode demo")
            
        elif args.mode == 'demo':
            # デモモード
            decision = ai_system.demo_mode()
            
        elif args.mode == 'evaluate':
            # 評価モード
            logger.info("📊 評価モード")
            if args.model_path:
                # TODO: 保存済みモデルの評価
                logger.info("保存済みモデルの評価は今後実装予定")
            else:
                logger.error("評価にはモデルパスが必要です: --model-path")
                
    except KeyboardInterrupt:
        logger.info("🛑 システム終了（ユーザー中断）")
    except Exception as e:
        logger.error(f"❌ システムエラー: {e}", exc_info=True)
    finally:
        logger.info("🏁 AI Trading System 停止")


if __name__ == "__main__":
    main()