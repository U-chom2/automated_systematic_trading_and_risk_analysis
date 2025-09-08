"""
株式売買AI - メインスクリプト

推論時の入力仕様:
1. 日経225の直近30日の日次データ（高値・安値・終値）
2. ターゲット企業の直近30日の日次データ（高値・安値・終値）
3. ターゲット企業の直近1ヶ月のIR情報
"""

import sys
from pathlib import Path
import numpy as np
import torch
import argparse
import logging

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.trading_model import TradingDecisionModel, MarketData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def generate_sample_data() -> MarketData:
    """ダミーデータを生成（すべて0で明確にダミー）"""
    logger.error("FALLBACK: Generating all-zero dummy data for training demo")
    
    # 日経225データ（30日分、すべて0）
    nikkei_high = np.zeros(30)
    nikkei_low = np.zeros(30)
    nikkei_close = np.zeros(30)
    
    # ターゲット株データ（30日分、すべて0）
    target_high = np.zeros(30)
    target_low = np.zeros(30)
    target_close = np.zeros(30)
    
    # IRニュース（サンプル）
    ir_news = [
        "2024年第3四半期決算：売上高は前年同期比15%増収",
        "新製品の販売が好調に推移",
        "通期業績予想を上方修正"
    ]
    
    return MarketData(
        nikkei_high=nikkei_high,
        nikkei_low=nikkei_low,
        nikkei_close=nikkei_close,
        target_high=target_high,
        target_low=target_low,
        target_close=target_close,
        ir_news=ir_news
    )


def run_inference(model_path: str = None, demo: bool = True, device: str = None):
    """推論実行
    
    Args:
        model_path: モデルファイルのパス
        demo: デモモードで実行するか
        device: 使用するデバイス ('mps', 'cuda', 'cpu', or None for auto-detect)
    """
    print("=" * 60)
    print("株式売買AI - 推論モード")
    print("=" * 60)
    
    # デバイス自動検出
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon GPU
        elif torch.cuda.is_available():
            device = 'cuda:0'  # NVIDIA GPU
        else:
            device = 'cpu'  # CPU fallback
        print(f"自動検出デバイス: {device}")
    else:
        print(f"指定デバイス: {device}")
    
    # モデル初期化
    model = TradingDecisionModel(device=device)
    
    # モデル読み込み（存在する場合）
    if model_path and Path(model_path).exists():
        model.load(model_path, device=device)
        print(f"モデルを読み込みました: {model_path}")
    
    # モデルをデバイスに移動
    model = model.to(device)
    model.eval()
    
    # データ取得
    if demo:
        market_data = generate_sample_data()
    else:
        # 実データを取得する場合はここに実装
        market_data = generate_sample_data()  # 現時点ではデモデータを使用
    
    print("\n【入力データ】")
    print(f"日経225 最新値: {market_data.nikkei_close[-1]:.0f}")
    print(f"日経225 30日変化率: {(market_data.nikkei_close[-1] / market_data.nikkei_close[0] - 1) * 100:.2f}%")
    print(f"ターゲット株 最新値: {market_data.target_close[-1]:.0f}")
    print(f"ターゲット株 30日変化率: {(market_data.target_close[-1] / market_data.target_close[0] - 1) * 100:.2f}%")
    print(f"IRニュース件数: {len(market_data.ir_news)}件")
    
    # 推論実行
    with torch.no_grad():
        decision = model(market_data)
    
    print("\n【売買判断結果】")
    print(f"推奨アクション: {decision['action']}")
    print(f"信頼度: {decision['confidence']*100:.1f}%")
    print("\n【詳細確率】")
    print(f"強売り: {decision['sell_prob']*100:.1f}%")
    print(f"ホールド: {decision['hold_prob']*100:.1f}%")
    print(f"少量買い: {decision['buy_small_prob']*100:.1f}%")
    print(f"強買い: {decision['buy_large_prob']*100:.1f}%")
    print(f"\n推奨ポジション: {decision['recommended_position']:.2f}")
    print("（-0.33=全売却, 0=ホールド, 1.0=全力買い）")
    
    # 入力仕様の表示
    print("\n" + "=" * 60)
    print("【推論時の入力仕様】")
    print("=" * 60)
    print("1. 日経225指数（直近30日の日次データ）")
    print("   - nikkei_high: float[30]  # 高値")
    print("   - nikkei_low: float[30]   # 安値")
    print("   - nikkei_close: float[30] # 終値")
    print("\n2. ターゲット企業株価（直近30日の日次データ）")
    print("   - target_high: float[30]  # 高値")
    print("   - target_low: float[30]   # 安値")
    print("   - target_close: float[30] # 終値")
    print("\n3. IR情報")
    print("   - ir_news: List[str]      # 直近1ヶ月のIRニュース")
    print("\n【出力】")
    print("   - action: str             # 推奨アクション")
    print("   - confidence: float       # 信頼度（0-1）")
    print("   - recommended_position: float # 推奨ポジション（-0.33～1.0）")
    print("=" * 60)
    
    return decision


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(description='株式売買AI')
    parser.add_argument('--mode', choices=['train', 'inference', 'demo'], 
                       default='demo', help='実行モード')
    parser.add_argument('--model', type=str, default='models/trading_model.pth',
                       help='モデルファイルパス')
    parser.add_argument('--data', type=str, help='データファイルパス')
    parser.add_argument('--device', type=str, choices=['mps', 'cuda', 'cuda:0', 'cpu'],
                       help='使用するデバイス (default: auto-detect)')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        decision = run_inference(demo=True, device=args.device)
    elif args.mode == 'inference':
        decision = run_inference(model_path=args.model, demo=False, device=args.device)
    elif args.mode == 'train':
        print("訓練モードは train.py を使用してください")
        print("使用方法: python train.py")
    
    # モデル保存（デモモードの場合）
    if args.mode == 'demo':
        # デバイス自動検出
        if args.device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
        else:
            device = args.device
            
        model = TradingDecisionModel(device=device)
        save_path = Path('models') / 'trading_model_demo.pth'
        save_path.parent.mkdir(exist_ok=True)
        model.save(str(save_path))
        print(f"\nデモモデルを {save_path} に保存しました")


if __name__ == "__main__":
    main()