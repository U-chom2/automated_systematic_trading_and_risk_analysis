"""
株式売買AI - PPOモデル推論スクリプト (東証グロース605社対応)

推論時の入力仕様:
1. 日経225の直近30日の日次データ（高値・安値・終値）
2. 東証グロース605社の当日データ（終値・出来高）
3. ポートフォリオ状態（現金比率、各銘柄ポジション）
4. IR情報（ModernBERT-jaセンチメント分析結果）
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import PPO components
from models.agents.ppo_agent import PPOTradingAgent
from models.environment.trading_env import TradingEnvironment
from stable_baselines3 import PPO

# Import from train.py
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", Path(__file__).parent / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

EnhancedTradingEnvironment = train_module.EnhancedTradingEnvironment
load_growth_stocks_symbols = train_module.load_growth_stocks_symbols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_model() -> Optional[str]:
    """最新の訓練済みモデルを検索"""
    model_dir = Path(__file__).parent / 'models' / 'rl'
    if not model_dir.exists():
        return None
    
    model_files = list(model_dir.glob('ppo_nikkei_model_*.zip'))
    if not model_files:
        return None
    
    # ファイル名の日時から最新のモデルを選択
    latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
    return str(latest_model)


def load_real_data(symbols: List[str], lookback_days: int = 365) -> Dict:
    """実際の市場データを取得"""
    logger.info("実際の市場データを取得中...")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    # 日経225データ
    logger.info("日経225データを取得中...")
    nikkei = yf.Ticker('^N225')
    nikkei_hist = nikkei.history(start=start_date, end=end_date)
    
    nikkei_data = pd.DataFrame({
        'date': nikkei_hist.index,
        'high': nikkei_hist['High'].values,
        'low': nikkei_hist['Low'].values,
        'close': nikkei_hist['Close'].values
    })
    
    # グロース株データ（最新30社のみサンプル取得）
    logger.info("グロース株データを取得中（サンプル30社）...")
    sample_symbols = symbols[:30]  # サンプルとして30社のみ
    
    stock_data = {}
    for symbol in sample_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                stock_data[symbol] = {
                    'close': hist['Close'].iloc[-1],  # 最新終値
                    'volume': hist['Volume'].iloc[-1]  # 最新出来高
                }
        except Exception as e:
            logger.debug(f"データ取得失敗: {symbol}, {e}")
    
    logger.info(f"取得完了: 日経225={len(nikkei_data)}日, 株式={len(stock_data)}社")
    
    return {
        'nikkei_data': nikkei_data,
        'stock_data': stock_data,
        'symbols': list(stock_data.keys())
    }


def create_demo_environment(symbols: List[str]) -> EnhancedTradingEnvironment:
    """デモ用の環境を作成"""
    logger.info("デモ環境を作成中...")
    
    # ダミーデータ生成
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='B')[:244]
    
    # 日経225ダミーデータ
    nikkei_data = pd.DataFrame({
        'date': dates,
        'high': np.random.normal(30000, 1000, len(dates)),
        'low': np.random.normal(29500, 1000, len(dates)),  
        'close': np.random.normal(29750, 1000, len(dates))
    })
    
    # 株式ダミーデータ（全605社）
    stock_data_list = []
    for symbol in symbols:  # 全605社
        for date in dates:
            stock_data_list.append({
                'date': date,
                'symbol': symbol,
                'open': np.random.normal(1000, 200),
                'high': np.random.normal(1100, 200),
                'low': np.random.normal(900, 200),
                'close': np.random.normal(1000, 200),
                'volume': np.random.randint(1000, 100000)
            })
    
    stock_df = pd.DataFrame(stock_data_list)
    stock_df.set_index(['date', 'symbol'], inplace=True)
    
    # IRニュースダミーデータ
    news_data_list = []
    for symbol in symbols:
        for i, date in enumerate(dates[::7]):  # 週1回のニュース
            news_data_list.append({
                'date': date,
                'symbol': symbol,
                'title': f'{symbol} 業績発表',
                'content': f'{symbol}の決算が発表されました',
                'category': 'earnings'
            })
    
    news_df = pd.DataFrame(news_data_list)
    if not news_df.empty:
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df.set_index('date', inplace=True)
    else:
        news_df = None
    
    # 環境作成（全605社、IRニュース付き）
    env = EnhancedTradingEnvironment(
        nikkei_data=nikkei_data,
        price_data=stock_df,
        symbols=symbols,  # 全605社
        initial_cash=10000000,
        commission_rate=0.001,
        window_size=30,
        news_data=news_df  # IRニュースデータ追加
    )
    
    return env


def run_inference(model_path: str = None, demo: bool = True, device: str = None):
    """PPOモデルで推論実行"""
    print("=" * 70)
    print("株式売買AI - PPO推論モード (東証グロース605社対応)")
    print("=" * 70)
    
    # デバイス自動検出
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        print(f"自動検出デバイス: {device}")
    
    # モデルパス検索
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            logger.error("訓練済みモデルが見つかりません")
            logger.info("先に 'uv run python train/train.py' で訓練を実行してください")
            return None
    
    print(f"使用モデル: {model_path}")
    
    # 銘柄リスト読み込み
    symbols = load_growth_stocks_symbols()
    print(f"対象銘柄数: {len(symbols)}社")
    
    # 環境とデータ準備
    if demo:
        print("\n【デモモード】ダミーデータで推論実行")
        env = create_demo_environment(symbols)
        current_symbols = symbols  # 全605社
    else:
        print("\n【実データモード】実際の市場データで推論実行")
        # 実データ取得機能は将来実装
        logger.warning("実データモードは未実装です。デモモードで実行します。")
        env = create_demo_environment(symbols)
        current_symbols = symbols  # 全605社
    
    # PPOモデル読み込み
    try:
        model = PPO.load(model_path, device=device)
        print(f"モデル読み込み完了: {Path(model_path).name}")
    except Exception as e:
        logger.error(f"モデル読み込みエラー: {e}")
        return None
    
    # 推論実行
    print(f"\n【推論実行】{len(current_symbols)}銘柄に対する売買判断")
    print("-" * 50)
    
    # 環境リセット
    obs, info = env.reset()
    
    # 推論実行
    with torch.no_grad():
        actions, _ = model.predict(obs, deterministic=True)
    
    # 結果表示
    print(f"\n【売買アクション結果】")
    print(f"アクション配列形状: {actions.shape}")
    print(f"アクション範囲: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # 主要銘柄のアクション表示
    print(f"\n【主要銘柄の推奨アクション】")
    for i, symbol in enumerate(current_symbols[:10]):  # 上位10銘柄
        action_val = actions[i] if i < len(actions) else 0
        if action_val > 0.3:
            recommendation = f"🟢 買い推奨 ({action_val:.3f})"
        elif action_val < -0.3:
            recommendation = f"🔴 売り推奨 ({action_val:.3f})"
        else:
            recommendation = f"⚪ ホールド ({action_val:.3f})"
        
        print(f"  {symbol}: {recommendation}")
    
    # 統計情報
    buy_signals = np.sum(actions > 0.3)
    sell_signals = np.sum(actions < -0.3)
    hold_signals = len(actions) - buy_signals - sell_signals
    
    print(f"\n【アクション統計】")
    print(f"買いシグナル: {buy_signals}銘柄 ({buy_signals/len(actions)*100:.1f}%)")
    print(f"売りシグナル: {sell_signals}銘柄 ({sell_signals/len(actions)*100:.1f}%)")
    print(f"ホールド: {hold_signals}銘柄 ({hold_signals/len(actions)*100:.1f}%)")
    
    # 入力仕様の表示
    print("\n" + "=" * 70)
    print("【モデル入力仕様】")
    print("=" * 70)
    print("1. 日経225指数（直近30日）: 90次元")
    print("   - 高値・安値・終値の正規化済み時系列データ")
    print(f"\n2. グロース{len(current_symbols)}社データ（当日）: {len(current_symbols)*2}次元")
    print("   - 各銘柄の終値・出来高（正規化・クリッピング済み）")
    print(f"\n3. ポートフォリオ状態: {len(current_symbols)+1}次元")
    print("   - 現金比率 + 各銘柄ポジション比率")
    print(f"\n4. IR情報（オプション）: {len(current_symbols)*10}次元")
    print("   - ModernBERT-jaセンチメント分析結果")
    print(f"\n総観測次元: 約{90 + len(current_symbols)*2 + len(current_symbols)+1 + len(current_symbols)*10}次元")
    print("=" * 70)
    
    return {
        'actions': actions,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'hold_signals': hold_signals,
        'symbols': current_symbols
    }


def main():
    parser = argparse.ArgumentParser(description="PPO株式売買AI推論")
    parser.add_argument('--model', type=str, help='モデルファイルパス')
    parser.add_argument('--demo', action='store_true', default=True, help='デモモード')
    parser.add_argument('--real', action='store_true', help='実データモード（未実装）')
    parser.add_argument('--device', type=str, choices=['mps', 'cuda', 'cpu'], 
                       help='使用デバイス')
    
    args = parser.parse_args()
    
    # デモモードの設定
    demo_mode = not args.real
    
    result = run_inference(
        model_path=args.model,
        demo=demo_mode,
        device=args.device
    )
    
    if result:
        print(f"\n✅ 推論完了: {len(result['symbols'])}銘柄を分析")
    else:
        print("\n❌ 推論失敗")


if __name__ == "__main__":
    main()