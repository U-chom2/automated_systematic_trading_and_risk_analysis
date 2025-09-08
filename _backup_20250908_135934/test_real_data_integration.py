"""
実データ統合テスト
ダミーデータを実データに置き換えた後の動作確認
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# モジュールインポート
from nikkei_data_fetcher import Nikkei225DataFetcher
from ppo_scoring_adapter import create_ppo_adapter
from investment_analyzer import InvestmentAnalyzer
from config import Config, TradingMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_nikkei_data_fetcher():
    """日経225実データ取得テスト"""
    print("\n=== 日経225実データ取得テスト ===")
    
    try:
        cache_dir = Path("cache/nikkei")
        fetcher = Nikkei225DataFetcher(cache_dir=cache_dir)
        
        # 30日分のデータ取得
        print("30日分の日経225データを取得中...")
        data = fetcher.fetch_for_window(30)
        
        if not data.empty:
            print(f"✅ データ取得成功: {len(data)}日分")
            print(f"期間: {data['date'].min().strftime('%Y-%m-%d')} ～ {data['date'].max().strftime('%Y-%m-%d')}")
            
            # 最新データ表示
            latest = data.iloc[-1]
            print(f"\n最新データ ({latest['date'].strftime('%Y-%m-%d')}):")
            print(f"  始値: ¥{latest['open']:,.0f}")
            print(f"  高値: ¥{latest['high']:,.0f}")
            print(f"  安値: ¥{latest['low']:,.0f}")
            print(f"  終値: ¥{latest['close']:,.0f}")
            print(f"  出来高: {latest['volume']:,.0f}")
            
            # データの統計情報
            print(f"\n統計情報:")
            print(f"  終値平均: ¥{data['close'].mean():,.0f}")
            print(f"  終値最高: ¥{data['close'].max():,.0f}")
            print(f"  終値最低: ¥{data['close'].min():,.0f}")
            print(f"  変動率: {((data['close'].max() - data['close'].min()) / data['close'].mean() * 100):.2f}%")
            
            return True
        else:
            print("❌ データ取得失敗")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def test_ppo_with_real_nikkei():
    """実日経データを使用したPPOモデルテスト"""
    print("\n=== PPOモデル実データテスト ===")
    
    try:
        # PPOアダプター初期化
        print("PPOアダプターを初期化中...")
        adapter = create_ppo_adapter()
        
        # モデル情報表示
        model_info = adapter.get_model_info()
        print(f"✅ PPOモデルロード成功")
        print(f"  モデル: {model_info['model_name']}")
        print(f"  デバイス: {model_info['device']}")
        
        # ダミーのテクニカル指標を作成
        from technical_analyzer import TechnicalIndicators
        indicators = TechnicalIndicators(
            sma_5=2800.0,
            sma_25=2750.0,
            sma_75=2700.0,
            rsi=55.0,
            macd=10.0,
            macd_signal=5.0,
            macd_histogram=5.0,
            bollinger_upper=2900.0,
            bollinger_middle=2800.0,
            bollinger_lower=2700.0,
            price_change_1d=1.5,
            price_change_5d=3.0,
            price_change_25d=5.0
        )
        
        # PPO予測実行（実日経データが内部で使用される）
        print("\n実日経データを使用してPPO予測実行中...")
        result = adapter.calculate_investment_score(
            indicators=indicators,
            current_price=2800.0,
            market_cap_millions=35000000,
            symbol="7203.T"
        )
        
        print(f"✅ PPO予測成功")
        print(f"  投資スコア: {result.total_score:.1f}点")
        print(f"  投資判断: {result.recommendation['judgment']}")
        print(f"  PPOアクション値: {result.analysis_details.get('ppo_action_value', 0):.3f}")
        print(f"  PPO判断: {result.analysis_details.get('ppo_action_interpretation', 'Unknown')}")
        
        # 日経データキャッシュの確認
        if adapter.nikkei_data_cache is not None:
            print(f"\n日経データキャッシュ確認:")
            print(f"  キャッシュサイズ: {len(adapter.nikkei_data_cache)}日分")
            latest_nikkei = adapter.nikkei_data_cache.iloc[-1]
            print(f"  最新日経終値: ¥{latest_nikkei['close']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_system_with_real_data():
    """完全システムテスト（実データ使用）"""
    print("\n=== 完全システム実データテスト ===")
    
    try:
        # システム初期化
        config = Config(TradingMode.DAY_TRADING)
        analyzer = InvestmentAnalyzer(config, use_ppo=True)
        
        print("システム初期化完了")
        trading_info = analyzer.get_trading_info()
        print(f"  取引モード: {trading_info['trading_mode']}")
        print(f"  分析手法: {trading_info['scoring_method']}")
        
        # 実際の銘柄分析（トヨタ）
        print("\n実銘柄分析実行中...")
        result = analyzer.analyze_single_stock("7203.T", "トヨタ自動車", 35000000)
        
        if result:
            print(f"✅ 分析成功 - トヨタ自動車")
            print(f"  現在株価: ¥{result['現在株価']:,.0f}")
            print(f"  投資スコア: {result['投資スコア']:.1f}点")
            print(f"  投資判断: {result['投資判断']}")
            print(f"  分析手法: {result.get('分析手法', 'Unknown')}")
            
            # PPO固有情報
            if 'PPOアクション値' in result:
                print(f"\nPPO分析詳細:")
                print(f"  アクション値: {result['PPOアクション値']:.3f}")
                print(f"  PPO判断: {result['PPO判断']}")
                print(f"  信頼度: {result['PPO信頼度']:.3f}")
            
            # テクニカル指標
            print(f"\nテクニカル指標:")
            print(f"  RSI: {result.get('RSI', 'N/A')}")
            print(f"  1日変化率: {result.get('1日変化率', 0):.2f}%")
            print(f"  5日変化率: {result.get('5日変化率', 0):.2f}%")
            
            return True
        else:
            print("❌ 分析失敗")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cache_status():
    """キャッシュ状態の確認"""
    print("\n=== キャッシュ状態確認 ===")
    
    cache_dir = Path("cache/nikkei")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.json"))
        print(f"キャッシュディレクトリ: {cache_dir}")
        print(f"キャッシュファイル数: {len(cache_files)}")
        
        for file in cache_files:
            size_kb = file.stat().st_size / 1024
            mod_time = datetime.fromtimestamp(file.stat().st_mtime)
            print(f"  - {file.name}: {size_kb:.1f}KB, 更新: {mod_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        print("キャッシュディレクトリが存在しません")


def main():
    """メインテスト実行"""
    print("=" * 80)
    print("🔍 実データ統合テスト開始")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    results = {}
    
    # 1. 日経225データ取得テスト
    results['nikkei_fetch'] = test_nikkei_data_fetcher()
    
    # 2. PPO実データテスト
    results['ppo_real_data'] = test_ppo_with_real_nikkei()
    
    # 3. 完全システムテスト
    results['full_system'] = test_full_system_with_real_data()
    
    # 4. キャッシュ状態確認
    check_cache_status()
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("📊 テスト結果サマリー")
    print("=" * 80)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 全テスト合格！実データ統合は正常に動作しています")
    else:
        print("\n⚠️ 一部のテストが失敗しました")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)