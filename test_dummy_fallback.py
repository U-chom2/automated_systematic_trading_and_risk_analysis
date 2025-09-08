"""
ダミーデータフォールバックテスト
すべて0のダミーデータでもエラーが発生しないことを確認
"""

import sys
from pathlib import Path
from datetime import datetime

# 強制的にダミーデータを使用するためのテスト用モジュール
def test_with_all_zero_dummy_data():
    """すべて0のダミーデータでシステムが動作することを確認"""
    print("\n=== ダミーデータフォールバックテスト ===")
    print("すべて0のダミーデータでエラーが発生しないことを確認中...")
    
    try:
        # 技術指標計算（すべて0の価格データで）
        from technical_analyzer import TechnicalAnalyzer
        import pandas as pd
        import numpy as np
        
        analyzer = TechnicalAnalyzer()
        
        # すべて0のダミー価格データ
        dates = pd.date_range(end=datetime.now(), periods=30)
        dummy_data = pd.DataFrame({
            'Date': dates,
            'Open': np.zeros(30),
            'High': np.zeros(30),
            'Low': np.zeros(30),
            'Close': np.zeros(30),
            'Volume': np.zeros(30)
        })
        dummy_data.set_index('Date', inplace=True)
        
        print("\n1. テクニカル指標計算テスト（すべて0の価格データ）")
        
        # 各種指標計算（ゼロ除算エラーが発生しないことを確認）
        prices = dummy_data['Close']
        
        # SMA計算
        sma = analyzer.calculate_sma(prices, 5)
        print(f"   SMA(5): {sma}")
        
        # RSI計算
        rsi = analyzer.calculate_rsi(prices)
        print(f"   RSI: {rsi}")
        
        # MACD計算
        macd, signal, hist = analyzer.calculate_macd(prices)
        print(f"   MACD: {macd}, Signal: {signal}, Hist: {hist}")
        
        # ボリンジャーバンド計算
        bb_upper, bb_middle, bb_lower = analyzer.calculate_bollinger_bands(prices)
        print(f"   BB Upper: {bb_upper}, Middle: {bb_middle}, Lower: {bb_lower}")
        
        # 価格変化率計算
        change_1d, change_5d, change_25d = analyzer.calculate_price_changes(prices)
        print(f"   変化率 - 1日: {change_1d}, 5日: {change_5d}, 25日: {change_25d}")
        
        # モメンタム計算
        momentum = analyzer.calculate_momentum(prices, 5)
        print(f"   モメンタム(5): {momentum}")
        
        # ボラティリティ計算
        volatility = analyzer.calculate_volatility(prices)
        print(f"   ボラティリティ: {volatility}")
        
        print("   ✅ テクニカル指標計算: エラーなし")
        
        # PPOアダプターのテスト
        print("\n2. PPOアダプターテスト（すべて0のダミーデータ）")
        
        from ppo_scoring_adapter import create_ppo_adapter
        from technical_analyzer import TechnicalIndicators
        
        adapter = create_ppo_adapter()
        
        # すべて0のテクニカル指標
        indicators = TechnicalIndicators(
            sma_5=0.0,
            sma_25=0.0,
            sma_75=0.0,
            rsi=50.0,  # RSIのデフォルト値
            macd=0.0,
            macd_signal=0.0,
            macd_histogram=0.0,
            bollinger_upper=0.0,
            bollinger_middle=0.0,
            bollinger_lower=0.0,
            price_change_1d=0.0,
            price_change_5d=0.0,
            price_change_25d=0.0
        )
        
        # PPO予測（ゼロ値でもエラーが発生しないことを確認）
        result = adapter.calculate_investment_score(
            indicators=indicators,
            current_price=0.0,  # 価格も0
            market_cap_millions=0,
            symbol="TEST"
        )
        
        print(f"   投資スコア: {result.total_score:.1f}点")
        print(f"   PPOアクション値: {result.analysis_details.get('ppo_action_value', 0):.3f}")
        print("   ✅ PPOアダプター: エラーなし")
        
        # 価格フェッチャーのダミーデータテスト
        print("\n3. 価格フェッチャーのダミーデータテスト")
        
        from src.data_collector.price_fetcher import PriceFetcher
        
        fetcher = PriceFetcher()
        
        # 存在しない銘柄でフォールバック動作を確認
        price = fetcher.get_current_price("DUMMY_SYMBOL")
        print(f"   ダミー銘柄の価格: {price}")
        
        # 過去データ取得（フォールバック）
        historical = fetcher.get_historical_data("DUMMY_SYMBOL", period="5d")
        if historical:
            print(f"   過去データ件数: {len(historical)}")
            if historical:
                first_record = historical[0]
                print(f"   最初のレコード - Close: {first_record['close']}, Volume: {first_record['volume']}")
        
        print("   ✅ 価格フェッチャー: エラーなし")
        
        print("\n✅ すべてのテスト合格！")
        print("ダミーデータ（すべて0）でもシステムは正常に動作します")
        return True
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("=" * 80)
    print("🔍 ダミーデータフォールバックテスト")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    success = test_with_all_zero_dummy_data()
    
    if success:
        print("\n🎉 ダミーデータ処理が正しく実装されています")
        print("エラーや齟齬なく実行できることを確認しました")
    else:
        print("\n⚠️ ダミーデータ処理に問題があります")
        print("上記のエラーを修正してください")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())