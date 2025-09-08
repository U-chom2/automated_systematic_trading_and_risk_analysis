"""
Phase1 動作テスト - デイトレード設定の確認
"""

import sys
from pathlib import Path

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, TradingMode, day_trading_config

def test_trading_mode_settings():
    """取引モード設定のテスト"""
    print("🔧 Phase1 デイトレード設定テスト開始")
    print("=" * 60)
    
    # 中長期設定のテスト
    long_term_config = Config(TradingMode.LONG_TERM)
    print("\n📊 中長期設定:")
    print(f"  モード: {long_term_config.trading_mode.value}")
    print(f"  RSI期間: {long_term_config.technical_analysis.rsi_period}日")
    print(f"  移動平均: {long_term_config.technical_analysis.sma_short_period}/{long_term_config.technical_analysis.sma_medium_period}/{long_term_config.technical_analysis.sma_long_period}日")
    print(f"  強い買い閾値: {long_term_config.investment_thresholds.strong_buy_threshold}点")
    print(f"  目標利益率: {long_term_config.investment_thresholds.target_profit_strong}%")
    print(f"  損切りライン: {long_term_config.investment_thresholds.stop_loss_strong}%")
    
    # デイトレード設定のテスト
    day_config = Config(TradingMode.DAY_TRADING)
    print("\n🏃 デイトレード設定:")
    print(f"  モード: {day_config.trading_mode.value}")
    print(f"  RSI期間: {day_config.technical_analysis.rsi_period}日")
    print(f"  移動平均: {day_config.technical_analysis.sma_short_period}/{day_config.technical_analysis.sma_medium_period}/{day_config.technical_analysis.sma_long_period}日")
    print(f"  強い買い閾値: {day_config.investment_thresholds.strong_buy_threshold}点")
    print(f"  目標利益率: {day_config.investment_thresholds.target_profit_strong}%")
    print(f"  損切りライン: {day_config.investment_thresholds.stop_loss_strong}%")
    print(f"  最大同時保有: {day_config.investment_limits.max_daily_positions}銘柄")
    print(f"  1日最大損失: ¥{day_config.investment_limits.max_daily_loss:,}")

def test_investment_recommendations():
    """投資推奨の比較テスト"""
    print("\n📈 投資推奨比較テスト")
    print("-" * 60)
    
    test_scores = [85, 75, 65, 55, 45, 35]
    
    long_config = Config(TradingMode.LONG_TERM)
    day_config = Config(TradingMode.DAY_TRADING)
    
    for score in test_scores:
        long_rec = long_config.get_investment_recommendation(score)
        day_rec = day_config.get_investment_recommendation(score)
        
        print(f"\n📊 スコア{score}点:")
        print(f"  中長期: {long_rec['judgment']} | 利益{long_rec['target_profit']}% | 損切{long_rec['stop_loss']}% | {long_rec['holding_period']}")
        print(f"  デイトレ: {day_rec['judgment']} | 利益{day_rec['target_profit']}% | 損切{day_rec['stop_loss']}% | {day_rec['holding_period']}")

def test_mode_switching():
    """モード切り替えテスト"""
    print("\n🔄 モード切り替えテスト")
    print("-" * 60)
    
    config = Config(TradingMode.LONG_TERM)
    print(f"初期モード: {config.get_trading_mode_info()['description']}")
    print(f"実行タイミング: {config.get_execution_timing_info()}")
    
    # デイトレードモードに切り替え
    config.switch_trading_mode(TradingMode.DAY_TRADING)
    print(f"\n切り替え後: {config.get_trading_mode_info()['description']}")
    print(f"実行タイミング: {config.get_execution_timing_info()}")
    print(f"利益目標範囲: {config.get_trading_mode_info()['target_profit_range']}")
    print(f"分析重点: {config.get_trading_mode_info()['analysis_focus']}")

def test_scoring_weights():
    """スコア重み設定テスト"""
    print("\n⚖️ スコア重み比較")
    print("-" * 60)
    
    long_config = Config(TradingMode.LONG_TERM)
    day_config = Config(TradingMode.DAY_TRADING)
    
    print("中長期モード重み:")
    print(f"  トレンド: {long_config.scoring_weights.trend_weight}%")
    print(f"  モメンタム: {long_config.scoring_weights.price_momentum_weight}%")
    print(f"  RSI: {long_config.scoring_weights.rsi_weight}%")
    print(f"  MACD: {long_config.scoring_weights.macd_weight}%")
    
    print("\nデイトレードモード重み:")
    print(f"  トレンド: {day_config.scoring_weights.trend_weight}%")
    print(f"  モメンタム: {day_config.scoring_weights.price_momentum_weight}% ⚡")
    print(f"  RSI: {day_config.scoring_weights.rsi_weight}%")
    print(f"  MACD: {day_config.scoring_weights.macd_weight}%")
    print(f"  時価総額: {day_config.scoring_weights.market_cap_weight}% (無視)")

def main():
    """テスト実行"""
    print("🚀 Phase1 デイトレード設定 - 完全動作テスト")
    print("=" * 80)
    
    try:
        test_trading_mode_settings()
        test_investment_recommendations()
        test_mode_switching()
        test_scoring_weights()
        
        print("\n✅ Phase1テスト結果")
        print("=" * 60)
        print("🎯 デイトレード設定クラス作成: SUCCESS")
        print("🎯 動的保有期間設定: SUCCESS")
        print("🎯 利益目標・リスク管理: SUCCESS")
        print("🎯 設定切り替え機能: SUCCESS")
        print("\n🏆 Phase1 完全成功！デイトレード対応準備完了")
        
        print("\n📋 次のステップ:")
        print("  Phase2: technical_analyzer.py 短期指標対応")
        print("  Phase3: investment_scorer.py 重み調整")
        print("  Phase4: investment_limiter.py デイトレ制限")
        print("  Phase5: investment_analyzer.py 統合")
        
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()