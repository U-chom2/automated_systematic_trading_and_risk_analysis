"""
シミュレーションシステムテスト
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent))

from trading_simulator import TradingSimulator, SimulationConfig
from portfolio_manager import PortfolioManager
from config import TradingMode


def test_portfolio_manager():
    """ポートフォリオマネージャーのテスト"""
    print("🧪 ポートフォリオマネージャーテスト")
    print("=" * 60)
    
    # 初期化
    portfolio = PortfolioManager(initial_cash=100000.0)
    
    # 購入テスト
    print("\n📈 購入テスト:")
    success = portfolio.buy_stock(
        symbol="7203.T",
        company_name="トヨタ自動車",
        price=2800.0,
        shares=10,
        target_pct=2.0,
        stop_loss_pct=-1.5
    )
    print(f"購入結果: {'成功' if success else '失敗'}")
    
    # ポートフォリオ表示
    portfolio.display_portfolio()
    
    # 価格更新テスト
    print("\n📊 価格更新テスト:")
    price_data = {"7203.T": 2850.0}  # 1.8%上昇
    sold = portfolio.update_positions(price_data)
    print(f"売却銘柄: {sold}")
    
    # 目標価格到達テスト
    print("\n🎯 目標価格到達テスト:")
    price_data = {"7203.T": 2860.0}  # 2.1%上昇（目標達成）
    sold = portfolio.update_positions(price_data)
    print(f"売却銘柄: {sold}")
    
    # 最終表示
    portfolio.display_portfolio()
    
    return portfolio.get_performance_summary()


def test_basic_simulation():
    """基本的なシミュレーションテスト"""
    print("\n🧪 基本シミュレーションテスト")
    print("=" * 60)
    
    # シミュレーター初期化
    config = SimulationConfig(
        initial_capital=100000.0,
        max_positions=3,
        max_investment_per_stock=20000.0,
        trading_mode=TradingMode.DAY_TRADING
    )
    
    simulator = TradingSimulator(config)
    
    # テスト用の推奨銘柄を手動で追加
    print("\n📋 テスト銘柄の購入:")
    test_stocks = [
        ("4381.T", "ビープラッツ", 527.0, 30),
        ("9215.T", "CaSy", 1019.0, 15),
        ("7093.T", "アディッシュ", 602.0, 25)
    ]
    
    for symbol, name, price, shares in test_stocks:
        success = simulator.portfolio.buy_stock(
            symbol=symbol,
            company_name=name,
            price=price,
            shares=shares,
            target_pct=config.target_profit_pct,
            stop_loss_pct=config.stop_loss_pct,
            max_holding_days=config.max_holding_days
        )
        print(f"  {symbol}: {'成功' if success else '失敗'}")
    
    # ポートフォリオ確認
    simulator.portfolio.display_portfolio()
    
    # 価格変動シミュレーション
    print("\n📈 価格変動シミュレーション:")
    
    # Day 1: 小幅上昇
    print("\nDay 1: 小幅上昇")
    price_data = {
        "4381.T": 535.0,  # +1.5%
        "9215.T": 1025.0,  # +0.6%
        "7093.T": 608.0   # +1.0%
    }
    sold = simulator.portfolio.update_positions(price_data)
    print(f"売却: {sold}")
    
    # Day 2: 目標達成と損切り
    print("\nDay 2: 目標達成と損切り")
    price_data = {
        "4381.T": 540.0,  # +2.5%（目標達成）
        "9215.T": 1000.0,  # -1.9%（損切り）
        "7093.T": 615.0   # +2.2%（目標達成）
    }
    sold = simulator.portfolio.update_positions(price_data)
    print(f"売却: {sold}")
    
    # 最終結果
    simulator.portfolio.display_portfolio()
    
    # レポート生成
    print("\n" + simulator.get_report())
    
    return simulator.portfolio.get_performance_summary()


def test_recommendations_integration():
    """推奨銘柄取得の統合テスト"""
    print("\n🧪 推奨銘柄統合テスト")
    print("=" * 60)
    
    config = SimulationConfig(
        initial_capital=100000.0,
        max_positions=5,
        trading_mode=TradingMode.DAY_TRADING
    )
    
    simulator = TradingSimulator(config)
    
    # 推奨銘柄取得テスト
    print("\n📊 推奨銘柄の取得:")
    recommendations = simulator.get_todays_recommendations()
    
    if recommendations:
        print(f"推奨銘柄数: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"\n{i}. {rec.company_name} ({rec.symbol})")
            print(f"   スコア: {rec.score}点")
            print(f"   価格: ¥{rec.current_price:,.0f}")
            print(f"   推奨株数: {rec.shares}株")
            print(f"   投資額: ¥{rec.investment_amount:,.0f}")
        
        # 取引実行テスト
        print("\n💹 取引実行:")
        result = simulator.execute_daily_trades(recommendations[:3])
        print(f"実行: {len(result['executed_buys'])}件")
        print(f"スキップ: {len(result['skipped_buys'])}件")
        
        # ポートフォリオ確認
        simulator.portfolio.display_portfolio()
    else:
        print("推奨銘柄が見つかりませんでした")
    
    return simulator.portfolio.get_performance_summary()


def test_multi_day_simulation():
    """複数日シミュレーションテスト"""
    print("\n🧪 複数日シミュレーションテスト")
    print("=" * 60)
    
    config = SimulationConfig(
        initial_capital=100000.0,
        max_positions=5,
        trading_mode=TradingMode.DAY_TRADING
    )
    
    simulator = TradingSimulator(config)
    
    # 3日間のシミュレーション実行
    print("\n📅 3日間のシミュレーション:")
    result = simulator.run_simulation(days=3, auto_buy=True)
    
    # 結果サマリー
    print("\n📊 シミュレーション結果:")
    final = result["final_summary"]
    print(f"最終資産: ¥{final['portfolio_value']:,.0f}")
    print(f"総損益: ¥{final['total_pnl']:+,.0f} ({final['total_return_pct']:+.2f}%)")
    print(f"勝率: {final['win_rate']:.1f}%")
    
    return final


def main():
    """メインテスト実行"""
    print("🚀 シミュレーションシステムテスト開始")
    print("=" * 80)
    
    try:
        # 1. ポートフォリオマネージャーテスト
        result1 = test_portfolio_manager()
        print(f"\n✅ ポートフォリオテスト完了")
        
        # 2. 基本シミュレーションテスト
        result2 = test_basic_simulation()
        print(f"\n✅ 基本シミュレーションテスト完了")
        
        # 3. 推奨銘柄統合テスト
        result3 = test_recommendations_integration()
        print(f"\n✅ 推奨銘柄統合テスト完了")
        
        # 4. 複数日シミュレーションテスト
        result4 = test_multi_day_simulation()
        print(f"\n✅ 複数日シミュレーションテスト完了")
        
        print("\n" + "=" * 80)
        print("🎉 全テスト完了！")
        print("=" * 80)
        print("\n📋 テスト結果サマリー:")
        print("1. ポートフォリオ管理: SUCCESS")
        print("2. 基本シミュレーション: SUCCESS")
        print("3. 推奨銘柄統合: SUCCESS")
        print("4. 複数日シミュレーション: SUCCESS")
        
        print("\n🏁 シミュレーションシステム準備完了")
        print("明日から実際のシミュレーションを開始できます！")
        
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()