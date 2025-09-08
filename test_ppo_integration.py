"""
PPO統合システムの動作確認テスト
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

from investment_analyzer import InvestmentAnalyzer
from config import Config, TradingMode
from technical_analyzer import TechnicalIndicators

# ログレベル設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ppo_initialization():
    """PPO統合システム初期化テスト"""
    print("=== PPO統合システム動作確認 ===")
    print()
    
    try:
        # PPOモデル使用での初期化
        config = Config(TradingMode.DAY_TRADING)
        analyzer = InvestmentAnalyzer(config, use_ppo=True)
        
        # システム情報確認
        trading_info = analyzer.get_trading_info()
        print("📊 システム情報:")
        print(f"  取引モード: {trading_info['trading_mode']}")
        print(f"  分析手法: {trading_info['scoring_method']}")
        
        if 'ppo_model_info' in trading_info:
            ppo_info = trading_info['ppo_model_info']
            print(f"  PPOモデル: {ppo_info['model_name']}")
            print(f"  デバイス: {ppo_info['device']}")
            print(f"  ロード状態: {ppo_info['is_loaded']}")
        
        print()
        print("✅ PPO統合システム初期化完了")
        return analyzer
        
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return None

def test_single_stock_analysis(analyzer):
    """単一銘柄分析テスト"""
    if not analyzer:
        return
        
    print("\n=== 単一銘柄分析テスト ===")
    
    try:
        # Toyota (7203.T) を分析
        result = analyzer.analyze_single_stock("7203.T", "トヨタ自動車", 35000000)
        
        if result:
            print("📈 分析結果 - トヨタ自動車 (7203.T):")
            print(f"  現在株価: ¥{result['現在株価']:,.0f}")
            print(f"  投資スコア: {result['投資スコア']:.1f}点")
            print(f"  投資判断: {result['投資判断']}")
            print(f"  分析手法: {result.get('分析手法', 'Unknown')}")
            
            # PPO特有情報の表示
            if 'PPOモデル' in result:
                print(f"  PPOモデル: {result['PPOモデル']}")
                print(f"  PPOアクション値: {result.get('PPOアクション値', 0):.3f}")
                print(f"  PPO判断: {result.get('PPO判断', 'Unknown')}")
                print(f"  PPO信頼度: {result.get('PPO信頼度', 0):.3f}")
            
            print("✅ 単一銘柄分析成功")
        else:
            print("❌ 分析結果なし")
            
    except Exception as e:
        print(f"❌ 単一銘柄分析エラー: {e}")

def test_multiple_stocks_analysis(analyzer):
    """複数銘柄分析テスト（軽量版）"""
    if not analyzer:
        return
        
    print("\n=== 複数銘柄分析テスト（3銘柄） ===")
    
    try:
        # データフェッチャーを使用して企業リストの最初の3社を取得
        companies_df = analyzer.data_fetcher.load_target_companies()
        test_companies = companies_df.head(3)
        
        results = []
        for _, company in test_companies.iterrows():
            symbol = analyzer.data_fetcher.create_symbol_from_code(company['証券コード'])
            company_name = company['企業名']
            market_cap = company.get('時価総額 (百万円)', 1500)
            
            print(f"📊 {company_name} ({symbol}) 分析中...")
            
            result = analyzer.analyze_single_stock(symbol, company_name, market_cap)
            
            if result:
                results.append(result)
                print(f"  投資スコア: {result['投資スコア']:.1f}点 - {result['投資判断']}")
                print(f"  分析手法: {result.get('分析手法', 'Unknown')}")
            else:
                print("  ❌ 分析失敗")
        
        print(f"\n✅ {len(results)}/3 銘柄の分析完了")
        
        # スコア順でソート
        if results:
            results.sort(key=lambda x: x['投資スコア'], reverse=True)
            print("\n📊 分析結果サマリー (スコア順):")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['企業名']}: {result['投資スコア']:.1f}点 ({result['投資判断']})")
        
    except Exception as e:
        print(f"❌ 複数銘柄分析エラー: {e}")

def test_ppo_vs_traditional():
    """PPOモデル vs 従来手法の比較テスト"""
    print("\n=== PPOモデル vs 従来手法 比較テスト ===")
    
    try:
        config = Config(TradingMode.DAY_TRADING)
        
        # PPOモデル使用
        ppo_analyzer = InvestmentAnalyzer(config, use_ppo=True)
        
        # 従来手法使用
        traditional_analyzer = InvestmentAnalyzer(config, use_ppo=False)
        
        test_symbol = "7203.T"
        test_company = "トヨタ自動車"
        
        print(f"📊 {test_company} ({test_symbol}) の比較分析:")
        
        # PPO分析
        ppo_result = ppo_analyzer.analyze_single_stock(test_symbol, test_company, 35000000)
        
        # 従来分析
        traditional_result = traditional_analyzer.analyze_single_stock(test_symbol, test_company, 35000000)
        
        if ppo_result and traditional_result:
            print("\n📈 PPOモデル結果:")
            print(f"  投資スコア: {ppo_result['投資スコア']:.1f}点")
            print(f"  投資判断: {ppo_result['投資判断']}")
            print(f"  分析手法: {ppo_result.get('分析手法', 'Unknown')}")
            
            print("\n📊 従来手法結果:")
            print(f"  投資スコア: {traditional_result['投資スコア']:.1f}点")
            print(f"  投資判断: {traditional_result['投資判断']}")
            print(f"  分析手法: {traditional_result.get('分析手法', 'Unknown')}")
            
            # 差分分析
            score_diff = ppo_result['投資スコア'] - traditional_result['投資スコア']
            print(f"\n🔍 分析結果差分:")
            print(f"  スコア差: {score_diff:+.1f}点")
            print(f"  PPO判断: {ppo_result['投資判断']}")
            print(f"  従来判断: {traditional_result['投資判断']}")
            
            print("\n✅ 比較分析完了")
        else:
            print("❌ 比較分析失敗")
            
    except Exception as e:
        print(f"❌ 比較テストエラー: {e}")

def main():
    """メイン実行関数"""
    print(f"🚀 PPO統合システム動作確認開始 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. システム初期化テスト
    analyzer = test_ppo_initialization()
    
    # 2. 単一銘柄分析テスト
    test_single_stock_analysis(analyzer)
    
    # 3. 複数銘柄分析テスト
    test_multiple_stocks_analysis(analyzer)
    
    # 4. PPO vs 従来手法比較テスト
    test_ppo_vs_traditional()
    
    print("\n" + "=" * 80)
    print("🎉 PPO統合システム動作確認完了")

if __name__ == "__main__":
    main()