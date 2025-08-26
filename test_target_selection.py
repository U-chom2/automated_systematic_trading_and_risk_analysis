"""Test script for target selection with integrated scoring.

Tests the complete flow from Excel loading to investment decisions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.system_core.target_selector import TargetSelector
from src.analysis_engine.integrated_scorer import IntegratedScorer
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_scoring_system():
    """Test the scoring system with sample data."""
    print("\n" + "=" * 80)
    print("📊 スコアリングシステムのテスト")
    print("=" * 80)
    
    scorer = IntegratedScorer()
    
    # Test case 1: High score scenario
    test_data_high = {
        "catalyst": {
            "title": "業績予想の上方修正に関するお知らせ",
            "keywords": ["上方修正", "業績"],
            "importance": "high"
        },
        "sentiment": {
            "positive_ratio": 0.85,
            "change_rate": 2.0,  # 100% increase
            "mention_count": 250
        },
        "technical": {
            "trend": "uptrend",
            "sector_performance": 0.02,
            "volume_ratio": 2.0,
            "rsi": 65,
            "ma_deviation": 0.10
        }
    }
    
    result_high = scorer.analyze_with_breakdown(test_data_high)
    
    print("\n【高スコアケース】")
    print(f"カタリストスコア: {result_high['scores']['catalyst_score']}/50")
    print(f"センチメントスコア: {result_high['scores']['sentiment_score']}/30")
    print(f"テクニカルスコア: {result_high['scores']['technical_score']}/20")
    print(f"合計スコア: {result_high['scores']['total_score']}/100")
    print(f"投資判断: {result_high['decision']} ({'実行' if result_high['execute'] else '見送り'})")
    print(f"理由: {result_high['decision_rationale']}")
    
    # Test case 2: Low score scenario
    test_data_low = {
        "catalyst": None,
        "sentiment": {
            "positive_ratio": 0.45,
            "change_rate": 0.8,
            "mention_count": 20
        },
        "technical": {
            "trend": "downtrend",
            "sector_performance": -0.01,
            "volume_ratio": 0.8,
            "rsi": 35,
            "ma_deviation": -0.05
        }
    }
    
    result_low = scorer.analyze_with_breakdown(test_data_low)
    
    print("\n【低スコアケース】")
    print(f"カタリストスコア: {result_low['scores']['catalyst_score']}/50")
    print(f"センチメントスコア: {result_low['scores']['sentiment_score']}/30")
    print(f"テクニカルスコア: {result_low['scores']['technical_score']}/20")
    print(f"合計スコア: {result_low['scores']['total_score']}/100")
    print(f"投資判断: {result_low['decision']} ({'実行' if result_low['execute'] else '見送り'})")
    print(f"理由: {result_low['decision_rationale']}")
    
    # Test case 3: High score but filtered (RSI overbought)
    test_data_filtered = test_data_high.copy()
    test_data_filtered["technical"]["rsi"] = 80  # Overbought
    
    result_filtered = scorer.analyze_with_breakdown(test_data_filtered)
    
    print("\n【フィルター条件該当ケース（RSI > 75）】")
    print(f"合計スコア: {result_filtered['scores']['total_score']}/100")
    print(f"投資判断: {result_filtered['decision']} ({'実行' if result_filtered['execute'] else '見送り'})")
    print(f"理由: {result_filtered['decision_rationale']}")


def test_target_selection():
    """Test target selection from Excel watchlist."""
    print("\n" + "=" * 80)
    print("🎯 ターゲット企業選定テスト")
    print("=" * 80)
    
    try:
        # Initialize selector
        selector = TargetSelector()
        
        # Load watchlist
        print("\n1. ウォッチリスト読み込み")
        print("-" * 40)
        companies = selector.load_targets()
        print(f"✅ {len(companies)} 社を読み込みました")
        
        # Show first 3 companies
        print("\n最初の3社:")
        for company in companies[:3]:
            print(f"  - {company.get('symbol')}: {company.get('company_name')}")
        
        # Analyze top targets (limited to 3 for demo)
        print("\n2. トップターゲット分析")
        print("-" * 40)
        print("分析中... (これには数分かかる場合があります)")
        
        # Note: This will make real API calls - limit to 3 for demo
        top_targets = selector.select_top_targets(max_targets=3)
        
        if top_targets:
            print(f"\n✅ {len(top_targets)} 社が買い推奨として選定されました")
            
            for i, target in enumerate(top_targets, 1):
                print(f"\n【推奨 {i}】")
                print(f"銘柄: {target.get('symbol')} - {target.get('company_name')}")
                print(f"スコア内訳:")
                print(f"  カタリスト: {target['scores']['catalyst_score']}/50")
                print(f"  センチメント: {target['scores']['sentiment_score']}/30")
                print(f"  テクニカル: {target['scores']['technical_score']}/20")
                print(f"  合計: {target['scores']['total_score']}/100")
                
                if target.get('metadata'):
                    meta = target['metadata']
                    print(f"企業情報:")
                    print(f"  時価総額: {meta.get('market_cap')} 百万円")
                    print(f"  主要テーマ: {meta.get('theme')}")
                    print(f"  業績トレンド: {meta.get('performance_trend')}")
        else:
            print("\n❌ 買い推奨基準を満たす銘柄はありませんでした")
        
        # Show summary
        print("\n3. 分析サマリー")
        print("-" * 40)
        summary = selector.get_analysis_summary()
        
        if summary.get("statistics"):
            stats = summary["statistics"]
            print(f"分析銘柄数: {stats.get('total_analyzed', 0)}")
            print(f"買い候補数: {stats.get('buy_candidates', 0)}")
            print(f"選定数: {stats.get('selected', 0)}")
            
            if summary.get("score_distribution"):
                print(f"\nスコア分布:")
                for range_name, count in summary["score_distribution"].items():
                    print(f"  {range_name}: {count}社")
        
        # Export results
        output_path = Path("analysis_results.json")
        if selector.export_results(output_path):
            print(f"\n✅ 分析結果を {output_path} に保存しました")
            
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    print("=" * 80)
    print("🚀 ターゲット企業選定システム 統合テスト")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Test scoring system first (no API calls)
    test_scoring_system()
    
    # Ask before running full analysis
    print("\n" + "=" * 80)
    print("⚠️  注意: 次のテストは実際のAPIを使用します")
    print("実行しますか？ (y/n): ", end="")
    
    user_input = input().strip().lower()
    if user_input == 'y':
        test_target_selection()
    else:
        print("テストをスキップしました")
    
    print("\n" + "=" * 80)
    print("✅ テスト完了")
    print("=" * 80)


if __name__ == "__main__":
    main()