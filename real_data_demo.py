#!/usr/bin/env python
"""リアルデータ統合デモンストレーション

実際のYahoo FinanceとTDnetのデータを使用して
システムの動作を確認します。
"""

import sys
import json
from datetime import datetime
from typing import Dict, Any, List

from src.system_integrator_real import SystemIntegratorReal


def print_section(title: str) -> None:
    """セクションタイトルを表示"""
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)


def run_real_data_demo() -> None:
    """リアルデータデモを実行"""
    
    print("\n🚀 自動売買システム リアルデータ統合デモ")
    print("━" * 60)
    
    # テスト用銘柄
    test_symbols = [
        "7203",  # トヨタ自動車
        "6758",  # ソニーグループ
        "9984",  # ソフトバンクグループ
    ]
    
    # リアルデータモードでシステムを初期化
    config = {
        "use_real_data": True,  # リアルデータを使用
        "max_positions": 3,
        "risk_limit": 0.02,
        "confidence_threshold": 0.7
    }
    
    print("\n⚙️ システム設定:")
    print(f"  • データソース: {'リアルデータ' if config['use_real_data'] else 'モックデータ'}")
    print(f"  • 分析対象銘柄: {', '.join(test_symbols)}")
    print(f"  • リスク上限: {config['risk_limit']*100:.1f}%")
    print(f"  • 信頼度閾値: {config['confidence_threshold']*100:.0f}%")
    
    # システム初期化
    print("\n🔄 システムを初期化中...")
    system = SystemIntegratorReal(config)
    
    # 1. 市場状態の確認
    print_section("市場状態確認")
    market_status = system.get_market_status()
    print(f"  • 市場: {market_status['market']}")
    print(f"  • 状態: {'🟢 開場中' if market_status['is_open'] else '🔴 閉場中'}")
    print(f"  • 現在時刻: {market_status['current_time']}")
    if market_status['next_open']:
        print(f"  • 次回開場: {market_status['next_open']}")
    if market_status['next_close']:
        print(f"  • 次回閉場: {market_status['next_close']}")
    
    # 2. 市場データ収集
    print_section("市場データ収集")
    print("📡 リアルタイムデータを取得中...")
    
    market_data = system.collect_market_data(test_symbols)
    
    for symbol, data in market_data.items():
        price_data = data.get("price_data", {})
        ir_releases = data.get("ir_releases", [])
        technical = data.get("technical_indicators", {})
        
        print(f"\n📌 {symbol}:")
        print(f"  価格情報:")
        print(f"    • 現在価格: ¥{price_data.get('current_price', 0):,.0f}")
        print(f"    • 出来高: {price_data.get('volume', 0):,}")
        print(f"    • 高値: ¥{price_data.get('high', 0):,.0f}")
        print(f"    • 安値: ¥{price_data.get('low', 0):,.0f}")
        
        if technical and technical.get('rsi') is not None:
            print(f"  テクニカル指標:")
            print(f"    • RSI: {technical.get('rsi', 0):.1f}")
            if technical.get('sma_20'):
                print(f"    • SMA(20): ¥{technical.get('sma_20'):,.0f}")
            if technical.get('macd', {}).get('histogram'):
                macd_signal = "買い" if technical['macd']['histogram'] > 0 else "売り"
                print(f"    • MACD: {macd_signal}シグナル")
        
        print(f"  IR情報: {len(ir_releases)}件の開示")
        if ir_releases:
            latest = ir_releases[0]
            print(f"    • 最新: {latest.get('title', '')[:40]}...")
    
    # 3. AI分析実行
    print_section("AI分析実行")
    print("🤖 データを分析中...")
    
    analysis_results = system.perform_ai_analysis(market_data)
    
    # 分析結果の表示
    recommendations = []
    
    for symbol, analysis in analysis_results.items():
        print(f"\n📈 {symbol} 分析結果:")
        print(f"  • カタリストスコア: {analysis['catalyst_score']}/50")
        print(f"  • センチメントスコア: {analysis['sentiment_score']}/30")
        print(f"  • テクニカルスコア: {analysis['technical_score']}/20")
        print(f"  • 総合スコア: {analysis['total_score']}/100")
        print(f"  • 信頼度: {analysis['confidence']*100:.1f}%")
        print(f"  • データソース: {analysis['data_source']}")
        
        # 投資判断
        if analysis['confidence'] >= config['confidence_threshold']:
            recommendations.append({
                'symbol': symbol,
                'confidence': analysis['confidence'],
                'score': analysis['total_score'],
                'price': analysis['current_price']
            })
    
    # 4. 投資推奨
    print_section("投資推奨")
    
    if recommendations:
        print("✅ 以下の銘柄を推奨します:\n")
        
        # スコア順にソート
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['symbol']}")
            print(f"     • 推奨度: {'⭐' * min(5, int(rec['score']/20))}")
            print(f"     • 信頼度: {rec['confidence']*100:.1f}%")
            print(f"     • 現在価格: ¥{rec['price']:,.0f}")
            
            # 推奨ポジションサイズ（リスク管理）
            position_size = min(1000000, int(1000000 * rec['confidence']))
            shares = position_size // rec['price'] if rec['price'] > 0 else 0
            print(f"     • 推奨株数: {shares:,}株")
            print(f"     • 推奨投資額: ¥{shares * rec['price']:,.0f}")
    else:
        print("⚠️ 現在、推奨条件を満たす銘柄はありません。")
        print("   市場状況を継続的に監視することをお勧めします。")
    
    # 5. システムサマリー
    print_section("実行サマリー")
    
    print("📊 分析結果:")
    print(f"  • 分析銘柄数: {len(test_symbols)}")
    print(f"  • 推奨銘柄数: {len(recommendations)}")
    
    if recommendations:
        total_investment = sum(rec['price'] * (1000000 // rec['price']) for rec in recommendations[:config['max_positions']])
        print(f"  • 推奨投資総額: ¥{total_investment:,.0f}")
        avg_confidence = sum(rec['confidence'] for rec in recommendations) / len(recommendations)
        print(f"  • 平均信頼度: {avg_confidence*100:.1f}%")
    
    print("\n💡 次のアクション:")
    if market_status['is_open']:
        print("  • 市場が開いているため、推奨銘柄の購入を検討してください")
    else:
        print("  • 市場が閉じているため、次回開場時に推奨銘柄の購入を検討してください")
    print("  • 市場状況を継続的に監視してください")
    print("  • リスク管理を徹底してください")
    
    print("\n" + "=" * 60)
    print("✨ リアルデータ統合デモ完了!")
    print("=" * 60)


def main() -> None:
    """メインエントリポイント"""
    try:
        run_real_data_demo()
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️ ユーザーによって中断されました")
        return 1
    except Exception as e:
        print(f"\n\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())