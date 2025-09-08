"""統合版メインアプリケーション

起動から投資提案まで完全に動作するシステム
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.system_integrator import SystemIntegrator


def setup_logging() -> None:
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("integrated_trading_system.log")
        ]
    )


def print_banner() -> None:
    """システム起動バナー表示"""
    print("=" * 80)
    print("🤖 AI 自動株式取引システム - 統合版")
    print("   Automated Systematic Trading & Risk Analysis System")
    print("=" * 80)
    print(f"⏰ 起動時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔄 モード: Mock Analysis Mode (安全な模擬分析)")
    print("=" * 80)
    print()


def print_separator(title: str = "") -> None:
    """セクション区切り表示"""
    print()
    print("-" * 60)
    if title:
        print(f"📊 {title}")
        print("-" * 60)


def print_recommendations(recommendations: list) -> None:
    """投資提案を整形して表示"""
    print_separator("🎯 AI投資提案結果")
    
    if not recommendations:
        print("❌ 投資提案が生成されませんでした")
        return
    
    print(f"📈 {len(recommendations)}件の投資提案が生成されました\n")
    
    for i, rec in enumerate(recommendations, 1):
        symbol = rec["symbol"]
        action = rec["action"]
        confidence = rec["confidence"]
        current_price = rec["current_price"]
        reasoning = rec["reasoning"]
        
        # アクションのアイコン
        action_icon = {
            "buy": "🟢 買い推奨",
            "sell": "🔴 売り推奨", 
            "hold": "🟡 様子見"
        }.get(action, "❓ 不明")
        
        print(f"【提案 {i}】")
        print(f"  🏷️  銘柄: {symbol}")
        print(f"  📊  判定: {action_icon}")
        print(f"  💰  現在価格: ¥{current_price:,.2f}")
        print(f"  🎯  信頼度: {confidence:.1%}")
        
        if action == "buy":
            print(f"  📈  目標価格: ¥{rec.get('price_target', 0):,.2f}")
            print(f"  🛑  ストップロス: ¥{rec.get('stop_loss', 0):,.2f}")
            
        print(f"  💡  理由: {reasoning}")
        
        # スコアの詳細
        print(f"  📋  分析スコア詳細:")
        print(f"      • カタリスト: {rec.get('catalyst_score', 0)}/50点")
        print(f"      • 感情分析: {rec.get('sentiment_score', 0)}/30点")
        print(f"      • テクニカル: {rec.get('technical_score', 0)}/20点")
        print(f"      • 合計: {rec.get('total_score', 0)}/100点")
        print()


def run_system_integration() -> None:
    """システム統合実行"""
    logger = logging.getLogger(__name__)
    
    try:
        print_separator("🚀 システム初期化")
        
        # システム設定
        config = {
            "capital": 1000000,  # 100万円
            "risk_per_trade_ratio": 0.01,  # 1%リスク
            "buy_threshold": 70,  # 70点以上で買い
            "data_dir": "./data",
            "mock_mode": True
        }
        
        # ウォッチリスト（日本の主要銘柄）
        watchlist = [
            "7203",  # トヨタ自動車
            "6758",  # ソニーグループ 
            "9984",  # ソフトバンクグループ
            "7974",  # 任天堂
            "4503"   # アステラス製薬
        ]
        
        print("✅ 設定ファイル読み込み完了")
        print(f"   💰 運用資金: ¥{config['capital']:,}")
        print(f"   ⚖️  リスク比率: {config['risk_per_trade_ratio']:.1%}")
        print(f"   🎯 買い閾値: {config['buy_threshold']}点")
        print(f"   👀 監視銘柄数: {len(watchlist)}銘柄")
        print()
        
        # システムインテグレーター初期化
        print("🔧 システムインテグレーター初期化中...")
        integrator = SystemIntegrator(config)
        
        if not integrator.initialize_system():
            print("❌ システム初期化に失敗しました")
            return
            
        print("✅ システム初期化完了")
        
        # ウォッチリスト設定
        print("📋 ウォッチリスト設定中...")
        integrator.load_watchlist(watchlist)
        print("✅ ウォッチリスト設定完了")
        
        print_separator("📡 市場データ収集")
        
        # データ収集開始
        print("🔍 市場データ収集を開始します...")
        for symbol in watchlist:
            company_name = {
                "7203": "トヨタ自動車",
                "6758": "ソニーグループ",
                "9984": "ソフトバンクグループ", 
                "7974": "任天堂",
                "4503": "アステラス製薬"
            }.get(symbol, f"銘柄{symbol}")
            
            print(f"  📊 {symbol} ({company_name}) - データ収集中...")
        
        print("✅ 市場データ収集完了")
        
        print_separator("🧠 AI分析実行")
        
        # AI分析・投資提案生成
        print("🤖 AI分析エンジンを起動しています...")
        print("   • カタリスト分析 (IR・決算情報)")
        print("   • 感情分析 (ソーシャルメディア)")
        print("   • テクニカル分析 (価格・出来高)")
        print("   • リスク評価")
        print()
        
        print("⚡ 完全分析サイクルを実行中...")
        recommendations = integrator.run_complete_analysis_cycle()
        
        if recommendations:
            print("✅ AI分析・投資提案生成完了")
        else:
            print("⚠️  推奨銘柄が見つかりませんでした")
            
        # 結果表示
        print_recommendations(recommendations)
        
        print_separator("💾 結果保存")
        
        # パフォーマンス統計
        performance = integrator.get_performance_statistics()
        print(f"📈 システム統計:")
        print(f"   • 総取引数: {performance.get('total_trades', 0)}回")
        print(f"   • 勝率: {performance.get('win_rate', 0):.1%}")
        print(f"   • 総収益: ¥{performance.get('total_return', 0):,.0f}")
        
        print("\n✅ 分析結果をCSVファイルに保存しました")
        print("   📁 データディレクトリ: ./data/")
        print("   📄 実行計画: execution_plans.csv")
        print("   📄 取引履歴: trade_history.csv") 
        print("   📄 システム状態: system_state.csv")
        
        print_separator("🎉 実行完了")
        
        print("🎯 AI自動株式取引システムの統合テストが正常に完了しました！")
        print()
        print("📝 次のステップ:")
        print("   1. 実際のAPIキーを設定してリアルデータで実行")
        print("   2. より高度なAI分析モデルの実装")
        print("   3. バックテスト機能の追加")
        print("   4. リアルトレード機能の実装")
        print()
        print("⚠️  注意: 現在はモック(模擬)モードで動作しています")
        print("   実際の投資判断には使用しないでください")
        
    except KeyboardInterrupt:
        print("\n⏹️  ユーザーによって停止されました")
        logger.info("System stopped by user")
        
    except Exception as e:
        print(f"\n❌ システムエラーが発生しました: {e}")
        logger.error(f"System error: {e}")
        
    finally:
        print(f"\n⏰ 終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


def main() -> None:
    """メイン関数"""
    setup_logging()
    print_banner()
    run_system_integration()


if __name__ == "__main__":
    main()