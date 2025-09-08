"""Demonstration of FR-01 and FR-02 Data Collection Components."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_collector.watchlist_manager import WatchlistManager, ScreeningCriteria
from src.data_collector.tdnet_scraper import TdnetScraper
from src.data_collector.x_streamer import XStreamer
from src.data_collector.yahoo_board_scraper import YahooBoardScraper
from src.data_collector.price_fetcher import PriceFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataCollectionDemo:
    """Demonstrates the integrated data collection system."""
    
    def __init__(self):
        """Initialize all data collection components."""
        # FR-01: Watchlist Management
        self.watchlist_manager = WatchlistManager("ターゲット企業.xlsx")
        
        # FR-02: Data Collection Components
        self.tdnet_scraper = TdnetScraper(polling_interval=1)
        self.x_streamer = XStreamer(
            api_key="demo_key",
            api_secret="demo_secret",
            access_token="demo_token",
            access_token_secret="demo_token_secret"
        )
        self.yahoo_scraper = YahooBoardScraper(request_delay=1.0)
        self.price_fetcher = PriceFetcher(
            api_endpoint="https://api.demo-broker.com",
            api_key="demo_api_key"
        )
        
        self.active_watchlist = []
        self.triggers_detected = []
    
    def demo_fr01_watchlist_management(self):
        """Demonstrate FR-01: Dynamic watchlist management."""
        print("\n" + "="*60)
        print("FR-01: 監視リスト管理 (Watchlist Management)")
        print("="*60)
        
        # Load companies from Excel
        if self.watchlist_manager.load_companies_from_excel():
            print(f"✅ Excelから{len(self.watchlist_manager.companies)}社を読み込みました")
            
            # Show sample companies
            print("\n📊 サンプル企業:")
            for i, (code, company) in enumerate(list(self.watchlist_manager.companies.items())[:3]):
                print(f"  {code}: {company.name}")
                print(f"    時価総額: {company.market_cap:,}百万円")
                print(f"    業績: {company.performance_trend}")
                print(f"    テーマ: {company.main_theme}")
            
            # Apply screening criteria
            print("\n🔍 スクリーニング条件を適用:")
            criteria = ScreeningCriteria(
                max_market_cap=2000,  # 20億円以下
                performance_trends=["赤字縮小・黒字化予想", "増収増益"],
                chart_conditions=["底値圏"]
            )
            
            filtered = self.watchlist_manager.apply_screening(criteria)
            print(f"  条件: 時価総額20億円以下、成長トレンド、底値圏")
            print(f"  結果: {len(filtered)}社が条件に合致")
            
            # Set active watchlist
            self.active_watchlist = filtered[:5]  # Top 5 companies
            self.watchlist_manager.update_active_watchlist(self.active_watchlist)
            
            print(f"\n✅ アクティブウォッチリスト: {self.active_watchlist}")
            
            return True
        else:
            print("❌ Excelファイルの読み込みに失敗しました")
            return False
    
    def demo_fr02_ir_trigger_detection(self):
        """Demonstrate FR-02: IR/Press release trigger detection."""
        print("\n" + "="*60)
        print("FR-02: IRトリガー検知 (TDnet Monitoring)")
        print("="*60)
        
        # Test various IR titles
        test_releases = [
            {
                "title": "2024年3月期決算上方修正に関するお知らせ",
                "company_code": self.active_watchlist[0] if self.active_watchlist else "7203",
                "content": "売上高を前回予想から10%上方修正いたします。"
            },
            {
                "title": "株式会社○○との業務提携について",
                "company_code": self.active_watchlist[1] if len(self.active_watchlist) > 1 else "6758",
                "content": "AI分野における戦略的パートナーシップを締結。"
            },
            {
                "title": "定期株主総会開催のお知らせ",
                "company_code": "9999",
                "content": "定期株主総会を開催いたします。"
            }
        ]
        
        print("\n📰 IR発表をチェック中...")
        for release in test_releases:
            # Check if it's a trigger
            is_trigger = self.tdnet_scraper.check_for_trigger_keywords(release["title"])
            
            # Check if it's relevant to watchlist
            is_relevant = self.tdnet_scraper.is_market_relevant_release(
                release, self.active_watchlist
            )
            
            status = "🚨 トリガー検知!" if (is_trigger and is_relevant) else "  通常のIR"
            print(f"\n{status}")
            print(f"  企業: {release['company_code']}")
            print(f"  タイトル: {release['title'][:30]}...")
            print(f"  重要キーワード: {'あり' if is_trigger else 'なし'}")
            print(f"  監視リスト該当: {'はい' if is_relevant else 'いいえ'}")
            
            if is_trigger and is_relevant:
                self.triggers_detected.append(release)
    
    def demo_fr02_social_media_anomaly(self):
        """Demonstrate FR-02: Social media anomaly detection."""
        print("\n" + "="*60)
        print("FR-02: SNS異常検知 (X/Twitter Monitoring)")
        print("="*60)
        
        # Simulate social media activity
        if self.active_watchlist:
            symbol = self.active_watchlist[0]
            company = self.watchlist_manager.get_company_by_code(symbol)
            
            print(f"\n📱 {symbol} ({company.name if company else '不明'}) のSNS活動を監視中...")
            
            # Simulate normal activity
            print("\n通常時のツイート:")
            normal_tweets = [
                f"{symbol}の株価をウォッチ中",
                f"#{symbol} 決算に注目",
            ]
            
            for tweet in normal_tweets:
                mentions = self.x_streamer.extract_stock_mentions(tweet)
                sentiment = self.x_streamer._analyze_basic_sentiment(tweet)
                print(f"  ツイート: {tweet}")
                print(f"    言及銘柄: {mentions}")
                print(f"    センチメント: {sentiment:.2f}")
            
            # Simulate anomaly
            print("\n異常な活動を検知！")
            anomaly_tweets = [
                f"速報！{symbol}が上方修正発表！買いだ！",
                f"#{symbol} 急騰中！まだ間に合う！",
                f"{symbol}爆上げきた！追加投資する！",
            ] * 5  # Simulate high volume
            
            # Record mentions
            for tweet in anomaly_tweets:
                for mention in self.x_streamer.extract_stock_mentions(tweet):
                    self.x_streamer._record_mention(mention)
            
            # Check for anomaly
            current_count = self.x_streamer.get_mention_count(symbol, 60)
            threshold = self.x_streamer.calculate_anomaly_threshold(symbol, 24)
            is_anomaly = current_count > threshold
            
            print(f"\n📊 統計分析:")
            print(f"  過去1時間の言及数: {current_count}")
            print(f"  異常検知閾値 (μ+3σ): {threshold:.2f}")
            print(f"  異常検知: {'🚨 はい' if is_anomaly else 'いいえ'}")
            
            if is_anomaly:
                self.triggers_detected.append({
                    "type": "sns_anomaly",
                    "symbol": symbol,
                    "mention_count": current_count
                })
    
    def demo_fr02_board_sentiment(self):
        """Demonstrate FR-02: Yahoo Finance board sentiment analysis."""
        print("\n" + "="*60)
        print("FR-02: 掲示板センチメント分析 (Yahoo Finance)")
        print("="*60)
        
        if self.active_watchlist:
            symbol = self.active_watchlist[0]
            company = self.watchlist_manager.get_company_by_code(symbol)
            
            print(f"\n💬 {symbol} ({company.name if company else '不明'}) の掲示板分析...")
            
            # Get board analytics (simulated)
            analytics = self.yahoo_scraper.get_board_analytics(symbol)
            
            print(f"\n📊 掲示板統計:")
            print(f"  過去1時間の投稿数: {analytics.posts_last_hour}")
            print(f"  過去24時間の投稿数: {analytics.posts_last_24h}")
            print(f"  平均センチメント: {analytics.average_sentiment:.2f}")
            print(f"  異常な活動: {'🚨 検知' if analytics.unusual_activity else '正常'}")
            
            # Sentiment distribution
            print(f"\nセンチメント分布:")
            print(f"  ポジティブ: {analytics.sentiment_distribution.get('positive', 0)}")
            print(f"  ネガティブ: {analytics.sentiment_distribution.get('negative', 0)}")
            print(f"  中立: {analytics.sentiment_distribution.get('neutral', 0)}")
            
            # Top keywords
            if analytics.top_keywords:
                print(f"\n頻出キーワード: {', '.join(analytics.top_keywords)}")
    
    def demo_fr02_price_monitoring(self):
        """Demonstrate FR-02: Real-time price monitoring."""
        print("\n" + "="*60)
        print("FR-02: リアルタイム価格監視 (Price Monitoring)")
        print("="*60)
        
        if self.active_watchlist:
            symbol = self.active_watchlist[0]
            company = self.watchlist_manager.get_company_by_code(symbol)
            
            print(f"\n💹 {symbol} ({company.name if company else '不明'}) の価格データ...")
            
            # Get price data (simulated)
            price_data = self.price_fetcher.get_price_data(symbol)
            
            print(f"\n現在の価格情報:")
            print(f"  現在値: ¥{price_data['current_price']:,.2f}")
            print(f"  始値: ¥{price_data['open_price']:,.2f}")
            print(f"  高値: ¥{price_data['high_price']:,.2f}")
            print(f"  安値: ¥{price_data['low_price']:,.2f}")
            print(f"  出来高: {price_data['volume']:,}")
            print(f"  前日比: {price_data['change']:+.2f} ({price_data['change_percent']:+.2f}%)")
            
            # Technical indicators
            volatility = self.price_fetcher.calculate_volatility(symbol, 60)
            atr = self.price_fetcher.calculate_atr(symbol, 14)
            
            print(f"\nテクニカル指標:")
            print(f"  ヒストリカル・ボラティリティ(60日): {volatility:.2%}")
            print(f"  ATR(14日): {atr:.2f}")
            
            # Market status
            is_open = self.price_fetcher.is_market_open()
            print(f"\n市場状態: {'🟢 開場中' if is_open else '🔴 閉場中'}")
    
    def show_trigger_summary(self):
        """Show summary of all detected triggers."""
        print("\n" + "="*60)
        print("トリガーサマリー (Trigger Summary)")
        print("="*60)
        
        if self.triggers_detected:
            print(f"\n🚨 検出されたトリガー: {len(self.triggers_detected)}件")
            
            for i, trigger in enumerate(self.triggers_detected, 1):
                print(f"\n{i}. ", end="")
                if "title" in trigger:  # IR trigger
                    print(f"IRトリガー")
                    print(f"   企業: {trigger['company_code']}")
                    print(f"   タイトル: {trigger['title'][:40]}...")
                elif trigger.get("type") == "sns_anomaly":  # SNS trigger
                    print(f"SNS異常トリガー")
                    print(f"   銘柄: {trigger['symbol']}")
                    print(f"   言及数: {trigger['mention_count']}")
            
            print("\n⚡ アクション: これらのトリガーは分析エンジンに送信されます")
        else:
            print("\n✅ 現在、検出されたトリガーはありません")
    
    def run_demo(self):
        """Run the complete demonstration."""
        print("\n" + "="*80)
        print("🤖 AIデイトレードシステム - データ収集層デモンストレーション")
        print("="*80)
        print("\nこのデモでは、FR-01（監視リスト管理）とFR-02（リアルタイムデータ収集）の")
        print("実装を実演します。")
        
        # FR-01: Watchlist Management
        if not self.demo_fr01_watchlist_management():
            print("\n❌ ウォッチリストの初期化に失敗しました")
            return
        
        # FR-02: Data Collection Components
        self.demo_fr02_ir_trigger_detection()
        self.demo_fr02_social_media_anomaly()
        self.demo_fr02_board_sentiment()
        self.demo_fr02_price_monitoring()
        
        # Summary
        self.show_trigger_summary()
        
        print("\n" + "="*80)
        print("✅ データ収集層のデモンストレーション完了")
        print("="*80)
        print("\n次のステップ:")
        print("1. 分析エンジン (AnalysisEngine) がトリガーを受け取り分析")
        print("2. 実行管理 (ExecutionManager) が取引判断を実行")
        print("3. システムコア (SystemCore) が全体を統合管理")


def main():
    """Main entry point for the demo."""
    demo = DataCollectionDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()