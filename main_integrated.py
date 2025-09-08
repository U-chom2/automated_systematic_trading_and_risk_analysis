#!/usr/bin/env python
"""統合メインシステム - AI駆動型自動売買・リスク分析"""
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.application.use_cases.ai_analysis import AIAnalysisUseCase, RealtimeAnalysisUseCase
from src.domain.services.signal_generator import SignalGenerator
from src.domain.services.risk_manager import RiskManager
from src.infrastructure.ai_models.ppo_integration import PPOModelIntegration
from src.common.logging import setup_logging, get_logger
from src.common.config import settings


# ロギング設定
setup_logging(log_level="INFO")
logger = get_logger(__name__)


class IntegratedTradingSystem:
    """統合取引システム"""
    
    def __init__(self):
        """初期化"""
        logger.info("Initializing Integrated Trading System...")
        
        # コンポーネント初期化
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.ppo_model = PPOModelIntegration()
        self.ai_analysis = AIAnalysisUseCase(
            signal_generator=self.signal_generator,
            ppo_model=self.ppo_model
        )
        
        logger.info("System initialization completed")
    
    async def run_daily_analysis(self) -> Dict:
        """日次分析を実行"""
        logger.info("=" * 80)
        logger.info(f"📊 日次分析開始 - {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
        logger.info("=" * 80)
        
        try:
            # 推奨銘柄を取得
            recommendations = await self.ai_analysis.get_recommendations(
                num_stocks=10,
                min_confidence=0.5
            )
            
            # 結果を表示
            self._display_recommendations(recommendations)
            
            # リスク評価
            risk_assessment = self._assess_portfolio_risk(recommendations)
            self._display_risk_assessment(risk_assessment)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "recommendations": recommendations,
                "risk_assessment": risk_assessment
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}
    
    def _display_recommendations(self, recommendations: List[Dict]):
        """推奨銘柄を表示"""
        print("\n" + "=" * 80)
        print("🎯 AI推奨銘柄（統合分析）")
        print("=" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n【第{i}位】 {rec.get('ticker', 'N/A')} - {rec.get('company_name', 'Unknown')}")
            print(f"  📈 総合スコア: {rec.get('total_score', 0):.1f}")
            print(f"  💰 現在株価: ¥{rec.get('current_price', 0):,.0f}")
            print(f"  📊 5日リターン: {rec.get('5d_return', 0):.2f}%")
            print(f"  🤖 AI信頼度: {rec.get('max_confidence', 0):.1%}")
            
            if rec.get('per', 0) > 0:
                print(f"  📊 PER: {rec['per']:.2f}")
            
            # シグナル詳細
            if rec.get('signals'):
                print("  📝 検出シグナル:")
                for signal in rec['signals'][:3]:  # 上位3つ
                    print(f"    • {signal['source']}: 信頼度 {signal['confidence']:.1%}")
    
    def _assess_portfolio_risk(self, recommendations: List[Dict]) -> Dict:
        """ポートフォリオリスクを評価"""
        if not recommendations:
            return {"risk_level": "N/A", "score": 0}
        
        # 簡易リスク評価
        avg_confidence = sum(r.get('max_confidence', 0) for r in recommendations) / len(recommendations)
        
        if avg_confidence > 0.7:
            risk_level = "低リスク"
            risk_score = 20
        elif avg_confidence > 0.5:
            risk_level = "中リスク"
            risk_score = 50
        else:
            risk_level = "高リスク"
            risk_score = 80
        
        return {
            "risk_level": risk_level,
            "score": risk_score,
            "avg_confidence": avg_confidence,
            "recommendation_count": len(recommendations)
        }
    
    def _display_risk_assessment(self, risk_assessment: Dict):
        """リスク評価を表示"""
        print("\n" + "=" * 80)
        print("🛡️ リスク評価")
        print("=" * 80)
        print(f"  リスクレベル: {risk_assessment['risk_level']}")
        print(f"  リスクスコア: {risk_assessment['score']}/100")
        print(f"  平均信頼度: {risk_assessment.get('avg_confidence', 0):.1%}")
        print(f"  分析銘柄数: {risk_assessment.get('recommendation_count', 0)}")
    
    async def start_realtime_monitoring(self, tickers: List[str] = None):
        """リアルタイム監視を開始"""
        if not tickers:
            tickers = [
                "7203.T",  # トヨタ
                "9984.T",  # ソフトバンクG
                "6758.T",  # ソニー
                "4661.T",  # オリエンタルランド
                "8058.T",  # 三菱商事
            ]
        
        logger.info(f"Starting real-time monitoring for: {tickers}")
        
        realtime = RealtimeAnalysisUseCase(
            ai_analysis=self.ai_analysis,
            update_interval=300  # 5分ごと
        )
        
        async def signal_callback(signals):
            """シグナル発生時のコールバック"""
            logger.info(f"⚡ Strong signals detected: {len(signals)}")
            for signal in signals:
                print(f"  {signal.ticker}: {signal.signal_type} "
                      f"(confidence: {signal.confidence:.1%})")
        
        await realtime.start_monitoring(tickers, signal_callback)


async def main():
    """メイン処理"""
    print("\n" + "=" * 80)
    print("🚀 AI駆動型自動売買・リスク分析システム")
    print("    Integrated Trading System v2.0")
    print("=" * 80)
    
    # システム初期化
    system = IntegratedTradingSystem()
    
    # メニュー表示
    print("\n実行モードを選択してください:")
    print("1. 日次分析を実行")
    print("2. リアルタイム監視を開始")
    print("3. 両方を実行")
    print("0. 終了")
    
    try:
        choice = input("\n選択 (0-3): ").strip()
    except KeyboardInterrupt:
        print("\n終了します")
        return
    
    if choice == "1":
        # 日次分析
        result = await system.run_daily_analysis()
        
        # 市場状況を表示
        print("\n" + "=" * 80)
        print("📊 市場状況")
        print("=" * 80)
        
        import yfinance as yf
        
        # 日経225
        nikkei = yf.Ticker("^N225")
        nikkei_hist = nikkei.history(period="2d")
        if not nikkei_hist.empty:
            latest = nikkei_hist['Close'].iloc[-1]
            prev = nikkei_hist['Close'].iloc[-2]
            change = (latest - prev) / prev * 100
            print(f"日経平均: ¥{latest:,.0f} ({change:+.2f}%)")
        
        # USD/JPY
        usdjpy = yf.Ticker("USDJPY=X")
        usdjpy_hist = usdjpy.history(period="2d")
        if not usdjpy_hist.empty:
            print(f"USD/JPY: ¥{usdjpy_hist['Close'].iloc[-1]:.2f}")
        
        print("\n" + "=" * 80)
        print("⚠️  投資に関する重要事項")
        print("=" * 80)
        print("• 本システムの分析は参考情報です")
        print("• 投資判断は自己責任で行ってください")
        print("• 適切なリスク管理を心がけてください")
        print("• 分散投資を推奨します")
        
    elif choice == "2":
        # リアルタイム監視
        await system.start_realtime_monitoring()
        
    elif choice == "3":
        # 両方実行
        result = await system.run_daily_analysis()
        print("\n5秒後にリアルタイム監視を開始します...")
        await asyncio.sleep(5)
        await system.start_realtime_monitoring()
        
    else:
        print("終了します")
    
    print(f"\n実行完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nシステムを終了しました")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"\nエラーが発生しました: {e}")