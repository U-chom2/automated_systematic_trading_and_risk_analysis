"""リアルデータ対応システム統合管理

Yahoo FinanceとTDnetの実データを使用した統合システム
"""

import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional

from .system_integrator import SystemIntegrator, MockDataCollector, MockAIAnalyzer
from .data_collector.yahoo_finance_client import YahooFinanceClient
from .data_collector.tdnet_real_scraper import TDnetRealScraper
from .persistence import (
    ExecutionPlanManager, 
    TradeHistoryManager, 
    SystemStateManager
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataCollector:
    """リアルデータ収集器"""
    
    def __init__(self) -> None:
        """初期化"""
        self.yahoo_client = YahooFinanceClient(enable_cache=True)
        self.tdnet_scraper = TDnetRealScraper(enable_cache=True)
        logger.info("RealDataCollector initialized")
    
    def collect_price_data(self, symbol: str) -> Dict[str, Any]:
        """株価データ収集（実データ）"""
        # 日本市場の銘柄コードをYahoo Finance形式に変換
        yahoo_symbol = f"{symbol}.T"
        
        # Yahoo Financeから実データを取得
        price_data = self.yahoo_client.get_current_price(yahoo_symbol)
        
        if price_data and 'error' not in price_data:
            return {
                "current_price": price_data.get('current_price', 0),
                "volume": price_data.get('volume', 0),
                "high": price_data.get('high', 0),
                "low": price_data.get('low', 0),
                "open": price_data.get('open', 0),
                "previous_close": price_data.get('previous_close', 0),
                "timestamp": price_data.get('timestamp', datetime.now().isoformat())
            }
        else:
            # エラー時はデフォルト値を返す
            logger.warning(f"Failed to fetch real price data for {symbol}, using defaults")
            return {
                "current_price": 0,
                "volume": 0,
                "high": 0,
                "low": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def collect_ir_releases(self, symbol: str) -> List[Dict[str, Any]]:
        """IR情報収集（実データ）"""
        # TDnetから実データを取得
        releases = self.tdnet_scraper.fetch_company_releases(
            company_code=symbol,
            days_back=7  # 過去7日間
        )
        
        # データ形式を統一
        formatted_releases = []
        for release in releases:
            formatted_release = {
                "title": release.get('title', ''),
                "content": release.get('title', ''),  # 詳細内容は別途取得が必要
                "timestamp": f"{release.get('release_date', '')} {release.get('release_time', '')}",
                "importance": self._determine_importance(release),
                "category": release.get('category', 'その他'),
                "pdf_url": release.get('pdf_url', '')
            }
            formatted_releases.append(formatted_release)
        
        return formatted_releases
    
    def collect_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """ソーシャル感情分析（実データ統合予定）"""
        # 現時点ではモックデータを返す
        # 将来的にはTwitter APIなどと統合
        return {
            "positive": 0.5,
            "negative": 0.3,
            "neutral": 0.2,
            "mention_count": 100,
            "sentiment_trend": "stable"
        }
    
    def collect_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """テクニカル指標収集（実データ）"""
        yahoo_symbol = f"{symbol}.T"
        
        # Yahoo Financeからテクニカル指標を取得
        indicators = self.yahoo_client.calculate_technical_indicators(yahoo_symbol)
        
        return indicators
    
    def _determine_importance(self, release: Dict[str, Any]) -> str:
        """IR情報の重要度を判定"""
        category = release.get('category', '')
        
        if category in ['決算短信', '業績予想の修正']:
            return 'high'
        elif category in ['配当予想の修正', 'M&A']:
            return 'medium'
        else:
            return 'low'


class HybridAIAnalyzer:
    """ハイブリッドAI分析エンジン（実データ＋AI）"""
    
    def __init__(self) -> None:
        """初期化"""
        self.yahoo_client = YahooFinanceClient()
        self.tdnet_scraper = TDnetRealScraper()
        logger.info("HybridAIAnalyzer initialized")
    
    def analyze_catalyst_impact(self, releases: List[Dict[str, Any]]) -> int:
        """カタリスト分析（実データベース）"""
        if not releases:
            return 0
        
        total_score = 0
        
        for release in releases:
            # TDnetスクレイパーの解析機能を使用
            parsed = self.tdnet_scraper.parse_release_content(release)
            importance_score = parsed.get('importance_score', 0)
            
            # 重要度に応じてスコア加算（最大50点）
            total_score += importance_score * 0.5
        
        return min(int(total_score), 50)
    
    def analyze_sentiment_score(self, sentiment_data: Dict[str, Any]) -> int:
        """感情スコア分析（実データベース）"""
        positive_ratio = sentiment_data.get("positive", 0.5)
        negative_ratio = sentiment_data.get("negative", 0.3)
        mention_count = sentiment_data.get("mention_count", 100)
        
        # 感情スコア計算（最大30点）
        sentiment_score = (positive_ratio - negative_ratio) * 30
        
        # メンション数による調整
        if mention_count > 500:
            sentiment_score *= 1.2
        elif mention_count < 50:
            sentiment_score *= 0.8
        
        return max(0, min(int(sentiment_score), 30))
    
    def analyze_technical_indicators(self, symbol: str, indicators: Dict[str, Any]) -> int:
        """テクニカル分析（実データベース）"""
        score = 0
        max_score = 20
        
        # RSI分析
        rsi = indicators.get('rsi')
        if rsi is not None:
            if 30 <= rsi <= 70:
                score += 5  # 適正範囲
            elif rsi < 30:
                score += 8  # 売られすぎ（買いシグナル）
            elif rsi > 70:
                score += 2  # 買われすぎ（売りシグナル）
        
        # 移動平均分析
        sma_20 = indicators.get('sma_20')
        sma_50 = indicators.get('sma_50')
        
        if sma_20 and sma_50:
            if sma_20 > sma_50:
                score += 5  # ゴールデンクロス傾向
            else:
                score += 2  # デッドクロス傾向
        
        # ボリンジャーバンド分析
        bb = indicators.get('bollinger_bands', {})
        if bb.get('upper') and bb.get('lower'):
            score += 3
        
        # MACD分析
        macd = indicators.get('macd', {})
        if macd.get('histogram'):
            histogram = macd['histogram']
            if histogram > 0:
                score += 5  # 買いシグナル
            else:
                score += 2  # 売りシグナル
        
        return min(score, max_score)


class SystemIntegratorReal(SystemIntegrator):
    """リアルデータ対応システム統合クラス
    
    実データとモックデータを切り替え可能な統合システム
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初期化
        
        Args:
            config: システム設定
        """
        super().__init__(config)
        
        # リアルデータモードの設定
        self.use_real_data = config.get("use_real_data", False)
        
        if self.use_real_data:
            # リアルデータコレクターとアナライザーを使用
            self.data_collector = RealDataCollector()
            self.ai_analyzer = HybridAIAnalyzer()
            logger.info("SystemIntegratorReal initialized with REAL DATA mode")
        else:
            # モックデータを使用（親クラスのまま）
            logger.info("SystemIntegratorReal initialized with MOCK mode")
    
    def collect_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """市場データ収集（リアル/モック切り替え対応）"""
        collected_data = {}
        
        for symbol in symbols:
            try:
                if self.use_real_data:
                    # リアルデータ収集
                    symbol_data = {
                        "price_data": self.data_collector.collect_price_data(symbol),
                        "ir_releases": self.data_collector.collect_ir_releases(symbol),
                        "social_sentiment": self.data_collector.collect_social_sentiment(symbol),
                        "technical_indicators": self.data_collector.collect_technical_indicators(symbol)
                    }
                else:
                    # モックデータ収集（親クラスのメソッド使用）
                    symbol_data = {
                        "price_data": self.data_collector.collect_price_data(symbol),
                        "ir_releases": self.data_collector.collect_ir_releases(symbol),
                        "social_sentiment": self.data_collector.collect_social_sentiment(symbol)
                    }
                
                collected_data[symbol] = symbol_data
                logger.info(f"Data collected for {symbol} (real_data={self.use_real_data})")
                
            except Exception as e:
                logger.error(f"Data collection failed for {symbol}: {e}")
                collected_data[symbol] = {
                    "price_data": {},
                    "ir_releases": [],
                    "social_sentiment": {},
                    "technical_indicators": {}
                }
        
        return collected_data
    
    def perform_ai_analysis(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """AI分析実行（リアル/モック切り替え対応）"""
        analysis_results = {}
        
        for symbol, data in market_data.items():
            try:
                if self.use_real_data:
                    # リアルデータベースの分析
                    catalyst_score = self.ai_analyzer.analyze_catalyst_impact(
                        data.get("ir_releases", [])
                    )
                    
                    sentiment_score = self.ai_analyzer.analyze_sentiment_score(
                        data.get("social_sentiment", {})
                    )
                    
                    technical_score = self.ai_analyzer.analyze_technical_indicators(
                        symbol,
                        data.get("technical_indicators", {})
                    )
                else:
                    # モック分析（親クラスのメソッド使用）
                    catalyst_score = self.ai_analyzer.analyze_catalyst_impact(
                        data.get("ir_releases", [])
                    )
                    
                    sentiment_score = self.ai_analyzer.analyze_sentiment_score(
                        data.get("social_sentiment", {})
                    )
                    
                    technical_score = self.ai_analyzer.analyze_technical_indicators(
                        data.get("price_data", {})
                    )
                
                total_score = catalyst_score + sentiment_score + technical_score
                confidence = min(total_score / 100, 1.0)
                
                # リスク評価
                volatility = 0.25  # デフォルト値
                if self.use_real_data and data.get("technical_indicators"):
                    # ボリンジャーバンドから推定
                    bb = data["technical_indicators"].get("bollinger_bands", {})
                    if bb.get("upper") and bb.get("lower") and bb.get("middle"):
                        volatility = (bb["upper"] - bb["lower"]) / (2 * bb["middle"])
                
                risk_assessment = self.ai_analyzer.calculate_risk_parameters(
                    symbol,
                    data.get("price_data", {}).get("current_price", 1000),
                    volatility
                ) if hasattr(self.ai_analyzer, 'calculate_risk_parameters') else {}
                
                analysis_results[symbol] = {
                    "catalyst_score": catalyst_score,
                    "sentiment_score": sentiment_score, 
                    "technical_score": technical_score,
                    "total_score": total_score,
                    "confidence": round(confidence, 3),
                    "current_price": data.get("price_data", {}).get("current_price", 0),
                    "risk_assessment": risk_assessment,
                    "data_source": "real" if self.use_real_data else "mock"
                }
                
                logger.info(f"Analysis completed for {symbol}: {total_score}/100 (source={analysis_results[symbol]['data_source']})")
                
            except Exception as e:
                logger.error(f"Analysis failed for {symbol}: {e}")
                analysis_results[symbol] = {
                    "catalyst_score": 0,
                    "sentiment_score": 0,
                    "technical_score": 0,
                    "total_score": 0,
                    "confidence": 0.0,
                    "data_source": "error"
                }
        
        return analysis_results
    
    def get_market_status(self) -> Dict[str, Any]:
        """市場状態を取得"""
        if self.use_real_data:
            yahoo_client = YahooFinanceClient()
            return yahoo_client.get_market_status('TSE')
        else:
            # モック市場状態
            return {
                'market': 'TSE',
                'is_open': True,
                'current_time': datetime.now().isoformat(),
                'next_open': None,
                'next_close': (datetime.now() + timedelta(hours=2)).isoformat()
            }
    
    def create_execution_plan(self, plan: Dict[str, Any]) -> bool:
        """実行計画を作成
        
        Args:
            plan: 実行計画の内容
            
        Returns:
            作成成功時True
        """
        return self.plan_manager.create_plan(plan)
    
    def save_system_state(self) -> None:
        """システム状態を保存"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'use_real_data': self.use_real_data,
            'config': self.config
        }
        self.state_manager.save_state(state)