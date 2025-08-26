"""データ収集→分析エンジン連携の統合テスト"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

from src.data_collector.tdnet_scraper import TdnetScraper
from src.data_collector.x_streamer import XStreamer
from src.data_collector.price_fetcher import PriceFetcher
from src.data_collector.watchlist_manager import WatchlistManager
from src.analysis_engine.nlp_analyzer import NlpAnalyzer, SentimentScore, ImportanceScore
from src.analysis_engine.technical_analyzer import TechnicalAnalyzer, TechnicalIndicators, MarketEnvironmentScore
from src.analysis_engine.risk_model import RiskModel


class TestDataAnalysisIntegration:
    """データ収集と分析エンジンの統合テストクラス"""
    
    @pytest.fixture
    def sample_ir_data(self) -> Dict[str, Any]:
        """テスト用IRデータ"""
        return {
            "title": "業績の上方修正に関するお知らせ",
            "content": """
            当社は、2024年3月期第3四半期決算において、
            業績予想の上方修正を発表いたします。
            売上高は前年同期比15%増となり、好調に推移しております。
            新商品の投入効果により、今後も成長が期待されます。
            """,
            "symbol": "7203",
            "timestamp": datetime.now(),
            "source": "TDnet"
        }
    
    @pytest.fixture
    def sample_sns_data(self) -> List[str]:
        """テスト用SNSデータ"""
        return [
            "7203の決算良かった！期待できそう",
            "トヨタ株買い時かも。業績上方修正は素晴らしい",
            "7203注目。新商品効果で株価上昇期待",
            "決算発表で株価急騰の予感",
            "好調な業績で長期保有したい銘柄"
        ]
    
    @pytest.fixture
    def sample_price_data(self) -> List[Dict[str, Any]]:
        """テスト用価格データ"""
        base_date = datetime.now() - timedelta(days=30)
        price_data = []
        
        for i in range(30):
            date = base_date + timedelta(days=i)
            # シンプルなトレンドを作成（上昇基調）
            base_price = 2400 + (i * 2)  # 2400から少しずつ上昇
            
            price_data.append({
                "date": date,
                "open": base_price,
                "high": base_price + 20,
                "low": base_price - 15,
                "close": base_price + 5,
                "volume": 800000 + (i * 5000)
            })
        
        return price_data
    
    @pytest.fixture
    def nlp_analyzer(self) -> NlpAnalyzer:
        """NlpAnalyzer インスタンス"""
        analyzer = NlpAnalyzer()
        # テスト用の簡易初期化
        analyzer.is_initialized = False  # 実際のモデルは使わない
        return analyzer
    
    @pytest.fixture
    def technical_analyzer(self) -> TechnicalAnalyzer:
        """TechnicalAnalyzer インスタンス"""
        return TechnicalAnalyzer()
    
    def test_ir_importance_analysis(self, nlp_analyzer: NlpAnalyzer, sample_ir_data: Dict[str, Any]) -> None:
        """IR重要度分析のテスト"""
        result = nlp_analyzer.analyze_ir_importance(sample_ir_data["content"])
        
        assert "score" in result
        assert "keywords" in result
        assert result["score"] > 0  # 上方修正キーワードが検出されるはず
        assert "上方修正" in result["keywords"]
    
    def test_sentiment_analysis(self, nlp_analyzer: NlpAnalyzer, sample_sns_data: List[str]) -> None:
        """センチメント分析のテスト"""
        result = nlp_analyzer.analyze_sentiment(sample_sns_data)
        
        assert "positive_ratio" in result
        assert "negative_ratio" in result
        assert "neutral_ratio" in result
        assert "change_rate" in result
        
        # ポジティブな投稿が多いため、ポジティブ比率が高いはず
        assert result["positive_ratio"] > 0.5
    
    def test_stock_mention_extraction(self, nlp_analyzer: NlpAnalyzer) -> None:
        """株式コード抽出のテスト"""
        test_text = "7203の決算が良く、(6758)も注目です。9984.Tも上昇中"
        
        result = nlp_analyzer.extract_stock_mentions(test_text)
        
        assert "7203" in result
        assert "6758" in result
        assert "9984.T" in result or "9984" in result
    
    def test_technical_indicators_calculation(self, technical_analyzer: TechnicalAnalyzer, 
                                            sample_price_data: List[Dict[str, Any]]) -> None:
        """テクニカル指標計算のテスト"""
        result = technical_analyzer.get_technical_indicators("7203", sample_price_data)
        
        assert isinstance(result, TechnicalIndicators)
        assert result.rsi > 0
        assert result.rsi <= 100
        assert result.moving_avg_deviation is not None
        assert result.volume_ratio > 0
    
    def test_rsi_calculation_with_dataframe(self, technical_analyzer: TechnicalAnalyzer, 
                                          sample_price_data: List[Dict[str, Any]]) -> None:
        """DataFrame使用のRSI計算テスト"""
        df = pd.DataFrame(sample_price_data)
        
        rsi = technical_analyzer.calculate_rsi(df)
        
        assert 0 <= rsi <= 100
        assert not pd.isna(rsi)
    
    def test_moving_average_deviation(self, technical_analyzer: TechnicalAnalyzer,
                                    sample_price_data: List[Dict[str, Any]]) -> None:
        """移動平均乖離率計算のテスト"""
        df = pd.DataFrame(sample_price_data)
        
        ma_deviation = technical_analyzer.calculate_moving_average_deviation(df)
        
        assert isinstance(ma_deviation, float)
        assert ma_deviation != 0  # 上昇トレンドなので乖離があるはず
    
    def test_bollinger_bands_calculation(self, technical_analyzer: TechnicalAnalyzer,
                                       sample_price_data: List[Dict[str, Any]]) -> None:
        """ボリンジャーバンド計算のテスト"""
        closes = [Decimal(str(data["close"])) for data in sample_price_data]
        
        upper, middle, lower = technical_analyzer.calculate_bollinger_bands(closes)
        
        assert upper > middle > lower  # 正常な順序
        assert upper > 0 and middle > 0 and lower > 0
    
    def test_macd_calculation(self, technical_analyzer: TechnicalAnalyzer,
                            sample_price_data: List[Dict[str, Any]]) -> None:
        """MACD計算のテスト"""
        closes = [Decimal(str(data["close"])) for data in sample_price_data]
        
        macd_line, signal_line, histogram = technical_analyzer.calculate_macd(closes)
        
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)
    
    def test_overheating_filter(self, technical_analyzer: TechnicalAnalyzer) -> None:
        """過熱感フィルターのテスト"""
        # 過熱状態のテクニカル指標
        overheated_indicators = TechnicalIndicators(
            rsi=85.0,  # > 75
            moving_avg_deviation=30.0,  # > 25%
            volume_ratio=2.0,
            atr=0.05,
            bollinger_upper=0.0,
            bollinger_lower=0.0,
            macd_line=0.0,
            macd_signal=0.0,
            stochastic_k=50.0,
            stochastic_d=50.0
        )
        
        is_overheated = technical_analyzer.check_overheating_filter(overheated_indicators)
        assert is_overheated is True
        
        # 正常状態のテクニカル指標
        normal_indicators = TechnicalIndicators(
            rsi=65.0,  # < 75
            moving_avg_deviation=15.0,  # < 25%
            volume_ratio=1.5,
            atr=0.05,
            bollinger_upper=0.0,
            bollinger_lower=0.0,
            macd_line=0.0,
            macd_signal=0.0,
            stochastic_k=50.0,
            stochastic_d=50.0
        )
        
        is_overheated = technical_analyzer.check_overheating_filter(normal_indicators)
        assert is_overheated is False
    
    def test_integrated_scoring_system(self, nlp_analyzer: NlpAnalyzer,
                                     technical_analyzer: TechnicalAnalyzer,
                                     sample_ir_data: Dict[str, Any],
                                     sample_sns_data: List[str],
                                     sample_price_data: List[Dict[str, Any]]) -> None:
        """統合スコアリングシステムのテスト"""
        # NLP分析
        ir_result = nlp_analyzer.analyze_ir_importance(sample_ir_data["content"])
        sentiment_result = nlp_analyzer.analyze_sentiment(sample_sns_data)
        
        # テクニカル分析
        technical_result = technical_analyzer.calculate_technical_score(pd.DataFrame(sample_price_data))
        
        # 統合スコア計算（要件定義書のスコアリング仕様に基づく）
        catalyst_score = ir_result["score"]  # Max 50
        sentiment_score = nlp_analyzer.calculate_sentiment_score(sentiment_result)  # Max 30
        technical_score = technical_result["market_environment_score"]  # Max 20
        
        total_score = catalyst_score + sentiment_score + technical_score
        
        # 買い判断閾値（80点以上）のテスト
        buy_decision = total_score >= 80 and technical_result["filter_passed"]
        
        assert catalyst_score >= 0
        assert sentiment_score >= 0
        assert technical_score >= 0
        assert total_score >= 0
        assert isinstance(buy_decision, bool)
        
        print(f"統合スコア: カタリスト={catalyst_score}, "
              f"センチメント={sentiment_score}, "
              f"テクニカル={technical_score}, "
              f"合計={total_score}, "
              f"買い判断={buy_decision}")
    
    def test_mention_anomaly_detection(self, nlp_analyzer: NlpAnalyzer) -> None:
        """メンション数異常検知のテスト"""
        # 過去24時間の平均的なメンション数
        historical_counts = [10, 12, 8, 15, 11, 9, 13, 10, 14, 8, 12, 11]
        
        # 異常な急増
        current_count = 50  # 3σを超える値
        
        result = nlp_analyzer.calculate_mention_anomaly(historical_counts, current_count)
        
        assert "is_anomaly" in result
        assert "z_score" in result
        assert "threshold" in result
        assert result["is_anomaly"] is True
        assert result["z_score"] > 3.0
    
    def test_risk_indicators_extraction(self, technical_analyzer: TechnicalAnalyzer,
                                      sample_price_data: List[Dict[str, Any]]) -> None:
        """リスクモデル用指標抽出のテスト"""
        df = pd.DataFrame(sample_price_data)
        
        risk_indicators = technical_analyzer.get_risk_indicators(df)
        
        assert "atr" in risk_indicators
        assert "historical_volatility" in risk_indicators
        assert "volume_ratio" in risk_indicators
        assert "price_momentum" in risk_indicators
        
        assert risk_indicators["atr"] >= 0
        assert risk_indicators["historical_volatility"] >= 0
        assert risk_indicators["volume_ratio"] >= 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_analysis_flow(self, nlp_analyzer: NlpAnalyzer,
                                                technical_analyzer: TechnicalAnalyzer,
                                                sample_ir_data: Dict[str, Any],
                                                sample_sns_data: List[str],
                                                sample_price_data: List[Dict[str, Any]]) -> None:
        """エンドツーエンドデータ分析フローのテスト"""
        symbol = sample_ir_data["symbol"]
        
        # ステップ1: データ収集（モック）
        ir_data = sample_ir_data
        sns_data = sample_sns_data
        price_data = sample_price_data
        
        # ステップ2: NLP分析
        importance_analysis = nlp_analyzer.analyze_ir_importance(ir_data["content"])
        sentiment_analysis = nlp_analyzer.analyze_sentiment(sns_data)
        mention_analysis = nlp_analyzer.calculate_mention_anomaly([5, 7, 6, 8, 5, 6], 25)
        
        # ステップ3: テクニカル分析
        technical_indicators = technical_analyzer.get_technical_indicators(symbol, price_data)
        technical_score = technical_analyzer.calculate_technical_score(pd.DataFrame(price_data))
        
        # ステップ4: 統合判断
        catalyst_score = importance_analysis["score"]
        sentiment_score = nlp_analyzer.calculate_sentiment_score(sentiment_analysis)
        market_score = technical_score["market_environment_score"]
        
        total_score = catalyst_score + sentiment_score + market_score
        buy_decision = total_score >= 80 and technical_score["filter_passed"]
        
        # フロー結果の検証
        analysis_result = {
            "symbol": symbol,
            "catalyst_score": catalyst_score,
            "sentiment_score": sentiment_score, 
            "market_environment_score": market_score,
            "total_score": total_score,
            "buy_decision": buy_decision,
            "filter_passed": technical_score["filter_passed"],
            "mention_anomaly": mention_analysis["is_anomaly"],
            "technical_indicators": technical_indicators
        }
        
        # 結果検証
        assert analysis_result["symbol"] == symbol
        assert analysis_result["total_score"] >= 0
        assert isinstance(analysis_result["buy_decision"], bool)
        assert isinstance(analysis_result["filter_passed"], bool)
        assert isinstance(analysis_result["mention_anomaly"], bool)
        assert isinstance(analysis_result["technical_indicators"], TechnicalIndicators)
        
        print(f"エンドツーエンド分析結果: {analysis_result}")
    
    def test_data_validation_and_error_handling(self, nlp_analyzer: NlpAnalyzer,
                                              technical_analyzer: TechnicalAnalyzer) -> None:
        """データバリデーションとエラーハンドリングのテスト"""
        # 空のデータでのテスト
        empty_ir_result = nlp_analyzer.analyze_ir_importance("")
        assert empty_ir_result["score"] == 0
        
        empty_sentiment_result = nlp_analyzer.analyze_sentiment([])
        assert empty_sentiment_result["positive_ratio"] == 0.0
        
        # 不正なデータでのテスト
        insufficient_price_data = [{"close": 100}]  # 不十分なデータ
        result = technical_analyzer.get_technical_indicators("TEST", insufficient_price_data)
        assert isinstance(result, TechnicalIndicators)  # エラーでも結果が返される
        
        # 異常値でのテスト
        try:
            nlp_analyzer.calculate_mention_anomaly([], 10)  # 空の履歴データ
        except ValueError:
            pass  # 期待される例外
        else:
            pytest.fail("Expected ValueError was not raised")