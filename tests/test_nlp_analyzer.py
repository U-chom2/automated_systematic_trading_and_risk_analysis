"""NlpAnalyzerのテストケース"""

import pytest
from typing import Dict, List, Any
from src.analysis_engine.nlp_analyzer import NlpAnalyzer


class TestNlpAnalyzer:
    """NlpAnalyzerのテストクラス"""
    
    @pytest.fixture
    def analyzer(self) -> NlpAnalyzer:
        """テスト用のNlpAnalyzerインスタンスを作成"""
        return NlpAnalyzer()
    
    def test_analyze_ir_importance(self, analyzer: NlpAnalyzer) -> None:
        """IR重要度分析のテスト"""
        # S級キーワードのテストケース
        ir_text = "本日、業績の上方修正を発表いたします。"
        result = analyzer.analyze_ir_importance(ir_text)
        
        assert isinstance(result, dict)
        assert "score" in result
        assert "keywords" in result
        assert result["score"] == 50  # 上方修正はS級キーワード
        assert "上方修正" in result["keywords"]
        
        # 業務提携のテストケース
        ir_text2 = "新規業務提携に関するお知らせ"
        result2 = analyzer.analyze_ir_importance(ir_text2)
        assert result2["score"] == 40  # 業務提携は40点
        assert "業務提携" in result2["keywords"]
        
        # キーワードなしのテストケース
        ir_text3 = "定時株主総会の開催について"
        result3 = analyzer.analyze_ir_importance(ir_text3)
        assert result3["score"] == 0
        assert len(result3["keywords"]) == 0
    
    def test_analyze_sentiment(self, analyzer: NlpAnalyzer) -> None:
        """センチメント分析のテスト"""
        # ポジティブなテキスト
        texts = [
            "この銘柄は素晴らしい！",
            "業績が好調で期待大",
            "買い時だと思う"
        ]
        result = analyzer.analyze_sentiment(texts)
        
        assert isinstance(result, dict)
        assert "positive_ratio" in result
        assert "negative_ratio" in result
        assert "neutral_ratio" in result
        assert "change_rate" in result
        assert result["positive_ratio"] > 0.5  # ポジティブ優勢
        
        # ネガティブなテキスト
        texts_negative = [
            "この銘柄は危険",
            "業績悪化が心配",
            "売り時かもしれない"
        ]
        result_negative = analyzer.analyze_sentiment(texts_negative)
        assert result_negative["negative_ratio"] > 0.5  # ネガティブ優勢
    
    def test_calculate_mention_anomaly(self, analyzer: NlpAnalyzer) -> None:
        """言及数異常検知のテスト"""
        # 通常の言及数
        mention_counts = [5, 6, 4, 5, 7, 6, 5, 4, 6, 5]
        current_count = 6
        result = analyzer.calculate_mention_anomaly(mention_counts, current_count)
        
        assert isinstance(result, dict)
        assert "is_anomaly" in result
        assert "z_score" in result
        assert "threshold" in result
        assert result["is_anomaly"] is False  # 通常範囲内
        
        # 異常な言及数（3σを超える）
        current_count_high = 20
        result_high = analyzer.calculate_mention_anomaly(mention_counts, current_count_high)
        assert result_high["is_anomaly"] is True  # 異常検知
        assert result_high["z_score"] > 3.0
    
    def test_calculate_sentiment_score(self, analyzer: NlpAnalyzer) -> None:
        """センチメントスコア計算のテスト"""
        # ポジティブ優勢で増加率も高い
        sentiment_result = {
            "positive_ratio": 0.7,
            "negative_ratio": 0.2,
            "neutral_ratio": 0.1,
            "change_rate": 0.5  # 50%増加
        }
        score = analyzer.calculate_sentiment_score(sentiment_result)
        
        assert isinstance(score, int)
        assert 0 <= score <= 30  # 最大30点
        assert score > 20  # 高スコア期待
        
        # ネガティブ優勢
        sentiment_negative = {
            "positive_ratio": 0.2,
            "negative_ratio": 0.7,
            "neutral_ratio": 0.1,
            "change_rate": -0.3
        }
        score_negative = analyzer.calculate_sentiment_score(sentiment_negative)
        assert score_negative < 10  # 低スコア
    
    def test_empty_input_handling(self, analyzer: NlpAnalyzer) -> None:
        """空入力のエラーハンドリングテスト"""
        # 空文字列
        result = analyzer.analyze_ir_importance("")
        assert result["score"] == 0
        assert len(result["keywords"]) == 0
        
        # 空リスト
        result_sentiment = analyzer.analyze_sentiment([])
        assert result_sentiment["positive_ratio"] == 0
        assert result_sentiment["negative_ratio"] == 0
        assert result_sentiment["neutral_ratio"] == 1.0  # デフォルトは中立
        
        # 空の言及数リスト
        with pytest.raises(ValueError):
            analyzer.calculate_mention_anomaly([], 5)