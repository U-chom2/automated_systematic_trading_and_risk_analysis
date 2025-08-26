"""TechnicalAnalyzerのテストケース"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src.analysis_engine.technical_analyzer import TechnicalAnalyzer


class TestTechnicalAnalyzer:
    """TechnicalAnalyzerのテストクラス"""
    
    @pytest.fixture
    def analyzer(self) -> TechnicalAnalyzer:
        """テスト用のTechnicalAnalyzerインスタンスを作成"""
        return TechnicalAnalyzer()
    
    @pytest.fixture
    def sample_price_data(self) -> pd.DataFrame:
        """テスト用の株価データを作成"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # トレンドのある価格データを生成
        np.random.seed(42)
        base_price = 1000
        returns = np.random.normal(0.001, 0.02, 100)  # 平均0.1%、標準偏差2%
        prices = [base_price]
        
        for i in range(1, 100):
            prices.append(prices[-1] * (1 + returns[i]))
        
        volumes = np.random.randint(10000, 100000, 100)
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': volumes
        })
    
    def test_calculate_rsi(self, analyzer: TechnicalAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """RSI計算のテスト"""
        rsi = analyzer.calculate_rsi(sample_price_data, period=14)
        
        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100
        
        # RSI 75以上で過熱判定のテスト
        is_overheated = analyzer.is_rsi_overheated(rsi, threshold=75)
        assert isinstance(is_overheated, bool)
    
    def test_calculate_moving_average_deviation(self, analyzer: TechnicalAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """移動平均乖離率計算のテスト"""
        deviation = analyzer.calculate_moving_average_deviation(sample_price_data, period=25)
        
        assert isinstance(deviation, float)
        assert -50 <= deviation <= 50  # 通常の乖離率範囲
        
        # 乖離率25%以上で過熱判定のテスト
        is_overheated = analyzer.is_ma_deviation_overheated(deviation, threshold=25)
        assert isinstance(is_overheated, bool)
    
    def test_calculate_volume_surge(self, analyzer: TechnicalAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """出来高急増検知のテスト"""
        surge_ratio = analyzer.calculate_volume_surge(sample_price_data, period=20)
        
        assert isinstance(surge_ratio, float)
        assert surge_ratio >= 0
        
        # 出来高2倍以上で急増判定のテスト
        is_surge = analyzer.is_volume_surge(surge_ratio, threshold=2.0)
        assert isinstance(is_surge, bool)
    
    def test_calculate_atr(self, analyzer: TechnicalAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """ATR（平均真の範囲）計算のテスト"""
        atr = analyzer.calculate_atr(sample_price_data, period=14)
        
        assert isinstance(atr, float)
        assert atr > 0  # ATRは正の値
    
    def test_calculate_historical_volatility(self, analyzer: TechnicalAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """ヒストリカル・ボラティリティ計算のテスト"""
        hv = analyzer.calculate_historical_volatility(sample_price_data, period=60)
        
        assert isinstance(hv, float)
        assert hv > 0  # ボラティリティは正の値
        assert hv < 1  # 年率換算で100%未満が通常
    
    def test_calculate_technical_score(self, analyzer: TechnicalAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """テクニカル分析総合スコア計算のテスト"""
        score = analyzer.calculate_technical_score(sample_price_data)
        
        assert isinstance(score, dict)
        assert "market_environment_score" in score
        assert "filter_passed" in score
        assert "rsi" in score
        assert "ma_deviation" in score
        assert "volume_surge_ratio" in score
        
        # スコアの範囲確認
        assert 0 <= score["market_environment_score"] <= 20
        assert isinstance(score["filter_passed"], bool)
        assert 0 <= score["rsi"] <= 100
    
    def test_check_filter_conditions(self, analyzer: TechnicalAnalyzer) -> None:
        """フィルター条件チェックのテスト"""
        # 過熱状態（RSI > 75, MA乖離 > 25%）
        overheated_result = analyzer.check_filter_conditions(rsi=80, ma_deviation=30)
        assert overheated_result is False  # フィルターで除外
        
        # 正常状態
        normal_result = analyzer.check_filter_conditions(rsi=50, ma_deviation=10)
        assert normal_result is True  # フィルターを通過
    
    def test_get_risk_indicators(self, analyzer: TechnicalAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """リスクモデル用指標取得のテスト"""
        indicators = analyzer.get_risk_indicators(sample_price_data)
        
        assert isinstance(indicators, dict)
        assert "atr" in indicators
        assert "historical_volatility" in indicators
        assert "volume_ratio" in indicators
        assert "price_momentum" in indicators
        
        # すべて数値であることを確認
        for key, value in indicators.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
    
    def test_empty_data_handling(self, analyzer: TechnicalAnalyzer) -> None:
        """空データのエラーハンドリングテスト"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            analyzer.calculate_rsi(empty_df)
        
        with pytest.raises(ValueError):
            analyzer.calculate_moving_average_deviation(empty_df)
    
    def test_insufficient_data_handling(self, analyzer: TechnicalAnalyzer) -> None:
        """データ不足時のエラーハンドリングテスト"""
        # 5日分のデータ（RSI計算には不十分）
        short_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        with pytest.raises(ValueError):
            analyzer.calculate_rsi(short_data, period=14)
    
    def test_sector_trend_analysis(self, analyzer: TechnicalAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """セクター指数トレンド分析のテスト"""
        # サンプルセクター指数データ
        sector_data = sample_price_data.copy()
        
        trend_score = analyzer.analyze_sector_trend(sector_data)
        
        assert isinstance(trend_score, int)
        assert 0 <= trend_score <= 20  # 市場環境スコアの最大値