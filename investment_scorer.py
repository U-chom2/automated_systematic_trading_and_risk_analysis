"""
Investment Scoring Module
投資スコアリングモジュール - テクニカル分析結果から投資スコアを計算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from config import config
from technical_analyzer import TechnicalIndicators


@dataclass
class ScoringResult:
    """スコアリング結果データクラス"""
    total_score: float
    component_scores: Dict[str, float]
    recommendation: Dict[str, any]
    analysis_details: Dict[str, any]


class InvestmentScorer:
    """投資スコアリングクラス"""
    
    def __init__(self):
        self.weights = config.scoring_weights
        self.thresholds = config.investment_thresholds
    
    def score_trend_analysis(self, indicators: TechnicalIndicators) -> float:
        """トレンド分析スコア（25点満点）"""
        score = 0.0
        
        if not indicators.sma_5 or not indicators.sma_25 or not indicators.sma_75:
            return 0.0
        
        # 上昇トレンド判定
        if indicators.sma_5 > indicators.sma_25 > indicators.sma_75:
            score += 15.0  # 強い上昇トレンド
        elif indicators.sma_5 > indicators.sma_25:
            score += 8.0   # 短期上昇トレンド
        
        # 長期トレンド判定
        if indicators.sma_25 > indicators.sma_75:
            score += 10.0  # 長期上昇トレンド
        
        return min(score, 25.0)
    
    def score_rsi_analysis(self, indicators: TechnicalIndicators) -> float:
        """RSI分析スコア（15点満点）"""
        if not indicators.rsi:
            return 0.0
        
        rsi = indicators.rsi
        
        # RSI買いゾーン判定
        if 30 <= rsi <= 70:
            # 適正範囲内
            if 50 <= rsi <= 65:
                return 15.0  # 最適な買いゾーン
            elif 45 <= rsi < 50:
                return 12.0  # やや弱気だが買い検討可能
            elif 65 < rsi <= 70:
                return 10.0  # やや強気だが買い可能
            else:
                return 8.0   # 買い検討範囲
        elif rsi < 30:
            return 5.0   # 売られすぎ（リスク高）
        else:  # rsi > 70
            return 0.0   # 買われすぎ（買い推奨しない）
    
    def score_macd_analysis(self, indicators: TechnicalIndicators) -> float:
        """MACD分析スコア（20点満点）"""
        if not indicators.macd or not indicators.macd_signal:
            return 0.0
        
        score = 0.0
        
        # MACD買いシグナル
        if indicators.macd > indicators.macd_signal:
            score += 12.0
        
        # MACDヒストグラム
        if indicators.macd_histogram and indicators.macd_histogram > 0:
            score += 8.0
        
        return min(score, 20.0)
    
    def score_price_momentum(self, indicators: TechnicalIndicators) -> float:
        """価格モメンタムスコア（15点満点）"""
        score = 0.0
        
        # 1日変化率
        if indicators.price_change_1d:
            if -1 <= indicators.price_change_1d <= 3:
                score += 3.0  # 適度な上昇
            elif indicators.price_change_1d > 3:
                score += 1.0  # 過度な上昇（リスク）
        
        # 5日変化率
        if indicators.price_change_5d:
            if 0 <= indicators.price_change_5d <= 5:
                score += 5.0  # 良好な短期トレンド
            elif indicators.price_change_5d > 5:
                score += 2.0  # 急激な上昇
        
        # 25日変化率
        if indicators.price_change_25d:
            if 0 <= indicators.price_change_25d <= 15:
                score += 7.0  # 健全な中期上昇
            elif indicators.price_change_25d > 15:
                score += 3.0  # 過度な上昇
        
        return min(score, 15.0)
    
    def score_volume_analysis(self, indicators: TechnicalIndicators) -> float:
        """出来高分析スコア（10点満点）"""
        # 簡易実装：出来高データが限定的なため固定スコア
        return 5.0
    
    def score_bollinger_bands(self, indicators: TechnicalIndicators, current_price: float) -> float:
        """ボリンジャーバンド分析スコア（10点満点）"""
        if not indicators.bollinger_upper or not indicators.bollinger_lower or not indicators.bollinger_middle:
            return 0.0
        
        # 現在価格のバンド内位置を計算
        band_width = indicators.bollinger_upper - indicators.bollinger_lower
        if band_width == 0:
            return 5.0
        
        position = (current_price - indicators.bollinger_lower) / band_width
        
        # バンド内の適正位置でスコアリング
        if 0.2 <= position <= 0.6:
            return 10.0  # 良好な買いゾーン
        elif 0.1 <= position < 0.2:
            return 8.0   # 下限付近（買い検討）
        elif 0.6 < position <= 0.8:
            return 6.0   # やや上限寄り
        else:
            return 2.0   # 極端な位置
    
    def score_market_cap_factor(self, market_cap_millions: float) -> float:
        """時価総額ファクタースコア（5点満点）"""
        if market_cap_millions <= 0:
            return 0.0
        
        # 小型株ボーナス（流動性とボラティリティのバランス）
        if 1000 <= market_cap_millions <= 5000:
            return 5.0  # 適度な小型株
        elif 500 <= market_cap_millions < 1000:
            return 3.0  # 極小型株
        elif 5000 < market_cap_millions <= 20000:
            return 4.0  # 中型株
        else:
            return 2.0  # 大型株または極小
    
    def calculate_investment_score(
        self, 
        indicators: TechnicalIndicators, 
        current_price: float,
        market_cap_millions: float = 1500.0
    ) -> ScoringResult:
        """総合投資スコアを計算"""
        
        # 各コンポーネントスコアを計算
        component_scores = {
            "trend": self.score_trend_analysis(indicators),
            "rsi": self.score_rsi_analysis(indicators),
            "macd": self.score_macd_analysis(indicators),
            "momentum": self.score_price_momentum(indicators),
            "volume": self.score_volume_analysis(indicators),
            "bollinger": self.score_bollinger_bands(indicators, current_price),
            "market_cap": self.score_market_cap_factor(market_cap_millions)
        }
        
        # 重み付き総合スコア計算
        total_score = sum(component_scores.values())
        
        # 投資推奨を取得
        recommendation = config.get_investment_recommendation(total_score)
        
        # 分析詳細
        analysis_details = {
            "current_price": current_price,
            "market_cap_millions": market_cap_millions,
            "rsi": indicators.rsi,
            "macd_signal": "買い" if (indicators.macd and indicators.macd_signal and indicators.macd > indicators.macd_signal) else "売り",
            "trend_signal": self._get_trend_signal(indicators),
            "price_changes": {
                "1d": indicators.price_change_1d,
                "5d": indicators.price_change_5d,
                "25d": indicators.price_change_25d
            }
        }
        
        return ScoringResult(
            total_score=total_score,
            component_scores=component_scores,
            recommendation=recommendation,
            analysis_details=analysis_details
        )
    
    def _get_trend_signal(self, indicators: TechnicalIndicators) -> str:
        """トレンドシグナルを取得"""
        if not indicators.sma_5 or not indicators.sma_25 or not indicators.sma_75:
            return "不明"
        
        if indicators.sma_5 > indicators.sma_25 > indicators.sma_75:
            return "強い上昇"
        elif indicators.sma_5 > indicators.sma_25:
            return "上昇"
        elif indicators.sma_5 < indicators.sma_25 < indicators.sma_75:
            return "下降"
        else:
            return "横ばい"
    
    def add_small_stock_bonus(self, base_score: float, market_cap_millions: float) -> float:
        """小型株ボーナスを追加"""
        if market_cap_millions < 2000:
            return min(base_score + 5.0, 100.0)  # 小型株ボーナス
        return base_score