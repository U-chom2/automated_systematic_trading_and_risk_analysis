"""
Investment Scoring Module
投資スコアリングモジュール - テクニカル分析結果から投資スコアを計算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from config import config, Config, TradingMode
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
    
    def __init__(self, config_instance: Optional[Config] = None):
        """初期化
        
        Args:
            config_instance: 設定インスタンス。Noneの場合はデフォルトconfig使用
        """
        self.config = config_instance or config
        self.weights = self.config.scoring_weights
        self.thresholds = self.config.investment_thresholds
        self.trading_mode = getattr(self.config, 'trading_mode', TradingMode.LONG_TERM)
    
    def score_trend_analysis(self, indicators: TechnicalIndicators) -> float:
        """トレンド分析スコア（取引モード対応）"""
        score = 0.0
        max_score = self.weights.trend_weight
        
        if self.trading_mode == TradingMode.DAY_TRADING:
            # デイトレード: 短期移動平均重視
            return self._score_daytrading_trend(indicators, max_score)
        else:
            # 中長期: 従来のロジック
            return self._score_longterm_trend(indicators, max_score)
    
    def _score_daytrading_trend(self, indicators: TechnicalIndicators, max_score: float) -> float:
        """デイトレード用トレンドスコア"""
        score = 0.0
        
        # 短期移動平均（10日/20日）を優先
        if indicators.sma_10 and indicators.sma_20:
            if indicators.sma_10 > indicators.sma_20:
                score += max_score * 0.6  # 短期上昇トレンド
        
        # 短期EMA（9日/21日）クロス
        if indicators.ema_9 and indicators.ema_21:
            if indicators.ema_9 > indicators.ema_21:
                score += max_score * 0.4  # EMA強気クロス
        
        return min(score, max_score)
    
    def _score_longterm_trend(self, indicators: TechnicalIndicators, max_score: float) -> float:
        """中長期用トレンドスコア（従来ロジック）"""
        score = 0.0
        
        if not indicators.sma_5 or not indicators.sma_25 or not indicators.sma_75:
            return 0.0
        
        # 上昇トレンド判定
        if indicators.sma_5 > indicators.sma_25 > indicators.sma_75:
            score += max_score * 0.6  # 強い上昇トレンド
        elif indicators.sma_5 > indicators.sma_25:
            score += max_score * 0.32  # 短期上昇トレンド
        
        # 長期トレンド判定
        if indicators.sma_25 > indicators.sma_75:
            score += max_score * 0.4  # 長期上昇トレンド
        
        return min(score, max_score)
    
    def score_rsi_analysis(self, indicators: TechnicalIndicators) -> float:
        """RSI分析スコア（取引モード対応）"""
        if not indicators.rsi:
            return 0.0
        
        rsi = indicators.rsi
        max_score = self.weights.rsi_weight
        
        if self.trading_mode == TradingMode.DAY_TRADING:
            # デイトレード: より広範囲での買い判定（9日RSI対応）
            return self._score_daytrading_rsi(rsi, max_score)
        else:
            # 中長期: 従来のロジック（14日RSI）
            return self._score_longterm_rsi(rsi, max_score)
    
    def _score_daytrading_rsi(self, rsi: float, max_score: float) -> float:
        """デイトレード用RSIスコア（9日RSI）"""
        # より反応の早い9日RSIに対応した閾値調整
        if 25 <= rsi <= 75:  # デイトレ用に範囲拡大
            if 45 <= rsi <= 65:
                return max_score  # 最適な買いゾーン
            elif 35 <= rsi < 45:
                return max_score * 0.8  # やや弱気だが買い検討
            elif 65 < rsi <= 75:
                return max_score * 0.6  # やや強気だが買い可能
            else:
                return max_score * 0.4  # 買い検討範囲
        elif rsi < 25:
            return max_score * 0.3  # 売られすぎ（短期反転狙い）
        else:  # rsi > 75
            return 0.0  # 買われすぎ（デイトレでは危険）
    
    def _score_longterm_rsi(self, rsi: float, max_score: float) -> float:
        """中長期用RSIスコア（14日RSI）"""
        # 従来のロジック
        if 30 <= rsi <= 70:
            if 50 <= rsi <= 65:
                return max_score  # 最適な買いゾーン
            elif 45 <= rsi < 50:
                return max_score * 0.8  # やや弱気だが買い検討可能
            elif 65 < rsi <= 70:
                return max_score * 0.67  # やや強気だが買い可能
            else:
                return max_score * 0.53  # 買い検討範囲
        elif rsi < 30:
            return max_score * 0.33  # 売られすぎ（リスク高）
        else:  # rsi > 70
            return 0.0  # 買われすぎ（買い推奨しない）
    
    def score_macd_analysis(self, indicators: TechnicalIndicators) -> float:
        """MACD分析スコア（取引モード対応）"""
        if not indicators.macd or not indicators.macd_signal:
            return 0.0
        
        max_score = self.weights.macd_weight
        score = 0.0
        
        # MACD買いシグナル
        if indicators.macd > indicators.macd_signal:
            score += max_score * 0.6
        
        # MACDヒストグラム
        if indicators.macd_histogram and indicators.macd_histogram > 0:
            score += max_score * 0.4
        
        return min(score, max_score)
    
    def score_price_momentum(self, indicators: TechnicalIndicators) -> float:
        """価格モメンタムスコア（取引モード対応）"""
        max_score = self.weights.price_momentum_weight
        
        if self.trading_mode == TradingMode.DAY_TRADING:
            # デイトレード: 短期モメンタム重視（35%）
            return self._score_daytrading_momentum(indicators, max_score)
        else:
            # 中長期: 従来のロジック（15%）
            return self._score_longterm_momentum(indicators, max_score)
    
    def _score_daytrading_momentum(self, indicators: TechnicalIndicators, max_score: float) -> float:
        """デイトレード用モメンタムスコア（最重要指標）"""
        score = 0.0
        
        # 3日モメンタム（最重要）
        if indicators.momentum_3d is not None:
            if 0.5 <= indicators.momentum_3d <= 2.0:
                score += max_score * 0.4  # 適度な短期上昇
            elif 2.0 < indicators.momentum_3d <= 4.0:
                score += max_score * 0.3  # 強い上昇（ややリスク）
            elif -0.5 <= indicators.momentum_3d < 0.5:
                score += max_score * 0.2  # 横ばい
        
        # 5日モメンタム
        if indicators.momentum_5d is not None:
            if 0.5 <= indicators.momentum_5d <= 3.0:
                score += max_score * 0.3  # 良好な短期トレンド
            elif 3.0 < indicators.momentum_5d <= 6.0:
                score += max_score * 0.2  # 急激な上昇
        
        # 短期ボラティリティ（リスク管理）
        if indicators.short_term_volatility is not None:
            if 1.0 <= indicators.short_term_volatility <= 3.0:
                score += max_score * 0.2  # 適度なボラティリティ
            elif indicators.short_term_volatility > 4.0:
                score *= 0.8  # 高ボラティリティペナルティ
        
        # 1日変化率（短期シグナル）
        if indicators.price_change_1d:
            if -0.5 <= indicators.price_change_1d <= 2.0:
                score += max_score * 0.1  # 適度な日内変動
        
        return min(score, max_score)
    
    def _score_longterm_momentum(self, indicators: TechnicalIndicators, max_score: float) -> float:
        """中長期用モメンタムスコア（従来ロジック）"""
        score = 0.0
        
        # 1日変化率
        if indicators.price_change_1d:
            if -1 <= indicators.price_change_1d <= 3:
                score += max_score * 0.2  # 適度な上昇
            elif indicators.price_change_1d > 3:
                score += max_score * 0.07  # 過度な上昇（リスク）
        
        # 5日変化率
        if indicators.price_change_5d:
            if 0 <= indicators.price_change_5d <= 5:
                score += max_score * 0.33  # 良好な短期トレンド
            elif indicators.price_change_5d > 5:
                score += max_score * 0.13  # 急激な上昇
        
        # 25日変化率
        if indicators.price_change_25d:
            if 0 <= indicators.price_change_25d <= 15:
                score += max_score * 0.47  # 健全な中期上昇
            elif indicators.price_change_25d > 15:
                score += max_score * 0.2  # 過度な上昇
        
        return min(score, max_score)
    
    def score_volume_analysis(self, indicators: TechnicalIndicators) -> float:
        """出来高分析スコア（取引モード対応）"""
        max_score = self.weights.volume_weight
        
        if self.trading_mode == TradingMode.DAY_TRADING:
            # デイトレード: 5日出来高比率を活用
            if indicators.volume_ratio_5d is not None:
                if 1.2 <= indicators.volume_ratio_5d <= 2.5:
                    return max_score  # 理想的な出来高増加
                elif 1.0 <= indicators.volume_ratio_5d < 1.2:
                    return max_score * 0.6  # 適度な出来高
                elif indicators.volume_ratio_5d > 3.0:
                    return max_score * 0.4  # 過度な出来高
                else:
                    return max_score * 0.2  # 低出来高
            return max_score * 0.5  # データなし
        else:
            # 中長期: 従来の簡易スコア
            return max_score * 0.5
    
    def score_bollinger_bands(self, indicators: TechnicalIndicators, current_price: float) -> float:
        """ボリンジャーバンド分析スコア（取引モード対応）"""
        if not indicators.bollinger_upper or not indicators.bollinger_lower or not indicators.bollinger_middle:
            return 0.0
        
        max_score = self.weights.bollinger_weight
        
        # 現在価格のバンド内位置を計算
        band_width = indicators.bollinger_upper - indicators.bollinger_lower
        if band_width == 0:
            return max_score * 0.5
        
        position = (current_price - indicators.bollinger_lower) / band_width
        
        if self.trading_mode == TradingMode.DAY_TRADING:
            # デイトレード: より激しい判定（10日BB）
            if 0.3 <= position <= 0.7:
                return max_score  # 良好な買いゾーン
            elif 0.1 <= position < 0.3:
                return max_score * 0.7  # 下限付近（反転狙い）
            elif 0.7 < position <= 0.9:
                return max_score * 0.3  # 上限付近（リスク）
            else:
                return 0.0  # 極端な位置（危険）
        else:
            # 中長期: 従来のロジック（20日BB）
            if 0.2 <= position <= 0.6:
                return max_score  # 良好な買いゾーン
            elif 0.1 <= position < 0.2:
                return max_score * 0.8  # 下限付近（買い検討）
            elif 0.6 < position <= 0.8:
                return max_score * 0.6  # やや上限寄り
            else:
                return max_score * 0.2  # 極端な位置
    
    def score_market_cap_factor(self, market_cap_millions: float) -> float:
        """時価総額ファクタースコア（取引モード対応）"""
        if market_cap_millions <= 0:
            return 0.0
        
        max_score = self.weights.market_cap_weight
        
        if self.trading_mode == TradingMode.DAY_TRADING:
            # デイトレード: 時価総額不要（重み0%）
            return 0.0
        else:
            # 中長期: 従来のロジック
            if 1000 <= market_cap_millions <= 5000:
                return max_score  # 適度な小型株
            elif 500 <= market_cap_millions < 1000:
                return max_score * 0.6  # 極小型株
            elif 5000 < market_cap_millions <= 20000:
                return max_score * 0.8  # 中型株
            else:
                return max_score * 0.4  # 大型株または極小
    
    def calculate_investment_score(
        self, 
        indicators: TechnicalIndicators, 
        current_price: float,
        market_cap_millions: float = 1500.0
    ) -> ScoringResult:
        """総合投資スコアを計算（取引モード対応）"""
        
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
        
        # 総合スコア計算（重み付き合計）
        total_score = sum(component_scores.values())
        
        # 投資推奨を取得（configインスタンス使用）
        recommendation = self.config.get_investment_recommendation(total_score)
        
        # 分析詳細（取引モード対応）
        analysis_details = self._build_analysis_details(
            indicators, current_price, market_cap_millions
        )
        
        return ScoringResult(
            total_score=total_score,
            component_scores=component_scores,
            recommendation=recommendation,
            analysis_details=analysis_details
        )
    
    def _build_analysis_details(
        self, 
        indicators: TechnicalIndicators, 
        current_price: float, 
        market_cap_millions: float
    ) -> Dict[str, any]:
        """分析詳細を構築（取引モード対応）"""
        base_details = {
            "trading_mode": self.trading_mode.value,
            "current_price": current_price,
            "market_cap_millions": market_cap_millions,
            "rsi": indicators.rsi,
            "macd_signal": "買い" if (indicators.macd and indicators.macd_signal and indicators.macd > indicators.macd_signal) else "売り",
            "trend_signal": self._get_trend_signal(indicators)
        }
        
        if self.trading_mode == TradingMode.DAY_TRADING:
            # デイトレード用詳細情報
            base_details.update({
                "short_term_momentum": {
                    "3d": indicators.momentum_3d,
                    "5d": indicators.momentum_5d
                },
                "short_term_volatility": indicators.short_term_volatility,
                "intraday_position": {
                    "high_ratio": indicators.intraday_high_ratio,
                    "low_ratio": indicators.intraday_low_ratio
                },
                "volume_ratio_5d": indicators.volume_ratio_5d,
                "price_changes": {
                    "1d": indicators.price_change_1d,
                    "3d": indicators.price_change_3d
                }
            })
        else:
            # 中長期用詳細情報
            base_details.update({
                "price_changes": {
                    "1d": indicators.price_change_1d,
                    "5d": indicators.price_change_5d,
                    "25d": indicators.price_change_25d
                }
            })
        
        return base_details
    
    def _get_trend_signal(self, indicators: TechnicalIndicators) -> str:
        """トレンドシグナルを取得（取引モード対応）"""
        if self.trading_mode == TradingMode.DAY_TRADING:
            # デイトレード: 短期指標ベース
            if indicators.sma_10 and indicators.sma_20:
                if indicators.sma_10 > indicators.sma_20:
                    if indicators.ema_9 and indicators.ema_21 and indicators.ema_9 > indicators.ema_21:
                        return "強い短期上昇"
                    return "短期上昇"
                else:
                    return "短期下降"
            return "不明"
        else:
            # 中長期: 従来ロジック
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