"""
Investment Analysis System Configuration
投資分析システム設定ファイル
"""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path


@dataclass
class InvestmentLimits:
    """投資制限設定"""
    max_investment_per_stock: float = 2000.0  # 1株当たりの投資上限
    base_investment_amount: float = 10000000.0  # ベース投資額（1000万円）
    strong_buy_multiplier: float = 0.8  # 強い買いの投資比率
    buy_multiplier: float = 0.5  # 買いの投資比率


@dataclass
class TechnicalAnalysisConfig:
    """テクニカル分析設定"""
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    sma_short_period: int = 5
    sma_medium_period: int = 25
    sma_long_period: int = 75
    
    ema_short_period: int = 12
    ema_long_period: int = 26
    macd_signal_period: int = 9
    
    bollinger_period: int = 20
    bollinger_std_dev: float = 2.0
    
    volume_sma_period: int = 20


@dataclass
class ScoringWeights:
    """投資スコア重み設定"""
    trend_weight: float = 25.0
    rsi_weight: float = 15.0
    macd_weight: float = 20.0
    price_momentum_weight: float = 15.0
    volume_weight: float = 10.0
    bollinger_weight: float = 10.0
    market_cap_weight: float = 5.0


@dataclass
class InvestmentThresholds:
    """投資判断閾値設定"""
    strong_buy_threshold: float = 85.0
    buy_threshold: float = 55.0
    hold_threshold: float = 45.0
    sell_threshold: float = 35.0
    
    target_profit_strong: float = 15.0  # 強い買いの目標利益率
    target_profit_buy: float = 10.0  # 買いの目標利益率
    
    stop_loss_strong: float = -8.0  # 強い買いの損切りライン
    stop_loss_buy: float = -5.0  # 買いの損切りライン


@dataclass
class SystemConfig:
    """システム全体設定"""
    target_companies_file: str = "ターゲット企業.xlsx"
    output_csv_file: str = "investment_analysis_results.csv"
    
    data_fetch_timeout: int = 30  # Yahoo Finance タイムアウト（秒）
    analysis_period_days: int = 100  # 分析期間（日数）
    
    warning_filters: List[str] = None
    
    def __post_init__(self):
        if self.warning_filters is None:
            self.warning_filters = [
                "ignore:.*timezone.*",
                "ignore:.*FutureWarning.*",
                "ignore:.*PerformanceWarning.*"
            ]


# システム全体の設定インスタンス
class Config:
    """統合設定クラス"""
    
    def __init__(self):
        self.investment_limits = InvestmentLimits()
        self.technical_analysis = TechnicalAnalysisConfig()
        self.scoring_weights = ScoringWeights()
        self.investment_thresholds = InvestmentThresholds()
        self.system = SystemConfig()
    
    @property
    def target_companies_path(self) -> Path:
        """ターゲット企業ファイルのパス"""
        return Path(self.system.target_companies_file)
    
    @property
    def output_csv_path(self) -> Path:
        """出力CSVファイルのパス"""
        return Path(self.system.output_csv_file)
    
    def get_investment_recommendation(self, score: float) -> Dict[str, any]:
        """スコアから投資推奨を取得"""
        thresholds = self.investment_thresholds
        limits = self.investment_limits
        
        if score >= thresholds.strong_buy_threshold:
            return {
                "judgment": "強い買い",
                "position_size": limits.strong_buy_multiplier,
                "target_profit": thresholds.target_profit_strong,
                "stop_loss": thresholds.stop_loss_strong,
                "holding_period": "2-6ヶ月"
            }
        elif score >= thresholds.buy_threshold:
            return {
                "judgment": "買い", 
                "position_size": limits.buy_multiplier,
                "target_profit": thresholds.target_profit_buy,
                "stop_loss": thresholds.stop_loss_buy,
                "holding_period": "2-6ヶ月"
            }
        elif score >= thresholds.hold_threshold:
            return {
                "judgment": "ホールド",
                "position_size": 0.0,
                "target_profit": 0.0,
                "stop_loss": 0.0,
                "holding_period": "様子見"
            }
        elif score >= thresholds.sell_threshold:
            return {
                "judgment": "小幅売り",
                "position_size": -0.3,
                "target_profit": 0.0,
                "stop_loss": 0.0,
                "holding_period": "現在売却推奨"
            }
        else:
            return {
                "judgment": "売り",
                "position_size": -0.6,
                "target_profit": 0.0,
                "stop_loss": 0.0,
                "holding_period": "現在売却推奨"
            }


# グローバル設定インスタンス
config = Config()