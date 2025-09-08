"""
Investment Analysis System Configuration
投資分析システム設定ファイル
"""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
from enum import Enum


class TradingMode(Enum):
    """取引スタイル設定"""
    LONG_TERM = "long_term"    # 中長期投資（従来）
    DAY_TRADING = "day_trading"  # デイトレード


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
class DayTradingTechnicalConfig:
    """デイトレード用テクニカル分析設定"""
    rsi_period: int = 9  # 短期RSI
    rsi_overbought: float = 75.0  # より高い売られすぎ線
    rsi_oversold: float = 25.0  # より低い買われすぎ線
    
    # 短期移動平均
    sma_short_period: int = 5
    sma_medium_period: int = 10
    sma_long_period: int = 20
    
    # 短期EMA設定
    ema_short_period: int = 9
    ema_long_period: int = 21
    macd_signal_period: int = 6
    
    # ボリンジャーバンド（短期）
    bollinger_period: int = 10
    bollinger_std_dev: float = 2.0
    
    # 出来高（短期）
    volume_sma_period: int = 10
    
    # デイトレード固有設定
    intraday_high_low_period: int = 5  # 当日高値・安値判定期間
    momentum_period: int = 3  # モメンタム計算期間


@dataclass
class DayTradingLimits:
    """デイトレード用投資制限"""
    max_investment_per_stock: float = 2000.0  # 1株当たりの投資上限（同じ）
    base_investment_amount: float = 10000000.0  # ベース投資額
    strong_buy_multiplier: float = 0.6  # 強い買い（デイトレードはより慎重）
    buy_multiplier: float = 0.4  # 買い（デイトレードはより慎重）
    
    # デイトレード固有制限
    max_daily_positions: int = 5  # 1日の最大同時保有数
    max_daily_loss: float = 5000.0  # 1日の最大損失額
    max_single_loss: float = 1000.0  # 1銘柄の最大損失額


@dataclass
class DayTradingThresholds:
    """デイトレード用投資判断閾値"""
    strong_buy_threshold: float = 75.0  # より低い閾値（機会を逃さない）
    buy_threshold: float = 60.0  # より積極的
    hold_threshold: float = 50.0  # ニュートラル
    sell_threshold: float = 40.0  # 早めの撤退
    
    # デイトレード用利益目標（小さく確実に）
    target_profit_strong: float = 3.0  # 強い買いの目標利益率
    target_profit_buy: float = 2.0  # 買いの目標利益率
    
    # デイトレード用損切り（早めに浅く）
    stop_loss_strong: float = -1.5  # 強い買いの損切りライン
    stop_loss_buy: float = -1.0  # 買いの損切りライン


@dataclass
class DayTradingScoreWeights:
    """デイトレード用スコア重み設定"""
    trend_weight: float = 20.0  # トレンド（短期重視）
    rsi_weight: float = 15.0  # RSI
    macd_weight: float = 15.0  # MACD
    price_momentum_weight: float = 35.0  # モメンタム（最重要）
    volume_weight: float = 10.0  # 出来高
    bollinger_weight: float = 5.0  # ボリンジャーバンド
    market_cap_weight: float = 0.0  # 時価総額（デイトレードでは不要）


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
    
    def __init__(self, trading_mode: TradingMode = TradingMode.LONG_TERM):
        self.trading_mode = trading_mode
        self.system = SystemConfig()
        
        # 取引モードに応じて設定を切り替え
        if trading_mode == TradingMode.DAY_TRADING:
            self.investment_limits = DayTradingLimits()
            self.technical_analysis = DayTradingTechnicalConfig()
            self.scoring_weights = DayTradingScoreWeights()
            self.investment_thresholds = DayTradingThresholds()
        else:
            # 従来の中長期設定
            self.investment_limits = InvestmentLimits()
            self.technical_analysis = TechnicalAnalysisConfig()
            self.scoring_weights = ScoringWeights()
            self.investment_thresholds = InvestmentThresholds()
    
    @property
    def target_companies_path(self) -> Path:
        """ターゲット企業ファイルのパス"""
        return Path(self.system.target_companies_file)
    
    @property
    def output_csv_path(self) -> Path:
        """出力CSVファイルのパス"""
        return Path(self.system.output_csv_file)
    
    def get_investment_recommendation(self, score: float) -> Dict[str, any]:
        """スコアから投資推奨を取得（取引モード対応）"""
        thresholds = self.investment_thresholds
        limits = self.investment_limits
        
        # 取引モードに応じた保有期間設定
        if self.trading_mode == TradingMode.DAY_TRADING:
            holding_periods = {
                "strong_buy": "1-3日",
                "buy": "当日のみ", 
                "hold": "監視継続",
                "sell": "即座に売却"
            }
        else:
            holding_periods = {
                "strong_buy": "2-6ヶ月",
                "buy": "2-6ヶ月",
                "hold": "様子見", 
                "sell": "現在売却推奨"
            }
        
        if score >= thresholds.strong_buy_threshold:
            return {
                "judgment": "強い買い",
                "position_size": limits.strong_buy_multiplier,
                "target_profit": thresholds.target_profit_strong,
                "stop_loss": thresholds.stop_loss_strong,
                "holding_period": holding_periods["strong_buy"]
            }
        elif score >= thresholds.buy_threshold:
            return {
                "judgment": "買い", 
                "position_size": limits.buy_multiplier,
                "target_profit": thresholds.target_profit_buy,
                "stop_loss": thresholds.stop_loss_buy,
                "holding_period": holding_periods["buy"]
            }
        elif score >= thresholds.hold_threshold:
            return {
                "judgment": "ホールド",
                "position_size": 0.0,
                "target_profit": 0.0,
                "stop_loss": 0.0,
                "holding_period": holding_periods["hold"]
            }
        elif score >= thresholds.sell_threshold:
            return {
                "judgment": "小幅売り",
                "position_size": -0.3,
                "target_profit": 0.0,
                "stop_loss": 0.0,
                "holding_period": holding_periods["sell"]
            }
        else:
            return {
                "judgment": "売り",
                "position_size": -0.6,
                "target_profit": 0.0,
                "stop_loss": 0.0,
                "holding_period": holding_periods["sell"]
            }
    
    def switch_trading_mode(self, new_mode: TradingMode) -> None:
        """取引モードを切り替える"""
        if new_mode == self.trading_mode:
            return  # 既に同じモード
        
        self.trading_mode = new_mode
        
        # 設定を再初期化
        if new_mode == TradingMode.DAY_TRADING:
            self.investment_limits = DayTradingLimits()
            self.technical_analysis = DayTradingTechnicalConfig()
            self.scoring_weights = DayTradingScoreWeights()
            self.investment_thresholds = DayTradingThresholds()
        else:
            self.investment_limits = InvestmentLimits()
            self.technical_analysis = TechnicalAnalysisConfig()
            self.scoring_weights = ScoringWeights()
            self.investment_thresholds = InvestmentThresholds()
    
    def get_trading_mode_info(self) -> Dict[str, any]:
        """現在の取引モード情報を取得"""
        return {
            "mode": self.trading_mode.value,
            "description": "デイトレード" if self.trading_mode == TradingMode.DAY_TRADING else "中長期投資",
            "target_profit_range": f"{self.investment_thresholds.target_profit_buy}-{self.investment_thresholds.target_profit_strong}%",
            "stop_loss_range": f"{self.investment_thresholds.stop_loss_buy}-{self.investment_thresholds.stop_loss_strong}%",
            "analysis_focus": "短期モメンタム重視" if self.trading_mode == TradingMode.DAY_TRADING else "中長期トレンド重視"
        }
    
    def is_day_trading_mode(self) -> bool:
        """デイトレードモードかどうかを判定"""
        return self.trading_mode == TradingMode.DAY_TRADING
    
    def get_execution_timing_info(self) -> str:
        """実行タイミングの推奨情報を取得"""
        if self.trading_mode == TradingMode.DAY_TRADING:
            return "毎日16:00実行 → 翌日寄り付きエントリー判断"
        else:
            return "週次/月次実行 → 中長期ポジション調整"


# グローバル設定インスタンス（デフォルト: 中長期モード）
config = Config()

# デイトレード用設定インスタンス
day_trading_config = Config(TradingMode.DAY_TRADING)