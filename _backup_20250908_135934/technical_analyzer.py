"""
Technical Analysis Module
テクニカル分析モジュール - 株価データから各種テクニカル指標を計算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
from dataclasses import dataclass

from config import config, TradingMode


@dataclass
class TechnicalIndicators:
    """テクニカル指標データクラス（デイトレード対応）"""
    # 基本指標
    rsi: Optional[float] = None
    sma_5: Optional[float] = None
    sma_25: Optional[float] = None
    sma_75: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_middle: Optional[float] = None
    volume_sma: Optional[float] = None
    price_change_1d: Optional[float] = None
    price_change_5d: Optional[float] = None
    price_change_25d: Optional[float] = None
    volatility: Optional[float] = None
    
    # デイトレード用短期指標
    sma_10: Optional[float] = None  # 10日移動平均
    sma_20: Optional[float] = None  # 20日移動平均
    ema_9: Optional[float] = None   # 9日EMA
    ema_21: Optional[float] = None  # 21日EMA
    short_term_volatility: Optional[float] = None  # 5日ボラティリティ
    
    # イントラデイ指標
    intraday_high_ratio: Optional[float] = None  # 当日高値からの位置
    intraday_low_ratio: Optional[float] = None   # 当日安値からの位置
    momentum_3d: Optional[float] = None          # 3日モメンタム
    momentum_5d: Optional[float] = None          # 5日モメンタム
    
    # 追加の短期変化率
    price_change_3d: Optional[float] = None      # 3日変化率
    
    # 出来高系指標
    volume_ratio_5d: Optional[float] = None      # 5日平均出来高比


class TechnicalAnalyzer:
    """テクニカル分析クラス（デイトレード対応）"""
    
    def __init__(self, trading_config=None):
        """
        テクニカル分析クラス初期化
        
        Args:
            trading_config: 取引設定（デフォルト：グローバルconfig使用）
        """
        if trading_config is not None:
            self.config = trading_config.technical_analysis
            self.trading_mode = trading_config.trading_mode
        else:
            self.config = config.technical_analysis
            self.trading_mode = config.trading_mode
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> Optional[float]:
        """RSI (Relative Strength Index) を計算"""
        if period is None:
            period = self.config.rsi_period
            
        if len(prices) < period:
            return None
        
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
        except (ZeroDivisionError, ValueError, IndexError):
            return None
    
    def calculate_sma(self, prices: pd.Series, period: int) -> Optional[float]:
        """単純移動平均 (SMA) を計算"""
        if len(prices) < period:
            return None
        
        try:
            sma = prices.rolling(window=period).mean()
            return float(sma.iloc[-1])
        except (ValueError, IndexError):
            return None
    
    def calculate_ema(self, prices: pd.Series, period: int) -> Optional[float]:
        """指数移動平均 (EMA) を計算"""
        if len(prices) < period:
            return None
        
        try:
            ema = prices.ewm(span=period, adjust=False).mean()
            return float(ema.iloc[-1])
        except (ValueError, IndexError):
            return None
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """MACD (Moving Average Convergence Divergence) を計算"""
        try:
            ema_12 = prices.ewm(span=self.config.ema_short_period).mean()
            ema_26 = prices.ewm(span=self.config.ema_long_period).mean()
            
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=self.config.macd_signal_period).mean()
            macd_histogram = macd - macd_signal
            
            return (
                float(macd.iloc[-1]),
                float(macd_signal.iloc[-1]),
                float(macd_histogram.iloc[-1])
            )
        except (ValueError, IndexError):
            return None, None, None
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = None, std_dev: float = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """ボリンジャーバンドを計算"""
        if period is None:
            period = self.config.bollinger_period
        if std_dev is None:
            std_dev = self.config.bollinger_std_dev
        
        if len(prices) < period:
            return None, None, None
        
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            bollinger_upper = sma + (std * std_dev)
            bollinger_lower = sma - (std * std_dev)
            
            return (
                float(bollinger_upper.iloc[-1]),
                float(bollinger_lower.iloc[-1]),
                float(sma.iloc[-1])
            )
        except (ValueError, IndexError):
            return None, None, None
    
    def calculate_price_changes(self, prices: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """価格変化率を計算"""
        try:
            current_price = prices.iloc[-1]
            
            # 1日変化率
            change_1d = None
            if len(prices) >= 2 and prices.iloc[-2] != 0:
                change_1d = ((current_price - prices.iloc[-2]) / prices.iloc[-2]) * 100
            elif len(prices) >= 2:
                change_1d = 0.0  # ゼロ除算回避
            
            # 5日変化率
            change_5d = None
            if len(prices) >= 6 and prices.iloc[-6] != 0:
                change_5d = ((current_price - prices.iloc[-6]) / prices.iloc[-6]) * 100
            elif len(prices) >= 6:
                change_5d = 0.0  # ゼロ除算回避
            
            # 25日変化率
            change_25d = None
            if len(prices) >= 26 and prices.iloc[-26] != 0:
                change_25d = ((current_price - prices.iloc[-26]) / prices.iloc[-26]) * 100
            elif len(prices) >= 26:
                change_25d = 0.0  # ゼロ除算回避
            
            return change_1d, change_5d, change_25d
        except (ValueError, IndexError):
            return None, None, None
    
    def calculate_volatility(self, prices: pd.Series, period: int = 20) -> Optional[float]:
        """ボラティリティ（価格変動の標準偏差）を計算"""
        if len(prices) < period:
            return None
        
        try:
            returns = prices.pct_change().dropna()
            if len(returns) < period:
                return None
            
            volatility = returns.tail(period).std() * 100  # パーセンテージ
            return float(volatility)
        except (ValueError, IndexError):
            return None
    
    def analyze_stock(self, data: pd.DataFrame) -> TechnicalIndicators:
        """株価データから全てのテクニカル指標を計算"""
        if data.empty or 'Close' not in data.columns:
            return TechnicalIndicators()
        
        prices = data['Close']
        volumes = data.get('Volume', pd.Series())
        
        # 各指標を計算
        indicators = TechnicalIndicators()
        
        # RSI
        indicators.rsi = self.calculate_rsi(prices)
        
        # 移動平均
        indicators.sma_5 = self.calculate_sma(prices, self.config.sma_short_period)
        indicators.sma_25 = self.calculate_sma(prices, self.config.sma_medium_period)
        indicators.sma_75 = self.calculate_sma(prices, self.config.sma_long_period)
        
        # 指数移動平均
        indicators.ema_12 = self.calculate_ema(prices, self.config.ema_short_period)
        indicators.ema_26 = self.calculate_ema(prices, self.config.ema_long_period)
        
        # MACD
        indicators.macd, indicators.macd_signal, indicators.macd_histogram = self.calculate_macd(prices)
        
        # ボリンジャーバンド
        indicators.bollinger_upper, indicators.bollinger_lower, indicators.bollinger_middle = self.calculate_bollinger_bands(prices)
        
        # 価格変化率
        indicators.price_change_1d, indicators.price_change_5d, indicators.price_change_25d = self.calculate_price_changes(prices)
        
        # ボラティリティ
        indicators.volatility = self.calculate_volatility(prices)
        
        # 出来高移動平均
        if not volumes.empty:
            indicators.volume_sma = self.calculate_sma(volumes, self.config.volume_sma_period)
        
        return indicators
    
    def get_technical_signals(self, indicators: TechnicalIndicators, current_price: float) -> List[str]:
        """テクニカル指標から売買シグナルを生成"""
        signals = []
        
        # トレンド判定
        if indicators.sma_5 and indicators.sma_25 and indicators.sma_75:
            if indicators.sma_5 > indicators.sma_25 > indicators.sma_75:
                signals.append("上昇トレンド")
            elif indicators.sma_5 < indicators.sma_25 < indicators.sma_75:
                signals.append("下降トレンド")
            else:
                signals.append("横ばい")
        
        # 長期トレンド
        if indicators.sma_25 and indicators.sma_75:
            if indicators.sma_25 > indicators.sma_75:
                signals.append("長期上昇")
            elif indicators.sma_25 < indicators.sma_75:
                signals.append("長期下降")
        
        # RSI判定
        if indicators.rsi:
            if indicators.rsi > self.config.rsi_overbought:
                signals.append("RSI売られすぎ")
            elif indicators.rsi >= 50:
                signals.append("RSI強気")
            elif indicators.rsi < self.config.rsi_oversold:
                signals.append("RSI買い場")
        
        # MACD判定
        if indicators.macd and indicators.macd_signal:
            if indicators.macd > indicators.macd_signal:
                signals.append("MACD買いシグナル")
            else:
                signals.append("MACD弱気")
        
        # ボリンジャーバンド判定
        if indicators.bollinger_upper and indicators.bollinger_lower:
            if current_price >= indicators.bollinger_upper * 0.98:  # 上限付近
                signals.append("BB上限付近")
            elif current_price <= indicators.bollinger_lower * 1.02:  # 下限付近
                signals.append("BB下限付近")
        
        # 価格変動判定
        if indicators.price_change_5d:
            if indicators.price_change_5d > 3:
                signals.append("5日上昇")
            elif indicators.price_change_5d < -3:
                signals.append("5日下落")
        
        if indicators.price_change_25d:
            if indicators.price_change_25d > 5:
                signals.append("25日好調")
            elif indicators.price_change_25d < -5:
                signals.append("25日軟調")
        
        # 出来高判定（ダミー）
        signals.append("出来高減少")  # 簡易実装
        
        return signals
    
    def calculate_short_term_volatility(self, prices: pd.Series, period: int = 5) -> Optional[float]:
        """短期ボラティリティを計算（デイトレード用）"""
        if len(prices) < period:
            return None
        
        try:
            returns = prices.pct_change().dropna()
            if len(returns) < period:
                return None
            
            volatility = returns.tail(period).std() * 100  # パーセンテージ
            return float(volatility)
        except (ValueError, IndexError):
            return None
    
    def calculate_intraday_indicators(self, data: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """イントラデイ指標を計算"""
        if data.empty or len(data) < 2:
            return None, None
        
        try:
            # 最新の高値・安値・終値
            latest = data.iloc[-1]
            current_price = latest['Close']
            daily_high = latest['High']
            daily_low = latest['Low']
            
            # 当日の値幅での位置を計算
            if daily_high != daily_low:
                high_ratio = (daily_high - current_price) / (daily_high - daily_low)  # 高値からの位置
                low_ratio = (current_price - daily_low) / (daily_high - daily_low)    # 安値からの位置
            else:
                high_ratio = 0.0
                low_ratio = 0.0
            
            return float(high_ratio), float(low_ratio)
        except (KeyError, ValueError, IndexError):
            return None, None
    
    def calculate_momentum(self, prices: pd.Series, period: int) -> Optional[float]:
        """モメンタムを計算"""
        if len(prices) < period + 1:
            return None
        
        try:
            current_price = prices.iloc[-1]
            past_price = prices.iloc[-(period + 1)]
            
            # ゼロ除算チェック
            if past_price != 0:
                momentum = ((current_price - past_price) / past_price) * 100
            else:
                momentum = 0.0  # ダミーデータの場合
            return float(momentum)
        except (ValueError, IndexError):
            return None
    
    def calculate_volume_ratio(self, volumes: pd.Series, period: int = 5) -> Optional[float]:
        """出来高比率を計算"""
        if volumes.empty or len(volumes) < period:
            return None
        
        try:
            current_volume = volumes.iloc[-1]
            avg_volume = volumes.tail(period).mean()
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                return float(volume_ratio)
            return None
        except (ValueError, IndexError):
            return None
    
    def analyze_stock_enhanced(self, data: pd.DataFrame) -> TechnicalIndicators:
        """株価データから全てのテクニカル指標を計算（デイトレード対応）"""
        if data.empty or 'Close' not in data.columns:
            return TechnicalIndicators()
        
        prices = data['Close']
        volumes = data.get('Volume', pd.Series())
        
        # 各指標を計算
        indicators = TechnicalIndicators()
        
        # 基本指標（従来通り）
        indicators.rsi = self.calculate_rsi(prices)
        
        # 移動平均（モード対応）
        indicators.sma_5 = self.calculate_sma(prices, self.config.sma_short_period)
        
        if hasattr(self.config, 'sma_medium_period'):
            indicators.sma_25 = self.calculate_sma(prices, self.config.sma_medium_period)
        if hasattr(self.config, 'sma_long_period'):
            indicators.sma_75 = self.calculate_sma(prices, self.config.sma_long_period)
        
        # デイトレード用短期移動平均
        if self.trading_mode == TradingMode.DAY_TRADING:
            indicators.sma_10 = self.calculate_sma(prices, 10)
            indicators.sma_20 = self.calculate_sma(prices, 20)
            indicators.ema_9 = self.calculate_ema(prices, 9)
            indicators.ema_21 = self.calculate_ema(prices, 21)
        
        # 指数移動平均
        indicators.ema_12 = self.calculate_ema(prices, self.config.ema_short_period)
        indicators.ema_26 = self.calculate_ema(prices, self.config.ema_long_period)
        
        # MACD
        indicators.macd, indicators.macd_signal, indicators.macd_histogram = self.calculate_macd(prices)
        
        # ボリンジャーバンド
        indicators.bollinger_upper, indicators.bollinger_lower, indicators.bollinger_middle = self.calculate_bollinger_bands(prices)
        
        # 価格変化率
        indicators.price_change_1d, indicators.price_change_5d, indicators.price_change_25d = self.calculate_price_changes(prices)
        
        # 基本ボラティリティ
        indicators.volatility = self.calculate_volatility(prices)
        
        # デイトレード用短期指標
        if self.trading_mode == TradingMode.DAY_TRADING:
            # 短期ボラティリティ
            indicators.short_term_volatility = self.calculate_short_term_volatility(prices)
            
            # イントラデイ指標
            indicators.intraday_high_ratio, indicators.intraday_low_ratio = self.calculate_intraday_indicators(data)
            
            # 短期モメンタム
            indicators.momentum_3d = self.calculate_momentum(prices, 3)
            indicators.momentum_5d = self.calculate_momentum(prices, 5)
            
            # 3日価格変化率
            if len(prices) >= 4:
                try:
                    current_price = prices.iloc[-1]
                    past_price = prices.iloc[-4]
                    if past_price != 0:
                        indicators.price_change_3d = ((current_price - past_price) / past_price) * 100
                    else:
                        indicators.price_change_3d = 0.0  # ゼロ除算回避
                except (ValueError, IndexError):
                    indicators.price_change_3d = None
            
            # 出来高比率
            if not volumes.empty:
                indicators.volume_ratio_5d = self.calculate_volume_ratio(volumes)
        
        # 出来高移動平均
        if not volumes.empty:
            indicators.volume_sma = self.calculate_sma(volumes, self.config.volume_sma_period)
        
        return indicators
    
    def get_enhanced_technical_signals(self, indicators: TechnicalIndicators, current_price: float) -> List[str]:
        """テクニカル指標から売買シグナルを生成（デイトレード対応）"""
        signals = []
        
        # デイトレードモードの場合は短期指標を重視
        if self.trading_mode == TradingMode.DAY_TRADING and indicators.sma_10 and indicators.sma_20:
            # 短期トレンド判定
            if current_price > indicators.sma_10 > indicators.sma_20:
                signals.append("短期上昇トレンド")
            elif current_price < indicators.sma_10 < indicators.sma_20:
                signals.append("短期下降トレンド")
            else:
                signals.append("短期レンジ")
        else:
            # 従来のトレンド判定
            if indicators.sma_5 and indicators.sma_25 and indicators.sma_75:
                if indicators.sma_5 > indicators.sma_25 > indicators.sma_75:
                    signals.append("上昇トレンド")
                elif indicators.sma_5 < indicators.sma_25 < indicators.sma_75:
                    signals.append("下降トレンド")
                else:
                    signals.append("横ばい")
        
        # 長期トレンド（共通）
        if indicators.sma_25 and indicators.sma_75:
            if indicators.sma_25 > indicators.sma_75:
                signals.append("長期上昇")
            elif indicators.sma_25 < indicators.sma_75:
                signals.append("長期下降")
        
        # RSI判定
        if indicators.rsi:
            if indicators.rsi > self.config.rsi_overbought:
                signals.append("RSI売られすぎ")
            elif indicators.rsi >= 50:
                signals.append("RSI強気")
            elif indicators.rsi < self.config.rsi_oversold:
                signals.append("RSI買い場")
        
        # MACD判定
        if indicators.macd and indicators.macd_signal:
            if indicators.macd > indicators.macd_signal:
                signals.append("MACD買いシグナル")
            else:
                signals.append("MACD弱気")
        
        # ボリンジャーバンド判定
        if indicators.bollinger_upper and indicators.bollinger_lower:
            if current_price >= indicators.bollinger_upper * 0.98:
                signals.append("BB上限付近")
            elif current_price <= indicators.bollinger_lower * 1.02:
                signals.append("BB下限付近")
        
        # デイトレード固有シグナル
        if self.trading_mode == TradingMode.DAY_TRADING:
            # 短期モメンタム
            if indicators.momentum_3d and indicators.momentum_3d > 2:
                signals.append("3日強いモメンタム")
            elif indicators.momentum_3d and indicators.momentum_3d < -2:
                signals.append("3日弱いモメンタム")
            
            # イントラデイ位置
            if indicators.intraday_high_ratio is not None:
                if indicators.intraday_high_ratio < 0.2:  # 高値付近
                    signals.append("日中高値圏")
                elif indicators.intraday_high_ratio > 0.8:  # 安値付近
                    signals.append("日中安値圏")
            
            # 短期ボラティリティ
            if indicators.short_term_volatility and indicators.short_term_volatility > 3:
                signals.append("高ボラティリティ")
            
            # 出来高異常
            if indicators.volume_ratio_5d and indicators.volume_ratio_5d > 1.5:
                signals.append("出来高増加")
            elif indicators.volume_ratio_5d and indicators.volume_ratio_5d < 0.5:
                signals.append("出来高減少")
        else:
            # 従来の価格変動判定
            if indicators.price_change_5d:
                if indicators.price_change_5d > 3:
                    signals.append("5日上昇")
                elif indicators.price_change_5d < -3:
                    signals.append("5日下落")
            
            if indicators.price_change_25d:
                if indicators.price_change_25d > 5:
                    signals.append("25日好調")
                elif indicators.price_change_25d < -5:
                    signals.append("25日軟調")
            
            signals.append("出来高減少")  # 簡易実装
        
        return signals