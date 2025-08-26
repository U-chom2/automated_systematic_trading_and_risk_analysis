"""Technical analysis module using TA-Lib and custom indicators."""

from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """Technical indicator values."""
    rsi: float
    moving_avg_deviation: float
    volume_ratio: float
    atr: float
    bollinger_upper: float
    bollinger_lower: float
    macd_line: float
    macd_signal: float
    stochastic_k: float
    stochastic_d: float


@dataclass
class MarketEnvironmentScore:
    """Market environment analysis result."""
    nikkei_trend: str  # "bullish", "bearish", "neutral"
    topix_trend: str
    sector_trend: str
    market_score: int  # 0-20 points
    risk_level: str   # "low", "medium", "high"


class TechnicalAnalyzer:
    """Technical analysis using TA-Lib and custom calculations."""
    
    def __init__(self) -> None:
        """Initialize Technical Analyzer."""
        self.indicators_cache: Dict[str, Dict] = {}
        logger.info("TechnicalAnalyzer initialized")
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            data: DataFrame with 'close' column
            period: RSI period
            
        Returns:
            RSI value (0-100)
        """
        if data.empty:
            raise ValueError("Data cannot be empty")
        if len(data) < period:
            raise ValueError(f"Insufficient data: need at least {period} periods")
        
        closes = data['close'].values
        deltas = np.diff(closes)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[:period])
        avg_losses = np.mean(losses[:period])
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_moving_average_deviation(self, data: pd.DataFrame, period: int = 25) -> float:
        """
        Calculate moving average deviation percentage.
        
        Args:
            data: DataFrame with 'close' column
            period: Moving average period
            
        Returns:
            Deviation percentage
        """
        if data.empty:
            raise ValueError("Data cannot be empty")
        if len(data) < period:
            raise ValueError(f"Insufficient data: need at least {period} periods")
        
        closes = data['close'].values
        current_price = closes[-1]
        ma = np.mean(closes[-period:])
        
        deviation = ((current_price - ma) / ma) * 100
        return float(deviation)
    
    def calculate_rsi_old(self, prices: List[Decimal], period: int = 14) -> float:
        """
        Calculate RSI (Legacy API).
        
        Args:
            prices: List of closing prices
            period: RSI period
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if insufficient data
        
        # Convert to numpy array
        price_array = np.array([float(p) for p in prices])
        
        # Calculate price changes
        deltas = np.diff(price_array)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[:period]) if len(gains) >= period else 0
        avg_losses = np.mean(losses[:period]) if len(losses) >= period else 0
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses if avg_losses != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_moving_average_deviation_legacy(self, current_price: Decimal, 
                                         prices: List[Decimal], 
                                         period: int = 25) -> float:
        """
        Calculate moving average deviation percentage (Legacy API).
        
        Args:
            current_price: Current stock price
            prices: Historical prices
            period: Moving average period
            
        Returns:
            Deviation percentage
        """
        if len(prices) < period:
            return 0.0
        
        price_array = np.array([float(p) for p in prices[-period:]])
        ma = np.mean(price_array)
        
        if ma == 0:
            return 0.0
        
        deviation = ((float(current_price) - ma) / ma) * 100
        return float(deviation)

    def calculate_volume_ratio(self, current_volume: int, 
                             historical_volumes: List[int], 
                             period: int = 20) -> float:
        """
        Calculate volume ratio compared to average.
        
        Args:
            current_volume: Current trading volume
            historical_volumes: Historical volume data
            period: Period for average calculation
            
        Returns:
            Volume ratio (1.0 = average)
        """
        if len(historical_volumes) < period:
            return 1.0
        
        avg_volume = np.mean(historical_volumes[-period:])
        
        if avg_volume == 0:
            return 1.0
        
        return float(current_volume / avg_volume)
    
    def calculate_atr_legacy(self, highs: List[Decimal], lows: List[Decimal], 
                     closes: List[Decimal], period: int = 14) -> Decimal:
        """
        Calculate Average True Range (ATR) (Legacy API).
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
            return Decimal("0.0")
        
        # Convert to numpy arrays
        highs_array = np.array([float(h) for h in highs])
        lows_array = np.array([float(l) for l in lows])
        closes_array = np.array([float(c) for c in closes])
        
        # Calculate True Range
        tr_list = []
        for i in range(1, len(highs_array)):
            tr1 = highs_array[i] - lows_array[i]
            tr2 = abs(highs_array[i] - closes_array[i-1])
            tr3 = abs(lows_array[i] - closes_array[i-1])
            tr_list.append(max(tr1, tr2, tr3))
        
        if len(tr_list) < period:
            atr = np.mean(tr_list) if tr_list else 0.0
        else:
            atr = np.mean(tr_list[-period:])
        
        return Decimal(str(atr))
    
    # Wrapper methods for backward compatibility
    def calculate_rsi(self, data, period: int = 14) -> float:
        """RSI calculation with dual API support."""
        if isinstance(data, pd.DataFrame):
            return self.calculate_rsi_new(data, period)
        elif isinstance(data, list):
            return self.calculate_rsi_old(data, period)
        else:
            raise ValueError("Data must be pandas DataFrame or list of Decimal")
    
    def calculate_moving_average_deviation(self, data, *args, **kwargs) -> float:
        """MA deviation with dual API support."""
        if isinstance(data, pd.DataFrame):
            period = args[0] if args else kwargs.get('period', 25)
            return self.calculate_moving_average_deviation_new(data, period)
        elif isinstance(data, Decimal):
            prices = args[0]
            period = args[1] if len(args) > 1 else kwargs.get('period', 25)
            return self.calculate_moving_average_deviation_legacy(data, prices, period)
        else:
            raise ValueError("Invalid arguments for moving_average_deviation")
    
    def calculate_atr(self, data, *args, **kwargs):
        """ATR calculation with dual API support."""
        if isinstance(data, pd.DataFrame):
            period = args[0] if args else kwargs.get('period', 14)
            return self.calculate_atr_new(data, period)
        elif isinstance(data, list) and len(args) >= 2:
            # Legacy API: highs, lows, closes, period
            lows = args[0]
            closes = args[1]
            period = args[2] if len(args) > 2 else kwargs.get('period', 14)
            return self.calculate_atr_legacy(data, lows, closes, period)
        else:
            raise ValueError("Invalid arguments for ATR calculation")
    
    # Rename methods for clarity
    def calculate_rsi_new(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI using pandas DataFrame."""
        if data.empty:
            raise ValueError("Data cannot be empty")
        if len(data) < period:
            raise ValueError(f"Insufficient data: need at least {period} periods")
        
        closes = data['close'].values
        deltas = np.diff(closes)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[:period])
        avg_losses = np.mean(losses[:period])
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_moving_average_deviation_new(self, data: pd.DataFrame, period: int = 25) -> float:
        """Calculate MA deviation using pandas DataFrame."""
        if data.empty:
            raise ValueError("Data cannot be empty")
        if len(data) < period:
            raise ValueError(f"Insufficient data: need at least {period} periods")
        
        closes = data['close'].values
        current_price = closes[-1]
        ma = np.mean(closes[-period:])
        
        deviation = ((current_price - ma) / ma) * 100
        return float(deviation)
    
    def calculate_atr_new(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR using pandas DataFrame."""
        if len(data) < 2:
            return 0.0
        
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # True Range calculation
        tr_list = []
        for i in range(1, len(data)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr_list.append(max(tr1, tr2, tr3))
        
        if len(tr_list) < period:
            return float(np.mean(tr_list))
        
        return float(np.mean(tr_list[-period:]))
    
    def calculate_bollinger_bands(self, prices: List[Decimal], 
                            period: int = 20, 
                            std_dev: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands using TA-Lib.
        
        Args:
            prices: Closing prices
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        try:
            import talib
            import numpy as np
            
            if len(prices) < period:
                return 0.0, 0.0, 0.0
            
            # Convert Decimal to float array
            price_array = np.array([float(p) for p in prices], dtype=np.float64)
            
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                price_array,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0
            )
            
            # Return the latest values
            latest_upper = float(upper[-1]) if not np.isnan(upper[-1]) else 0.0
            latest_middle = float(middle[-1]) if not np.isnan(middle[-1]) else 0.0
            latest_lower = float(lower[-1]) if not np.isnan(lower[-1]) else 0.0
            
            logger.debug(f"Bollinger Bands: upper={latest_upper:.2f}, "
                        f"middle={latest_middle:.2f}, lower={latest_lower:.2f}")
            
            return latest_upper, latest_middle, latest_lower
            
        except ImportError:
            logger.warning("TA-Lib not available, using simple calculation")
            return self._simple_bollinger_bands(prices, period, std_dev)
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            return 0.0, 0.0, 0.0
    
    def _simple_bollinger_bands(self, prices: List[Decimal], 
                               period: int, 
                               std_dev: float) -> Tuple[float, float, float]:
        """
        Simple Bollinger Bands calculation fallback.
        
        Args:
            prices: Closing prices
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(prices) < period:
            return 0.0, 0.0, 0.0
        
        import numpy as np
        
        # Get recent prices
        recent_prices = [float(p) for p in prices[-period:]]
        
        # Calculate middle band (SMA)
        middle = np.mean(recent_prices)
        
        # Calculate standard deviation
        std = np.std(recent_prices)
        
        # Calculate upper and lower bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return float(upper), float(middle), float(lower)  # Placeholder
    
    def calculate_macd(self, prices: List[Decimal], 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD indicators using TA-Lib.
        
        Args:
            prices: Closing prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        try:
            import talib
            import numpy as np
            
            if len(prices) < slow_period:
                return 0.0, 0.0, 0.0
            
            # Convert Decimal to float array
            price_array = np.array([float(p) for p in prices], dtype=np.float64)
            
            # Calculate MACD
            macd_line, signal_line, histogram = talib.MACD(
                price_array,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            
            # Return the latest values
            latest_macd = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else 0.0
            latest_signal = float(signal_line[-1]) if not np.isnan(signal_line[-1]) else 0.0
            latest_histogram = float(histogram[-1]) if not np.isnan(histogram[-1]) else 0.0
            
            logger.debug(f"MACD: line={latest_macd:.4f}, "
                        f"signal={latest_signal:.4f}, histogram={latest_histogram:.4f}")
            
            return latest_macd, latest_signal, latest_histogram
            
        except ImportError:
            logger.warning("TA-Lib not available, using simple calculation")
            return self._simple_macd(prices, fast_period, slow_period, signal_period)
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return 0.0, 0.0, 0.0
    
    def _simple_macd(self, prices: List[Decimal], 
                     fast_period: int, 
                     slow_period: int, 
                     signal_period: int) -> Tuple[float, float, float]:
        """
        Simple MACD calculation fallback.
        
        Args:
            prices: Closing prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(prices) < slow_period:
            return 0.0, 0.0, 0.0
        
        import numpy as np
        
        price_array = np.array([float(p) for p in prices])
        
        # Calculate EMAs
        def calculate_ema(data, period):
            alpha = 2 / (period + 1)
            ema = [data[0]]
            for i in range(1, len(data)):
                ema.append(alpha * data[i] + (1 - alpha) * ema[i-1])
            return ema
        
        # Calculate fast and slow EMAs
        fast_ema = calculate_ema(price_array, fast_period)
        slow_ema = calculate_ema(price_array, slow_period)
        
        # Calculate MACD line
        macd_values = []
        start_idx = max(0, len(price_array) - len(slow_ema))
        for i in range(len(slow_ema)):
            if i < len(fast_ema):
                macd_values.append(fast_ema[start_idx + i] - slow_ema[i])
        
        if len(macd_values) < signal_period:
            return 0.0, 0.0, 0.0
        
        # Calculate signal line (EMA of MACD)
        signal_values = calculate_ema(macd_values, signal_period)
        
        # Calculate histogram
        histogram = macd_values[-1] - signal_values[-1] if signal_values else 0.0
        
        return float(macd_values[-1]), float(signal_values[-1]), float(histogram)  # Placeholder
    
    def get_technical_indicators(self, symbol: str, 
                           price_data: List[Dict[str, Any]]) -> TechnicalIndicators:
        """
        Calculate all technical indicators for a symbol.
        
        Args:
            symbol: Stock symbol
            price_data: Historical price data
            
        Returns:
            Technical indicators object
        """
        try:
            if not price_data or len(price_data) < 14:
                logger.warning(f"Insufficient data for {symbol}: {len(price_data)} records")
                return self._get_default_indicators()
            
            # Convert price data to pandas DataFrame if it's not already
            import pandas as pd
            from decimal import Decimal
            
            # Prepare data
            df = pd.DataFrame(price_data)
            if 'close' not in df.columns:
                logger.error(f"Missing 'close' column in price data for {symbol}")
                return self._get_default_indicators()
            
            # Convert to appropriate types
            closes = [Decimal(str(row['close'])) for row in price_data]
            volumes = [int(row.get('volume', 0)) for row in price_data]
            
            # Calculate RSI
            try:
                rsi = self.calculate_rsi(df)
            except Exception as e:
                logger.warning(f"RSI calculation failed for {symbol}: {e}")
                rsi = 50.0
            
            # Calculate Moving Average Deviation
            try:
                ma_deviation = self.calculate_moving_average_deviation(df)
            except Exception as e:
                logger.warning(f"MA deviation calculation failed for {symbol}: {e}")
                ma_deviation = 0.0
            
            # Calculate Volume Ratio
            try:
                volume_ratio = self.calculate_volume_ratio(
                    volumes[-1] if volumes else 0, 
                    volumes[:-1] if len(volumes) > 1 else [0]
                )
            except Exception as e:
                logger.warning(f"Volume ratio calculation failed for {symbol}: {e}")
                volume_ratio = 1.0
            
            # Calculate ATR
            try:
                if 'high' in df.columns and 'low' in df.columns:
                    atr = self.calculate_atr(df)
                else:
                    # Fallback: use close prices for simplified ATR
                    atr = float(self.calculate_atr_legacy(closes, closes, closes))
            except Exception as e:
                logger.warning(f"ATR calculation failed for {symbol}: {e}")
                atr = 0.0
            
            # Calculate Bollinger Bands
            try:
                bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes)
            except Exception as e:
                logger.warning(f"Bollinger Bands calculation failed for {symbol}: {e}")
                bb_upper, bb_middle, bb_lower = 0.0, 0.0, 0.0
            
            # Calculate MACD
            try:
                macd_line, macd_signal, macd_histogram = self.calculate_macd(closes)
            except Exception as e:
                logger.warning(f"MACD calculation failed for {symbol}: {e}")
                macd_line, macd_signal, macd_histogram = 0.0, 0.0, 0.0
            
            # Calculate Stochastic (simple implementation)
            try:
                stoch_k, stoch_d = self._calculate_stochastic(df)
            except Exception as e:
                logger.warning(f"Stochastic calculation failed for {symbol}: {e}")
                stoch_k, stoch_d = 50.0, 50.0
            
            logger.debug(f"Technical indicators calculated for {symbol}: "
                        f"RSI={rsi:.1f}, MA_dev={ma_deviation:.2f}%")
            
            return TechnicalIndicators(
                rsi=rsi,
                moving_avg_deviation=ma_deviation,
                volume_ratio=volume_ratio,
                atr=atr,
                bollinger_upper=bb_upper,
                bollinger_lower=bb_lower,
                macd_line=macd_line,
                macd_signal=macd_signal,
                stochastic_k=stoch_k,
                stochastic_d=stoch_d
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators for {symbol}: {e}")
            return self._get_default_indicators()
    
    def _get_default_indicators(self) -> TechnicalIndicators:
        """Return default technical indicators when calculation fails."""
        return TechnicalIndicators(
            rsi=50.0,
            moving_avg_deviation=0.0,
            volume_ratio=1.0,
            atr=0.0,
            bollinger_upper=0.0,
            bollinger_lower=0.0,
            macd_line=0.0,
            macd_signal=0.0,
            stochastic_k=50.0,
            stochastic_d=50.0
        )
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """
        Calculate Stochastic oscillator.
        
        Args:
            df: DataFrame with high, low, close columns
            k_period: %K period
            d_period: %D smoothing period
            
        Returns:
            Tuple of (%K, %D)
        """
        if len(df) < k_period:
            return 50.0, 50.0
        
        try:
            import talib
            import numpy as np
            
            if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
                high = df['high'].values.astype(np.float64)
                low = df['low'].values.astype(np.float64)
                close = df['close'].values.astype(np.float64)
                
                slowk, slowd = talib.STOCH(high, low, close, 
                                          fastk_period=k_period, 
                                          slowk_period=d_period, 
                                          slowd_period=d_period)
                
                return float(slowk[-1] if not np.isnan(slowk[-1]) else 50.0), \
                       float(slowd[-1] if not np.isnan(slowd[-1]) else 50.0)
            else:
                # Fallback using only close prices
                return self._simple_stochastic(df['close'].values, k_period)
                
        except ImportError:
            return self._simple_stochastic(df['close'].values, k_period)
        except Exception as e:
            logger.warning(f"Stochastic calculation failed: {e}")
            return 50.0, 50.0
    
    def _simple_stochastic(self, closes, period: int) -> Tuple[float, float]:
        """Simple stochastic calculation fallback."""
        if len(closes) < period:
            return 50.0, 50.0
        
        # Use close prices as high and low for simplified calculation
        recent_closes = closes[-period:]
        current_close = closes[-1]
        highest = max(recent_closes)
        lowest = min(recent_closes)
        
        if highest == lowest:
            k_value = 50.0
        else:
            k_value = ((current_close - lowest) / (highest - lowest)) * 100
        
        return float(k_value), float(k_value)  # Simplified: %D = %K
    
    def analyze_market_environment(self) -> MarketEnvironmentScore:
        """
        Analyze overall market environment (Nikkei, TOPIX, sector).
        
        Returns:
            Market environment analysis result
        """
        # TODO: Implement market environment analysis
        logger.debug("Analyzing market environment")
        
        return MarketEnvironmentScore(
            nikkei_trend="neutral",
            topix_trend="neutral",
            sector_trend="neutral",
            market_score=10,  # 0-20 points
            risk_level="medium"
        )
    
    def check_overheating_filter(self, indicators: TechnicalIndicators) -> bool:
        """
        Check if stock is overheated (filter condition).
        
        Args:
            indicators: Technical indicators
            
        Returns:
            True if overheated (should skip trading)
        """
        # Filter conditions from requirements:
        # RSI > 75 OR moving_avg_deviation > +25%
        is_overheated = (
            indicators.rsi > 75.0 or 
            indicators.moving_avg_deviation > 25.0
        )
        
        logger.debug(f"Overheating check: RSI={indicators.rsi:.1f}, "
                    f"MA_dev={indicators.moving_avg_deviation:.1f}% -> {is_overheated}")
        
        return is_overheated
    
    def calculate_technical_score_for_trading(self, symbol: str,
                                            price_data: List[Dict[str, Any]]) -> int:
        """
        Calculate technical score for trading decision.
        
        Args:
            symbol: Stock symbol
            price_data: Price data
            
        Returns:
            Combined technical + market score (0-50 points)
        """
        indicators = self.get_technical_indicators(symbol, price_data)
        market_env = self.analyze_market_environment()
        
        # Check overheating filter first
        if self.check_overheating_filter(indicators):
            logger.warning(f"Overheating detected for {symbol}, returning 0 score")
            return 0
        
        # TODO: Implement sophisticated scoring logic
        # For now, return market environment score (max 20 points)
        technical_score = market_env.market_score
        
        logger.debug(f"Technical score for {symbol}: {technical_score}/20")
        return technical_score
    
    def calculate_volume_surge(self, data: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate volume surge ratio.
        
        Args:
            data: DataFrame with 'volume' column
            period: Period for average calculation
            
        Returns:
            Volume surge ratio
        """
        if len(data) < period:
            return 1.0
        
        volumes = data['volume'].values
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-period:])
        
        if avg_volume == 0:
            return 1.0
        
        return float(current_volume / avg_volume)
    
    def calculate_historical_volatility(self, data: pd.DataFrame, period: int = 60) -> float:
        """
        Calculate historical volatility.
        
        Args:
            data: DataFrame with 'close' column
            period: Period for calculation
            
        Returns:
            Annualized volatility
        """
        if len(data) < 2:
            return 0.0
        
        closes = data['close'].values
        returns = np.diff(np.log(closes))
        
        if len(returns) < period:
            vol = np.std(returns)
        else:
            vol = np.std(returns[-period:])
        
        # Annualize (252 trading days)
        annual_vol = vol * np.sqrt(252)
        return float(annual_vol)
    
    def is_rsi_overheated(self, rsi: float, threshold: float = 75) -> bool:
        """Check if RSI indicates overheating."""
        return rsi > threshold
    
    def is_ma_deviation_overheated(self, deviation: float, threshold: float = 25) -> bool:
        """Check if MA deviation indicates overheating."""
        return deviation > threshold
    
    def is_volume_surge(self, surge_ratio: float, threshold: float = 2.0) -> bool:
        """Check if volume surge is significant."""
        return surge_ratio > threshold
    
    def check_filter_conditions(self, rsi: float, ma_deviation: float) -> bool:
        """
        Check if stock passes filter conditions.
        
        Args:
            rsi: RSI value
            ma_deviation: Moving average deviation
            
        Returns:
            True if passes filter (can trade)
        """
        # Filter out overheated stocks
        if rsi > 75 or ma_deviation > 25:
            return False
        return True
    
    def calculate_technical_score(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive technical score.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            Technical analysis results
        """
        rsi = self.calculate_rsi(data)
        ma_deviation = self.calculate_moving_average_deviation(data)
        volume_surge_ratio = self.calculate_volume_surge(data)
        
        # Market environment score (simplified)
        market_score = 10  # Base score
        if rsi < 70 and ma_deviation < 20:
            market_score += 5
        if volume_surge_ratio > 1.5:
            market_score += 5
        
        filter_passed = self.check_filter_conditions(rsi, ma_deviation)
        
        return {
            "market_environment_score": min(market_score, 20),
            "filter_passed": filter_passed,
            "rsi": rsi,
            "ma_deviation": ma_deviation,
            "volume_surge_ratio": volume_surge_ratio
        }
    
    def get_risk_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Get indicators for risk model.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            Dictionary of risk indicators
        """
        atr = self.calculate_atr(data)
        hv = self.calculate_historical_volatility(data)
        volume_ratio = self.calculate_volume_surge(data)
        
        # Price momentum (simple calculation)
        closes = data['close'].values
        if len(closes) >= 5:
            momentum = (closes[-1] / closes[-5] - 1) * 100
        else:
            momentum = 0.0
        
        return {
            "atr": float(atr),
            "historical_volatility": float(hv),
            "volume_ratio": float(volume_ratio),
            "price_momentum": float(momentum)
        }
    
    def analyze_daily_technicals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive daily technical analysis for 16:15 scheduled processing.
        
        Args:
            market_data: End-of-day market data including price, volume, indicators
            
        Returns:
            Comprehensive technical analysis result
        """
        logger.info("Starting daily technical analysis")
        
        # Extract key data
        close_price = float(market_data.get("close_price", 0))
        volume = market_data.get("volume", 0)
        volatility = market_data.get("volatility", 0.25)
        
        # Calculate technical indicators if not already present
        rsi = market_data.get("rsi")
        if rsi is None and "historical_data" in market_data:
            rsi = self.calculate_rsi(market_data["historical_data"])
        
        ma_deviation = market_data.get("ma_deviation")
        if ma_deviation is None and "historical_data" in market_data:
            ma_deviation = self.calculate_moving_average_deviation(
                market_data["historical_data"], period=20
            )
        
        # Calculate technical score (0-20)
        technical_score = 10  # Base score
        
        # RSI scoring
        if rsi:
            if 40 <= rsi <= 60:  # Neutral zone
                technical_score += 5
            elif 30 <= rsi < 40:  # Oversold, potential bounce
                technical_score += 3
            elif rsi > 75:  # Overbought
                technical_score -= 5
        
        # MA deviation scoring
        if ma_deviation:
            if 0 < ma_deviation < 10:  # Mild positive deviation
                technical_score += 5
            elif ma_deviation > 25:  # Overextended
                technical_score -= 5
        
        # Volume analysis
        avg_volume = market_data.get("avg_volume", volume)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        if volume_ratio > 1.5:  # Volume surge
            technical_score += 2
        
        # Cap at 20
        technical_score = min(max(technical_score, 0), 20)
        
        # Check filter conditions
        filter_passed = True
        filter_reasons = []
        
        if rsi and rsi > 75:
            filter_passed = False
            filter_reasons.append("RSI overheated")
        
        if ma_deviation and ma_deviation > 25:
            filter_passed = False
            filter_reasons.append("MA deviation overheated")
        
        return {
            "technical_score": technical_score,
            "rsi": rsi,
            "ma_deviation": ma_deviation,
            "volume_ratio": volume_ratio,
            "volatility": volatility,
            "filter_passed": filter_passed,
            "filter_reasons": filter_reasons,
            "analysis_timestamp": datetime.now()
        }
    
    def calculate_market_environment_score(self, index_data: Dict[str, Any]) -> int:
        """
        Calculate market environment score based on index trends.
        
        Args:
            index_data: Market index data (Nikkei, TOPIX, sector indices)
            
        Returns:
            Market environment score (0-20)
        """
        score = 10  # Base score
        
        # Check Nikkei trend
        nikkei_change = index_data.get("nikkei_change_percent", 0)
        if nikkei_change > 1.0:
            score += 3
        elif nikkei_change < -1.0:
            score -= 3
        
        # Check sector trend
        sector_change = index_data.get("sector_change_percent", 0)
        if sector_change > 1.5:
            score += 5
        elif sector_change > 0:
            score += 2
        elif sector_change < -1.5:
            score -= 5
        
        # Market breadth
        advance_decline_ratio = index_data.get("advance_decline_ratio", 1.0)
        if advance_decline_ratio > 2.0:  # Strong breadth
            score += 2
        elif advance_decline_ratio < 0.5:  # Weak breadth
            score -= 2
        
        return min(max(score, 0), 20)

    def analyze_sector_trend(self, sector_data: pd.DataFrame) -> int:
        """
        Analyze sector trend for market environment score.
        
        Args:
            sector_data: Sector index data
            
        Returns:
            Trend score (0-20)
        """
        if len(sector_data) < 20:
            return 10  # Neutral score
        
        closes = sector_data['close'].values
        recent_avg = np.mean(closes[-5:])
        longer_avg = np.mean(closes[-20:])
        
        if recent_avg > longer_avg * 1.02:  # 2% above
            return 20  # Strong uptrend
        elif recent_avg > longer_avg:
            return 15  # Mild uptrend
        elif recent_avg < longer_avg * 0.98:  # 2% below
            return 0   # Downtrend
        else:
            return 10  # Neutral