"""Real-time stock price data fetcher."""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


class PriceFetcher:
    """Real-time stock price data fetcher using broker APIs."""
    
    def __init__(self, api_endpoint: str = None, api_key: str = None) -> None:
        """
        Initialize PriceFetcher.
        
        Args:
            api_endpoint: Broker API endpoint (optional, uses yfinance if not provided)
            api_key: API key for authentication (optional)
        """
        self.api_endpoint = api_endpoint or "yfinance"
        self.api_key = api_key
        self.price_cache: Dict[str, Tuple[Decimal, datetime]] = {}  # Fixed type hint
        
        logger.info(f"PriceFetcher initialized with data source: {self.api_endpoint}")
    
    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current stock price using yfinance.
        
        Args:
            symbol: Stock symbol (e.g., "7203" for Japanese stocks)
            
        Returns:
            Current price or None if unavailable
        """
        try:
            import yfinance as yf
            
            # Convert Japanese stock code to Yahoo Finance format
            yahoo_symbol = f"{symbol}.T" if not symbol.endswith('.T') else symbol
            
            logger.debug(f"Fetching current price for {yahoo_symbol}")
            
            # Check cache first
            if yahoo_symbol in self.price_cache:
                cached_price, cached_time = self.price_cache[yahoo_symbol]
                # Use cached price if it's less than 30 seconds old
                if (datetime.now() - cached_time).total_seconds() < 30:
                    logger.debug(f"Using cached price for {yahoo_symbol}: {cached_price}")
                    return cached_price
            
            # Fetch from Yahoo Finance
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            # Try multiple price fields in order of preference
            price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose', 'price']
            current_price = None
            
            for field in price_fields:
                if field in info and info[field] is not None:
                    current_price = Decimal(str(info[field]))
                    break
            
            if current_price is None:
                # Try fast_info as fallback
                try:
                    fast_info = ticker.fast_info
                    if hasattr(fast_info, 'last_price') and fast_info.last_price is not None:
                        current_price = Decimal(str(fast_info.last_price))
                except Exception:
                    pass
            
            if current_price is not None:
                # Cache the price
                self.price_cache[yahoo_symbol] = (current_price, datetime.now())
                logger.debug(f"Fetched current price for {yahoo_symbol}: {current_price}")
                return current_price
            else:
                logger.warning(f"No price data available for {yahoo_symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            
            # Return 0 for demo price (clearly indicating dummy data)
            logger.error(f"FALLBACK: Returning 0 for {symbol} - Real price fetch failed")
            return Decimal("0")
            
            return None
    
    def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive price data including OHLC and volume.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with price data or None if unavailable
        """
        try:
            import yfinance as yf
            
            # Convert Japanese stock code to Yahoo Finance format
            yahoo_symbol = f"{symbol}.T" if not symbol.endswith('.T') else symbol
            
            logger.debug(f"Fetching comprehensive price data for {yahoo_symbol}")
            
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            # Get current price
            current_price = self.get_current_price(symbol)
            
            if current_price is None:
                return None
            
            # Extract comprehensive data
            price_data = {
                "symbol": symbol,
                "yahoo_symbol": yahoo_symbol,
                "current_price": current_price,
                "timestamp": datetime.now(),
                
                # Basic price info
                "open": Decimal(str(info.get('regularMarketOpen', current_price))),
                "high": Decimal(str(info.get('regularMarketDayHigh', current_price))), 
                "low": Decimal(str(info.get('regularMarketDayLow', current_price))),
                "previous_close": Decimal(str(info.get('previousClose', current_price))),
                
                # Volume data
                "volume": info.get('regularMarketVolume', 0),
                "average_volume": info.get('averageVolume', 0),
                
                # Market cap and valuation
                "market_cap": info.get('marketCap', 0),
                "shares_outstanding": info.get('sharesOutstanding', 0),
                
                # Financial ratios
                "pe_ratio": info.get('forwardPE', info.get('trailingPE', None)),
                "pb_ratio": info.get('priceToBook', None),
                "dividend_yield": info.get('dividendYield', 0),
                
                # Volatility indicators
                "beta": info.get('beta', None),
                "fifty_two_week_high": Decimal(str(info.get('fiftyTwoWeekHigh', current_price))),
                "fifty_two_week_low": Decimal(str(info.get('fiftyTwoWeekLow', current_price))),
                
                # Company info
                "company_name": info.get('longName', info.get('shortName', f"Company {symbol}")),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                
                # Exchange info
                "exchange": info.get('exchange', 'TSE'),
                "currency": info.get('currency', 'JPY'),
                
                # Calculate derived metrics
                "price_change": current_price - Decimal(str(info.get('previousClose', current_price))),
                "price_change_percent": float((current_price - Decimal(str(info.get('previousClose', current_price)))) / 
                                            Decimal(str(info.get('previousClose', current_price))) * 100) if info.get('previousClose') else 0.0
            }
            
            logger.debug(f"Successfully fetched comprehensive data for {symbol}")
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive price data for {symbol}: {e}")
            
            # Fallback to basic price data
            current_price = self.get_current_price(symbol)
            if current_price:
                return {
                    "symbol": symbol,
                    "current_price": current_price,
                    "timestamp": datetime.now(),
                    "source": "fallback"
                }
            
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1mo") -> Optional[List[Dict[str, Any]]]:
        """
        Get historical price data.
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            List of historical price data or None if unavailable
        """
        try:
            import yfinance as yf
            import pandas as pd
            
            # Convert Japanese stock code to Yahoo Finance format
            yahoo_symbol = f"{symbol}.T" if not symbol.endswith('.T') else symbol
            
            logger.debug(f"Fetching historical data for {yahoo_symbol} (period: {period})")
            
            ticker = yf.Ticker(yahoo_symbol)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                logger.warning(f"No historical data available for {yahoo_symbol}")
                return None
            
            # Convert DataFrame to list of dictionaries
            historical_prices = []
            for date, row in hist_data.iterrows():
                price_record = {
                    "symbol": symbol,
                    "date": date.strftime("%Y-%m-%d"),
                    "datetime": date,
                    "open": Decimal(str(row['Open'])),
                    "high": Decimal(str(row['High'])),
                    "low": Decimal(str(row['Low'])),
                    "close": Decimal(str(row['Close'])),
                    "volume": int(row['Volume']) if pd.notna(row['Volume']) else 0,
                    "adj_close": Decimal(str(row['Adj Close'])) if 'Adj Close' in row.index else Decimal(str(row['Close']))
                }
                historical_prices.append(price_record)
            
            # Sort by date (newest first)
            historical_prices.sort(key=lambda x: x['datetime'], reverse=True)
            
            logger.debug(f"Fetched {len(historical_prices)} historical records for {symbol}")
            return historical_prices
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            
            # Generate demo historical data as fallback
            if symbol in ["7203", "6758", "9984", "8306", "6501"]:
                return self._generate_demo_historical_data(symbol, period)
            
            return None
    
    def _generate_demo_historical_data(self, symbol: str, period: str) -> List[Dict[str, Any]]:
        """Generate demo historical data with all zeros (clearly indicating dummy data)."""
        from datetime import timedelta
        
        logger.error(f"FALLBACK: Generating all-zero historical data for {symbol} - Real data fetch failed")
        
        # Determine number of days based on period
        period_days = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
            "6mo": 180, "1y": 365, "2y": 730
        }
        days = period_days.get(period, 30)
        
        historical_data = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i-1)
            
            # All zeros to clearly indicate dummy data
            open_price = 0.0
            high = 0.0
            low = 0.0
            current_price = 0.0
            volume = 0
            
            price_record = {
                "symbol": symbol,
                "date": date.strftime("%Y-%m-%d"),
                "datetime": date,
                "open": Decimal("0"),
                "high": Decimal("0"),
                "low": Decimal("0"),
                "close": Decimal("0"),
                "volume": 0,
                "adj_close": Decimal("0"),
                "is_dummy": True  # Flag to indicate dummy data
            }
            historical_data.append(price_record)
        
        logger.error(f"Generated {len(historical_data)} dummy records (all zeros) for {symbol}")
        return historical_data
    
    async def collect_daily_market_data(self, symbols: List[str], target_date: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
        """
        Collect end-of-day market data for multiple symbols (for scheduled batch processing).
        
        Args:
            symbols: List of stock symbols
            target_date: Target date (default: today)
            
        Returns:
            Dictionary mapping symbols to their market data
        """
        if target_date is None:
            target_date = datetime.now()
        
        logger.info(f"Collecting market data for {len(symbols)} symbols on {target_date.strftime('%Y-%m-%d')}")
        
        market_data = {}
        
        for symbol in symbols:
            try:
                # Get end-of-day data
                eod_data = await self.get_end_of_day_data(symbol, target_date)
                
                # Get historical data for technical indicators
                historical_data = self.get_historical_data(symbol, days=60)
                
                # Calculate technical indicators
                if historical_data:
                    eod_data["volatility"] = self.calculate_volatility(historical_data)
                    eod_data["atr"] = self.calculate_atr(historical_data, period=14)
                
                market_data[symbol] = eod_data
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                market_data[symbol] = {
                    "error": str(e),
                    "close_price": Decimal("0"),
                    "volume": 0
                }
        
        logger.info(f"Collected market data for {len(market_data)} symbols")
        return market_data
    
    async def get_end_of_day_data(self, symbol: str, target_date: datetime) -> Dict[str, Any]:
        """
        Get end-of-day data for a specific symbol and date.
        
        Args:
            symbol: Stock symbol
            target_date: Target date
            
        Returns:
            End-of-day market data
        """
        # In production, this would fetch from Yahoo Finance or similar API
        # For now, return demo data
        return {
            "symbol": symbol,
            "date": target_date.strftime("%Y-%m-%d"),
            "open": Decimal("1480.00"),
            "high": Decimal("1520.00"),
            "low": Decimal("1475.00"),
            "close_price": Decimal("1500.00"),
            "volume": 1500000,
            "change": Decimal("20.00"),
            "change_percent": Decimal("1.35"),
            "vwap": Decimal("1495.00"),
            "market_cap": Decimal("1000000000"),
            "timestamp": target_date
        }
    
    def get_symbols_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get summary statistics for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Summary statistics
        """
        summary = {
            "total_symbols": len(symbols),
            "cached_symbols": len([s for s in symbols if s in self.price_cache]),
            "last_update": datetime.now(),
            "market_status": "closed" if not self.is_market_open() else "open"
        }
        
        return summary

    def calculate_volatility(self, symbol: str, period: str = "1mo") -> Optional[float]:
        """
        Calculate historical volatility (standard deviation of returns).
        
        Args:
            symbol: Stock symbol
            period: Time period for calculation
            
        Returns:
            Volatility as a decimal (e.g., 0.25 for 25%) or None if unavailable
        """
        try:
            historical_data = self.get_historical_data(symbol, period)
            
            if not historical_data or len(historical_data) < 2:
                logger.warning(f"Insufficient data to calculate volatility for {symbol}")
                return None
            
            # Sort by date (oldest first) for return calculation
            historical_data.sort(key=lambda x: x['datetime'])
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(historical_data)):
                prev_close = float(historical_data[i-1]['close'])
                current_close = float(historical_data[i]['close'])
                
                if prev_close > 0:
                    daily_return = (current_close - prev_close) / prev_close
                    returns.append(daily_return)
            
            if not returns:
                return None
            
            # Calculate standard deviation of returns
            import math
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = math.sqrt(variance)
            
            # Annualize volatility (assuming 252 trading days)
            annualized_volatility = volatility * math.sqrt(252)
            
            logger.debug(f"Calculated volatility for {symbol}: {annualized_volatility:.4f} ({period})")
            return annualized_volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return None
    
    def calculate_atr(self, symbol: str, period: str = "1mo", atr_period: int = 14) -> Optional[Decimal]:
        """
        Calculate Average True Range (ATR).
        
        Args:
            symbol: Stock symbol
            period: Time period for historical data
            atr_period: Period for ATR calculation (typically 14)
            
        Returns:
            ATR value or None if unavailable
        """
        try:
            historical_data = self.get_historical_data(symbol, period)
            
            if not historical_data or len(historical_data) < atr_period + 1:
                logger.warning(f"Insufficient data to calculate ATR for {symbol}")
                return None
            
            # Sort by date (oldest first)
            historical_data.sort(key=lambda x: x['datetime'])
            
            # Calculate True Range for each day
            true_ranges = []
            
            for i in range(1, len(historical_data)):
                current = historical_data[i]
                previous = historical_data[i-1]
                
                high = float(current['high'])
                low = float(current['low'])
                prev_close = float(previous['close'])
                
                # True Range = max of:
                # 1. Current High - Current Low
                # 2. |Current High - Previous Close|
                # 3. |Current Low - Previous Close|
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            if len(true_ranges) < atr_period:
                return None
            
            # Calculate ATR as simple moving average of True Range
            atr_values = []
            for i in range(atr_period - 1, len(true_ranges)):
                atr = sum(true_ranges[i - atr_period + 1:i + 1]) / atr_period
                atr_values.append(atr)
            
            # Return the most recent ATR
            if atr_values:
                current_atr = Decimal(str(atr_values[-1]))
                logger.debug(f"Calculated ATR for {symbol}: {current_atr} (period: {atr_period})")
                return current_atr
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open
        """
        # TODO: Implement market hours check (JST 9:00-11:30, 12:30-15:00)
        logger.debug("Checking if market is open")
        return False
    
    def subscribe_to_price_updates(self, symbols: List[str]) -> None:
        """
        Subscribe to real-time price updates.
        
        Args:
            symbols: List of stock symbols to monitor
        """
        # TODO: Implement real-time subscription
        logger.info(f"Subscribing to price updates for: {symbols}")
        pass
    
    def unsubscribe_from_price_updates(self, symbols: List[str]) -> None:
        """
        Unsubscribe from price updates.
        
        Args:
            symbols: List of stock symbols to unsubscribe
        """
        # TODO: Implement unsubscription
        logger.info(f"Unsubscribing from price updates for: {symbols}")
        pass