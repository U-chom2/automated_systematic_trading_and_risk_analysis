"""Generate training data for risk model neural network - Fixed version.

This script generates training data by analyzing historical stock data
to determine optimal stop-loss percentages based on market conditions.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collector.yahoo_finance_client import YahooFinanceClient
from src.analysis_engine.technical_analyzer import TechnicalAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generate training data for risk model."""
    
    def __init__(self) -> None:
        """Initialize data generator."""
        self.yahoo_client = YahooFinanceClient()
        self.technical_analyzer = TechnicalAnalyzer()
        self.training_data = []
        
    def calculate_historical_volatility(self, prices: pd.Series, window: int = 20) -> float:
        """Calculate historical volatility.
        
        Args:
            prices: Price series
            window: Rolling window for volatility calculation
            
        Returns:
            Historical volatility (annualized)
        """
        try:
            # Calculate daily returns
            returns = prices.pct_change().dropna()
            
            # Calculate rolling standard deviation
            vol = returns.rolling(window=window).std()
            
            # Annualize (assuming 252 trading days)
            annual_vol = vol * np.sqrt(252)
            
            return float(annual_vol.iloc[-1]) if not vol.empty and not pd.isna(vol.iloc[-1]) else 0.05
        except Exception as e:
            logger.debug(f"Error calculating volatility: {e}")
            return 0.05
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR value
        """
        try:
            tr_list = []
            for i in range(1, len(close)):
                hl = high.iloc[i] - low.iloc[i]
                hc = abs(high.iloc[i] - close.iloc[i-1])
                lc = abs(low.iloc[i] - close.iloc[i-1])
                tr = max(hl, hc, lc)
                tr_list.append(tr)
            
            if tr_list:
                return float(np.mean(tr_list[-period:]))
            return 0.02
        except Exception as e:
            logger.debug(f"Error calculating ATR: {e}")
            return 0.02
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI manually.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI value (0-100)
        """
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            if avg_loss.iloc[-1] == 0:
                return 100.0
            
            rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
            rsi = 100 - (100 / (1 + rs))
            return float(rsi) if not pd.isna(rsi) else 50.0
        except Exception as e:
            logger.debug(f"Error calculating RSI: {e}")
            return 50.0
    
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta (market sensitivity).
        
        Args:
            stock_returns: Stock returns
            market_returns: Market index returns
            
        Returns:
            Beta value
        """
        try:
            if len(stock_returns) < 20 or len(market_returns) < 20:
                return 1.0
            
            # Align series
            aligned_stock = stock_returns.iloc[-20:]
            aligned_market = market_returns.iloc[-20:]
            
            # Calculate covariance and variance
            covariance = aligned_stock.cov(aligned_market)
            market_variance = aligned_market.var()
            
            if market_variance == 0:
                return 1.0
            
            return float(covariance / market_variance)
        except Exception as e:
            logger.debug(f"Error calculating beta: {e}")
            return 1.0
    
    def calculate_optimal_stop_loss(self, price_data: pd.DataFrame, 
                                   lookforward_days: int = 10) -> float:
        """Calculate optimal stop-loss based on future price movements.
        
        This is the target value for training. In real trading, we predict this.
        
        Args:
            price_data: Historical price data
            lookforward_days: Days to look forward for optimal stop
            
        Returns:
            Optimal stop-loss percentage
        """
        try:
            if len(price_data) < lookforward_days:
                return 0.05  # Default 5%
            
            # Get current price and future prices
            current_price = float(price_data['Close'].iloc[0])
            future_prices = price_data['Close'].iloc[1:lookforward_days+1]
            
            # Calculate maximum drawdown in the period
            min_price = float(future_prices.min())
            max_drawdown = (current_price - min_price) / current_price
            
            # Optimal stop-loss should be slightly larger than typical drawdown
            # but not too large to avoid unnecessary losses
            if max_drawdown < 0.02:  # Very stable
                optimal_stop = 0.02
            elif max_drawdown < 0.05:  # Moderate volatility
                optimal_stop = max_drawdown * 1.2
            elif max_drawdown < 0.10:  # High volatility
                optimal_stop = max_drawdown * 1.1
            else:  # Extreme volatility
                optimal_stop = min(0.15, max_drawdown)  # Cap at 15%
            
            return float(optimal_stop)
        except Exception as e:
            logger.debug(f"Error calculating optimal stop loss: {e}")
            return 0.05
    
    def generate_sample(self, symbol: str, date: datetime) -> Optional[Dict]:
        """Generate a single training sample.
        
        Args:
            symbol: Stock symbol
            date: Date for sample
            
        Returns:
            Training sample dictionary or None if failed
        """
        try:
            # Get historical data (60 days before the date)
            start_date = date - timedelta(days=90)
            end_date = date + timedelta(days=20)  # Need future data for target
            
            # Download stock data
            ticker = yf.Ticker(symbol)
            stock_data = ticker.history(start=start_date, end=end_date)
            
            if stock_data.empty or len(stock_data) < 30:
                return None
            
            # Find the date index
            stock_data.index = pd.to_datetime(stock_data.index)
            date_ts = pd.Timestamp(date)
            
            # Find closest date in data
            date_idx = None
            for i, idx_date in enumerate(stock_data.index):
                if idx_date.date() >= date_ts.date():
                    date_idx = i
                    break
            
            if date_idx is None or date_idx < 20 or date_idx >= len(stock_data) - 10:
                return None
            
            historical_data = stock_data.iloc[:date_idx+1]
            future_data = stock_data.iloc[date_idx:]
            
            # Calculate features
            hv = self.calculate_historical_volatility(historical_data['Close'])
            
            # Ensure we have the right columns
            if 'High' not in historical_data.columns or 'Low' not in historical_data.columns:
                return None
                
            atr = self.calculate_atr(
                historical_data['High'],
                historical_data['Low'],
                historical_data['Close']
            )
            
            # RSI (manual calculation)
            rsi = self.calculate_rsi(historical_data['Close'], period=14)
            
            # Volume ratio
            if 'Volume' in historical_data.columns and len(historical_data) >= 20:
                avg_volume = historical_data['Volume'].rolling(20).mean().iloc[-1]
                current_volume = historical_data['Volume'].iloc[-1]
                volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            # MA deviation
            ma_20 = historical_data['Close'].rolling(20).mean().iloc[-1]
            current_price = historical_data['Close'].iloc[-1]
            ma_deviation = float((current_price - ma_20) / ma_20) if ma_20 > 0 else 0.0
            
            # Beta (using SPY as market proxy)
            try:
                market_ticker = yf.Ticker('^GSPC')
                market_data = market_ticker.history(start=start_date, end=date)
                if not market_data.empty:
                    stock_returns = historical_data['Close'].pct_change().dropna()
                    market_returns = market_data['Close'].pct_change().dropna()
                    beta = self.calculate_beta(stock_returns, market_returns)
                else:
                    beta = 1.0
            except:
                beta = 1.0
            
            # Calculate target (optimal stop-loss)
            target_stop_loss = self.calculate_optimal_stop_loss(future_data)
            
            # Create sample
            sample = {
                'symbol': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'historical_volatility': float(hv) if not pd.isna(hv) else 0.2,
                'atr': float(atr / current_price) if atr > 0 else 0.02,  # Normalize by price
                'rsi': float(rsi / 100.0),  # Normalize to 0-1
                'volume_ratio': min(float(volume_ratio), 5.0) / 5.0,  # Cap and normalize
                'ma_deviation': max(-0.5, min(0.5, float(ma_deviation))),  # Clip to [-0.5, 0.5]
                'beta': min(float(beta), 3.0) / 3.0,  # Cap at 3 and normalize
                'target_stop_loss': float(target_stop_loss)
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"Failed to generate sample for {symbol} on {date}: {e}")
            return None
    
    def generate_dataset(self, symbols: List[str], 
                        start_date: datetime,
                        end_date: datetime,
                        samples_per_symbol: int = 50) -> List[Dict]:
        """Generate complete training dataset.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for sampling
            end_date: End date for sampling
            samples_per_symbol: Number of samples per symbol
            
        Returns:
            List of training samples
        """
        dataset = []
        
        for symbol in symbols:
            logger.info(f"Generating samples for {symbol}")
            
            # Generate random dates within range
            total_days = (end_date - start_date).days
            
            successful_samples = 0
            attempts = 0
            max_attempts = samples_per_symbol * 3  # Try 3x to get desired samples
            
            while successful_samples < samples_per_symbol and attempts < max_attempts:
                # Random date
                random_days = np.random.randint(0, total_days)
                sample_date = start_date + timedelta(days=random_days)
                
                # Skip weekends
                if sample_date.weekday() >= 5:
                    attempts += 1
                    continue
                
                sample = self.generate_sample(symbol, sample_date)
                if sample:
                    dataset.append(sample)
                    successful_samples += 1
                    logger.debug(f"Generated sample: {sample['symbol']} on {sample['date']}")
                
                attempts += 1
            
            logger.info(f"Generated {successful_samples} samples for {symbol}")
        
        logger.info(f"Total samples generated: {len(dataset)}")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_path: Path) -> None:
        """Save dataset to JSON file.
        
        Args:
            dataset: Training dataset
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Saved dataset to {output_path}")
    
    def load_dataset(self, input_path: Path) -> List[Dict]:
        """Load dataset from JSON file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Training dataset
        """
        with open(input_path, 'r') as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} samples from {input_path}")
        return dataset


def main():
    """Main function to generate training data."""
    generator = TrainingDataGenerator()
    
    # Use popular high-liquidity stocks for training
    training_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'JPM', 'JNJ', 'V', 'PG', 'HD'
    ]
    
    # Date range (use historical data that's guaranteed to exist)
    end_date = datetime(2024, 1, 1)
    start_date = datetime(2022, 1, 1)
    
    logger.info("Starting training data generation")
    logger.info(f"Symbols: {len(training_symbols)}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Generate dataset
    dataset = generator.generate_dataset(
        symbols=training_symbols,
        start_date=start_date,
        end_date=end_date,
        samples_per_symbol=20  # 20 samples per symbol = 200 total samples
    )
    
    if not dataset:
        logger.error("No data generated. Check network connection and date range.")
        return
    
    # Split into train and validation
    np.random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    # Save datasets
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    generator.save_dataset(train_data, output_dir / 'train_data.json')
    generator.save_dataset(val_data, output_dir / 'val_data.json')
    
    # Print statistics
    print("\n" + "="*60)
    print("Training Data Generation Complete")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    if dataset:
        # Show sample statistics
        stop_losses = [s['target_stop_loss'] for s in dataset]
        print(f"\nTarget stop-loss statistics:")
        print(f"  Mean: {np.mean(stop_losses):.3f}")
        print(f"  Std: {np.std(stop_losses):.3f}")
        print(f"  Min: {np.min(stop_losses):.3f}")
        print(f"  Max: {np.max(stop_losses):.3f}")


if __name__ == "__main__":
    main()