"""Train reinforcement learning agent for stock trading with Nikkei 225 as market indicator."""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from typing import List, Dict, Optional, Tuple
from gymnasium import spaces

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# 簡略化した訓練用のインポート
from models.trading_model import TradingDecisionModel, MarketData
from models.environment.trading_env import TradingEnvironment
from models.agents.ppo_agent import PPOTradingAgent, TradingCallback
try:
    from src.data_collector.yahoo_finance_client import YahooFinanceClient
except ImportError:
    YahooFinanceClient = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Nikkei225TradingPipeline:
    """Training pipeline with Nikkei 225 as market indicator."""
    
    def __init__(
        self,
        target_symbols: List[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 10000000,  # 10 million yen
        commission_rate: float = 0.001,
        window_size: int = 30,  # 1 month of daily data
        model_save_dir: Path = None
    ):
        """Initialize training pipeline.
        
        Args:
            target_symbols: List of target stock symbols (Japanese format: XXXX.T)
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            initial_cash: Initial cash amount
            commission_rate: Transaction cost rate
            window_size: Historical window size (30 days = 1 month)
            model_save_dir: Directory to save models
        """
        self.target_symbols = target_symbols
        self.nikkei_symbol = '^N225'  # Nikkei 225 index
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.window_size = window_size
        
        self.model_save_dir = model_save_dir or Path(__file__).parent / 'models' / 'rl'
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # YahooFinanceClient is optional, not needed for yfinance usage
        if YahooFinanceClient is not None:
            self.yahoo_client = YahooFinanceClient()
        else:
            self.yahoo_client = None
        
        logger.info(f"Pipeline initialized for {len(target_symbols)} target symbols with Nikkei 225 indicator")
    
    def fetch_nikkei_data(self) -> pd.DataFrame:
        """Fetch Nikkei 225 index data.
        
        Returns:
            DataFrame with Nikkei 225 OHLC data
        """
        logger.info(f"Fetching Nikkei 225 data from {self.start_date} to {self.end_date}")
        
        try:
            ticker = yf.Ticker(self.nikkei_symbol)
            hist = ticker.history(start=self.start_date, end=self.end_date)
            
            if hist.empty:
                raise ValueError("No Nikkei 225 data found")
            
            # Prepare data with high, low, close only
            nikkei_data = pd.DataFrame({
                'date': hist.index,
                'high': hist['High'].values,
                'low': hist['Low'].values,
                'close': hist['Close'].values
            })
            
            logger.info(f"Fetched {len(nikkei_data)} records for Nikkei 225")
            return nikkei_data
            
        except Exception as e:
            logger.error(f"Error fetching Nikkei 225 data: {e}")
            raise
    
    def fetch_target_stocks_data(self) -> pd.DataFrame:
        """Fetch target stocks price data.
        
        Returns:
            DataFrame with MultiIndex (date, symbol) and OHLCV columns
        """
        logger.info(f"Fetching target stocks data for {len(self.target_symbols)} symbols")
        
        all_data = []
        failed_symbols = []
        
        # Process in batches to avoid overwhelming the API
        batch_size = 50
        for i in range(0, len(self.target_symbols), batch_size):
            batch = self.target_symbols[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(self.target_symbols) + batch_size - 1)//batch_size}")
            
            for symbol in batch:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=self.start_date, end=self.end_date)
                    
                    if hist.empty:
                        logger.debug(f"No data found for {symbol}")
                        failed_symbols.append(symbol)
                        continue
                    
                    # Prepare data
                    hist['symbol'] = symbol
                    hist.reset_index(inplace=True)
                    hist.columns = [col.lower() for col in hist.columns]
                    
                    all_data.append(hist)
                    
                except Exception as e:
                    logger.debug(f"Error fetching data for {symbol}: {e}")
                    failed_symbols.append(symbol)
        
        logger.info(f"Successfully fetched data for {len(all_data)} symbols")
        if failed_symbols:
            logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols")
        
        if not all_data:
            raise ValueError("No data fetched for any target symbol")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.set_index(['date', 'symbol'], inplace=True)
        combined_df.sort_index(inplace=True)
        
        return combined_df
    
    def fetch_ir_news_mock(self) -> pd.DataFrame:
        """Fetch IR news data (mock implementation).
        
        Returns:
            DataFrame with news data
        """
        logger.info("Generating mock IR news data")
        
        # Generate mock news for demonstration
        news_data = []
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='W')
        
        for symbol in self.target_symbols:
            for date in dates[::3]:  # News every 3 weeks
                news_data.append({
                    'date': date,
                    'symbol': symbol,
                    'title': f'{symbol} 決算発表',
                    'content': f'{symbol}の四半期決算が発表されました。売上高は前年同期比で増加。',
                    'category': 'earnings'
                })
        
        if news_data:
            news_df = pd.DataFrame(news_data)
            news_df['date'] = pd.to_datetime(news_df['date'])
            news_df.set_index('date', inplace=True)
            logger.info(f"Generated {len(news_df)} mock news items")
            return news_df
        else:
            return pd.DataFrame()
    
    def create_enhanced_environment(
        self,
        nikkei_data: pd.DataFrame,
        target_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame] = None
    ) -> 'EnhancedTradingEnvironment':
        """Create enhanced trading environment with Nikkei 225 indicator.
        
        Args:
            nikkei_data: Nikkei 225 index data
            target_data: Target stocks price data
            news_data: Optional IR news data
            
        Returns:
            Enhanced trading environment
        """
        logger.info("Creating enhanced trading environment")
        
        # Create custom environment that includes Nikkei 225 as market indicator
        env = EnhancedTradingEnvironment(
            nikkei_data=nikkei_data,
            price_data=target_data,
            symbols=self.target_symbols,
            initial_cash=self.initial_cash,
            commission_rate=self.commission_rate,
            window_size=self.window_size,
            news_data=news_data
        )
        
        logger.info(f"Enhanced environment created with {env.max_steps} max steps")
        
        return env
    
    def train(
        self,
        total_timesteps: int = 50000,
        learning_rate: float = 3e-4,
        n_steps: int = 1024,
        batch_size: int = 32,
        n_epochs: int = 10,
        device: str = None
    ) -> PPOTradingAgent:
        """Train the RL agent.
        
        Args:
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            n_steps: Steps per update
            batch_size: Batch size
            n_epochs: Epochs per update
            device: Device to use
            
        Returns:
            Trained agent
        """
        logger.info("Starting training pipeline")
        
        # Set device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
            elif torch.cuda.is_available():
                device = "cuda:0"  # NVIDIA GPU
            else:
                device = "cpu"  # CPU fallback
            logger.info(f"Auto-detected device: {device}")
        
        # Fetch all data
        nikkei_data = self.fetch_nikkei_data()
        target_data = self.fetch_target_stocks_data()
        news_data = self.fetch_ir_news_mock()
        
        # Adjust window size if not enough data
        available_days = len(nikkei_data)
        adjusted_window_size = min(self.window_size, max(5, available_days - 10))  # At least 5 days, leaving more buffer
        
        if adjusted_window_size != self.window_size:
            logger.warning(f"Adjusted window_size from {self.window_size} to {adjusted_window_size} due to limited data ({available_days} days)")
            self.window_size = adjusted_window_size
        
        # Create enhanced environment
        env = self.create_enhanced_environment(nikkei_data, target_data, news_data)
        
        # Create agent
        agent = PPOTradingAgent(
            env=env,
            num_stocks=len(self.target_symbols),
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            device=device,
            tensorboard_log=None  # Disable tensorboard for now
        )
        
        # Create callback
        callback = TradingCallback(verbose=1)
        
        # Train agent
        agent.train(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save model
        model_path = self.model_save_dir / f"ppo_nikkei_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        agent.save(str(model_path))
        
        # Evaluate agent
        metrics = agent.evaluate(n_episodes=10)
        
        # Save metrics
        metrics_path = self.model_save_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Training completed. Model saved to {model_path}")
        logger.info(f"Evaluation metrics: {metrics}")
        
        return agent


class EnhancedTradingEnvironment(TradingEnvironment):
    """Enhanced trading environment with Nikkei 225 as market indicator."""
    
    def __init__(
        self,
        nikkei_data: pd.DataFrame,
        price_data: pd.DataFrame,
        symbols: List[str],
        initial_cash: float = 10000000,
        commission_rate: float = 0.001,
        window_size: int = 30,
        news_data: Optional[pd.DataFrame] = None
    ):
        """Initialize enhanced environment.
        
        Args:
            nikkei_data: Nikkei 225 index data
            price_data: Target stocks price data
            symbols: List of stock symbols
            initial_cash: Starting cash amount
            commission_rate: Transaction cost rate
            window_size: Number of historical days (30 = 1 month)
            news_data: Optional IR news data
        """
        super().__init__(
            price_data=price_data,
            symbols=symbols,
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            window_size=window_size,
            news_data=news_data
        )
        
        self.nikkei_data = nikkei_data
        
        # Adjust observation space for ALL 605 stocks
        # Nikkei: 3 features (high, low, close) * window_size
        # All stocks: 605 symbols * 2 features (price, volume) - current day only
        # Portfolio: 605 stocks + 1 (cash)
        # News: if available
        nikkei_features = 3 * window_size
        stock_features = len(symbols) * 2  # All 605 stocks, 2 features each, current day only
        portfolio_features = len(symbols) + 1
        news_features = len(symbols) * 10 if news_data is not None else 0
        
        obs_dim = nikkei_features + stock_features + portfolio_features + news_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        logger.info(f"Enhanced environment with Nikkei 225 indicator initialized")
    
    
    def _get_observation(self) -> np.ndarray:
        """Get observation including Nikkei 225 data.
        
        Returns:
            Observation vector with market indicator
        """
        obs_list = []
        
        # Get historical window indices
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        # 1. Add Nikkei 225 data (market indicator)
        for i in range(start_idx, end_idx):
            if i < len(self.nikkei_data):
                row = self.nikkei_data.iloc[i]
                # Normalize and validate Nikkei data
                high = row['high'] if not np.isnan(row['high']) else 30000
                low = row['low'] if not np.isnan(row['low']) else 30000
                close = row['close'] if not np.isnan(row['close']) else 30000
                
                obs_list.extend([
                    np.clip(high / 30000, 0, 5),  # Normalize with clipping
                    np.clip(low / 30000, 0, 5),
                    np.clip(close / 30000, 0, 5)
                ])
            else:
                obs_list.extend([1.0, 1.0, 1.0])  # Use neutral values instead of 0
        
        # 2. Add ALL 605 stocks data with compressed representation
        for symbol in self.symbols:
            # Use only latest price and volume for each stock to manage dimension
            if self.current_step < len(self.dates):
                date = self.dates[self.current_step]
                try:
                    row = self.price_data.loc[(date, symbol)]
                    # Use only essential features: price and volume
                    close_price = row['close'] if not np.isnan(row['close']) else 1000
                    volume = row['volume'] if not np.isnan(row['volume']) else 1000
                    
                    obs_list.extend([
                        np.clip(close_price / 1000, 0, 50),  # Normalized price
                        np.clip(volume / 1e6, 0, 100)        # Normalized volume
                    ])
                except KeyError:
                    obs_list.extend([1.0, 0.1])  # Default values
            else:
                obs_list.extend([1.0, 0.1])  # Default values
        
        # 3. Add portfolio state
        current_prices = self._get_current_prices(self.dates[min(self.current_step, len(self.dates)-1)])
        total_value = self.portfolio.get_total_value(current_prices)
        
        # Cash ratio
        obs_list.append(self.portfolio.cash / total_value if total_value > 0 else 1.0)
        
        # Position ratios
        for symbol in self.symbols:
            position_value = self.portfolio.holdings[symbol] * current_prices.get(symbol, 0)
            obs_list.append(position_value / total_value if total_value > 0 else 0.0)
        
        # 4. Add news features if available
        if self.news_data is not None:
            obs_list.extend(self._get_news_features())
        
        return np.array(obs_list, dtype=np.float32)


def load_growth_stocks_symbols(max_symbols: int = 605) -> List[str]:
    """Load Tokyo Stock Exchange Growth Market symbols from CSV.
    
    Args:
        max_symbols: Maximum number of symbols to load for manageable training
    
    Returns:
        List of stock symbols in Yahoo Finance format (XXXX.T)
    """
    import csv
    csv_path = Path(__file__).parent.parent / 'quick_test' / 'growth_stocks_complete_20250909_191236.csv'
    
    symbols = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row['企業ID'] and i < max_symbols:  # Limit number of symbols
                symbols.append(f"{row['企業ID']}.T")
    
    logger.info(f"Loaded {len(symbols)} growth stock symbols (limited from 605 for training)")
    return symbols


def main():
    """Main training function."""
    
    # Load all Tokyo Growth Market stocks (605 companies)
    target_symbols = load_growth_stocks_symbols()
    
    # Training period - 1 year of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Create training pipeline
    pipeline = Nikkei225TradingPipeline(
        target_symbols=target_symbols,
        start_date=start_date,
        end_date=end_date,
        initial_cash=10000000,  # 10 million yen
        commission_rate=0.001,
        window_size=30  # 1 month of daily data
    )
    
    # Check for available device
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = "cuda:0"  # NVIDIA GPU
    else:
        device = "cpu"  # CPU fallback
    logger.info(f"Using device: {device}")
    
    # Train agent with parameters optimized for 605 companies
    trained_agent = pipeline.train(
        total_timesteps=80000,  # Increased for larger dataset
        learning_rate=1e-4,  # Lower LR for stability with many stocks
        n_steps=1024,  # Moderate steps
        batch_size=64,  # Larger batch for better gradient estimates
        n_epochs=4,  # Conservative epochs to prevent overfitting
        device=device
    )
    
    logger.info("Training completed successfully")
    
    # Display input specification
    print("\n" + "="*60)
    print("推論時の入力仕様:")
    print("="*60)
    print("1. 日経225指数データ (直近30日):")
    print("   - 高値 (High)")
    print("   - 安値 (Low)")
    print("   - 終値 (Close)")
    print("\n2. ターゲット企業データ (直近30日):")
    print("   - 高値 (High)")
    print("   - 安値 (Low)")  
    print("   - 終値 (Close)")
    print("   - 出来高 (Volume)")
    print("   - テクニカル指標")
    print("\n3. IR情報:")
    print("   - ターゲット企業の直近1ヶ月のIRニュース")
    print("   - ModernBERT-jaによるセンチメント分析")
    print("\n4. 出力:")
    print("   - 各銘柄の売買アクション [-1, 1]")
    print("   - -1: 全売却, 0: ホールド, +1: 全力買い")
    print("="*60)


if __name__ == "__main__":
    main()