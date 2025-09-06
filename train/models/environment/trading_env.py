"""Trading Environment for Reinforcement Learning.

This module provides a Gymnasium-compatible trading environment for training RL agents
on stock trading tasks with portfolio management capabilities.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """Portfolio management class for tracking holdings and cash."""
    
    def __init__(self, initial_cash: float, symbols: List[str]):
        """Initialize portfolio.
        
        Args:
            initial_cash: Starting cash amount
            symbols: List of tradable symbols
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {symbol: 0 for symbol in symbols}
        self.symbols = symbols
        
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value.
        
        Args:
            prices: Current prices for each symbol
            
        Returns:
            Total portfolio value (cash + holdings value)
        """
        holdings_value = sum(
            self.holdings[symbol] * prices.get(symbol, 0) 
            for symbol in self.symbols
        )
        return self.cash + holdings_value
    
    def execute_trade(self, symbol: str, action: float, price: float, commission_rate: float) -> bool:
        """Execute a trade.
        
        Args:
            symbol: Symbol to trade
            action: Action value [-1, 1] where -1=sell all, 0=hold, 1=buy with all cash
            price: Current price
            commission_rate: Commission rate
            
        Returns:
            True if trade executed successfully
        """
        try:
            if action > 0.1:  # Buy
                # Calculate how much to buy based on action strength
                cash_to_use = self.cash * action
                commission = cash_to_use * commission_rate
                net_cash = cash_to_use - commission
                
                if net_cash > 0:
                    shares_to_buy = net_cash / price
                    self.holdings[symbol] += shares_to_buy
                    self.cash -= cash_to_use
                    return True
                    
            elif action < -0.1:  # Sell
                # Sell a portion based on action strength
                shares_to_sell = self.holdings[symbol] * abs(action)
                
                if shares_to_sell > 0:
                    gross_proceeds = shares_to_sell * price
                    commission = gross_proceeds * commission_rate
                    net_proceeds = gross_proceeds - commission
                    
                    self.holdings[symbol] -= shares_to_sell
                    self.cash += net_proceeds
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Trade execution failed for {symbol}: {e}")
            return False
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        for symbol in self.symbols:
            self.holdings[symbol] = 0


class TradingEnvironment(gym.Env):
    """Gymnasium-compatible trading environment."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        symbols: List[str],
        initial_cash: float = 10000000,
        commission_rate: float = 0.001,
        window_size: int = 30,
        news_data: Optional[pd.DataFrame] = None
    ):
        """Initialize trading environment.
        
        Args:
            price_data: Historical price data with MultiIndex (date, symbol)
            symbols: List of tradable symbols
            initial_cash: Starting cash amount
            commission_rate: Transaction commission rate
            window_size: Number of historical days to include in observation
            news_data: Optional news data
        """
        super().__init__()
        
        self.price_data = price_data
        self.symbols = symbols
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.window_size = window_size
        self.news_data = news_data
        
        # Get unique dates
        self.dates = sorted(price_data.index.get_level_values(0).unique())
        self.max_steps = len(self.dates) - window_size
        
        # Initialize portfolio
        self.portfolio = Portfolio(initial_cash, symbols)
        
        # Define action space: continuous actions for each symbol [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(symbols),), dtype=np.float32
        )
        
        # Define observation space
        # Features per symbol per day: OHLCV (5) + technical indicators (5) = 10
        stock_features = len(symbols) * 10 * window_size
        portfolio_features = len(symbols) + 1  # holdings + cash ratio
        news_features = len(symbols) * 10 if news_data is not None else 0
        
        obs_dim = stock_features + portfolio_features + news_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.initial_portfolio_value = initial_cash
        
        logger.info(f"TradingEnvironment initialized with {len(symbols)} symbols, "
                   f"{self.max_steps} max steps, observation dim: {obs_dim}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        self.current_step = self.window_size  # Start after window
        self.portfolio.reset()
        self.initial_portfolio_value = self.initial_cash
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step.
        
        Args:
            action: Action array for each symbol [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current prices
        current_date = self.dates[self.current_step]
        current_prices = self._get_current_prices(current_date)
        
        # Store portfolio value before trades
        portfolio_value_before = self.portfolio.get_total_value(current_prices)
        
        # Execute trades
        for i, symbol in enumerate(self.symbols):
            if symbol in current_prices:
                self.portfolio.execute_trade(
                    symbol, action[i], current_prices[symbol], self.commission_rate
                )
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        portfolio_value_after = self.portfolio.get_total_value(current_prices)
        reward = self._calculate_reward(portfolio_value_before, portfolio_value_after, current_prices)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get new observation
        observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape)
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Observation vector
        """
        obs_list = []
        
        # Get historical window indices
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        # Add stock data for each symbol
        for symbol in self.symbols:
            for i in range(start_idx, end_idx):
                if i < len(self.dates):
                    date = self.dates[i]
                    try:
                        row = self.price_data.loc[(date, symbol)]
                        # Normalize price data
                        obs_list.extend([
                            row['open'] / 10000,
                            row['high'] / 10000,
                            row['low'] / 10000,
                            row['close'] / 10000,
                            row['volume'] / 1e6
                        ])
                        # Add technical indicators
                        obs_list.extend(self._calculate_technical_indicators(symbol, date))
                    except KeyError:
                        obs_list.extend([0] * 10)
                else:
                    obs_list.extend([0] * 10)
        
        # Add portfolio state
        current_prices = self._get_current_prices(self.dates[min(self.current_step, len(self.dates)-1)])
        total_value = self.portfolio.get_total_value(current_prices)
        
        # Cash ratio
        obs_list.append(self.portfolio.cash / total_value if total_value > 0 else 1.0)
        
        # Position ratios
        for symbol in self.symbols:
            position_value = self.portfolio.holdings[symbol] * current_prices.get(symbol, 0)
            obs_list.append(position_value / total_value if total_value > 0 else 0.0)
        
        # Add news features if available
        if self.news_data is not None:
            obs_list.extend(self._get_news_features())
        
        return np.array(obs_list, dtype=np.float32)
    
    def _get_current_prices(self, date: pd.Timestamp) -> Dict[str, float]:
        """Get current prices for all symbols.
        
        Args:
            date: Current date
            
        Returns:
            Dictionary of symbol -> price
        """
        prices = {}
        for symbol in self.symbols:
            try:
                price = self.price_data.loc[(date, symbol), 'close']
                prices[symbol] = price
            except KeyError:
                # Use last available price or 0
                prices[symbol] = 0.0
        
        return prices
    
    def _calculate_technical_indicators(self, symbol: str, date: pd.Timestamp) -> List[float]:
        """Calculate technical indicators for a symbol.
        
        Args:
            symbol: Stock symbol
            date: Current date
            
        Returns:
            List of technical indicator values
        """
        # Simple technical indicators (placeholder implementation)
        # In a real implementation, you would calculate RSI, MACD, etc.
        try:
            row = self.price_data.loc[(date, symbol)]
            price = row['close']
            volume = row['volume']
            
            # Simple indicators (normalized)
            return [
                0.5,  # RSI placeholder (normalized to 0-1)
                0.0,  # MACD placeholder
                0.0,  # Moving average deviation placeholder
                min(volume / 1e6, 1.0),  # Volume ratio (capped at 1)
                0.5   # ATR placeholder
            ]
        except:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _get_news_features(self) -> List[float]:
        """Get news features for current date.
        
        Returns:
            List of news-based features
        """
        # Placeholder for news features
        # In a real implementation, this would process news data
        return [0.0] * (len(self.symbols) * 10)
    
    def _calculate_reward(self, value_before: float, value_after: float, current_prices: Dict[str, float]) -> float:
        """Calculate reward for the current step.
        
        Args:
            value_before: Portfolio value before trades
            value_after: Portfolio value after trades
            current_prices: Current market prices
            
        Returns:
            Reward value
        """
        # Basic reward: percentage change in portfolio value
        if value_before > 0:
            return_rate = (value_after - value_before) / value_before
        else:
            return_rate = 0.0
        
        # Scale reward
        reward = return_rate * 100
        
        # Add penalty for excessive trading (optional)
        # This could be enhanced based on specific trading strategy requirements
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary.
        
        Returns:
            Info dictionary with current state information
        """
        if self.current_step < len(self.dates):
            current_date = self.dates[self.current_step]
            current_prices = self._get_current_prices(current_date)
            total_value = self.portfolio.get_total_value(current_prices)
        else:
            current_date = None
            total_value = self.portfolio.cash
        
        return {
            'current_step': self.current_step,
            'current_date': current_date,
            'portfolio_value': total_value,
            'cash': self.portfolio.cash,
            'holdings': self.portfolio.holdings.copy(),
            'return_rate': (total_value - self.initial_portfolio_value) / self.initial_portfolio_value if self.initial_portfolio_value > 0 else 0.0
        }
    
    def render(self, mode: str = 'human') -> Optional[Any]:
        """Render environment state.
        
        Args:
            mode: Render mode
            
        Returns:
            Rendered output (if any)
        """
        if mode == 'human':
            info = self._get_info()
            print(f"Step: {info['current_step']}, "
                  f"Portfolio Value: {info['portfolio_value']:.2f}, "
                  f"Return: {info['return_rate']:.2%}")
        
        return None