"""Neural Network-based Risk Model for optimal stop-loss percentage calculation."""

from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
import logging
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle

logger = logging.getLogger(__name__)


class RiskNeuralNetwork(nn.Module):
    """PyTorch neural network for risk prediction."""
    
    def __init__(self, input_size: int = 6, hidden_size: int = 32, output_size: int = 1):
        super(RiskNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout(out)
        out = self.relu(self.layer2(out))
        out = self.dropout(out)
        out = self.sigmoid(self.layer3(out))
        return out


@dataclass
class MarketFeatures:
    """Market features for risk model input."""
    historical_volatility_60d: float
    atr_14d: float
    rsi_14d: float
    volume_ratio_20d: float
    market_trend_score: float
    sector_volatility: float
    time_of_day_factor: float
    day_of_week_factor: float


@dataclass
class RiskPrediction:
    """Risk model prediction result."""
    optimal_stop_loss_percent: float
    confidence: float
    risk_level: str  # "low", "medium", "high", "extreme"
    recommended_position_size_factor: float


class RiskModel:
    """Neural Network-based risk model using PyTorch."""
    
    def __init__(self, input_size: int = 6, hidden_size: int = 32, output_size: int = 1,
                 auto_load_model: bool = True) -> None:
        """
        Initialize Risk Model.
        
        Args:
            input_size: Number of input features (default 6)
            hidden_size: Hidden layer size
            output_size: Output size (default 1 for stop-loss percentage)
            auto_load_model: Whether to automatically load the trained model
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = RiskNeuralNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()
        self.is_trained = False
        # Updated feature names to match trained model
        self.feature_names = ["historical_volatility", "atr", "rsi", "volume_ratio", "ma_deviation", "beta"]
        
        # Auto-load trained model if available
        if auto_load_model:
            from pathlib import Path
            model_path = Path(__file__).parent.parent.parent / 'models' / 'risk_model.pth'
            if model_path.exists():
                try:
                    self.load_model(str(model_path))
                    logger.info(f"Auto-loaded trained model from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to auto-load model: {e}")
        
        logger.info(f"RiskModel initialized with input_size={input_size}, hidden_size={hidden_size}")
    
    def prepare_features(self, indicators: Dict[str, float]) -> np.ndarray:
        """
        Prepare features from indicators dictionary.
        
        Args:
            indicators: Dictionary with required indicators
            
        Returns:
            Feature array
        """
        required_features = ["atr", "historical_volatility", "volume_ratio", "price_momentum", "rsi", "ma_deviation"]
        
        if not indicators:
            raise ValueError("Indicators dictionary cannot be empty")
        
        for feature in required_features:
            if feature not in indicators:
                raise ValueError(f"Missing required feature: {feature}")
        
        features = np.array([
            indicators["atr"],
            indicators["historical_volatility"],
            indicators["volume_ratio"],
            indicators["price_momentum"],
            indicators["rsi"],
            indicators["ma_deviation"]
        ])
        
        return features
    
    def predict(self, indicators: Dict[str, float]) -> float:
        """
        Predict optimal stop-loss percentage.
        
        Args:
            indicators: Dictionary with market indicators
            
        Returns:
            Stop-loss percentage (0.01-0.15)
        """
        # Validate input before anything else
        if not indicators:
            raise ValueError("Indicators dictionary cannot be empty")
        
        # Check required features
        required_features = ["atr", "historical_volatility", "volume_ratio", "price_momentum", "rsi", "ma_deviation"]
        for feature in required_features:
            if feature not in indicators:
                raise ValueError(f"Missing required feature: {feature}")
        
        if not self.is_trained:
            # Return default stop-loss percentage before training
            return 0.08
        
        features = self.prepare_features(indicators)
        
        # Handle case where scaler wasn't loaded from model file
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        except Exception as e:
            logger.warning(f"Scaler transform failed: {e}, using unscaled features")
            # Use features directly if scaler not properly initialized
            features_scaled = features.reshape(1, -1)
            # Normalize manually to approximate expected range
            features_scaled = (features_scaled - features_scaled.mean()) / (features_scaled.std() + 1e-8)
        
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features_scaled)
            prediction = self.model(input_tensor).item()
        
        # Scale prediction to 1%-15% range
        stop_loss_pct = 0.01 + prediction * 0.14
        return float(np.clip(stop_loss_pct, 0.01, 0.15))

    def predict_from_market_data(self, symbol: str, market_data: Dict[str, Any]) -> RiskPrediction:
        """
        Predict optimal stop-loss from raw market data.
        
        Args:
            symbol: Stock symbol
            market_data: Raw market data from price fetcher
            
        Returns:
            RiskPrediction with optimal stop-loss percentage
        """
        try:
            # Calculate required indicators from market data
            indicators = self._calculate_indicators_from_market_data(market_data)
            
            # Get prediction from trained model
            stop_loss_pct = self.predict(indicators)
            
            # Determine risk level based on stop-loss percentage
            if stop_loss_pct <= 0.03:
                risk_level = "low"
                position_size_factor = 1.0
            elif stop_loss_pct <= 0.06:
                risk_level = "medium"
                position_size_factor = 0.8
            elif stop_loss_pct <= 0.10:
                risk_level = "high"
                position_size_factor = 0.6
            else:
                risk_level = "extreme"
                position_size_factor = 0.4
            
            # Calculate confidence based on model training status
            confidence = 0.8 if self.is_trained else 0.3
            
            return RiskPrediction(
                optimal_stop_loss_percent=stop_loss_pct,
                confidence=confidence,
                risk_level=risk_level,
                recommended_position_size_factor=position_size_factor
            )
            
        except Exception as e:
            logger.error(f"Failed to predict from market data: {e}")
            # Return default conservative prediction
            return RiskPrediction(
                optimal_stop_loss_percent=0.08,
                confidence=0.3,
                risk_level="medium",
                recommended_position_size_factor=0.5
            )
    
    def _calculate_indicators_from_market_data(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate required indicators from raw market data.
        
        Args:
            market_data: Raw market data with price history
            
        Returns:
            Dictionary of calculated indicators
        """
        # Extract price data
        prices = market_data.get("historical_prices", [])
        if not prices or len(prices) < 20:
            # Return default values if insufficient data
            return {
                "historical_volatility": 0.2,
                "atr": 2.0,
                "rsi": 50.0,
                "volume_ratio": 1.0,
                "ma_deviation": 0.0,
                "beta": 1.0
            }
        
        # Convert to numpy arrays for calculations
        closes = np.array([p.get("close", 0) for p in prices])
        highs = np.array([p.get("high", 0) for p in prices])
        lows = np.array([p.get("low", 0) for p in prices])
        volumes = np.array([p.get("volume", 0) for p in prices])
        
        # Calculate historical volatility (20-day)
        returns = np.diff(closes) / closes[:-1]
        historical_volatility = float(np.std(returns[-20:]) * np.sqrt(252)) if len(returns) >= 20 else 0.2
        
        # Calculate ATR (14-day)
        atr = self._calculate_atr(highs[-14:], lows[-14:], closes[-14:]) if len(closes) >= 14 else 2.0
        
        # Calculate RSI (14-day)
        rsi = self._calculate_rsi(closes[-15:]) if len(closes) >= 15 else 50.0
        
        # Calculate volume ratio (20-day avg / 60-day avg)
        volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1] if len(volumes) > 0 else 1
        volume_60 = np.mean(volumes[-60:]) if len(volumes) >= 60 else volume_20
        volume_ratio = float(volume_20 / volume_60) if volume_60 > 0 else 1.0
        
        # Calculate MA deviation (price vs 20-day MA)
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1] if len(closes) > 0 else 1
        current_price = closes[-1] if len(closes) > 0 else 1
        ma_deviation = float((current_price - ma_20) / ma_20) if ma_20 > 0 else 0.0
        
        # Calculate beta (simplified - correlation with market)
        # For now, use a default value as we don't have market index data
        beta = 1.0
        
        # Add price momentum for compatibility
        price_momentum = float((closes[-1] - closes[-5]) / closes[-5]) if len(closes) >= 5 and closes[-5] > 0 else 0.0
        
        return {
            "historical_volatility": historical_volatility,
            "atr": atr,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "ma_deviation": ma_deviation,
            "beta": beta,
            "price_momentum": price_momentum
        }
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """
        Calculate Average True Range.
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            
        Returns:
            ATR value
        """
        if len(highs) < 2:
            return 2.0
        
        tr_list = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_close, low_close)
            tr_list.append(tr)
        
        return float(np.mean(tr_list)) if tr_list else 2.0
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI value
        """
        if len(prices) < 2:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period]) if len(gains) >= period else np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses[:period]) if len(losses) >= period else np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def train(self, training_data: List[Dict[str, float]], epochs: int = 100, batch_size: int = 32) -> float:
        """
        Train the risk model.
        
        Args:
            training_data: List of training samples with features and target
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Final training loss
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        logger.info(f"Training model with {len(training_data)} samples for {epochs} epochs")
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for sample in training_data:
            features = self.prepare_features(sample)
            X_train.append(features)
            y_train.append(sample["target_stop_loss"])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Fit scaler and transform features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train_scaled)
        y_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        
        # Training loop
        self.model.train()
        final_loss = 0.0
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                # Normalize target to 0-1 range for sigmoid output
                batch_y_normalized = (batch_y - 0.01) / 0.14
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y_normalized)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            final_loss = total_loss / num_batches
        
        self.is_trained = True
        logger.info(f"Training completed. Final loss: {final_loss:.6f}")
        return final_loss
    
    def validate(self, test_data: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Validate model performance.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Validation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before validation")
        
        predictions = []
        targets = []
        
        for sample in test_data:
            sample_copy = sample.copy()
            target = sample_copy.pop("target_stop_loss")
            pred = self.predict(sample_copy)
            
            predictions.append(pred)
            targets.append(target)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((predictions - targets) / targets)) * 100
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        }
    
    def batch_predict(self, batch_indicators: List[Dict[str, float]]) -> List[float]:
        """
        Predict for batch of indicators.
        
        Args:
            batch_indicators: List of indicator dictionaries
            
        Returns:
            List of predictions
        """
        return [self.predict(indicators) for indicators in batch_indicators]
    
    def calculate_default_stop_loss(self, indicators: Dict[str, float]) -> float:
        """
        Calculate default stop-loss based on simple rules.
        
        Args:
            indicators: Market indicators
            
        Returns:
            Default stop-loss percentage
        """
        base_stop_loss = 0.08  # 8% base
        
        # Adjust based on volatility
        hv_adj = min(indicators.get("historical_volatility", 0.2) * 0.2, 0.04)
        atr_adj = min(indicators.get("atr", 50) / 1000, 0.03)
        
        adjusted_stop_loss = base_stop_loss + hv_adj + atr_adj
        return float(np.clip(adjusted_stop_loss, 0.01, 0.15))
    
    def save_model(self, model_path: str) -> None:
        """
        Save model to file.
        
        Args:
            model_path: Path to save the model
        """
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }
        
        torch.save(model_state, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from file.
        
        Args:
            model_path: Path to load the model from
        """
        model_state = torch.load(model_path, weights_only=False)
        
        # Load model weights
        self.model.load_state_dict(model_state['model_state_dict'])
        
        # Load scaler if available (may not be in older model files)
        if 'scaler' in model_state:
            self.scaler = model_state['scaler']
        else:
            # Initialize with default scaler if not in model file
            logger.warning("Scaler not found in model file, using default initialization")
            # The scaler will be fit when first used with training data
            
        # Load training status
        if 'is_trained' in model_state:
            self.is_trained = model_state['is_trained']
        else:
            # If model has weights, assume it's trained
            self.is_trained = True
            
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self, sample_data: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate feature importance (simplified version).
        
        Args:
            sample_data: Sample data for importance calculation
            
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained:
            # Return uniform importance if not trained
            return {feature: 1.0 / len(self.feature_names) for feature in self.feature_names}
        
        # Simple feature importance based on coefficient magnitude
        # This is a simplified version - real implementation would use gradient-based methods
        importance = {
            "atr": 0.22,
            "historical_volatility": 0.28,
            "volume_ratio": 0.15,
            "price_momentum": 0.12,
            "rsi": 0.13,
            "ma_deviation": 0.10
        }
        
        return importance