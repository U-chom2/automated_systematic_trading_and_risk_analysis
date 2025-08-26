"""Test and demonstrate the trained risk model.

This script loads a trained model and demonstrates prediction capabilities.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis_engine.risk_model import RiskModel, RiskNeuralNetwork

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTester:
    """Test trained risk model."""
    
    def __init__(self, model_path: Path = None) -> None:
        """Initialize tester.
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path or Path(__file__).parent / 'models' / 'best_risk_model.pth'
        self.model = None
        self.model_config = None
        
    def load_model(self) -> bool:
        """Load trained model.
        
        Returns:
            True if successful
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # Get model configuration
            self.model_config = checkpoint.get('model_config', {
                'input_size': 6,
                'hidden_size': 32,
                'output_size': 1
            })
            
            # Initialize model
            self.model = RiskNeuralNetwork(
                input_size=self.model_config['input_size'],
                hidden_size=self.model_config['hidden_size'],
                output_size=self.model_config['output_size']
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"Loaded model from {self.model_path}")
            if 'val_loss' in checkpoint:
                logger.info(f"Model validation loss: {checkpoint['val_loss']:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: dict) -> float:
        """Make prediction with the model.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Predicted stop-loss percentage
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Prepare input tensor
        input_features = [
            features.get('historical_volatility', 0.2),
            features.get('atr', 0.02),
            features.get('rsi', 0.5),
            features.get('volume_ratio', 0.2),
            features.get('ma_deviation', 0.0),
            features.get('beta', 0.33)
        ]
        
        input_tensor = torch.FloatTensor([input_features])
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Convert to percentage
        stop_loss_pct = float(prediction.item())
        
        # Ensure reasonable bounds
        stop_loss_pct = max(0.01, min(0.15, stop_loss_pct))
        
        return stop_loss_pct
    
    def test_scenarios(self) -> None:
        """Test model with different market scenarios."""
        scenarios = [
            {
                'name': 'Low Volatility Market',
                'features': {
                    'historical_volatility': 0.15,
                    'atr': 0.01,
                    'rsi': 0.5,
                    'volume_ratio': 0.2,
                    'ma_deviation': 0.02,
                    'beta': 0.3
                }
            },
            {
                'name': 'High Volatility Market',
                'features': {
                    'historical_volatility': 0.45,
                    'atr': 0.04,
                    'rsi': 0.7,
                    'volume_ratio': 0.6,
                    'ma_deviation': 0.15,
                    'beta': 0.5
                }
            },
            {
                'name': 'Overbought Conditions',
                'features': {
                    'historical_volatility': 0.25,
                    'atr': 0.02,
                    'rsi': 0.85,
                    'volume_ratio': 0.8,
                    'ma_deviation': 0.25,
                    'beta': 0.4
                }
            },
            {
                'name': 'Oversold Conditions',
                'features': {
                    'historical_volatility': 0.30,
                    'atr': 0.025,
                    'rsi': 0.20,
                    'volume_ratio': 0.5,
                    'ma_deviation': -0.15,
                    'beta': 0.35
                }
            },
            {
                'name': 'Normal Market',
                'features': {
                    'historical_volatility': 0.20,
                    'atr': 0.015,
                    'rsi': 0.55,
                    'volume_ratio': 0.2,
                    'ma_deviation': 0.05,
                    'beta': 0.33
                }
            }
        ]
        
        print("\n" + "="*70)
        print("Risk Model Predictions for Different Market Scenarios")
        print("="*70)
        
        for scenario in scenarios:
            prediction = self.predict(scenario['features'])
            
            print(f"\n{scenario['name']}:")
            print("-" * 40)
            print("Input Features:")
            for key, value in scenario['features'].items():
                # Denormalize for display
                display_value = value
                if key == 'historical_volatility':
                    display_value = value
                elif key == 'rsi':
                    display_value = value * 100
                elif key == 'volume_ratio':
                    display_value = value * 5
                elif key == 'beta':
                    display_value = value * 3
                    
                print(f"  {key:20s}: {display_value:.3f}")
            
            print(f"\n  📊 Predicted Stop-Loss: {prediction*100:.2f}%")
    
    def test_with_validation_data(self, val_data_path: Path) -> None:
        """Test model with validation data.
        
        Args:
            val_data_path: Path to validation data
        """
        # Load validation data
        with open(val_data_path, 'r') as f:
            val_data = json.load(f)
        
        if not val_data:
            logger.error("No validation data found")
            return
        
        # Make predictions
        predictions = []
        actuals = []
        errors = []
        
        for sample in val_data[:20]:  # Test on first 20 samples
            prediction = self.predict(sample)
            actual = sample['target_stop_loss']
            
            predictions.append(prediction)
            actuals.append(actual)
            errors.append(abs(prediction - actual))
        
        # Calculate statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        print("\n" + "="*70)
        print("Model Performance on Validation Data (First 20 Samples)")
        print("="*70)
        
        print(f"\nError Statistics:")
        print(f"  Mean Absolute Error: {mean_error*100:.3f}%")
        print(f"  Std Dev of Error: {std_error*100:.3f}%")
        
        print(f"\nSample Predictions:")
        print("-" * 50)
        print(f"{'Symbol':10s} {'Date':12s} {'Predicted':>10s} {'Actual':>10s} {'Error':>10s}")
        print("-" * 50)
        
        for i in range(min(10, len(predictions))):
            sample = val_data[i]
            print(f"{sample['symbol']:10s} {sample['date']:12s} "
                  f"{predictions[i]*100:9.2f}% {actuals[i]*100:9.2f}% "
                  f"{errors[i]*100:9.2f}%")
    
    def demonstrate_integration(self) -> None:
        """Demonstrate integration with RiskModel class."""
        print("\n" + "="*70)
        print("Integration with RiskModel Class")
        print("="*70)
        
        # Create RiskModel instance
        risk_model = RiskModel()
        
        # Load the trained model weights into RiskModel
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            risk_model.model.load_state_dict(checkpoint['model_state_dict'])
            risk_model.is_trained = True
            
            print("\n✅ Model loaded into RiskModel class")
            
            # Test prediction
            test_features = {
                'historical_volatility': 0.25,
                'atr': 14.5,
                'rsi': 65,
                'volume_ratio': 1.5,
                'ma_deviation': 0.08,
                'beta': 1.2
            }
            
            prediction = risk_model.predict(test_features)
            
            print("\nTest Prediction:")
            print(f"  Stop-loss percentage: {prediction.stop_loss_percentage*100:.2f}%")
            print(f"  Confidence score: {prediction.confidence_score:.2f}")
            print(f"  Risk level: {prediction.risk_level}")
            
            # Save model in RiskModel format
            model_save_path = Path(__file__).parent.parent / 'models' / 'risk_model.pth'
            model_save_path.parent.mkdir(exist_ok=True)
            risk_model.save_model(model_save_path)
            print(f"\n✅ Model saved for production use at: {model_save_path}")
            
        else:
            print("\n❌ No trained model found")


def main():
    """Main testing function."""
    tester = ModelTester()
    
    # Load model
    if not tester.load_model():
        print("Failed to load model. Please train the model first using train_risk_model.py")
        return
    
    print("\n" + "="*70)
    print("Risk Model Testing and Demonstration")
    print("="*70)
    
    # Test different scenarios
    tester.test_scenarios()
    
    # Test with validation data if available
    val_data_path = Path(__file__).parent / 'data' / 'val_data.json'
    if val_data_path.exists():
        tester.test_with_validation_data(val_data_path)
    
    # Demonstrate integration
    tester.demonstrate_integration()
    
    print("\n" + "="*70)
    print("Testing Complete")
    print("="*70)


if __name__ == "__main__":
    main()