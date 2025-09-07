"""Test script to verify trained model integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis_engine.risk_model import RiskModel
from decimal import Decimal
import numpy as np

def test_model_loading():
    """Test that the model loads automatically."""
    print("\n" + "="*60)
    print("Testing Model Auto-Loading")
    print("="*60)
    
    # Create RiskModel instance
    risk_model = RiskModel()
    
    # Check if model is trained
    if risk_model.is_trained:
        print("✓ Model successfully loaded from file")
        print(f"  Input size: {risk_model.input_size}")
        print(f"  Hidden size: {risk_model.hidden_size}")
        print(f"  Output size: {risk_model.output_size}")
    else:
        print("✗ Model failed to load")
        return False
    
    return True


def test_direct_prediction():
    """Test direct prediction with indicators."""
    print("\n" + "="*60)
    print("Testing Direct Prediction")
    print("="*60)
    
    risk_model = RiskModel()
    
    # Test case 1: Low volatility environment
    indicators_low = {
        "historical_volatility": 0.15,
        "atr": 1.5,
        "rsi": 45,
        "volume_ratio": 0.9,
        "ma_deviation": -0.02,
        "beta": 0.8,
        "price_momentum": 0.01
    }
    
    prediction_low = risk_model.predict(indicators_low)
    print(f"\nLow volatility scenario:")
    print(f"  Stop-loss: {prediction_low*100:.2f}%")
    
    # Test case 2: High volatility environment
    indicators_high = {
        "historical_volatility": 0.45,
        "atr": 5.0,
        "rsi": 75,
        "volume_ratio": 2.5,
        "ma_deviation": 0.15,
        "beta": 1.8,
        "price_momentum": 0.08
    }
    
    prediction_high = risk_model.predict(indicators_high)
    print(f"\nHigh volatility scenario:")
    print(f"  Stop-loss: {prediction_high*100:.2f}%")
    
    # Test case 3: Normal market conditions
    indicators_normal = {
        "historical_volatility": 0.25,
        "atr": 2.5,
        "rsi": 55,
        "volume_ratio": 1.1,
        "ma_deviation": 0.03,
        "beta": 1.0,
        "price_momentum": 0.02
    }
    
    prediction_normal = risk_model.predict(indicators_normal)
    print(f"\nNormal market scenario:")
    print(f"  Stop-loss: {prediction_normal*100:.2f}%")
    
    return True


def test_market_data_prediction():
    """Test prediction from market data."""
    print("\n" + "="*60)
    print("Testing Market Data Prediction")
    print("="*60)
    
    risk_model = RiskModel()
    
    # Create simulated market data
    np.random.seed(42)
    base_price = 100.0
    prices = []
    
    for i in range(60):
        # Generate price with some volatility
        change = np.random.normal(0, 2)
        price = base_price + change
        prices.append({
            "date": f"2024-{(i//30)+1:02d}-{(i%30)+1:02d}",
            "open": price - np.random.uniform(0, 1),
            "high": price + np.random.uniform(0, 2),
            "low": price - np.random.uniform(0, 2),
            "close": price,
            "volume": int(1000000 + np.random.uniform(-500000, 500000))
        })
        base_price = price
    
    market_data = {
        "historical_prices": prices
    }
    
    # Get prediction
    prediction = risk_model.predict_from_market_data("TEST", market_data)
    
    print(f"\nPrediction from market data:")
    print(f"  Symbol: TEST")
    print(f"  Stop-loss: {prediction.optimal_stop_loss_percent*100:.2f}%")
    print(f"  Risk level: {prediction.risk_level}")
    print(f"  Confidence: {prediction.confidence:.2f}")
    print(f"  Position size factor: {prediction.recommended_position_size_factor:.2f}")
    
    return True


def test_position_sizing():
    """Test position sizing with trained model predictions."""
    print("\n" + "="*60)
    print("Testing Position Sizing Integration")
    print("="*60)
    
    from src.execution_manager.order_manager import OrderManager
    
    # Create order manager instance
    order_manager = OrderManager()
    
    # Test parameters
    capital = Decimal("100000")
    entry_price = Decimal("150.50")
    risk_per_trade = 0.01  # 1% risk
    
    # Get stop-loss from trained model
    risk_model = RiskModel()
    indicators = {
        "historical_volatility": 0.30,
        "atr": 3.5,
        "rsi": 60,
        "volume_ratio": 1.3,
        "ma_deviation": 0.05,
        "beta": 1.2,
        "price_momentum": 0.03
    }
    
    stop_loss_pct = risk_model.predict(indicators)
    
    # Calculate position size
    position_result = order_manager.calculate_position_size(
        capital=capital,
        risk_per_trade_ratio=risk_per_trade,
        entry_price=entry_price,
        stop_loss_percentage=stop_loss_pct
    )
    
    print(f"\nPosition sizing calculation:")
    print(f"  Capital: ${capital:,.2f}")
    print(f"  Entry price: ${entry_price:.2f}")
    print(f"  Stop-loss: {stop_loss_pct*100:.2f}%")
    print(f"  Risk per trade: {risk_per_trade*100:.1f}%")
    print(f"\nResults:")
    print(f"  Position size: {position_result['position_size']} shares")
    print(f"  Max loss: ${position_result['max_loss']:.2f}")
    print(f"  Stop-loss price: ${position_result['stop_loss_price']:.2f}")
    print(f"  Risk per share: ${position_result['risk_per_share']:.2f}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TRAINED MODEL INTEGRATION TEST")
    print("="*60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Direct Prediction", test_direct_prediction),
        ("Market Data Prediction", test_market_data_prediction),
        ("Position Sizing", test_position_sizing)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! Model integration successful.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)