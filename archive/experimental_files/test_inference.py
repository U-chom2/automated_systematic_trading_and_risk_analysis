"""Test inference with device detection."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from train.models.trading_model import TradingDecisionModel, MarketData
from train.main import run_inference

def test_basic_inference():
    """Test basic inference functionality."""
    print("=" * 60)
    print("Testing Basic Inference with Device Detection")
    print("=" * 60)
    
    # Check available devices
    print("\nDevice Information:")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Auto-detect device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f"Selected device: {device}")
    print("-" * 60)
    
    # Create model
    print("\nCreating model...")
    model = TradingDecisionModel(device=device)
    print(f"✓ Model created on {device}")
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    
    market_data = MarketData(
        nikkei_high=np.random.randn(30) * 100 + 28000,
        nikkei_low=np.random.randn(30) * 100 + 27800,
        nikkei_close=np.random.randn(30) * 100 + 27900,
        target_high=np.random.randn(30) * 20 + 3020,
        target_low=np.random.randn(30) * 20 + 2980,
        target_close=np.random.randn(30) * 20 + 3000,
        ir_news=[
            "第3四半期決算発表：増収増益を達成",
            "新製品の販売が好調",
            "通期業績予想を上方修正"
        ]
    )
    print("✓ Sample data generated")
    
    # Run inference
    print("\nRunning inference...")
    model.eval()
    with torch.no_grad():
        decision = model(market_data)
    
    print("\n" + "=" * 60)
    print("Inference Results:")
    print("=" * 60)
    print(f"Recommended Action: {decision['action']}")
    print(f"Confidence: {decision['confidence']*100:.1f}%")
    print(f"\nDetailed Probabilities:")
    print(f"  Strong Sell: {decision['sell_prob']*100:.1f}%")
    print(f"  Hold: {decision['hold_prob']*100:.1f}%")
    print(f"  Small Buy: {decision['buy_small_prob']*100:.1f}%")
    print(f"  Strong Buy: {decision['buy_large_prob']*100:.1f}%")
    print(f"\nRecommended Position: {decision['recommended_position']:.2f}")
    print("(-0.33=Full Sell, 0=Hold, 1.0=Full Buy)")
    
    return True

def test_demo_mode():
    """Test demo mode from main.py."""
    print("\n" + "=" * 60)
    print("Testing Demo Mode")
    print("=" * 60)
    
    try:
        # Run demo inference
        decision = run_inference(demo=True, device=None)
        print("\n✓ Demo mode completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Demo mode failed: {e}")
        return False

def test_model_save_load():
    """Test model saving and loading with device handling."""
    print("\n" + "=" * 60)
    print("Testing Model Save/Load")
    print("=" * 60)
    
    # Auto-detect device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # Create and save model
    print(f"\nCreating model on {device}...")
    model1 = TradingDecisionModel(device=device)
    
    # Save model
    save_path = Path('test_model.pth')
    print(f"Saving model to {save_path}...")
    model1.save(str(save_path))
    print("✓ Model saved")
    
    # Load model
    print(f"Loading model on {device}...")
    model2 = TradingDecisionModel(device=device)
    model2.load(str(save_path), device=device)
    print("✓ Model loaded")
    
    # Test inference with loaded model
    print("Testing inference with loaded model...")
    np.random.seed(42)
    market_data = MarketData(
        nikkei_high=np.random.randn(30) * 100 + 28000,
        nikkei_low=np.random.randn(30) * 100 + 27800,
        nikkei_close=np.random.randn(30) * 100 + 27900,
        target_high=np.random.randn(30) * 20 + 3020,
        target_low=np.random.randn(30) * 20 + 2980,
        target_close=np.random.randn(30) * 20 + 3000,
        ir_news=["Test news"]
    )
    
    model2.eval()
    with torch.no_grad():
        decision = model2(market_data)
    
    print(f"✓ Inference successful: {decision['action']}")
    
    # Clean up
    if save_path.exists():
        save_path.unlink()
        print("✓ Test file cleaned up")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting Inference Tests with Device Detection")
    print("=" * 80)
    
    tests = [
        ("Basic Inference", test_basic_inference),
        ("Demo Mode", test_demo_mode),
        ("Model Save/Load", test_model_save_load)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary:")
    print("=" * 80)
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! Inference with device detection works correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)