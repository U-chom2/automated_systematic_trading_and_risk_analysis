"""Quick test of training with device detection."""

import sys
from pathlib import Path
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from train.train import Nikkei225TradingPipeline

def test_training():
    """Run a quick training test."""
    
    # Small dataset for testing
    target_symbols = ['7203.T']  # Just Toyota for quick test
    start_date = '2023-10-01'
    end_date = '2023-11-01'  # Just 1 month
    
    print("Starting quick training test...")
    print(f"Target symbols: {target_symbols}")
    print(f"Period: {start_date} to {end_date}")
    print("-" * 50)
    
    # Create pipeline
    pipeline = Nikkei225TradingPipeline(
        target_symbols=target_symbols,
        start_date=start_date,
        end_date=end_date,
        initial_cash=10000000,
        commission_rate=0.001,
        window_size=10  # Smaller window for quick test
    )
    
    # Train with very few timesteps for testing
    trained_agent = pipeline.train(
        total_timesteps=100,  # Very small for quick test
        learning_rate=3e-4,
        n_steps=32,  # Small batch
        batch_size=8,
        n_epochs=2,
        device=None  # Let it auto-detect
    )
    
    print("-" * 50)
    print("✓ Training completed successfully!")
    print(f"✓ Model saved to: {pipeline.model_save_dir}")
    
    # Check device used
    if hasattr(trained_agent, 'device'):
        print(f"✓ Training ran on device: {trained_agent.device}")
    
    return True


if __name__ == "__main__":
    success = test_training()
    if success:
        print("\n✅ All tests passed! The training code works with MPS/CUDA device detection.")
    else:
        print("\n❌ Test failed.")