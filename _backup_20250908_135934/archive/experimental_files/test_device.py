"""Test device detection for training code."""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_device_detection():
    """Test device detection logic."""
    print("Testing device detection...")
    print("-" * 50)
    
    # Check available devices
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CPU available: True (always)")
    print("-" * 50)
    
    # Auto-detect device
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        torch_device = torch.device('mps')
    elif torch.cuda.is_available():
        device = "cuda:0"  # NVIDIA GPU
        torch_device = torch.device('cuda:0')
    else:
        device = "cpu"  # CPU fallback
        torch_device = torch.device('cpu')
    
    print(f"Selected device: {device}")
    print(f"Torch device: {torch_device}")
    print("-" * 50)
    
    # Test tensor operations
    try:
        test_tensor = torch.randn(10, 10).to(torch_device)
        result = torch.matmul(test_tensor, test_tensor)
        print(f"✓ Tensor operations work on {device}")
        print(f"  Test tensor shape: {test_tensor.shape}")
        print(f"  Result shape: {result.shape}")
    except Exception as e:
        print(f"✗ Error on {device}: {e}")
    
    print("-" * 50)
    
    # Test with modules from training code
    try:
        from train.models.agents.ppo_agent import PPOTradingAgent
        print("✓ PPOTradingAgent imported successfully")
    except ImportError as e:
        print(f"✗ Could not import PPOTradingAgent: {e}")
    
    try:
        from train.models.modernbert_encoder import ModernBERTNewsEncoder
        print("✓ ModernBERTNewsEncoder imported successfully")
    except ImportError as e:
        print(f"✗ Could not import ModernBERTNewsEncoder: {e}")
    
    try:
        from src.analysis_engine.nlp_analyzer import NlpAnalyzer
        analyzer = NlpAnalyzer()
        print(f"✓ NlpAnalyzer initialized on device: {analyzer.device if hasattr(analyzer, 'device') else 'unknown'}")
    except Exception as e:
        print(f"✗ Could not initialize NlpAnalyzer: {e}")
    
    print("-" * 50)
    print("Device detection test completed!")


if __name__ == "__main__":
    test_device_detection()