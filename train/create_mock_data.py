"""Create mock training data for quick testing."""

import json
import numpy as np
from pathlib import Path

def create_mock_data(n_samples=100):
    """Create mock training data."""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        sample = {
            'symbol': f'STOCK{i % 10}',
            'date': f'2024-{(i % 12 + 1):02d}-{(i % 28 + 1):02d}',
            'historical_volatility': np.random.uniform(0.1, 0.5),
            'atr': np.random.uniform(0.01, 0.05),
            'rsi': np.random.uniform(0.2, 0.8),
            'volume_ratio': np.random.uniform(0.1, 0.9),
            'ma_deviation': np.random.uniform(-0.2, 0.2),
            'beta': np.random.uniform(0.1, 0.6),
            'target_stop_loss': np.random.uniform(0.02, 0.12)
        }
        data.append(sample)
    
    return data

def main():
    """Create and save mock data."""
    # Create data directory
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Generate data
    all_data = create_mock_data(100)
    
    # Split into train and validation
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Save data
    with open(data_dir / 'mock_train_data.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(data_dir / 'mock_val_data.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print("Mock data created successfully!")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Show sample
    print("\nSample data point:")
    for key, value in train_data[0].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()