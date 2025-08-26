"""Generate training data for Japanese stocks from target list.

This script generates training data using stocks from ターゲット企業.xlsx
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collector.target_watchlist_loader import TargetWatchlistLoader
from train.generate_training_data import TrainingDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JapanTrainingDataGenerator(TrainingDataGenerator):
    """Generate training data for Japanese stocks."""
    
    def __init__(self) -> None:
        """Initialize Japanese data generator."""
        super().__init__()
        self.watchlist_loader = TargetWatchlistLoader()
    
    def get_japanese_symbols(self) -> list:
        """Get Japanese stock symbols from Excel.
        
        Returns:
            List of Japanese stock symbols
        """
        companies = self.watchlist_loader.load_watchlist()
        symbols = self.watchlist_loader.get_symbols()
        
        logger.info(f"Loaded {len(symbols)} Japanese stocks from Excel")
        
        # Show first 10 symbols
        print("\nTarget symbols (first 10):")
        for i, symbol in enumerate(symbols[:10], 1):
            company = next((c for c in companies if c['symbol'] == symbol), {})
            print(f"  {i:2d}. {symbol}: {company.get('company_name', 'Unknown')}")
        
        return symbols


def main():
    """Main function to generate Japanese stock training data."""
    generator = JapanTrainingDataGenerator()
    
    # Get Japanese symbols
    japanese_symbols = generator.get_japanese_symbols()
    
    if not japanese_symbols:
        logger.error("No Japanese symbols loaded")
        return
    
    # Also include some major Japanese indices for comparison
    additional_symbols = [
        '^N225',  # Nikkei 225
        'EWJ',    # iShares MSCI Japan ETF
    ]
    
    # For training, we'll use a subset to speed up the process
    # In production, you can use all symbols
    training_symbols = japanese_symbols[:15]  # Use first 15 stocks
    
    # Date range (avoid too recent dates for Japanese markets)
    end_date = datetime.now() - timedelta(days=30)
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    logger.info("Starting Japanese training data generation")
    logger.info(f"Symbols: {len(training_symbols)}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Generate dataset
    print("\n" + "="*60)
    print("Generating training data...")
    print("="*60)
    print("This may take 10-20 minutes depending on network speed.")
    print("")
    
    dataset = []
    
    for i, symbol in enumerate(training_symbols, 1):
        print(f"Processing {i}/{len(training_symbols)}: {symbol}")
        
        # Generate 20 samples per symbol
        symbol_samples = generator.generate_dataset(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            samples_per_symbol=20
        )
        
        dataset.extend(symbol_samples)
        print(f"  Generated {len(symbol_samples)} samples")
    
    if not dataset:
        logger.error("No data generated. Check network connection and symbol validity.")
        return
    
    # Add metadata
    for sample in dataset:
        sample['market'] = 'japan'
    
    # Shuffle and split
    np.random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    # Save datasets
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    # Save with different names to keep original US data
    generator.save_dataset(train_data, output_dir / 'japan_train_data.json')
    generator.save_dataset(val_data, output_dir / 'japan_val_data.json')
    
    # Also create a combined dataset if US data exists
    us_train_path = output_dir / 'train_data.json'
    us_val_path = output_dir / 'val_data.json'
    
    if us_train_path.exists() and us_val_path.exists():
        print("\nCombining with US data...")
        
        # Load US data
        us_train = generator.load_dataset(us_train_path)
        us_val = generator.load_dataset(us_val_path)
        
        # Add market metadata to US data
        for sample in us_train:
            if 'market' not in sample:
                sample['market'] = 'us'
        for sample in us_val:
            if 'market' not in sample:
                sample['market'] = 'us'
        
        # Combine datasets
        combined_train = us_train + train_data
        combined_val = us_val + val_data
        
        # Shuffle combined data
        np.random.shuffle(combined_train)
        np.random.shuffle(combined_val)
        
        # Save combined datasets
        generator.save_dataset(combined_train, output_dir / 'combined_train_data.json')
        generator.save_dataset(combined_val, output_dir / 'combined_val_data.json')
        
        print(f"\nCombined dataset created:")
        print(f"  Combined training samples: {len(combined_train)}")
        print(f"  Combined validation samples: {len(combined_val)}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Japanese Training Data Generation Complete")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    if dataset:
        # Show sample statistics
        stop_losses = [s['target_stop_loss'] for s in dataset]
        print(f"\nTarget stop-loss statistics (Japanese stocks):")
        print(f"  Mean: {np.mean(stop_losses):.3f}")
        print(f"  Std: {np.std(stop_losses):.3f}")
        print(f"  Min: {np.min(stop_losses):.3f}")
        print(f"  Max: {np.max(stop_losses):.3f}")
        
        # Compare with typical Japanese market volatility
        volatilities = [s['historical_volatility'] for s in dataset]
        print(f"\nHistorical volatility (Japanese stocks):")
        print(f"  Mean: {np.mean(volatilities):.3f}")
        print(f"  Std: {np.std(volatilities):.3f}")
    
    print("\n💡 Tip: To train with Japanese data, use:")
    print("  uv run python train_risk_model.py")
    print("  with train_data_path = 'data/japan_train_data.json'")
    print("\n  Or use combined data for better generalization:")
    print("  with train_data_path = 'data/combined_train_data.json'")


if __name__ == "__main__":
    main()