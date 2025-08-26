"""Test script to verify Excel watchlist loading."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_collector.target_watchlist_loader import TargetWatchlistLoader
import pandas as pd


def main():
    """Test Excel watchlist loading."""
    print("=" * 60)
    print("📊 ターゲット企業リスト読み込みテスト")
    print("=" * 60)
    
    # First, let's see what's in the Excel file
    excel_path = Path("ターゲット企業.xlsx")
    
    if not excel_path.exists():
        print(f"❌ Excelファイルが見つかりません: {excel_path}")
        return
    
    # Read Excel to check structure
    print("\n1. Excelファイル構造の確認")
    print("-" * 40)
    
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
        print(f"行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print(f"\n列名: {list(df.columns)}")
        print(f"\n最初の5行:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Excel読み込みエラー: {e}")
        return
    
    # Test the loader
    print("\n2. WatchlistLoaderのテスト")
    print("-" * 40)
    
    loader = TargetWatchlistLoader()
    watchlist = loader.load_watchlist()
    
    if watchlist:
        print(f"✅ {len(watchlist)} 社の企業を読み込みました")
        
        # Show sample data
        print(f"\n最初の3社:")
        for i, company in enumerate(watchlist[:3], 1):
            print(f"\n企業 {i}:")
            for key, value in company.items():
                print(f"  {key}: {value}")
        
        # Get symbols
        symbols = loader.get_symbols()
        print(f"\n読み込んだシンボル (最初の10個):")
        for symbol in symbols[:10]:
            print(f"  - {symbol}")
        
        # Show statistics
        stats = loader.get_watchlist_stats()
        print(f"\n統計情報:")
        print(f"  総企業数: {stats['total_companies']}")
        print(f"  セクター別:")
        for sector, count in stats.get('sectors', {}).items():
            print(f"    - {sector}: {count}社")
            
    else:
        print("❌ 企業リストの読み込みに失敗しました")


if __name__ == "__main__":
    main()