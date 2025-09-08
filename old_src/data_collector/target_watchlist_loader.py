"""Target watchlist loader for Excel file.

Loads and manages target companies from Excel spreadsheet.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TargetWatchlistLoader:
    """Loads and manages target companies from Excel file."""
    
    def __init__(self, excel_path: Optional[Path] = None) -> None:
        """Initialize watchlist loader.
        
        Args:
            excel_path: Path to Excel file. Defaults to 'ターゲット企業.xlsx'
        """
        if excel_path is None:
            excel_path = Path("ターゲット企業.xlsx")
        
        self.excel_path = Path(excel_path)
        self.watchlist: List[Dict[str, Any]] = []
        self.loaded_at: Optional[datetime] = None
        
    def load_watchlist(self) -> List[Dict[str, Any]]:
        """Load watchlist from Excel file.
        
        Returns:
            List of target companies with their details
        """
        try:
            if not self.excel_path.exists():
                logger.error(f"Excel file not found: {self.excel_path}")
                return []
            
            # Read Excel file
            df = pd.read_excel(self.excel_path, engine='openpyxl')
            logger.info(f"Loaded {len(df)} companies from {self.excel_path}")
            
            # Convert to list of dictionaries
            self.watchlist = []
            for _, row in df.iterrows():
                company_data = row.to_dict()
                
                # Clean and standardize data
                cleaned_data = self._clean_company_data(company_data)
                if cleaned_data:
                    self.watchlist.append(cleaned_data)
            
            self.loaded_at = datetime.now()
            logger.info(f"Successfully loaded {len(self.watchlist)} target companies")
            
            return self.watchlist
            
        except Exception as e:
            logger.error(f"Failed to load watchlist: {e}")
            return []
    
    def _clean_company_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and standardize company data.
        
        Args:
            data: Raw company data from Excel
            
        Returns:
            Cleaned company data or None if invalid
        """
        try:
            # Expected columns might include: 証券コード, 企業名, 業種, etc.
            cleaned = {}
            
            # Map common column names
            column_mapping = {
                '証券コード': 'symbol',
                'コード': 'symbol',
                '銘柄コード': 'symbol',
                '企業名': 'company_name',
                '会社名': 'company_name',
                '銘柄名': 'company_name',
                '業種': 'sector',
                'セクター': 'sector',
                '市場': 'market',
                '取引所': 'market',
                '時価総額': 'market_cap',
                'メモ': 'notes',
                '備考': 'notes'
            }
            
            for key, value in data.items():
                # Skip NaN values
                if pd.isna(value):
                    continue
                    
                # Map column names
                mapped_key = column_mapping.get(key, key.lower())
                
                # Clean symbol (ensure it's a string and add .T for Tokyo)
                if mapped_key == 'symbol':
                    symbol_str = str(int(value)) if isinstance(value, float) else str(value)
                    # Add .T suffix for Tokyo Stock Exchange if not present
                    if not symbol_str.endswith('.T'):
                        symbol_str += '.T'
                    cleaned['symbol'] = symbol_str
                else:
                    cleaned[mapped_key] = value
            
            # Validate required fields
            if 'symbol' not in cleaned:
                logger.warning(f"No symbol found in data: {data}")
                return None
            
            # Add metadata
            cleaned['source'] = 'excel_watchlist'
            cleaned['loaded_at'] = datetime.now().isoformat()
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to clean company data: {e}")
            return None
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols from watchlist.
        
        Returns:
            List of stock symbols
        """
        if not self.watchlist:
            self.load_watchlist()
        
        return [company['symbol'] for company in self.watchlist if 'symbol' in company]
    
    def get_companies_by_sector(self, sector: str) -> List[Dict[str, Any]]:
        """Get companies filtered by sector.
        
        Args:
            sector: Sector name to filter by
            
        Returns:
            List of companies in specified sector
        """
        if not self.watchlist:
            self.load_watchlist()
        
        return [
            company for company in self.watchlist
            if company.get('sector', '').lower() == sector.lower()
        ]
    
    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information for specific company.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company information or None if not found
        """
        if not self.watchlist:
            self.load_watchlist()
        
        # Handle both with and without .T suffix
        symbol_variants = [symbol, symbol.replace('.T', ''), symbol + '.T']
        
        for company in self.watchlist:
            if company.get('symbol') in symbol_variants:
                return company
        
        return None
    
    def refresh_watchlist(self) -> bool:
        """Reload watchlist from Excel file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.load_watchlist()
            return len(self.watchlist) > 0
        except Exception as e:
            logger.error(f"Failed to refresh watchlist: {e}")
            return False
    
    def get_watchlist_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded watchlist.
        
        Returns:
            Dictionary with watchlist statistics
        """
        if not self.watchlist:
            self.load_watchlist()
        
        stats = {
            'total_companies': len(self.watchlist),
            'loaded_at': self.loaded_at.isoformat() if self.loaded_at else None,
            'source_file': str(self.excel_path),
            'sectors': {}
        }
        
        # Count by sector
        for company in self.watchlist:
            sector = company.get('sector', 'Unknown')
            stats['sectors'][sector] = stats['sectors'].get(sector, 0) + 1
        
        return stats