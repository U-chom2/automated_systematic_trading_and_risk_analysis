"""Watchlist management for monitoring target companies."""

from typing import Dict, List, Any, Optional, Set
from decimal import Decimal
from datetime import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CompanyInfo:
    """Company information from Excel data."""
    code: str  # 証券コード
    name: str  # 企業名
    market_cap: int  # 時価総額（百万円）
    main_theme: str  # 主要テーマ
    performance_trend: str  # 業績トレンド
    chart_volume: str  # チャート/出来高
    ir_history: str  # 主要なIR実績
    added_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.added_at is None:
            self.added_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class ScreeningCriteria:
    """Screening criteria for dynamic watchlist generation."""
    min_market_cap: Optional[int] = None  # 最小時価総額（百万円）
    max_market_cap: Optional[int] = None  # 最大時価総額（百万円）
    required_themes: Optional[List[str]] = None  # 必須テーマ
    excluded_themes: Optional[List[str]] = None  # 除外テーマ
    performance_trends: Optional[List[str]] = None  # 業績トレンド条件
    chart_conditions: Optional[List[str]] = None  # チャート条件
    has_ir_history: Optional[bool] = None  # IR実績の有無


class WatchlistManager:
    """Manages watchlist of target companies for monitoring."""
    
    def __init__(self, excel_file_path: str) -> None:
        """
        Initialize Watchlist Manager.
        
        Args:
            excel_file_path: Path to Excel file containing company data
        """
        self.excel_file_path = Path(excel_file_path)
        self.companies: Dict[str, CompanyInfo] = {}
        self.active_watchlist: Set[str] = set()
        self.screening_criteria: Optional[ScreeningCriteria] = None
        self.last_loaded: Optional[datetime] = None
        logger.info(f"WatchlistManager initialized with {excel_file_path}")
    
    def load_companies_from_excel(self) -> bool:
        """
        Load company data from Excel file.
        
        Returns:
            True if loaded successfully
        """
        try:
            if not self.excel_file_path.exists():
                logger.error(f"Excel file not found: {self.excel_file_path}")
                return False
            
            df = pd.read_excel(self.excel_file_path)
            
            # Validate required columns
            required_columns = ['証券コード', '企業名', '時価総額 (百万円)', 
                              '主要テーマ', '業績トレンド', 'チャート/出来高', '主要なIR実績']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns in Excel file: {missing_columns}")
                return False
            
            # Load company data
            companies_loaded = 0
            for _, row in df.iterrows():
                code = str(row['証券コード']).strip()
                
                if not code or code == 'nan':
                    continue
                
                company = CompanyInfo(
                    code=code,
                    name=str(row['企業名']).strip(),
                    market_cap=int(row['時価総額 (百万円)']),
                    main_theme=str(row['主要テーマ']).strip(),
                    performance_trend=str(row['業績トレンド']).strip(),
                    chart_volume=str(row['チャート/出来高']).strip(),
                    ir_history=str(row['主要なIR実績']).strip()
                )
                
                self.companies[code] = company
                companies_loaded += 1
            
            self.last_loaded = datetime.now()
            logger.info(f"Loaded {companies_loaded} companies from Excel file")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Excel file: {e}")
            return False
    
    def get_all_companies(self) -> Dict[str, CompanyInfo]:
        """
        Get all loaded companies.
        
        Returns:
            Dictionary of company code -> CompanyInfo
        """
        return self.companies.copy()
    
    def get_company_by_code(self, code: str) -> Optional[CompanyInfo]:
        """
        Get company information by code.
        
        Args:
            code: Company code (e.g., "7203")
            
        Returns:
            Company information or None
        """
        return self.companies.get(code)
    
    def set_screening_criteria(self, criteria: ScreeningCriteria) -> None:
        """
        Set screening criteria for dynamic watchlist generation.
        
        Args:
            criteria: Screening criteria
        """
        self.screening_criteria = criteria
        logger.info("Updated screening criteria")
    
    def apply_screening(self, criteria: Optional[ScreeningCriteria] = None) -> List[str]:
        """
        Apply screening criteria to generate filtered company list.
        
        Args:
            criteria: Screening criteria (uses stored criteria if None)
            
        Returns:
            List of company codes meeting criteria
        """
        if criteria is None:
            criteria = self.screening_criteria
        
        if criteria is None:
            logger.warning("No screening criteria provided")
            return list(self.companies.keys())
        
        filtered_companies = []
        
        for code, company in self.companies.items():
            if self._meets_criteria(company, criteria):
                filtered_companies.append(code)
        
        logger.info(f"Screening applied: {len(filtered_companies)} companies selected")
        return filtered_companies
    
    def _meets_criteria(self, company: CompanyInfo, 
                       criteria: ScreeningCriteria) -> bool:
        """
        Check if company meets screening criteria.
        
        Args:
            company: Company information
            criteria: Screening criteria
            
        Returns:
            True if company meets criteria
        """
        # Market cap filtering
        if criteria.min_market_cap and company.market_cap < criteria.min_market_cap:
            return False
        if criteria.max_market_cap and company.market_cap > criteria.max_market_cap:
            return False
        
        # Theme filtering
        if criteria.required_themes:
            if not any(theme in company.main_theme 
                      for theme in criteria.required_themes):
                return False
        
        if criteria.excluded_themes:
            if any(theme in company.main_theme 
                  for theme in criteria.excluded_themes):
                return False
        
        # Performance trend filtering
        if criteria.performance_trends:
            if company.performance_trend not in criteria.performance_trends:
                return False
        
        # Chart condition filtering
        if criteria.chart_conditions:
            if not any(condition in company.chart_volume 
                      for condition in criteria.chart_conditions):
                return False
        
        # IR history filtering
        if criteria.has_ir_history is not None:
            has_ir = company.ir_history and company.ir_history.strip() != "―"
            if criteria.has_ir_history != has_ir:
                return False
        
        return True
    
    def update_active_watchlist(self, company_codes: List[str]) -> bool:
        """
        Update active watchlist.
        
        Args:
            company_codes: List of company codes to monitor
            
        Returns:
            True if updated successfully
        """
        # Validate that all codes exist
        invalid_codes = [code for code in company_codes 
                        if code not in self.companies]
        
        if invalid_codes:
            logger.error(f"Invalid company codes: {invalid_codes}")
            return False
        
        self.active_watchlist = set(company_codes)
        logger.info(f"Updated active watchlist: {len(company_codes)} companies")
        return True
    
    def get_active_watchlist(self) -> List[str]:
        """
        Get current active watchlist.
        
        Returns:
            List of company codes in active watchlist
        """
        return list(self.active_watchlist)
    
    def add_to_watchlist(self, code: str) -> bool:
        """
        Add company to active watchlist.
        
        Args:
            code: Company code
            
        Returns:
            True if added successfully
        """
        if code not in self.companies:
            logger.error(f"Company code {code} not found in loaded companies")
            return False
        
        self.active_watchlist.add(code)
        logger.info(f"Added {code} to active watchlist")
        return True
    
    def remove_from_watchlist(self, code: str) -> bool:
        """
        Remove company from active watchlist.
        
        Args:
            code: Company code
            
        Returns:
            True if removed successfully
        """
        if code in self.active_watchlist:
            self.active_watchlist.remove(code)
            logger.info(f"Removed {code} from active watchlist")
            return True
        else:
            logger.warning(f"Company code {code} not in active watchlist")
            return False
    
    def get_watchlist_with_info(self) -> List[CompanyInfo]:
        """
        Get active watchlist with complete company information.
        
        Returns:
            List of CompanyInfo objects for active watchlist
        """
        return [self.companies[code] for code in self.active_watchlist 
                if code in self.companies]
    
    def reload_from_excel(self) -> bool:
        """
        Reload company data from Excel file.
        
        Returns:
            True if reloaded successfully
        """
        logger.info("Reloading company data from Excel file")
        
        # Save current active watchlist
        current_watchlist = self.active_watchlist.copy()
        
        # Clear and reload
        self.companies.clear()
        self.active_watchlist.clear()
        
        if self.load_companies_from_excel():
            # Restore active watchlist (only valid codes)
            valid_codes = [code for code in current_watchlist 
                          if code in self.companies]
            self.active_watchlist = set(valid_codes)
            
            if len(valid_codes) != len(current_watchlist):
                logger.warning(f"Some watchlist companies were removed during reload")
            
            return True
        else:
            return False
    
    def export_watchlist_to_dict(self) -> Dict[str, Any]:
        """
        Export active watchlist to dictionary format.
        
        Returns:
            Dictionary containing watchlist data
        """
        return {
            "watchlist_codes": list(self.active_watchlist),
            "company_details": {
                code: {
                    "name": self.companies[code].name,
                    "market_cap": self.companies[code].market_cap,
                    "main_theme": self.companies[code].main_theme,
                    "performance_trend": self.companies[code].performance_trend
                }
                for code in self.active_watchlist
                if code in self.companies
            },
            "total_companies": len(self.active_watchlist),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_companies_by_theme(self, theme: str) -> List[CompanyInfo]:
        """
        Get companies by main theme.
        
        Args:
            theme: Theme keyword to search for
            
        Returns:
            List of companies with matching theme
        """
        matching_companies = []
        
        for company in self.companies.values():
            if theme.lower() in company.main_theme.lower():
                matching_companies.append(company)
        
        logger.debug(f"Found {len(matching_companies)} companies with theme: {theme}")
        return matching_companies
    
    def get_companies_by_performance_trend(self, trend: str) -> List[CompanyInfo]:
        """
        Get companies by performance trend.
        
        Args:
            trend: Performance trend to filter by
            
        Returns:
            List of companies with matching trend
        """
        matching_companies = []
        
        for company in self.companies.values():
            if trend in company.performance_trend:
                matching_companies.append(company)
        
        logger.debug(f"Found {len(matching_companies)} companies with trend: {trend}")
        return matching_companies