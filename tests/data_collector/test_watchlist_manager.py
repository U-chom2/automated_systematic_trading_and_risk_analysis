"""Tests for WatchlistManager (FR-01)."""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.data_collector.watchlist_manager import (
    WatchlistManager,
    CompanyInfo,
    ScreeningCriteria
)


class TestWatchlistManager:
    """Test cases for WatchlistManager."""
    
    @pytest.fixture
    def sample_excel_data(self) -> pd.DataFrame:
        """Create sample Excel data for testing."""
        return pd.DataFrame({
            '証券コード': ['7203', '6758', '9984', '7974', '4503'],
            '企業名': ['トヨタ自動車', 'ソニーグループ', 'ソフトバンクグループ', '任天堂', 'アステラス製薬'],
            '時価総額 (百万円)': [37000000, 15000000, 8000000, 6000000, 4500000],
            '主要テーマ': ['自動車・EV', 'エンタメ・半導体', 'テレコム・AI', 'ゲーム・娯楽', '医薬品・バイオ'],
            '業績トレンド': ['増収増益', '増収増益', '赤字縮小・黒字化予想', '増収増益', '増収増益'],
            'チャート/出来高': ['上昇トレンド・高出来高', '底値圏・低出来高', '調整局面・中出来高', '上昇トレンド・高出来高', '横ばい・低出来高'],
            '主要なIR実績': ['EV戦略発表', 'PS5好調', 'AI投資拡大', '新作ゲーム', '新薬開発']
        })
    
    @pytest.fixture
    def temp_excel_file(self, sample_excel_data: pd.DataFrame) -> str:
        """Create temporary Excel file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            sample_excel_data.to_excel(tmp.name, index=False)
            return tmp.name
    
    @pytest.fixture
    def watchlist_manager(self, temp_excel_file: str) -> WatchlistManager:
        """Create WatchlistManager instance for testing."""
        return WatchlistManager(temp_excel_file)
    
    def test_initialization(self, temp_excel_file: str) -> None:
        """Test WatchlistManager initialization."""
        manager = WatchlistManager(temp_excel_file)
        
        assert manager.excel_file_path == Path(temp_excel_file)
        assert len(manager.companies) == 0
        assert len(manager.active_watchlist) == 0
        assert manager.screening_criteria is None
        assert manager.last_loaded is None
    
    def test_load_companies_from_excel_success(self, 
                                             watchlist_manager: WatchlistManager) -> None:
        """Test successful loading of companies from Excel."""
        result = watchlist_manager.load_companies_from_excel()
        
        assert result is True
        assert len(watchlist_manager.companies) == 5
        assert "7203" in watchlist_manager.companies
        assert "6758" in watchlist_manager.companies
        
        # Check company data
        toyota = watchlist_manager.companies["7203"]
        assert toyota.name == "トヨタ自動車"
        assert toyota.market_cap == 37000000
        assert toyota.main_theme == "自動車・EV"
        assert toyota.added_at is not None
        assert toyota.last_updated is not None
    
    def test_load_companies_from_nonexistent_file(self) -> None:
        """Test loading from non-existent Excel file."""
        manager = WatchlistManager("nonexistent.xlsx")
        result = manager.load_companies_from_excel()
        
        assert result is False
        assert len(manager.companies) == 0
    
    def test_get_company_by_code(self, watchlist_manager: WatchlistManager) -> None:
        """Test getting company by code."""
        watchlist_manager.load_companies_from_excel()
        
        # Existing company
        toyota = watchlist_manager.get_company_by_code("7203")
        assert toyota is not None
        assert toyota.name == "トヨタ自動車"
        
        # Non-existent company
        unknown = watchlist_manager.get_company_by_code("9999")
        assert unknown is None
    
    def test_add_to_watchlist(self, watchlist_manager: WatchlistManager) -> None:
        """Test adding company to watchlist."""
        watchlist_manager.load_companies_from_excel()
        
        # Add valid company
        result = watchlist_manager.add_to_watchlist("7203")
        assert result is True
        assert "7203" in watchlist_manager.active_watchlist
        assert len(watchlist_manager.active_watchlist) == 1
        
        # Add invalid company
        result = watchlist_manager.add_to_watchlist("9999")
        assert result is False
        assert len(watchlist_manager.active_watchlist) == 1
    
    def test_remove_from_watchlist(self, watchlist_manager: WatchlistManager) -> None:
        """Test removing company from watchlist."""
        watchlist_manager.load_companies_from_excel()
        watchlist_manager.add_to_watchlist("7203")
        
        # Remove existing company
        result = watchlist_manager.remove_from_watchlist("7203")
        assert result is True
        assert "7203" not in watchlist_manager.active_watchlist
        
        # Remove non-existent company
        result = watchlist_manager.remove_from_watchlist("7203")
        assert result is False
    
    def test_update_active_watchlist(self, watchlist_manager: WatchlistManager) -> None:
        """Test updating active watchlist with multiple companies."""
        watchlist_manager.load_companies_from_excel()
        
        # Update with valid codes
        codes = ["7203", "6758", "9984"]
        result = watchlist_manager.update_active_watchlist(codes)
        assert result is True
        assert watchlist_manager.active_watchlist == set(codes)
        
        # Update with invalid codes
        invalid_codes = ["7203", "9999"]
        result = watchlist_manager.update_active_watchlist(invalid_codes)
        assert result is False
    
    def test_screening_criteria_market_cap(self, 
                                         watchlist_manager: WatchlistManager) -> None:
        """Test screening by market cap."""
        watchlist_manager.load_companies_from_excel()
        
        # Filter for large cap (>10B yen)
        criteria = ScreeningCriteria(min_market_cap=10000000)
        filtered = watchlist_manager.apply_screening(criteria)
        
        # Should include Toyota (37B), Sony (15B) but not others
        assert "7203" in filtered  # Toyota
        assert "6758" in filtered  # Sony
        assert "9984" not in filtered  # SoftBank (8B)
    
    def test_screening_criteria_theme(self, watchlist_manager: WatchlistManager) -> None:
        """Test screening by theme."""
        watchlist_manager.load_companies_from_excel()
        
        # Filter for auto/EV theme
        criteria = ScreeningCriteria(required_themes=["自動車"])
        filtered = watchlist_manager.apply_screening(criteria)
        
        assert "7203" in filtered  # Toyota (自動車・EV)
        assert "6758" not in filtered  # Sony (エンタメ・半導体)
    
    def test_screening_criteria_performance_trend(self, 
                                                watchlist_manager: WatchlistManager) -> None:
        """Test screening by performance trend."""
        watchlist_manager.load_companies_from_excel()
        
        # Filter for profitable companies
        criteria = ScreeningCriteria(performance_trends=["増収増益"])
        filtered = watchlist_manager.apply_screening(criteria)
        
        # Should include profitable companies, exclude SoftBank (loss recovery)
        assert "7203" in filtered  # Toyota
        assert "6758" in filtered  # Sony
        assert "9984" not in filtered  # SoftBank
        assert "7974" in filtered  # Nintendo
    
    def test_screening_criteria_ir_history(self, 
                                         watchlist_manager: WatchlistManager) -> None:
        """Test screening by IR history."""
        watchlist_manager.load_companies_from_excel()
        
        # Filter companies with IR history
        criteria = ScreeningCriteria(has_ir_history=True)
        filtered = watchlist_manager.apply_screening(criteria)
        
        # All sample companies have IR history
        assert len(filtered) == 5
    
    def test_get_companies_by_theme(self, watchlist_manager: WatchlistManager) -> None:
        """Test getting companies by theme."""
        watchlist_manager.load_companies_from_excel()
        
        auto_companies = watchlist_manager.get_companies_by_theme("自動車")
        assert len(auto_companies) == 1
        assert auto_companies[0].code == "7203"
    
    def test_get_companies_by_performance_trend(self, 
                                              watchlist_manager: WatchlistManager) -> None:
        """Test getting companies by performance trend."""
        watchlist_manager.load_companies_from_excel()
        
        profitable_companies = watchlist_manager.get_companies_by_performance_trend("増収増益")
        profitable_codes = [c.code for c in profitable_companies]
        
        assert "7203" in profitable_codes
        assert "6758" in profitable_codes
        assert "7974" in profitable_codes
        assert "9984" not in profitable_codes  # 赤字縮小
    
    def test_get_watchlist_with_info(self, watchlist_manager: WatchlistManager) -> None:
        """Test getting watchlist with complete company information."""
        watchlist_manager.load_companies_from_excel()
        watchlist_manager.update_active_watchlist(["7203", "6758"])
        
        watchlist_info = watchlist_manager.get_watchlist_with_info()
        
        assert len(watchlist_info) == 2
        codes = [company.code for company in watchlist_info]
        assert "7203" in codes
        assert "6758" in codes
    
    def test_export_watchlist_to_dict(self, watchlist_manager: WatchlistManager) -> None:
        """Test exporting watchlist to dictionary."""
        watchlist_manager.load_companies_from_excel()
        watchlist_manager.update_active_watchlist(["7203", "6758"])
        
        export_data = watchlist_manager.export_watchlist_to_dict()
        
        assert export_data["total_companies"] == 2
        assert "7203" in export_data["watchlist_codes"]
        assert "6758" in export_data["watchlist_codes"]
        assert "company_details" in export_data
        assert "7203" in export_data["company_details"]
        assert export_data["company_details"]["7203"]["name"] == "トヨタ自動車"
    
    def test_complex_screening_criteria(self, 
                                      watchlist_manager: WatchlistManager) -> None:
        """Test complex screening with multiple criteria."""
        watchlist_manager.load_companies_from_excel()
        
        # Complex criteria: Large cap + profitable + high volume
        criteria = ScreeningCriteria(
            min_market_cap=10000000,  # >10B yen
            performance_trends=["増収増益"],
            chart_conditions=["高出来高"]
        )
        
        filtered = watchlist_manager.apply_screening(criteria)
        
        # Should only include Toyota (large, profitable, high volume)
        assert "7203" in filtered  # Toyota meets all criteria
        assert "6758" not in filtered  # Sony has low volume
        assert "9984" not in filtered  # SoftBank not profitable
    
    def test_reload_from_excel(self, watchlist_manager: WatchlistManager) -> None:
        """Test reloading data from Excel file."""
        # Initial load
        watchlist_manager.load_companies_from_excel()
        watchlist_manager.update_active_watchlist(["7203", "6758"])
        
        initial_count = len(watchlist_manager.companies)
        initial_watchlist = watchlist_manager.active_watchlist.copy()
        
        # Reload
        result = watchlist_manager.reload_from_excel()
        
        assert result is True
        assert len(watchlist_manager.companies) == initial_count
        assert watchlist_manager.active_watchlist == initial_watchlist