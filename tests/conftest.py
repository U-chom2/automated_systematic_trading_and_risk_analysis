"""Pytest configuration and fixtures for the test suite."""

import pytest
import sys
from pathlib import Path
from decimal import Decimal
from typing import Dict, Any, List

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture
def sample_capital() -> Decimal:
    """Fixture for sample capital amount."""
    return Decimal("100000")  # 10万円


@pytest.fixture
def sample_stock_price() -> Decimal:
    """Fixture for sample stock price."""
    return Decimal("1000")  # 1000円


@pytest.fixture
def sample_api_keys() -> Dict[str, str]:
    """Fixture for sample API keys (for testing only)."""
    return {
        "x_api_key": "test_x_api_key",
        "x_api_secret": "test_x_api_secret",
        "x_access_token": "test_x_access_token",
        "x_access_token_secret": "test_x_access_token_secret",
        "broker_api_key": "test_broker_api_key",
        "broker_secret": "test_broker_secret",
        "broker_api_endpoint": "https://api.test-broker.com",
        "paper_trading": True
    }


@pytest.fixture
def sample_ir_data() -> Dict[str, Any]:
    """Fixture for sample IR/press release data."""
    return {
        "title": "2024年3月期決算上方修正に関するお知らせ",
        "content": "当社は、2024年3月期の業績予想を上方修正いたします。",
        "company_code": "7203",
        "timestamp": "2024-01-15T15:00:00",
        "category": "決算"
    }


@pytest.fixture
def sample_price_history() -> List[Dict[str, Any]]:
    """Fixture for sample price history data."""
    return [
        {
            "symbol": "7203",
            "date": "2024-01-15",
            "open": Decimal("980"),
            "high": Decimal("1020"),
            "low": Decimal("975"),
            "close": Decimal("1000"),
            "volume": 1000000
        },
        {
            "symbol": "7203",
            "date": "2024-01-14",
            "open": Decimal("990"),
            "high": Decimal("1010"),
            "low": Decimal("985"),
            "close": Decimal("995"),
            "volume": 800000
        }
    ]


@pytest.fixture
def sample_social_posts() -> List[str]:
    """Fixture for sample social media posts."""
    return [
        "トヨタの決算が好調らしい #7203",
        "7203は今後も上昇しそう",
        "トヨタ株を買い増し予定 $7203"
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify collected test items."""
    # Mark tests without explicit markers as unit tests
    for item in items:
        if not any(marker.name in ["unit", "integration", "slow"] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)