"""テスト共通設定"""
import pytest
import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator, Dict, Any, List
from datetime import datetime, date
from decimal import Decimal
from uuid import uuid4

# srcディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.database.connection import DatabaseConnection
from src.infrastructure.cache.redis_client import RedisClient
from src.common.config import Settings


@pytest.fixture(scope="session")
def event_loop():
    """イベントループフィクスチャ"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """テスト用設定"""
    return Settings(
        environment="testing",
        debug=True,
        postgres_url="postgresql+asyncpg://test:test@localhost/test_trading_db",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
    )


@pytest.fixture
async def db_connection(test_settings) -> AsyncGenerator[DatabaseConnection, None]:
    """データベース接続フィクスチャ"""
    connection = DatabaseConnection(test_settings.get_database_url())
    await connection.connect()
    
    # テスト用のテーブルを作成
    await connection.execute("CREATE SCHEMA IF NOT EXISTS test")
    
    yield connection
    
    # クリーンアップ
    await connection.execute("DROP SCHEMA IF EXISTS test CASCADE")
    await connection.disconnect()


@pytest.fixture
async def redis_client(test_settings) -> AsyncGenerator[RedisClient, None]:
    """Redisクライアントフィクスチャ"""
    client = RedisClient(test_settings.get_redis_url())
    await client.connect()
    
    yield client
    
    # クリーンアップ
    await client.flush_all()
    await client.disconnect()


@pytest.fixture
def sample_portfolio_data():
    """サンプルポートフォリオデータ"""
    return {
        "id": uuid4(),
        "name": "Test Portfolio",
        "initial_capital": Decimal("10000000"),
        "current_capital": Decimal("10000000"),
        "is_active": True,
        "created_at": datetime.utcnow(),
    }


@pytest.fixture
def sample_stock_data():
    """サンプル株式データ"""
    return {
        "id": uuid4(),
        "ticker": "7203.T",
        "name": "トヨタ自動車",
        "market": "TSE",
        "sector": "Automobile",
        "is_active": True,
    }


@pytest.fixture
def sample_trade_data():
    """サンプル取引データ"""
    return {
        "id": uuid4(),
        "portfolio_id": uuid4(),
        "ticker": "7203.T",
        "side": "BUY",
        "quantity": 100,
        "price": Decimal("2500"),
        "commission": Decimal("250"),
        "executed_at": datetime.utcnow(),
    }


@pytest.fixture
def sample_signal_data():
    """サンプルシグナルデータ"""
    return {
        "id": uuid4(),
        "ticker": "7203.T",
        "signal_type": "BUY",
        "strength": 0.85,
        "confidence": 0.92,
        "source": "AI_MODEL",
        "generated_at": datetime.utcnow(),
    }


@pytest.fixture
def sample_market_data():
    """サンプル市場データ"""
    return {
        "ticker": "7203.T",
        "date": date.today(),
        "open": Decimal("2480"),
        "high": Decimal("2520"),
        "low": Decimal("2470"),
        "close": Decimal("2500"),
        "volume": 1000000,
    }


# 既存のフィクスチャも維持（互換性のため）
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