"""Main entry point for AI Day Trading System."""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.system_core.trading_system import TradingSystem, SystemConfig


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("trading_system.log")
        ]
    )


def load_config() -> SystemConfig:
    """Load system configuration."""
    return SystemConfig(
        capital=Decimal("100000"),  # 10万円
        risk_per_trade=0.01,  # 1%
        buy_threshold=80,  # 80点以上で買い
        max_positions=5,
        tdnet_polling_interval=1,  # 1秒
        price_update_interval=5,  # 5秒
        log_level="INFO"
    )


def load_api_keys() -> Dict[str, Any]:
    """Load API keys from environment or config file."""
    # TODO: Load from environment variables or secure config
    return {
        "x_api_key": "your_x_api_key",
        "x_api_secret": "your_x_api_secret",
        "x_access_token": "your_x_access_token",
        "x_access_token_secret": "your_x_access_token_secret",
        "broker_api_key": "your_broker_api_key",
        "broker_secret": "your_broker_secret",
        "broker_api_endpoint": "https://api.your-broker.com",
        "paper_trading": True
    }


def load_watchlist() -> list[str]:
    """Load watchlist of symbols to monitor."""
    # TODO: Load from database or config file
    return [
        "7203",  # トヨタ自動車
        "6758",  # ソニーグループ
        "9984",  # ソフトバンクグループ
        "7974",  # 任天堂
        "4503"   # アステラス製薬
    ]


async def main() -> None:
    """Main application entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting AI Day Trading System...")
    
    try:
        # Load configuration
        config = load_config()
        api_keys = load_api_keys()
        watchlist = load_watchlist()
        
        # Initialize trading system
        trading_system = TradingSystem(config)
        
        # Initialize components
        if not await trading_system.initialize_components(api_keys):
            logger.error("Failed to initialize system components")
            return
        
        # Load watchlist
        trading_system.load_watchlist(watchlist)
        
        # Start monitoring
        logger.info("System initialized successfully. Starting monitoring...")
        await trading_system.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down...")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        logger.info("AI Day Trading System stopped.")


def demo_mode() -> None:
    """Run system in demo mode (no real trading)."""
    print("=== AI Day Trading System Demo ===")
    print()
    print("This is a demo of the AI Day Trading System framework.")
    print("All modules are implemented as dummy functions for now.")
    print()
    print("System Components:")
    print("1. DataCollector - Collect data from TDnet, X, and price APIs")
    print("2. AnalysisEngine - NLP, Technical Analysis, Risk Model")
    print("3. ExecutionManager - Order management and position tracking")
    print("4. SystemCore - Main orchestration and workflow management")
    print()
    print("Next Steps:")
    print("- Implement actual data collection logic")
    print("- Train machine learning models")
    print("- Connect to real broker APIs")
    print("- Implement comprehensive backtesting")
    print()
    print("To run tests: pytest tests/")
    print("To start development: Follow TDD approach in CLAUDE.md")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    else:
        asyncio.run(main())
