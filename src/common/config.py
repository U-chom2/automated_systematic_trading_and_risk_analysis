"""設定管理モジュール"""
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, PostgresDsn, RedisDsn, Field
from functools import lru_cache
import os
from pathlib import Path


class Settings(BaseSettings):
    """アプリケーション設定"""
    
    # 基本設定
    app_name: str = "Automated Trading System"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API設定
    api_prefix: str = "/api/v1"
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # データベース設定
    postgres_url: PostgresDsn = Field(
        default="postgresql+asyncpg://user:password@localhost/trading_db",
        env="DATABASE_URL"
    )
    redis_url: RedisDsn = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # TimescaleDB設定
    enable_timescaledb: bool = Field(default=True, env="ENABLE_TIMESCALEDB")
    timescale_chunk_interval: str = Field(default="1 month", env="TIMESCALE_CHUNK_INTERVAL")
    
    # 市場データプロバイダー設定
    yfinance_enabled: bool = Field(default=True, env="YFINANCE_ENABLED")
    yahoo_finance_api_key: Optional[str] = Field(default=None, env="YAHOO_FINANCE_API_KEY")
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    quandl_api_key: Optional[str] = Field(default=None, env="QUANDL_API_KEY")
    
    # AI/ML設定
    model_path: Path = Field(
        default=Path("./models"),
        env="MODEL_PATH"
    )
    ppo_model_checkpoint: Optional[str] = Field(default=None, env="PPO_MODEL_CHECKPOINT")
    sentiment_model_name: str = Field(
        default="cl-tohoku/bert-base-japanese-v3",
        env="SENTIMENT_MODEL_NAME"
    )
    
    # 取引設定
    default_commission_rate: float = Field(default=0.001, env="DEFAULT_COMMISSION_RATE")
    default_slippage_rate: float = Field(default=0.0005, env="DEFAULT_SLIPPAGE_RATE")
    max_positions: int = Field(default=20, env="MAX_POSITIONS")
    min_position_size: float = Field(default=10000.0, env="MIN_POSITION_SIZE")
    max_position_size: float = Field(default=1000000.0, env="MAX_POSITION_SIZE")
    
    # リスク管理設定
    max_portfolio_risk: float = Field(default=0.15, env="MAX_PORTFOLIO_RISK")
    max_position_risk: float = Field(default=0.02, env="MAX_POSITION_RISK")
    stop_loss_percentage: float = Field(default=0.05, env="STOP_LOSS_PERCENTAGE")
    take_profit_percentage: float = Field(default=0.10, env="TAKE_PROFIT_PERCENTAGE")
    
    # バックテスト設定
    backtest_initial_capital: float = Field(default=10000000.0, env="BACKTEST_INITIAL_CAPITAL")
    backtest_rebalance_frequency: str = Field(default="weekly", env="BACKTEST_REBALANCE_FREQUENCY")
    
    # ログ設定
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_file: Optional[Path] = Field(default=None, env="LOG_FILE")
    
    # セキュリティ設定
    secret_key: str = Field(
        default="your-secret-key-here-change-in-production",
        env="SECRET_KEY"
    )
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # キャッシュ設定
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # seconds
    market_data_cache_ttl: int = Field(default=60, env="MARKET_DATA_CACHE_TTL")
    
    # モニタリング設定
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=8001, env="METRICS_PORT")
    
    # ワーカー設定
    worker_count: int = Field(default=4, env="WORKER_COUNT")
    worker_class: str = Field(default="uvicorn.workers.UvicornWorker", env="WORKER_CLASS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_database_url(self, async_driver: bool = True) -> str:
        """データベースURLを取得"""
        if async_driver:
            return str(self.postgres_url)
        else:
            # 同期ドライバー用のURL
            return str(self.postgres_url).replace("+asyncpg", "")
    
    def get_redis_url(self) -> str:
        """Redis URLを取得"""
        return str(self.redis_url)
    
    def is_production(self) -> bool:
        """本番環境かどうか"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """開発環境かどうか"""
        return self.environment == "development"
    
    def is_testing(self) -> bool:
        """テスト環境かどうか"""
        return self.environment == "testing"
    
    def get_model_path(self, model_type: str) -> Path:
        """モデルパスを取得"""
        return self.model_path / model_type
    
    def get_log_config(self) -> Dict[str, Any]:
        """ログ設定を取得"""
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": self.log_format,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": self.log_level,
                }
            },
            "root": {
                "level": self.log_level,
                "handlers": ["console"],
            },
        }
        
        if self.log_file:
            config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "filename": str(self.log_file),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": self.log_level,
            }
            config["root"]["handlers"].append("file")
        
        return config


@lru_cache()
def get_settings() -> Settings:
    """設定シングルトンを取得"""
    return Settings()


# グローバル設定インスタンス
settings = get_settings()