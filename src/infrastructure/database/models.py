"""SQLAlchemyモデル定義"""
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4
from sqlalchemy import (
    Column, String, Integer, Numeric, Boolean, DateTime, Date,
    ForeignKey, Index, UniqueConstraint, Text, JSON
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class StockModel(Base):
    """株式モデル"""
    __tablename__ = "stocks"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    company_name = Column(String(255), nullable=False)
    company_name_jp = Column(String(255))
    exchange = Column(String(20), nullable=False, index=True)
    sector = Column(String(100), index=True)
    industry = Column(String(100))
    market_cap = Column(Numeric(20, 2))
    currency = Column(String(3), default="JPY")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # リレーション
    trades = relationship("TradeModel", back_populates="stock")
    positions = relationship("PositionModel", back_populates="stock")
    market_data = relationship("MarketDataModel", back_populates="stock")
    
    __table_args__ = (
        Index("idx_stock_ticker_exchange", "ticker", "exchange"),
    )


class PortfolioModel(Base):
    """ポートフォリオモデル"""
    __tablename__ = "portfolios"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    initial_capital = Column(Numeric(20, 2), nullable=False)
    current_capital = Column(Numeric(20, 2), nullable=False)
    currency = Column(String(3), default="JPY")
    strategy_type = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # リレーション
    positions = relationship("PositionModel", back_populates="portfolio")
    trades = relationship("TradeModel", back_populates="portfolio")
    performance_history = relationship("PerformanceModel", back_populates="portfolio")


class PositionModel(Base):
    """ポジションモデル"""
    __tablename__ = "positions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id = Column(PGUUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    stock_id = Column(PGUUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    average_cost = Column(Numeric(20, 4), nullable=False)
    current_price = Column(Numeric(20, 4))
    opened_at = Column(DateTime, default=datetime.now)
    closed_at = Column(DateTime)
    
    # リレーション
    portfolio = relationship("PortfolioModel", back_populates="positions")
    stock = relationship("StockModel", back_populates="positions")
    
    __table_args__ = (
        UniqueConstraint("portfolio_id", "stock_id", "opened_at", name="uq_portfolio_stock_position"),
        Index("idx_position_portfolio_ticker", "portfolio_id", "ticker"),
    )


class TradeModel(Base):
    """取引モデル"""
    __tablename__ = "trades"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id = Column(PGUUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    stock_id = Column(PGUUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)  # BUY, SELL, SHORT, COVER
    order_type = Column(String(15), nullable=False)  # MARKET, LIMIT, STOP, STOP_LIMIT
    quantity = Column(Integer, nullable=False)
    order_price = Column(Numeric(20, 4))
    executed_price = Column(Numeric(20, 4))
    commission = Column(Numeric(15, 4), default=0)
    status = Column(String(15), nullable=False, index=True)  # PENDING, SUBMITTED, FILLED, etc.
    signal_id = Column(PGUUID(as_uuid=True))
    execution_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)
    submitted_at = Column(DateTime)
    executed_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    
    # リレーション
    portfolio = relationship("PortfolioModel", back_populates="trades")
    stock = relationship("StockModel", back_populates="trades")
    
    __table_args__ = (
        Index("idx_trade_portfolio_status", "portfolio_id", "status"),
        Index("idx_trade_executed_at", "executed_at"),
    )


class MarketDataModel(Base):
    """市場データモデル（TimescaleDB用）"""
    __tablename__ = "market_data"
    
    time = Column(DateTime, primary_key=True)
    stock_id = Column(PGUUID(as_uuid=True), ForeignKey("stocks.id"), primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    open = Column(Numeric(20, 4), nullable=False)
    high = Column(Numeric(20, 4), nullable=False)
    low = Column(Numeric(20, 4), nullable=False)
    close = Column(Numeric(20, 4), nullable=False)
    volume = Column(Integer, nullable=False)
    interval = Column(String(5), nullable=False, default="1d")  # 1m, 5m, 1h, 1d, etc.
    
    # リレーション
    stock = relationship("StockModel", back_populates="market_data")
    
    __table_args__ = (
        Index("idx_market_data_ticker_time", "ticker", "time"),
        Index("idx_market_data_interval", "interval"),
    )


class SignalModel(Base):
    """シグナルモデル"""
    __tablename__ = "signals"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    stock_id = Column(PGUUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)  # TECHNICAL, FUNDAMENTAL, etc.
    direction = Column(String(10), nullable=False)  # LONG, SHORT, NEUTRAL
    strength = Column(Numeric(5, 2), nullable=False)  # 0-100
    confidence = Column(Numeric(5, 2), nullable=False)  # 0-100
    target_price = Column(Numeric(20, 4))
    stop_loss = Column(Numeric(20, 4))
    time_horizon = Column(String(10), default="medium")
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)
    expires_at = Column(DateTime)
    
    __table_args__ = (
        Index("idx_signal_ticker_created", "ticker", "created_at"),
        Index("idx_signal_type_direction", "signal_type", "direction"),
    )


class PerformanceModel(Base):
    """パフォーマンス履歴モデル"""
    __tablename__ = "performance_history"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id = Column(PGUUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    date = Column(Date, nullable=False)
    total_value = Column(Numeric(20, 2), nullable=False)
    cash_balance = Column(Numeric(20, 2), nullable=False)
    positions_value = Column(Numeric(20, 2), nullable=False)
    daily_return = Column(Numeric(10, 6))
    cumulative_return = Column(Numeric(10, 6))
    realized_pnl = Column(Numeric(20, 2))
    unrealized_pnl = Column(Numeric(20, 2))
    sharpe_ratio = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 6))
    created_at = Column(DateTime, default=datetime.now)
    
    # リレーション
    portfolio = relationship("PortfolioModel", back_populates="performance_history")
    
    __table_args__ = (
        UniqueConstraint("portfolio_id", "date", name="uq_portfolio_performance_date"),
        Index("idx_performance_portfolio_date", "portfolio_id", "date"),
    )


class NewsModel(Base):
    """ニュースモデル"""
    __tablename__ = "news"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(Text, nullable=False)
    content = Column(Text)
    source = Column(String(100))
    url = Column(Text)
    ticker = Column(String(10), index=True)
    sentiment_score = Column(Numeric(5, 4))  # -1.0 to 1.0
    relevance_score = Column(Numeric(5, 4))  # 0.0 to 1.0
    published_at = Column(DateTime, nullable=False)
    fetched_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index("idx_news_ticker_published", "ticker", "published_at"),
        Index("idx_news_sentiment", "sentiment_score"),
    )


class AIModelModel(Base):
    """AIモデルメタデータモデル"""
    __tablename__ = "ai_models"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    model_name = Column(String(255), nullable=False, unique=True)
    model_type = Column(String(50), nullable=False)  # PPO, DQN, BERT, etc.
    version = Column(String(20), nullable=False)
    description = Column(Text)
    input_features = Column(JSON, nullable=False)
    output_dimension = Column(Integer, nullable=False)
    hyperparameters = Column(JSON)
    training_metrics = Column(JSON)
    model_path = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        Index("idx_ai_model_type_active", "model_type", "is_active"),
    )