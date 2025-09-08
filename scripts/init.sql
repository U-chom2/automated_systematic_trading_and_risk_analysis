-- データベース初期化スクリプト

-- TimescaleDBエクステンションを有効化
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- スキーマを作成
CREATE SCHEMA IF NOT EXISTS trading;

-- テーブル作成
SET search_path TO trading, public;

-- 株式マスタ
CREATE TABLE IF NOT EXISTS stocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    market VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap DECIMAL(20, 2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ポートフォリオ
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    initial_capital DECIMAL(20, 2) NOT NULL,
    current_capital DECIMAL(20, 2) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 取引履歴
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    stock_id UUID NOT NULL REFERENCES stocks(id),
    ticker VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    price DECIMAL(20, 4) NOT NULL CHECK (price > 0),
    commission DECIMAL(20, 4) DEFAULT 0,
    slippage DECIMAL(20, 4) DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    order_type VARCHAR(20) DEFAULT 'MARKET',
    status VARCHAR(20) DEFAULT 'EXECUTED',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ポジション
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    stock_id UUID NOT NULL REFERENCES stocks(id),
    ticker VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(20, 4) NOT NULL,
    current_price DECIMAL(20, 4),
    stop_loss DECIMAL(20, 4),
    take_profit DECIMAL(20, 4),
    is_open BOOLEAN DEFAULT TRUE,
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(portfolio_id, ticker, is_open)
);

-- シグナル
CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES stocks(id),
    ticker VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    strength DECIMAL(5, 4) CHECK (strength >= -1 AND strength <= 1),
    confidence DECIMAL(5, 4) CHECK (confidence >= 0 AND confidence <= 1),
    source VARCHAR(50),
    metadata JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- 市場データ（時系列）
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    stock_id UUID NOT NULL REFERENCES stocks(id),
    ticker VARCHAR(20) NOT NULL,
    open DECIMAL(20, 4),
    high DECIMAL(20, 4),
    low DECIMAL(20, 4),
    close DECIMAL(20, 4) NOT NULL,
    volume BIGINT,
    adjusted_close DECIMAL(20, 4),
    PRIMARY KEY (time, stock_id)
);

-- TimescaleDBのハイパーテーブルに変換
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

-- インデックスを作成
CREATE INDEX IF NOT EXISTS idx_stocks_ticker ON stocks(ticker);
CREATE INDEX IF NOT EXISTS idx_stocks_market ON stocks(market);
CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks(sector);

CREATE INDEX IF NOT EXISTS idx_trades_portfolio_id ON trades(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_trades_stock_id ON trades(stock_id);
CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);

CREATE INDEX IF NOT EXISTS idx_positions_portfolio_id ON positions(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_positions_stock_id ON positions(stock_id);
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_positions_is_open ON positions(is_open);

CREATE INDEX IF NOT EXISTS idx_signals_stock_id ON signals(stock_id);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_generated_at ON signals(generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_is_active ON signals(is_active);

CREATE INDEX IF NOT EXISTS idx_market_data_ticker ON market_data(ticker, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_stock_id_time ON market_data(stock_id, time DESC);

-- トリガー関数：updated_atを自動更新
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- トリガーを設定
CREATE TRIGGER update_stocks_updated_at BEFORE UPDATE ON stocks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_portfolios_updated_at BEFORE UPDATE ON portfolios
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ビューを作成
CREATE OR REPLACE VIEW portfolio_summary AS
SELECT 
    p.id,
    p.name,
    p.initial_capital,
    p.current_capital,
    COUNT(DISTINCT pos.id) as open_positions,
    SUM(pos.quantity * pos.current_price) as positions_value,
    p.current_capital + COALESCE(SUM(pos.quantity * pos.current_price), 0) as total_value,
    ((p.current_capital + COALESCE(SUM(pos.quantity * pos.current_price), 0) - p.initial_capital) / p.initial_capital * 100) as return_percentage
FROM portfolios p
LEFT JOIN positions pos ON p.id = pos.portfolio_id AND pos.is_open = TRUE
WHERE p.is_active = TRUE
GROUP BY p.id, p.name, p.initial_capital, p.current_capital;

-- 権限設定
GRANT ALL PRIVILEGES ON SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trading_user;