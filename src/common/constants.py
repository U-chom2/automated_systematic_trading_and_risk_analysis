"""共通定数定義"""
from enum import Enum
from decimal import Decimal


# 市場関連
class Market(str, Enum):
    """市場"""
    TSE = "TSE"  # 東京証券取引所
    NYSE = "NYSE"  # ニューヨーク証券取引所
    NASDAQ = "NASDAQ"  # ナスダック
    OTHER = "OTHER"


class MarketSegment(str, Enum):
    """市場区分"""
    PRIME = "PRIME"  # プライム市場
    STANDARD = "STANDARD"  # スタンダード市場
    GROWTH = "GROWTH"  # グロース市場
    TOPIX = "TOPIX"  # TOPIX
    NIKKEI225 = "NIKKEI225"  # 日経225


# 取引関連
class OrderType(str, Enum):
    """注文タイプ"""
    MARKET = "MARKET"  # 成行
    LIMIT = "LIMIT"  # 指値
    STOP = "STOP"  # 逆指値
    STOP_LIMIT = "STOP_LIMIT"  # 逆指値指値


class OrderSide(str, Enum):
    """売買区分"""
    BUY = "BUY"  # 買い
    SELL = "SELL"  # 売り


class OrderStatus(str, Enum):
    """注文ステータス"""
    PENDING = "PENDING"  # 待機中
    SUBMITTED = "SUBMITTED"  # 発注済み
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # 部分約定
    FILLED = "FILLED"  # 約定済み
    CANCELLED = "CANCELLED"  # キャンセル
    REJECTED = "REJECTED"  # 拒否
    EXPIRED = "EXPIRED"  # 期限切れ


class ExecutionType(str, Enum):
    """執行条件"""
    DAY = "DAY"  # 当日限り
    GTC = "GTC"  # 無期限
    IOC = "IOC"  # 即時約定または失効
    FOK = "FOK"  # 全量約定または失効


# ポジション関連
class PositionSide(str, Enum):
    """ポジション方向"""
    LONG = "LONG"  # ロング
    SHORT = "SHORT"  # ショート


class PositionStatus(str, Enum):
    """ポジションステータス"""
    OPEN = "OPEN"  # オープン
    CLOSED = "CLOSED"  # クローズ
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"  # 部分決済


# シグナル関連
class SignalType(str, Enum):
    """シグナルタイプ"""
    BUY = "BUY"  # 買いシグナル
    SELL = "SELL"  # 売りシグナル
    HOLD = "HOLD"  # 保有継続
    NEUTRAL = "NEUTRAL"  # 中立


class SignalSource(str, Enum):
    """シグナルソース"""
    TECHNICAL = "TECHNICAL"  # テクニカル分析
    FUNDAMENTAL = "FUNDAMENTAL"  # ファンダメンタル分析
    SENTIMENT = "SENTIMENT"  # センチメント分析
    AI_MODEL = "AI_MODEL"  # AIモデル
    COMBINED = "COMBINED"  # 複合


class SignalStrength(str, Enum):
    """シグナル強度"""
    VERY_STRONG = "VERY_STRONG"  # 非常に強い
    STRONG = "STRONG"  # 強い
    MODERATE = "MODERATE"  # 中程度
    WEAK = "WEAK"  # 弱い
    VERY_WEAK = "VERY_WEAK"  # 非常に弱い


# 戦略関連
class StrategyType(str, Enum):
    """戦略タイプ"""
    MOMENTUM = "MOMENTUM"  # モメンタム
    MEAN_REVERSION = "MEAN_REVERSION"  # 平均回帰
    ARBITRAGE = "ARBITRAGE"  # 裁定取引
    PAIRS_TRADING = "PAIRS_TRADING"  # ペアトレード
    CORE_SATELLITE = "CORE_SATELLITE"  # コアサテライト
    AI_DRIVEN = "AI_DRIVEN"  # AI駆動


class RebalanceFrequency(str, Enum):
    """リバランス頻度"""
    DAILY = "DAILY"  # 日次
    WEEKLY = "WEEKLY"  # 週次
    MONTHLY = "MONTHLY"  # 月次
    QUARTERLY = "QUARTERLY"  # 四半期
    ANNUALLY = "ANNUALLY"  # 年次
    NEVER = "NEVER"  # なし


# リスク関連
class RiskLevel(str, Enum):
    """リスクレベル"""
    VERY_LOW = "VERY_LOW"  # 非常に低い
    LOW = "LOW"  # 低い
    MEDIUM = "MEDIUM"  # 中程度
    HIGH = "HIGH"  # 高い
    VERY_HIGH = "VERY_HIGH"  # 非常に高い


class RiskMetric(str, Enum):
    """リスク指標"""
    VAR = "VAR"  # Value at Risk
    CVAR = "CVAR"  # Conditional VaR
    SHARPE_RATIO = "SHARPE_RATIO"  # シャープレシオ
    SORTINO_RATIO = "SORTINO_RATIO"  # ソルティノレシオ
    MAX_DRAWDOWN = "MAX_DRAWDOWN"  # 最大ドローダウン
    BETA = "BETA"  # ベータ
    VOLATILITY = "VOLATILITY"  # ボラティリティ


# 時間枠関連
class TimeFrame(str, Enum):
    """時間枠"""
    TICK = "TICK"  # ティック
    SECOND_1 = "1s"  # 1秒
    MINUTE_1 = "1m"  # 1分
    MINUTE_5 = "5m"  # 5分
    MINUTE_15 = "15m"  # 15分
    MINUTE_30 = "30m"  # 30分
    HOUR_1 = "1h"  # 1時間
    HOUR_4 = "4h"  # 4時間
    DAY_1 = "1d"  # 1日
    WEEK_1 = "1w"  # 1週間
    MONTH_1 = "1M"  # 1月


# テクニカル指標
class TechnicalIndicator(str, Enum):
    """テクニカル指標"""
    SMA = "SMA"  # 単純移動平均
    EMA = "EMA"  # 指数移動平均
    RSI = "RSI"  # RSI
    MACD = "MACD"  # MACD
    BOLLINGER_BANDS = "BB"  # ボリンジャーバンド
    STOCHASTIC = "STOCH"  # ストキャスティクス
    ATR = "ATR"  # Average True Range
    ADX = "ADX"  # Average Directional Index
    ICHIMOKU = "ICHIMOKU"  # 一目均衡表
    VOLUME = "VOLUME"  # 出来高


# デフォルト値
class DefaultValues:
    """デフォルト値"""
    
    # 取引関連
    MIN_ORDER_SIZE = 100  # 最小注文数（株）
    DEFAULT_COMMISSION_RATE = Decimal("0.001")  # デフォルト手数料率（0.1%）
    DEFAULT_SLIPPAGE_RATE = Decimal("0.0005")  # デフォルトスリッページ率（0.05%）
    
    # ポートフォリオ関連
    DEFAULT_INITIAL_CAPITAL = Decimal("10000000")  # デフォルト初期資金（1000万円）
    MAX_POSITION_SIZE = Decimal("0.1")  # 最大ポジションサイズ（資金の10%）
    MAX_POSITIONS = 20  # 最大ポジション数
    
    # リスク管理
    DEFAULT_STOP_LOSS = Decimal("0.05")  # デフォルトストップロス（5%）
    DEFAULT_TAKE_PROFIT = Decimal("0.10")  # デフォルト利益確定（10%）
    MAX_PORTFOLIO_RISK = Decimal("0.15")  # 最大ポートフォリオリスク（15%）
    MAX_DAILY_LOSS = Decimal("0.03")  # 最大日次損失（3%）
    
    # バックテスト
    BACKTEST_START_YEAR = 2020  # バックテスト開始年
    BACKTEST_LOOKBACK_DAYS = 365  # バックテスト参照期間（日）
    
    # キャッシュ
    CACHE_TTL_SECONDS = 300  # キャッシュ有効期限（秒）
    MARKET_DATA_CACHE_TTL = 60  # 市場データキャッシュ有効期限（秒）
    
    # API
    API_RATE_LIMIT = 100  # APIレート制限（リクエスト/分）
    API_TIMEOUT_SECONDS = 30  # APIタイムアウト（秒）
    
    # モデル
    MODEL_CONFIDENCE_THRESHOLD = 0.7  # モデル信頼度閾値
    SENTIMENT_SCORE_THRESHOLD = 0.6  # センチメントスコア閾値


# エラーコード
class ErrorCode(str, Enum):
    """エラーコード"""
    
    # 一般エラー（1000番台）
    UNKNOWN_ERROR = "E1000"
    VALIDATION_ERROR = "E1001"
    NOT_FOUND = "E1002"
    ALREADY_EXISTS = "E1003"
    PERMISSION_DENIED = "E1004"
    
    # 取引エラー（2000番台）
    INSUFFICIENT_FUNDS = "E2001"
    INVALID_ORDER = "E2002"
    ORDER_NOT_FOUND = "E2003"
    POSITION_NOT_FOUND = "E2004"
    MARKET_CLOSED = "E2005"
    
    # データエラー（3000番台）
    DATA_NOT_AVAILABLE = "E3001"
    INVALID_DATA_FORMAT = "E3002"
    DATA_FETCH_ERROR = "E3003"
    
    # システムエラー（4000番台）
    DATABASE_ERROR = "E4001"
    CACHE_ERROR = "E4002"
    EXTERNAL_API_ERROR = "E4003"
    MODEL_ERROR = "E4004"
    
    # リスクエラー（5000番台）
    RISK_LIMIT_EXCEEDED = "E5001"
    STOP_LOSS_TRIGGERED = "E5002"
    MARGIN_CALL = "E5003"


# メッセージテンプレート
class MessageTemplate:
    """メッセージテンプレート"""
    
    # 成功メッセージ
    ORDER_CREATED = "注文を作成しました: {order_id}"
    ORDER_EXECUTED = "注文を執行しました: {order_id}"
    ORDER_CANCELLED = "注文をキャンセルしました: {order_id}"
    POSITION_OPENED = "ポジションをオープンしました: {ticker}"
    POSITION_CLOSED = "ポジションをクローズしました: {ticker}"
    
    # エラーメッセージ
    INSUFFICIENT_FUNDS_MSG = "資金が不足しています: 必要額={required}, 利用可能額={available}"
    INVALID_ORDER_MSG = "無効な注文です: {reason}"
    RISK_LIMIT_MSG = "リスク限度を超えています: {risk_type}"
    
    # 警告メッセージ
    HIGH_VOLATILITY_WARNING = "高ボラティリティを検出: {ticker}"
    LOW_LIQUIDITY_WARNING = "低流動性を検出: {ticker}"
    STOP_LOSS_WARNING = "ストップロスに接近: {ticker}"