"""ユーティリティ関数"""
from typing import Any, Dict, List, Optional, TypeVar, Union
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from uuid import UUID, uuid4
import hashlib
import json
import base64
from functools import wraps
import time
import asyncio
from pathlib import Path


T = TypeVar("T")


# 日付・時刻関連
def now_utc() -> datetime:
    """現在のUTC時刻を取得"""
    return datetime.utcnow()


def now_jst() -> datetime:
    """現在のJST時刻を取得"""
    return datetime.utcnow() + timedelta(hours=9)


def to_jst(dt: datetime) -> datetime:
    """UTC時刻をJSTに変換"""
    return dt + timedelta(hours=9)


def from_jst(dt: datetime) -> datetime:
    """JST時刻をUTCに変換"""
    return dt - timedelta(hours=9)


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """市場が開いているかチェック（東京証券取引所）
    
    Args:
        dt: チェック対象時刻（JST）
    
    Returns:
        市場が開いているか
    """
    if dt is None:
        dt = now_jst()
    
    # 土日は休場
    if dt.weekday() >= 5:
        return False
    
    # 取引時間: 9:00-11:30, 12:30-15:00
    time = dt.time()
    morning_open = time >= datetime.strptime("09:00", "%H:%M").time()
    morning_close = time <= datetime.strptime("11:30", "%H:%M").time()
    afternoon_open = time >= datetime.strptime("12:30", "%H:%M").time()
    afternoon_close = time <= datetime.strptime("15:00", "%H:%M").time()
    
    return (morning_open and morning_close) or (afternoon_open and afternoon_close)


def get_next_market_open() -> datetime:
    """次の市場開始時刻を取得（JST）"""
    dt = now_jst()
    
    # 今日が平日で9時前なら今日の9時
    if dt.weekday() < 5 and dt.hour < 9:
        return dt.replace(hour=9, minute=0, second=0, microsecond=0)
    
    # 次の平日の9時を探す
    dt = dt + timedelta(days=1)
    while dt.weekday() >= 5:
        dt = dt + timedelta(days=1)
    
    return dt.replace(hour=9, minute=0, second=0, microsecond=0)


def date_range(start_date: date, end_date: date) -> List[date]:
    """日付範囲を生成"""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def business_days(start_date: date, end_date: date) -> List[date]:
    """営業日のみを取得"""
    dates = date_range(start_date, end_date)
    return [d for d in dates if d.weekday() < 5]


# 数値・金額関連
def round_price(price: Decimal, tick_size: Decimal = Decimal("1")) -> Decimal:
    """価格を呼値単位に丸める"""
    return (price / tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick_size


def format_currency(amount: Union[float, Decimal], currency: str = "JPY") -> str:
    """通貨フォーマット"""
    if currency == "JPY":
        return f"¥{amount:,.0f}"
    elif currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def calculate_return(initial: float, final: float) -> float:
    """収益率を計算"""
    if initial == 0:
        return 0.0
    return (final - initial) / initial


def calculate_annualized_return(
    total_return: float, days: int
) -> float:
    """年率換算収益率を計算"""
    if days == 0:
        return 0.0
    years = days / 365.25
    return (1 + total_return) ** (1 / years) - 1


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.01,
    periods_per_year: int = 252
) -> float:
    """シャープレシオを計算"""
    if not returns:
        return 0.0
    
    import numpy as np
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / periods_per_year
    
    if len(excess_returns) < 2:
        return 0.0
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    return mean_excess / std_excess * np.sqrt(periods_per_year)


def calculate_max_drawdown(values: List[float]) -> float:
    """最大ドローダウンを計算"""
    if not values:
        return 0.0
    
    max_value = values[0]
    max_dd = 0.0
    
    for value in values:
        max_value = max(max_value, value)
        drawdown = (max_value - value) / max_value if max_value > 0 else 0
        max_dd = max(max_dd, drawdown)
    
    return max_dd


# 文字列・ハッシュ関連
def generate_id() -> str:
    """一意のIDを生成"""
    return str(uuid4())


def hash_string(text: str) -> str:
    """文字列をハッシュ化"""
    return hashlib.sha256(text.encode()).hexdigest()


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """文字列を切り詰める"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def to_snake_case(text: str) -> str:
    """キャメルケースをスネークケースに変換"""
    import re
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", text)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def to_camel_case(text: str) -> str:
    """スネークケースをキャメルケースに変換"""
    components = text.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


# JSON関連
def json_encode(obj: Any) -> str:
    """オブジェクトをJSON文字列に変換"""
    def default(o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        elif isinstance(o, Decimal):
            return float(o)
        elif isinstance(o, UUID):
            return str(o)
        elif hasattr(o, "to_dict"):
            return o.to_dict()
        elif hasattr(o, "__dict__"):
            return o.__dict__
        else:
            return str(o)
    
    return json.dumps(obj, default=default, ensure_ascii=False)


def json_decode(text: str) -> Any:
    """JSON文字列をオブジェクトに変換"""
    return json.loads(text)


def base64_encode(data: bytes) -> str:
    """Base64エンコード"""
    return base64.b64encode(data).decode("utf-8")


def base64_decode(text: str) -> bytes:
    """Base64デコード"""
    return base64.b64decode(text.encode("utf-8"))


# デコレーター
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """リトライデコレーター
    
    Args:
        max_attempts: 最大試行回数
        delay: 初回遅延（秒）
        backoff: 遅延の倍率
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def memoize(func):
    """メモ化デコレーター"""
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def timing(func):
    """実行時間計測デコレーター"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            print(f"{func.__name__} took {elapsed:.3f} seconds")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            print(f"{func.__name__} took {elapsed:.3f} seconds")
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# ファイル・パス関連
def ensure_directory(path: Path) -> Path:
    """ディレクトリが存在しない場合は作成"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_file(path: Path) -> Dict[str, Any]:
    """JSONファイルを読み込む"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, data: Dict[str, Any]) -> None:
    """JSONファイルに書き込む"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_file_size(path: Path) -> int:
    """ファイルサイズを取得（バイト）"""
    return path.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """ファイルサイズを人間が読みやすい形式にフォーマット"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


# バッチ処理
def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """リストをチャンクに分割"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested: List[List[T]]) -> List[T]:
    """ネストしたリストをフラット化"""
    return [item for sublist in nested for item in sublist]


def deduplicate_list(lst: List[T], key: Optional[callable] = None) -> List[T]:
    """リストから重複を除去（順序を保持）"""
    seen = set()
    result = []
    for item in lst:
        k = key(item) if key else item
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


# 辞書操作
def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """辞書を深くマージ"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def filter_none_values(d: Dict) -> Dict:
    """Noneの値を除外"""
    return {k: v for k, v in d.items() if v is not None}


def get_nested_value(d: Dict, path: str, default: Any = None) -> Any:
    """ネストした辞書から値を取得
    
    Args:
        d: 辞書
        path: パス（ドット区切り）
        default: デフォルト値
    
    Returns:
        値
    """
    keys = path.split(".")
    value = d
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value