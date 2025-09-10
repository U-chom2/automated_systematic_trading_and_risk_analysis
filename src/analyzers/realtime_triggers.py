"""リアルタイム分析トリガーシステム

市場の変動や条件に基づいて自動的に分析を実行するトリガーシステム
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import yfinance as yf
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor

from src.utils.logger_utils import create_dual_logger
from .news_collector import NewsCollector, NewsItem
from .sentiment_analyzer import ModernBERTSentimentAnalyzer
from .stock_analyzer import StockAnalyzer

logger = create_dual_logger(__name__, console_output=True)


class TriggerType(Enum):
    """トリガータイプ"""
    PRICE_CHANGE = "price_change"
    VOLUME_SPIKE = "volume_spike"
    RSI_EXTREME = "rsi_extreme"
    NEWS_ALERT = "news_alert"
    SENTIMENT_CHANGE = "sentiment_change"
    TECHNICAL_SIGNAL = "technical_signal"


@dataclass
class TriggerCondition:
    """トリガー条件"""
    trigger_type: TriggerType
    ticker: str
    company_name: str
    threshold: float  # 閾値（価格変動率、RSI値など）
    time_window: int = 300  # 監視時間窓（秒）
    cooldown: int = 1800  # クールダウン期間（秒）
    is_active: bool = True
    last_triggered: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerEvent:
    """トリガーイベント"""
    condition: TriggerCondition
    triggered_at: datetime
    value: float  # トリガーされた値
    context: Dict[str, Any]  # 追加コンテキスト情報
    analysis_result: Optional[Any] = None


class RealtimeAnalysisTriggers:
    """リアルタイム分析トリガーシステム"""
    
    def __init__(
        self, 
        stock_analyzer: StockAnalyzer,
        news_collector: Optional[NewsCollector] = None,
        sentiment_analyzer: Optional[ModernBERTSentimentAnalyzer] = None,
        max_workers: int = 4
    ):
        """初期化
        
        Args:
            stock_analyzer: 株式分析エンジン
            news_collector: ニュース収集システム
            sentiment_analyzer: センチメント分析システム
            max_workers: 並行実行可能な最大ワーカー数
        """
        self.stock_analyzer = stock_analyzer
        self.news_collector = news_collector or NewsCollector()
        self.sentiment_analyzer = sentiment_analyzer or ModernBERTSentimentAnalyzer()
        
        # トリガー管理
        self.trigger_conditions: Dict[str, TriggerCondition] = {}
        self.active_monitors: Set[str] = set()
        self.trigger_callbacks: List[Callable[[TriggerEvent], None]] = []
        
        # 実行管理
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # キャッシュ
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.volume_cache: Dict[str, List[float]] = {}
        self.sentiment_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("リアルタイムトリガーシステム初期化完了")
    
    def add_price_change_trigger(
        self,
        ticker: str,
        company_name: str,
        threshold_percent: float,
        time_window: int = 300,
        cooldown: int = 1800
    ) -> str:
        """価格変動トリガーを追加
        
        Args:
            ticker: ティッカーシンボル
            company_name: 企業名
            threshold_percent: 変動率閾値（%）
            time_window: 監視時間窓（秒）
            cooldown: クールダウン期間（秒）
            
        Returns:
            トリガーID
        """
        trigger_id = f"price_{ticker}_{threshold_percent}"
        
        condition = TriggerCondition(
            trigger_type=TriggerType.PRICE_CHANGE,
            ticker=ticker,
            company_name=company_name,
            threshold=threshold_percent,
            time_window=time_window,
            cooldown=cooldown,
            metadata={"direction": "both"}  # both, up, down
        )
        
        self.trigger_conditions[trigger_id] = condition
        logger.info(f"価格変動トリガー追加: {ticker} ±{threshold_percent}%")
        
        return trigger_id
    
    def add_volume_spike_trigger(
        self,
        ticker: str,
        company_name: str,
        volume_multiplier: float,
        time_window: int = 300,
        cooldown: int = 1800
    ) -> str:
        """出来高急増トリガーを追加
        
        Args:
            ticker: ティッカーシンボル
            company_name: 企業名
            volume_multiplier: 平均出来高に対する倍数
            time_window: 監視時間窓（秒）
            cooldown: クールダウン期間（秒）
            
        Returns:
            トリガーID
        """
        trigger_id = f"volume_{ticker}_{volume_multiplier}"
        
        condition = TriggerCondition(
            trigger_type=TriggerType.VOLUME_SPIKE,
            ticker=ticker,
            company_name=company_name,
            threshold=volume_multiplier,
            time_window=time_window,
            cooldown=cooldown
        )
        
        self.trigger_conditions[trigger_id] = condition
        logger.info(f"出来高急増トリガー追加: {ticker} {volume_multiplier}x")
        
        return trigger_id
    
    def add_rsi_extreme_trigger(
        self,
        ticker: str,
        company_name: str,
        rsi_threshold: float,
        extreme_type: str = "both",  # oversold, overbought, both
        cooldown: int = 3600
    ) -> str:
        """RSI極値トリガーを追加
        
        Args:
            ticker: ティッカーシンボル
            company_name: 企業名
            rsi_threshold: RSI閾値（30=売られすぎ, 70=買われすぎ）
            extreme_type: 極値タイプ
            cooldown: クールダウン期間（秒）
            
        Returns:
            トリガーID
        """
        trigger_id = f"rsi_{ticker}_{rsi_threshold}_{extreme_type}"
        
        condition = TriggerCondition(
            trigger_type=TriggerType.RSI_EXTREME,
            ticker=ticker,
            company_name=company_name,
            threshold=rsi_threshold,
            cooldown=cooldown,
            metadata={"extreme_type": extreme_type}
        )
        
        self.trigger_conditions[trigger_id] = condition
        logger.info(f"RSI極値トリガー追加: {ticker} RSI {rsi_threshold} ({extreme_type})")
        
        return trigger_id
    
    def add_news_alert_trigger(
        self,
        ticker: str,
        company_name: str,
        importance_threshold: float = 0.7,
        cooldown: int = 3600
    ) -> str:
        """ニュースアラートトリガーを追加
        
        Args:
            ticker: ティッカーシンボル
            company_name: 企業名
            importance_threshold: 重要度閾値
            cooldown: クールダウン期間（秒）
            
        Returns:
            トリガーID
        """
        trigger_id = f"news_{ticker}_{importance_threshold}"
        
        condition = TriggerCondition(
            trigger_type=TriggerType.NEWS_ALERT,
            ticker=ticker,
            company_name=company_name,
            threshold=importance_threshold,
            cooldown=cooldown
        )
        
        self.trigger_conditions[trigger_id] = condition
        logger.info(f"ニュースアラートトリガー追加: {ticker} 重要度 {importance_threshold}+")
        
        return trigger_id
    
    def add_sentiment_change_trigger(
        self,
        ticker: str,
        company_name: str,
        sentiment_threshold: float = 0.3,
        cooldown: int = 7200
    ) -> str:
        """センチメント変化トリガーを追加
        
        Args:
            ticker: ティッカーシンボル
            company_name: 企業名
            sentiment_threshold: センチメント変化の閾値
            cooldown: クールダウン期間（秒）
            
        Returns:
            トリガーID
        """
        trigger_id = f"sentiment_{ticker}_{sentiment_threshold}"
        
        condition = TriggerCondition(
            trigger_type=TriggerType.SENTIMENT_CHANGE,
            ticker=ticker,
            company_name=company_name,
            threshold=sentiment_threshold,
            cooldown=cooldown
        )
        
        self.trigger_conditions[trigger_id] = condition
        logger.info(f"センチメント変化トリガー追加: {ticker} 変化 {sentiment_threshold}+")
        
        return trigger_id
    
    def add_callback(self, callback: Callable[[TriggerEvent], None]):
        """トリガー発火時のコールバックを追加
        
        Args:
            callback: コールバック関数
        """
        self.trigger_callbacks.append(callback)
        logger.info("トリガーコールバック追加")
    
    def start_monitoring(self, check_interval: int = 30):
        """監視開始
        
        Args:
            check_interval: チェック間隔（秒）
        """
        if self.is_running:
            logger.warning("監視は既に実行中です")
            return
        
        self.is_running = True
        
        def monitor_loop():
            logger.info("リアルタイム監視開始")
            
            while self.is_running:
                try:
                    # 全トリガー条件をチェック
                    for trigger_id, condition in self.trigger_conditions.items():
                        if condition.is_active and not self._is_in_cooldown(condition):
                            self._check_trigger_condition(trigger_id, condition)
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"監視ループエラー: {e}")
                    time.sleep(check_interval)
            
            logger.info("リアルタイム監視停止")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """監視停止"""
        if not self.is_running:
            logger.warning("監視は実行されていません")
            return
        
        self.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("監視停止完了")
    
    def _is_in_cooldown(self, condition: TriggerCondition) -> bool:
        """クールダウン期間中かチェック"""
        if condition.last_triggered is None:
            return False
        
        elapsed = (datetime.now() - condition.last_triggered).total_seconds()
        return elapsed < condition.cooldown
    
    def _check_trigger_condition(self, trigger_id: str, condition: TriggerCondition):
        """トリガー条件をチェック"""
        try:
            if condition.trigger_type == TriggerType.PRICE_CHANGE:
                self._check_price_change(trigger_id, condition)
            elif condition.trigger_type == TriggerType.VOLUME_SPIKE:
                self._check_volume_spike(trigger_id, condition)
            elif condition.trigger_type == TriggerType.RSI_EXTREME:
                self._check_rsi_extreme(trigger_id, condition)
            elif condition.trigger_type == TriggerType.NEWS_ALERT:
                self._check_news_alert(trigger_id, condition)
            elif condition.trigger_type == TriggerType.SENTIMENT_CHANGE:
                self._check_sentiment_change(trigger_id, condition)
                
        except Exception as e:
            logger.warning(f"トリガーチェックエラー {trigger_id}: {e}")
    
    def _check_price_change(self, trigger_id: str, condition: TriggerCondition):
        """価格変動チェック"""
        try:
            stock = yf.Ticker(condition.ticker)
            hist = stock.history(period="1d", interval="1m")
            
            if len(hist) < 2:
                return
            
            # 現在価格と開始価格を比較
            current_price = hist['Close'].iloc[-1]
            start_price = hist['Close'].iloc[0]
            
            price_change_percent = ((current_price - start_price) / start_price) * 100
            
            if abs(price_change_percent) >= condition.threshold:
                context = {
                    "current_price": current_price,
                    "start_price": start_price,
                    "change_percent": price_change_percent
                }
                
                self._trigger_event(trigger_id, condition, price_change_percent, context)
                
        except Exception as e:
            logger.debug(f"価格変動チェックエラー {condition.ticker}: {e}")
    
    def _check_volume_spike(self, trigger_id: str, condition: TriggerCondition):
        """出来高急増チェック"""
        try:
            stock = yf.Ticker(condition.ticker)
            hist = stock.history(period="5d", interval="1m")
            
            if len(hist) < 100:  # 十分なデータが必要
                return
            
            # 直近の平均出来高
            recent_volume = hist['Volume'].iloc[-10:].mean()
            
            # 過去平均出来高
            avg_volume = hist['Volume'].iloc[:-10].mean()
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio >= condition.threshold:
                context = {
                    "recent_volume": recent_volume,
                    "average_volume": avg_volume,
                    "volume_ratio": volume_ratio
                }
                
                self._trigger_event(trigger_id, condition, volume_ratio, context)
                
        except Exception as e:
            logger.debug(f"出来高急増チェックエラー {condition.ticker}: {e}")
    
    def _check_rsi_extreme(self, trigger_id: str, condition: TriggerCondition):
        """RSI極値チェック"""
        try:
            stock = yf.Ticker(condition.ticker)
            hist = stock.history(period="1mo")
            
            if len(hist) < 14:  # RSI計算に必要
                return
            
            # RSI計算
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            extreme_type = condition.metadata.get("extreme_type", "both")
            should_trigger = False
            
            if extreme_type in ["oversold", "both"] and current_rsi <= condition.threshold:
                should_trigger = True
            elif extreme_type in ["overbought", "both"] and current_rsi >= (100 - condition.threshold):
                should_trigger = True
            
            if should_trigger:
                context = {
                    "rsi": current_rsi,
                    "extreme_type": "oversold" if current_rsi <= 30 else "overbought"
                }
                
                self._trigger_event(trigger_id, condition, current_rsi, context)
                
        except Exception as e:
            logger.debug(f"RSI極値チェックエラー {condition.ticker}: {e}")
    
    def _check_news_alert(self, trigger_id: str, condition: TriggerCondition):
        """ニュースアラートチェック"""
        try:
            news_items = self.news_collector.collect_all_news(
                condition.ticker, 
                condition.company_name,
                days_back=1
            )
            
            for news in news_items:
                if news.importance >= condition.threshold:
                    context = {
                        "news_title": news.title,
                        "news_source": news.source,
                        "importance": news.importance,
                        "published_date": news.published_date.isoformat()
                    }
                    
                    self._trigger_event(trigger_id, condition, news.importance, context)
                    break  # 1つ見つけたら十分
                    
        except Exception as e:
            logger.debug(f"ニュースアラートチェックエラー {condition.ticker}: {e}")
    
    def _check_sentiment_change(self, trigger_id: str, condition: TriggerCondition):
        """センチメント変化チェック"""
        try:
            # 最新ニュースを取得
            news_items = self.news_collector.collect_all_news(
                condition.ticker,
                condition.company_name,
                days_back=1
            )
            
            if not news_items:
                return
            
            # センチメント分析
            sentiment_results = self.sentiment_analyzer.analyze_news_batch(news_items)
            
            if not sentiment_results:
                return
            
            # 全体センチメントを計算
            overall_sentiment, avg_confidence = self.sentiment_analyzer.calculate_overall_sentiment(
                sentiment_results
            )
            
            # 過去のセンチメントと比較
            cache_key = condition.ticker
            if cache_key in self.sentiment_cache:
                prev_sentiment = self.sentiment_cache[cache_key].get("sentiment", 0)
                sentiment_change = abs(overall_sentiment - prev_sentiment)
                
                if sentiment_change >= condition.threshold:
                    context = {
                        "current_sentiment": overall_sentiment,
                        "previous_sentiment": prev_sentiment,
                        "sentiment_change": sentiment_change,
                        "confidence": avg_confidence
                    }
                    
                    self._trigger_event(trigger_id, condition, sentiment_change, context)
            
            # キャッシュ更新
            self.sentiment_cache[cache_key] = {
                "sentiment": overall_sentiment,
                "confidence": avg_confidence,
                "updated_at": datetime.now()
            }
            
        except Exception as e:
            logger.debug(f"センチメント変化チェックエラー {condition.ticker}: {e}")
    
    def _trigger_event(self, trigger_id: str, condition: TriggerCondition, value: float, context: Dict[str, Any]):
        """トリガーイベントを発生"""
        # トリガー時刻を更新
        condition.last_triggered = datetime.now()
        
        # 分析を実行（非同期）
        future = self.executor.submit(
            self._execute_triggered_analysis,
            condition.ticker,
            condition.company_name
        )
        
        # イベント作成
        event = TriggerEvent(
            condition=condition,
            triggered_at=condition.last_triggered,
            value=value,
            context=context
        )
        
        logger.info(f"トリガー発火: {condition.company_name} ({condition.trigger_type.value}) - 値: {value}")
        
        # コールバック実行
        for callback in self.trigger_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"コールバックエラー: {e}")
        
        # 分析結果を待って設定
        try:
            analysis_result = future.result(timeout=60)
            event.analysis_result = analysis_result
        except Exception as e:
            logger.warning(f"トリガー分析エラー: {e}")
    
    def _execute_triggered_analysis(self, ticker: str, company_name: str):
        """トリガーされた分析を実行"""
        try:
            return self.stock_analyzer.analyze_stock(ticker, company_name)
        except Exception as e:
            logger.error(f"トリガー分析実行エラー {ticker}: {e}")
            return None
    
    def get_trigger_status(self) -> Dict[str, Any]:
        """トリガー状態を取得"""
        active_triggers = sum(1 for c in self.trigger_conditions.values() if c.is_active)
        cooldown_triggers = sum(1 for c in self.trigger_conditions.values() if self._is_in_cooldown(c))
        
        return {
            "is_monitoring": self.is_running,
            "total_triggers": len(self.trigger_conditions),
            "active_triggers": active_triggers,
            "cooldown_triggers": cooldown_triggers,
            "trigger_types": {
                trigger_type.value: sum(1 for c in self.trigger_conditions.values() 
                                      if c.trigger_type == trigger_type)
                for trigger_type in TriggerType
            }
        }
    
    def remove_trigger(self, trigger_id: str) -> bool:
        """トリガーを削除"""
        if trigger_id in self.trigger_conditions:
            del self.trigger_conditions[trigger_id]
            logger.info(f"トリガー削除: {trigger_id}")
            return True
        return False
    
    def disable_trigger(self, trigger_id: str) -> bool:
        """トリガーを無効化"""
        if trigger_id in self.trigger_conditions:
            self.trigger_conditions[trigger_id].is_active = False
            logger.info(f"トリガー無効化: {trigger_id}")
            return True
        return False
    
    def enable_trigger(self, trigger_id: str) -> bool:
        """トリガーを有効化"""
        if trigger_id in self.trigger_conditions:
            self.trigger_conditions[trigger_id].is_active = True
            logger.info(f"トリガー有効化: {trigger_id}")
            return True
        return False
    
    def shutdown(self):
        """システムシャットダウン"""
        logger.info("リアルタイムトリガーシステムシャットダウン開始")
        
        self.stop_monitoring()
        self.executor.shutdown(wait=True)
        
        logger.info("リアルタイムトリガーシステムシャットダウン完了")