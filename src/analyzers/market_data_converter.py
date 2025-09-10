"""市場データ変換 - Yahoo FinanceデータをPPOモデル用観測値に変換"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path

from src.utils.logger_utils import create_dual_logger

logger = create_dual_logger(__name__, console_output=True)


class MarketDataConverter:
    """Yahoo Finance市場データをPPOモデル用観測値に変換"""
    
    def __init__(self):
        """初期化"""
        self.nikkei_symbols = ["^N225"]  # 日経225
        self.observation_cache: Dict[str, Any] = {}
        self.cache_timeout = 300  # 5分キャッシュ
    
    def get_observation_for_ticker(self, ticker: str, use_cache: bool = True) -> np.ndarray:
        """指定ティッカーの観測値を生成
        
        Args:
            ticker: ティッカーシンボル (例: "7203.T")
            use_cache: キャッシュを使用するか
            
        Returns:
            PPO用観測値 (shape: [1838])
        """
        cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        if use_cache and cache_key in self.observation_cache:
            return self.observation_cache[cache_key]
        
        try:
            # 日経225データ取得 (30日分)
            nikkei_data = self._get_nikkei_data()
            
            # 個別株データ取得
            stock_data = self._get_stock_data(ticker)
            
            # ポートフォリオ状態取得
            portfolio_data = self._get_portfolio_state()
            
            # IR/センチメント情報（現在は仮実装）
            sentiment_data = self._get_sentiment_data(ticker)
            
            # 観測値を構築
            observation = self._build_observation(
                nikkei_data=nikkei_data,
                stock_data=stock_data,
                portfolio_data=portfolio_data,
                sentiment_data=sentiment_data,
                ticker=ticker
            )
            
            # キャッシュに保存
            if use_cache:
                self.observation_cache[cache_key] = observation
            
            return observation
            
        except Exception as e:
            logger.warning(f"観測値生成エラー {ticker}: {e}")
            return self._get_fallback_observation()
    
    def _get_nikkei_data(self) -> Dict[str, Any]:
        """日経225の30日データを取得
        
        Returns:
            日経225データ
        """
        try:
            nikkei = yf.Ticker("^N225")
            hist = nikkei.history(period="30d")
            
            if hist.empty:
                raise ValueError("日経225データが取得できませんでした")
            
            # 最新30日分の高値・安値・終値
            data = {
                "high": hist["High"].tail(30).values.tolist(),
                "low": hist["Low"].tail(30).values.tolist(), 
                "close": hist["Close"].tail(30).values.tolist(),
                "volume": hist["Volume"].tail(30).values.tolist(),
            }
            
            # データの正規化
            for key in ["high", "low", "close"]:
                if data[key]:
                    mean_val = np.mean(data[key])
                    std_val = np.std(data[key]) + 1e-8
                    data[key] = [(val - mean_val) / std_val for val in data[key]]
            
            return data
            
        except Exception as e:
            logger.warning(f"日経225データ取得エラー: {e}")
            return {
                "high": [0.0] * 30,
                "low": [0.0] * 30,
                "close": [0.0] * 30,
                "volume": [0.0] * 30,
            }
    
    def _get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """個別株の当日データを取得
        
        Args:
            ticker: ティッカーシンボル
            
        Returns:
            個別株データ
        """
        try:
            stock = yf.Ticker(ticker)
            
            # 過去5日のデータを取得（当日含む）
            hist = stock.history(period="5d")
            info = stock.info
            
            if hist.empty:
                raise ValueError(f"株価データが取得できませんでした: {ticker}")
            
            latest = hist.iloc[-1]
            
            # 基本価格情報
            data = {
                "current_price": float(latest["Close"]),
                "volume": float(latest["Volume"]),
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
            }
            
            # 変化率計算
            if len(hist) >= 2:
                prev_close = hist.iloc[-2]["Close"]
                data["change_pct"] = float((latest["Close"] - prev_close) / prev_close * 100)
            else:
                data["change_pct"] = 0.0
            
            # 企業情報
            data.update({
                "market_cap": info.get("marketCap", 0) / 1e9 if info.get("marketCap") else 0.0,  # 10億円単位
                "pe_ratio": info.get("trailingPE", 0.0) or 0.0,
                "pb_ratio": info.get("priceToBook", 0.0) or 0.0,
                "dividend_yield": info.get("dividendYield", 0.0) or 0.0,
            })
            
            return data
            
        except Exception as e:
            logger.warning(f"個別株データ取得エラー {ticker}: {e}")
            return {
                "current_price": 1000.0,
                "volume": 10000.0,
                "open": 1000.0,
                "high": 1000.0,
                "low": 1000.0,
                "change_pct": 0.0,
                "market_cap": 50.0,  # 50億円
                "pe_ratio": 15.0,
                "pb_ratio": 1.0,
                "dividend_yield": 0.02,
            }
    
    def _get_portfolio_state(self) -> Dict[str, Any]:
        """現在のポートフォリオ状態を取得
        
        Returns:
            ポートフォリオデータ
        """
        try:
            # 実装は後で詳細化 - 現在は仮実装
            portfolio_path = Path("data/portfolio.json")
            
            if portfolio_path.exists():
                import json
                with open(portfolio_path, 'r', encoding='utf-8') as f:
                    portfolio = json.load(f)
                
                return {
                    "cash_ratio": portfolio.get("cash_ratio", 0.5),
                    "position_count": len(portfolio.get("positions", [])),
                    "total_value": portfolio.get("total_value", 1000000.0),  # 100万円
                }
            else:
                return {
                    "cash_ratio": 0.5,  # 現金比率50%
                    "position_count": 0,
                    "total_value": 1000000.0,
                }
            
        except Exception as e:
            logger.warning(f"ポートフォリオ取得エラー: {e}")
            return {
                "cash_ratio": 0.5,
                "position_count": 0,
                "total_value": 1000000.0,
            }
    
    def _get_sentiment_data(self, ticker: str) -> Dict[str, Any]:
        """IR/センチメント情報を取得（仮実装）
        
        Args:
            ticker: ティッカーシンボル
            
        Returns:
            センチメントデータ
        """
        # ModernBERT-ja センチメント分析の結果を想定（後で実装）
        return {
            "sentiment_score": 0.0,  # -1.0 to 1.0
            "news_count": 0,
            "ir_score": 0.0,
        }
    
    def _build_observation(
        self, 
        nikkei_data: Dict[str, Any],
        stock_data: Dict[str, Any], 
        portfolio_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        ticker: str
    ) -> np.ndarray:
        """各データから観測値を構築
        
        Args:
            nikkei_data: 日経225データ
            stock_data: 個別株データ
            portfolio_data: ポートフォリオデータ
            sentiment_data: センチメントデータ
            ticker: ティッカーシンボル
            
        Returns:
            統合観測値 [1838次元]
        """
        observation_parts = []
        
        # 1. 日経225データ (30日 × 3指標 = 90次元)
        for key in ["high", "low", "close"]:
            observation_parts.extend(nikkei_data[key])
        
        # 2. 個別株データ (10次元)
        observation_parts.extend([
            stock_data["current_price"] / 1000.0,  # 正規化
            stock_data["volume"] / 100000.0,
            stock_data["change_pct"] / 100.0,
            stock_data["market_cap"] / 100.0,
            stock_data["pe_ratio"] / 50.0,
            stock_data["pb_ratio"] / 5.0,
            stock_data["dividend_yield"] * 100.0,
            (stock_data["high"] - stock_data["low"]) / stock_data["current_price"],  # ボラティリティ
            stock_data["open"] / stock_data["current_price"],  # 始値比率
            1.0,  # バイアス項
        ])
        
        # 3. ポートフォリオ状態 (3次元)
        observation_parts.extend([
            portfolio_data["cash_ratio"],
            portfolio_data["position_count"] / 10.0,  # 正規化
            portfolio_data["total_value"] / 1000000.0,
        ])
        
        # 4. センチメントデータ (3次元)
        observation_parts.extend([
            sentiment_data["sentiment_score"],
            sentiment_data["news_count"] / 10.0,
            sentiment_data["ir_score"],
        ])
        
        # 5. 残り次元をパディング（7956 - 106 = 7850次元）
        current_size = len(observation_parts)
        target_size = 7956  # PPOモデルが期待するサイズに合わせる
        
        if current_size < target_size:
            padding_size = target_size - current_size
            observation_parts.extend([0.0] * padding_size)
        elif current_size > target_size:
            observation_parts = observation_parts[:target_size]
        
        observation = np.array(observation_parts, dtype=np.float32)
        
        # NaN/Inf チェック
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 値の範囲制限
        observation = np.clip(observation, -10.0, 10.0)
        
        logger.debug(f"観測値生成完了 {ticker}: shape={observation.shape}")
        
        return observation
    
    def _get_fallback_observation(self) -> np.ndarray:
        """フォールバック用の観測値
        
        Returns:
            デフォルト観測値
        """
        return np.zeros(7956, dtype=np.float32)
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self.observation_cache.clear()
        logger.info("観測値キャッシュをクリアしました")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得
        
        Returns:
            キャッシュ統計
        """
        return {
            "cache_size": len(self.observation_cache),
            "cache_keys": list(self.observation_cache.keys())
        }