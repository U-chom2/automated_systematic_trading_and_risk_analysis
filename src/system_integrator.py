"""システム統合管理

起動から投資提案まで完全なフローを管理するクラス
"""

import json
import logging
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional

from .persistence import (
    ExecutionPlanManager, 
    TradeHistoryManager, 
    SystemStateManager
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDataCollector:
    """モックデータ収集器"""
    
    def collect_price_data(self, symbol: str) -> Dict[str, Any]:
        """株価データ収集（モック）"""
        base_price = {
            "7203": 2450.0,  # トヨタ
            "6758": 11500.0,  # ソニー
            "9984": 5200.0   # SBG
        }.get(symbol, 1000.0)
        
        # ランダムな変動を追加
        variation = random.uniform(-0.05, 0.05)
        current_price = base_price * (1 + variation)
        
        return {
            "current_price": round(current_price, 2),
            "volume": random.randint(500000, 2000000),
            "high": round(current_price * 1.02, 2),
            "low": round(current_price * 0.98, 2),
            "timestamp": datetime.now().isoformat()
        }
    
    def collect_ir_releases(self, symbol: str) -> List[Dict[str, Any]]:
        """IR情報収集（モック）"""
        sample_releases = [
            {
                "title": "2024年第3四半期決算発表",
                "content": "売上高前年同期比15%増、営業利益20%増の好調な業績",
                "timestamp": datetime.now().isoformat(),
                "importance": random.choice(["high", "medium", "low"])
            },
            {
                "title": "新製品発表のお知らせ",
                "content": "革新的な新技術を搭載した次世代製品を発表",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "importance": random.choice(["high", "medium"])
            }
        ]
        
        return random.sample(sample_releases, random.randint(0, len(sample_releases)))
    
    def collect_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """ソーシャル感情分析（モック）"""
        return {
            "positive": round(random.uniform(0.3, 0.8), 2),
            "negative": round(random.uniform(0.1, 0.3), 2),
            "neutral": round(random.uniform(0.1, 0.4), 2),
            "mention_count": random.randint(50, 500),
            "sentiment_trend": random.choice(["improving", "stable", "declining"])
        }


class MockAIAnalyzer:
    """モックAI分析エンジン"""
    
    def analyze_catalyst_impact(self, releases: List[Dict[str, Any]]) -> int:
        """カタリスト分析（モック）"""
        if not releases:
            return 0
            
        total_score = 0
        for release in releases:
            importance = release.get("importance", "low")
            if "決算" in release.get("title", ""):
                base_score = 30
            elif "新製品" in release.get("title", ""):
                base_score = 25
            else:
                base_score = 15
            
            # 重要度による調整
            multiplier = {"high": 1.5, "medium": 1.0, "low": 0.7}.get(importance, 1.0)
            total_score += int(base_score * multiplier)
        
        return min(total_score, 50)  # 最大50点
    
    def analyze_sentiment_score(self, sentiment_data: Dict[str, Any]) -> int:
        """感情スコア分析（モック）"""
        positive_ratio = sentiment_data.get("positive", 0.5)
        negative_ratio = sentiment_data.get("negative", 0.3)
        mention_count = sentiment_data.get("mention_count", 100)
        
        # ポジティブ比率に基づくベーススコア
        base_score = (positive_ratio - negative_ratio) * 30
        
        # メンション数による調整（話題性）
        volume_multiplier = min(mention_count / 200, 1.5)
        
        final_score = int(base_score * volume_multiplier)
        return max(0, min(final_score, 30))  # 0-30点の範囲
    
    def analyze_technical_indicators(self, price_data: Dict[str, Any]) -> int:
        """テクニカル分析（モック）"""
        current_price = price_data.get("current_price", 0)
        high = price_data.get("high", current_price)
        low = price_data.get("low", current_price)
        volume = price_data.get("volume", 0)
        
        # 価格位置分析（高値に近いほど高得点）
        price_position = (current_price - low) / (high - low) if high != low else 0.5
        position_score = price_position * 10
        
        # 出来高分析（平均より多いと高得点）
        volume_score = min(volume / 1000000, 2) * 5  # 100万株基準
        
        # ランダムなテクニカル要素
        rsi_score = random.uniform(0, 5)
        
        total_score = int(position_score + volume_score + rsi_score)
        return max(0, min(total_score, 20))  # 0-20点の範囲
    
    def calculate_risk_parameters(self, symbol: str, price: float, 
                                volatility: float) -> Dict[str, Any]:
        """リスク計算（モック）"""
        # ボラティリティに基づくストップロス
        stop_loss_percent = min(volatility * 20, 10)  # 最大10%
        
        # 利益目標（リスクの2-3倍）
        profit_target_percent = stop_loss_percent * random.uniform(2.0, 3.0)
        
        return {
            "stop_loss_percent": round(stop_loss_percent, 2),
            "profit_target_percent": round(profit_target_percent, 2),
            "confidence_level": round(random.uniform(0.7, 0.95), 2)
        }


class SystemIntegrator:
    """システム統合クラス
    
    起動から投資提案まで完全なフローを管理します。
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初期化
        
        Args:
            config: システム設定
        """
        self.config = config
        self.data_dir = config.get("data_dir", "./data")
        self.mock_mode = config.get("mock_mode", True)
        
        # 永続化層の初期化
        self.execution_plan_manager = ExecutionPlanManager(self.data_dir)
        self.trade_history_manager = TradeHistoryManager(self.data_dir)
        self.system_state_manager = SystemStateManager(self.data_dir)
        
        # モックコンポーネント
        self.data_collector = MockDataCollector()
        self.ai_analyzer = MockAIAnalyzer()
        
        # ウォッチリスト
        self.watchlist: List[str] = []
        
        logger.info("SystemIntegrator initialized")
    
    def initialize_system(self) -> bool:
        """システム初期化
        
        Returns:
            初期化成功時True
        """
        try:
            # システム状態を保存
            initial_state = {
                "state_id": f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "system_status": "initializing",
                "last_update": datetime.now().isoformat(),
                "portfolio_value": float(self.config.get("capital", 100000)),
                "cash_balance": float(self.config.get("capital", 100000)),
                "positions": {},
                "active_plans": []
            }
            
            self.system_state_manager.save_state(initial_state)
            logger.info("System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def load_watchlist(self, symbols: List[str]) -> None:
        """ウォッチリスト設定
        
        Args:
            symbols: 監視対象シンボルリスト
        """
        self.watchlist = symbols
        logger.info(f"Watchlist loaded: {len(symbols)} symbols")
    
    def collect_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """市場データ収集
        
        Args:
            symbols: データ収集対象シンボル
            
        Returns:
            収集されたデータ
        """
        collected_data = {}
        
        for symbol in symbols:
            try:
                symbol_data = {
                    "price_data": self.data_collector.collect_price_data(symbol),
                    "ir_releases": self.data_collector.collect_ir_releases(symbol),
                    "social_sentiment": self.data_collector.collect_social_sentiment(symbol)
                }
                
                collected_data[symbol] = symbol_data
                logger.info(f"Data collected for {symbol}")
                
            except Exception as e:
                logger.error(f"Data collection failed for {symbol}: {e}")
                collected_data[symbol] = {
                    "price_data": {},
                    "ir_releases": [],
                    "social_sentiment": {}
                }
        
        return collected_data
    
    def perform_ai_analysis(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """AI分析実行
        
        Args:
            market_data: 市場データ
            
        Returns:
            分析結果
        """
        analysis_results = {}
        
        for symbol, data in market_data.items():
            try:
                # 各分析の実行
                catalyst_score = self.ai_analyzer.analyze_catalyst_impact(
                    data.get("ir_releases", [])
                )
                
                sentiment_score = self.ai_analyzer.analyze_sentiment_score(
                    data.get("social_sentiment", {})
                )
                
                technical_score = self.ai_analyzer.analyze_technical_indicators(
                    data.get("price_data", {})
                )
                
                total_score = catalyst_score + sentiment_score + technical_score
                confidence = min(total_score / 100, 1.0)  # 100点満点で正規化
                
                analysis_results[symbol] = {
                    "catalyst_score": catalyst_score,
                    "sentiment_score": sentiment_score, 
                    "technical_score": technical_score,
                    "total_score": total_score,
                    "confidence": round(confidence, 3),
                    "current_price": data.get("price_data", {}).get("current_price", 0),
                    "risk_assessment": self.ai_analyzer.calculate_risk_parameters(
                        symbol, 
                        data.get("price_data", {}).get("current_price", 1000),
                        0.25  # 仮のボラティリティ
                    )
                }
                
                logger.info(f"Analysis completed for {symbol}: {total_score}/100")
                
            except Exception as e:
                logger.error(f"Analysis failed for {symbol}: {e}")
                analysis_results[symbol] = {
                    "catalyst_score": 0,
                    "sentiment_score": 0,
                    "technical_score": 0,
                    "total_score": 0,
                    "confidence": 0.0
                }
        
        return analysis_results
    
    def generate_investment_recommendations(self, analysis_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """投資提案生成
        
        Args:
            analysis_results: AI分析結果
            
        Returns:
            投資提案リスト
        """
        recommendations = []
        buy_threshold = self.config.get("buy_threshold", 70)
        
        for symbol, analysis in analysis_results.items():
            total_score = analysis.get("total_score", 0)
            confidence = analysis.get("confidence", 0)
            current_price = analysis.get("current_price", 0)
            risk_assessment = analysis.get("risk_assessment", {})
            
            # 投資判断
            if total_score >= buy_threshold and confidence > 0.6:
                action = "buy"
                reasoning = f"高スコア ({total_score}/100) により買い推奨"
            elif total_score <= 30:
                action = "sell"
                reasoning = f"低スコア ({total_score}/100) により売り推奨"
            else:
                action = "hold"
                reasoning = f"中程度スコア ({total_score}/100) により様子見"
            
            # 目標価格とストップロス計算
            stop_loss_percent = risk_assessment.get("stop_loss_percent", 5.0)
            profit_target_percent = risk_assessment.get("profit_target_percent", 15.0)
            
            if action == "buy":
                price_target = current_price * (1 + profit_target_percent / 100)
                stop_loss = current_price * (1 - stop_loss_percent / 100)
            else:
                price_target = current_price
                stop_loss = current_price
            
            recommendation = {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "current_price": current_price,
                "price_target": round(price_target, 2),
                "stop_loss": round(stop_loss, 2),
                "reasoning": reasoning,
                "total_score": total_score,
                "catalyst_score": analysis.get("catalyst_score", 0),
                "sentiment_score": analysis.get("sentiment_score", 0),
                "technical_score": analysis.get("technical_score", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            recommendations.append(recommendation)
            logger.info(f"Recommendation generated for {symbol}: {action}")
        
        # 信頼度でソート（降順）
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return recommendations
    
    def run_complete_analysis_cycle(self, raise_on_error: bool = False) -> List[Dict[str, Any]]:
        """完全分析サイクル実行
        
        Args:
            raise_on_error: エラー時に例外を再発生させるかどうか
            
        Returns:
            投資提案リスト
        """
        logger.info("Starting complete analysis cycle")
        
        try:
            # 1. データ収集
            logger.info("Step 1: Collecting market data")
            market_data = self.collect_market_data(self.watchlist)
            
            # 2. AI分析
            logger.info("Step 2: Performing AI analysis")
            analysis_results = self.perform_ai_analysis(market_data)
            
            # 3. 投資提案生成
            logger.info("Step 3: Generating investment recommendations")
            recommendations = self.generate_investment_recommendations(analysis_results)
            
            # 4. 結果の保存
            self._save_analysis_results(recommendations)
            
            logger.info(f"Analysis cycle completed: {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}")
            if raise_on_error:
                raise  # 例外を再発生
            return []
    
    def run_complete_analysis_cycle_with_error_handling(self) -> Dict[str, Any]:
        """エラーハンドリング付き完全分析サイクル
        
        Returns:
            実行結果（エラー情報含む）
        """
        try:
            # エラーを再発生させるモードで実行
            recommendations = self.run_complete_analysis_cycle(raise_on_error=True)
            return {
                "success": True,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Analysis cycle with error handling failed: {e}")
            
            error_info = {
                "success": False,
                "error": str(e),
                "recovery_action": "システムを安全モードで再起動します",
                "timestamp": datetime.now().isoformat()
            }
            
            # エラーを永続化
            try:
                self.system_state_manager.record_error({
                    "error_id": f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "severity": "high",
                    "component": "analysis_cycle",
                    "resolved": False
                })
            except Exception as log_error:
                logger.error(f"Failed to log error: {log_error}")
            
            return error_info
    
    def calculate_risk_parameters(self, symbol: str, current_price: float, 
                                volatility: float) -> Dict[str, Any]:
        """リスク計算
        
        Args:
            symbol: シンボル
            current_price: 現在価格
            volatility: ボラティリティ
            
        Returns:
            リスク計算結果
        """
        capital = self.config.get("capital", 100000)
        risk_per_trade_ratio = self.config.get("risk_per_trade_ratio", 0.01)
        
        # ポジションサイズ計算
        max_loss_amount = capital * risk_per_trade_ratio
        stop_loss_percent = min(volatility * 20, 10)  # 最大10%
        stop_loss_price = current_price * (1 - stop_loss_percent / 100)
        loss_per_share = current_price - stop_loss_price
        
        position_size = int(max_loss_amount / loss_per_share) if loss_per_share > 0 else 0
        
        return {
            "position_size": position_size,
            "stop_loss_price": round(stop_loss_price, 2),
            "max_loss_amount": round(max_loss_amount, 2),
            "risk_per_trade_ratio": risk_per_trade_ratio
        }
    
    def record_simulated_trade(self, trade_data: Dict[str, Any]) -> None:
        """模擬取引記録
        
        Args:
            trade_data: 取引データ
        """
        trade_record = {
            "trade_id": f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "symbol": trade_data["symbol"],
            "side": "buy" if trade_data["action"] == "buy" else "sell",
            "quantity": trade_data.get("quantity", 100),
            "price": Decimal(str(trade_data["price"])),
            "executed_at": trade_data.get("timestamp", datetime.now().isoformat()),
            "order_type": "market",
            "commission": Decimal("500"),  # 仮の手数料
            "total_amount": Decimal(str(trade_data["price"] * trade_data.get("quantity", 100) + 500)),
            "status": "filled"
        }
        
        self.trade_history_manager.record_trade(trade_record)
        logger.info(f"Simulated trade recorded: {trade_record['trade_id']}")
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """パフォーマンス統計取得
        
        Returns:
            パフォーマンス統計
        """
        summary = self.trade_history_manager.get_trading_summary()
        
        return {
            "total_trades": summary.get("total_trades", 0),
            "win_rate": 0.65 if summary.get("total_trades", 0) > 0 else 0.0,  # モック値
            "total_return": float(summary.get("net_pnl", 0)),
            "avg_return_per_trade": float(summary.get("net_pnl", 0)) / max(summary.get("total_trades", 1), 1)
        }
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """現在の設定取得
        
        Returns:
            現在のシステム設定
        """
        return self.config.copy()
    
    def save_all_data(self, data: Dict[str, Any]) -> bool:
        """全データ保存
        
        Args:
            data: 保存データ
            
        Returns:
            保存成功時True
        """
        try:
            # 実行計画保存
            if "execution_plan" in data:
                plan_data = data["execution_plan"].copy()
                # 必要なフィールドを追加
                if "created_at" not in plan_data:
                    plan_data["created_at"] = datetime.now().isoformat()
                if "status" not in plan_data:
                    plan_data["status"] = "active"
                self.execution_plan_manager.save_plan(plan_data)
            
            # 取引記録保存
            if "trade_record" in data:
                trade_data = data["trade_record"].copy()
                # 必要なフィールドを追加・変換
                required_fields = {
                    "trade_id": f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "side": "buy" if trade_data.get("action") == "buy" else "sell",
                    "quantity": trade_data.get("quantity", 100),
                    "price": Decimal(str(trade_data.get("price", 0))),
                    "executed_at": datetime.now().isoformat(),
                    "order_type": "market",
                    "commission": Decimal("500"),
                    "status": "filled"
                }
                
                for key, default_value in required_fields.items():
                    if key not in trade_data:
                        trade_data[key] = default_value
                
                # total_amountを計算
                if "total_amount" not in trade_data:
                    price = trade_data["price"] if isinstance(trade_data["price"], Decimal) else Decimal(str(trade_data["price"]))
                    quantity = trade_data["quantity"]
                    commission = trade_data["commission"]
                    trade_data["total_amount"] = price * quantity + commission
                
                self.trade_history_manager.record_trade(trade_data)
            
            # システム状態保存
            if "system_state" in data:
                state_data = data["system_state"].copy()
                # 必要なフィールドを追加
                if "state_id" not in state_data:
                    state_data["state_id"] = f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if "last_update" not in state_data:
                    state_data["last_update"] = datetime.now().isoformat()
                if "system_status" not in state_data:
                    state_data["system_status"] = state_data.get("status", "active")
                
                self.system_state_manager.save_state(state_data)
            
            return True
        except Exception as e:
            logger.error(f"Data save failed: {e}")
            return False
    
    def load_all_data(self) -> Dict[str, Any]:
        """全データ読み込み
        
        Returns:
            読み込みデータ
        """
        try:
            loaded_data = {}
            
            # 実行計画読み込み（最新1件）
            plans = self.execution_plan_manager.list_plans()
            if plans:
                loaded_data["execution_plan"] = plans[0]
            
            # 取引記録読み込み（サマリー）
            loaded_data["trade_record"] = self.trade_history_manager.get_trading_summary()
            
            # システム状態読み込み（最新）
            latest_state = self.system_state_manager.load_latest_state()
            if latest_state:
                loaded_data["system_state"] = latest_state
            
            return loaded_data
        except Exception as e:
            logger.error(f"Data load failed: {e}")
            return {}
    
    def _save_analysis_results(self, recommendations: List[Dict[str, Any]]) -> None:
        """分析結果保存
        
        Args:
            recommendations: 投資提案リスト
        """
        try:
            # 実行計画として保存
            for i, rec in enumerate(recommendations):
                if rec["action"] == "buy" and rec["confidence"] > 0.7:
                    plan = {
                        "id": f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                        "name": f"{rec['symbol']} 投資計画",
                        "strategy": "ai_recommendation",
                        "parameters": {
                            "symbol": rec["symbol"],
                            "target_price": rec["price_target"],
                            "stop_loss": rec["stop_loss"],
                            "confidence": rec["confidence"]
                        },
                        "created_at": datetime.now().isoformat(),
                        "status": "pending"
                    }
                    
                    self.execution_plan_manager.save_plan(plan)
                    
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")