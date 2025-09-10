"""
個別銘柄分析実行クラス
"""
import csv
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from src.utils.logger_utils import create_dual_logger

from .models import (
    TradingRecommendation, 
    TodoItem, 
    AnalysisResult,
    ActionType,
    RiskLevel
)
from .ai_model_loader import PPOModelLoader
from .market_data_converter import MarketDataConverter
from .news_collector import NewsCollector
from .sentiment_analyzer import ModernBERTSentimentAnalyzer
from .risk_evaluator import ComprehensiveRiskEvaluator, RiskAssessment


logger = create_dual_logger(__name__, console_output=True)


class StockAnalyzer:
    """個別銘柄分析と取引推奨生成"""
    
    def __init__(self, 
                 target_csv_path: str = "data/target.csv",
                 portfolio_path: str = "data/portfolio.json",
                 model_path: Optional[str] = None,
                 enable_ai: bool = True,
                 enable_news: bool = True,
                 enable_risk_evaluation: bool = True):
        """
        Args:
            target_csv_path: target.csvのパス
            portfolio_path: ポートフォリオファイルのパス
            model_path: PPOモデルパス（Noneの場合はデフォルト使用）
            enable_ai: AI予測を有効にするか
            enable_news: ニュース・センチメント分析を有効にするか
            enable_risk_evaluation: 包括的リスク評価を有効にするか
        """
        self.target_csv_path = Path(target_csv_path)
        self.portfolio_path = Path(portfolio_path)
        self.enable_ai = enable_ai
        self.enable_news = enable_news
        self.enable_risk_evaluation = enable_risk_evaluation
        
        # PPOモデルローダーを初期化
        if self.enable_ai:
            try:
                self.ai_model = PPOModelLoader(model_path) if model_path else PPOModelLoader()
                self.market_converter = MarketDataConverter()
                logger.info("AIモデル統合完了")
            except Exception as e:
                logger.warning(f"AIモデル初期化失敗: {e}")
                self.ai_model = None
                self.market_converter = None
        else:
            self.ai_model = None
            self.market_converter = None
        
        # ニュース・センチメント分析システム初期化
        if self.enable_news:
            try:
                self.news_collector = NewsCollector()
                self.sentiment_analyzer = ModernBERTSentimentAnalyzer()
                logger.info("ニュース・センチメント分析システム初期化完了")
            except Exception as e:
                logger.warning(f"ニュース・センチメント分析初期化失敗: {e}")
                self.news_collector = None
                self.sentiment_analyzer = None
        else:
            self.news_collector = None
            self.sentiment_analyzer = None
        
        # 包括的リスク評価システム初期化
        if self.enable_risk_evaluation:
            try:
                self.risk_evaluator = ComprehensiveRiskEvaluator()
                logger.info("包括的リスク評価システム初期化完了")
            except Exception as e:
                logger.warning(f"リスク評価システム初期化失敗: {e}")
                self.risk_evaluator = None
        else:
            self.risk_evaluator = None
        
        self.portfolio = self._load_portfolio()
        
        # システムヘルスチェック
        if self.enable_ai:
            self._check_ai_health()
        if self.enable_news:
            self._check_news_health()
        if self.enable_risk_evaluation:
            self._check_risk_health()
        
    def load_target_companies(self) -> pd.DataFrame:
        """target.csvから対象企業を読み込み"""
        if not self.target_csv_path.exists():
            logger.error(f"target.csv が存在しません: {self.target_csv_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.target_csv_path, encoding='utf-8')
            logger.info(f"target.csv から {len(df)} 社を読み込みました")
            return df
        except Exception as e:
            logger.error(f"target.csv の読み込みエラー: {e}")
            raise
    
    def analyze_stock(self, ticker: str, company_name: str) -> AnalysisResult:
        """個別銘柄を分析"""
        logger.info(f"分析開始: {company_name} ({ticker})")
        
        try:
            # テクニカル分析
            technical_score = self._analyze_technical(ticker)
            
            # ファンダメンタル分析
            fundamental_score = self._analyze_fundamental(ticker)
            
            # センチメント分析（ニュースベース）
            sentiment_score, news_sentiment_details = self._analyze_sentiment(ticker, company_name)
            
            # AI予測（PPOモデル）
            ai_prediction = self._get_ai_prediction(ticker)
            
            # 包括的リスク評価
            risk_assessment = self._evaluate_comprehensive_risk(ticker, company_name)
            
            # 指標計算
            indicators = self._calculate_indicators(ticker)
            
            # リスク調整後スコア計算
            risk_adjustment_factor = 1.0
            if risk_assessment and risk_assessment.overall_risk_score > 0:
                risk_adjustment_factor = max(0.5, 1.0 - (risk_assessment.overall_risk_score - 50) / 100)
            
            result = AnalysisResult(
                ticker=ticker,
                company_name=company_name,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                sentiment_score=sentiment_score,
                ai_prediction=ai_prediction,
                indicators=indicators,
                risk_assessment=risk_assessment,
                news_sentiment_details=news_sentiment_details,
                risk_adjustment_factor=risk_adjustment_factor
            )
            
            logger.info(f"分析完了: {company_name} - 総合スコア: {result.total_score:.1f} (リスク調整: {risk_adjustment_factor:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"分析エラー {ticker}: {e}")
            # エラー時はデフォルト値を返す
            return AnalysisResult(
                ticker=ticker,
                company_name=company_name,
                technical_score=50,
                fundamental_score=50,
                sentiment_score=0,
                ai_prediction=None,
                indicators={},
                risk_assessment=None,
                news_sentiment_details=None,
                risk_adjustment_factor=1.0
            )
    
    def _analyze_technical(self, ticker: str) -> float:
        """テクニカル分析スコア計算"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")
            
            if hist.empty:
                return 50.0
            
            scores = []
            
            # RSI計算
            rsi = self._calculate_rsi(hist['Close'])
            if rsi is not None:
                # RSI 30以下で買い、70以上で売り
                if rsi <= 30:
                    scores.append(90)
                elif rsi >= 70:
                    scores.append(10)
                else:
                    scores.append(50 + (40 - rsi) * 1.25)
            
            # 移動平均線
            ma20 = hist['Close'].rolling(window=20).mean()
            ma50 = hist['Close'].rolling(window=50).mean()
            
            if not ma20.empty and not ma50.empty:
                current_price = hist['Close'].iloc[-1]
                ma20_current = ma20.iloc[-1]
                ma50_current = ma50.iloc[-1]
                
                # ゴールデンクロス/デッドクロス判定
                if ma20_current > ma50_current and current_price > ma20_current:
                    scores.append(80)
                elif ma20_current < ma50_current and current_price < ma20_current:
                    scores.append(20)
                else:
                    scores.append(50)
            
            # ボリューム分析
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].iloc[-5:].mean()
            
            if recent_volume > avg_volume * 1.5:
                scores.append(70)  # 出来高増加
            elif recent_volume < avg_volume * 0.5:
                scores.append(30)  # 出来高減少
            else:
                scores.append(50)
            
            return np.mean(scores) if scores else 50.0
            
        except Exception as e:
            logger.warning(f"テクニカル分析エラー {ticker}: {e}")
            return 50.0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """RSI計算"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not rsi.empty else None
        except:
            return None
    
    def _analyze_fundamental(self, ticker: str) -> float:
        """ファンダメンタル分析スコア計算"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            scores = []
            
            # PER（株価収益率）
            pe_ratio = info.get('trailingPE', 0)
            if pe_ratio > 0:
                if pe_ratio < 15:
                    scores.append(80)
                elif pe_ratio > 30:
                    scores.append(20)
                else:
                    scores.append(50)
            
            # PBR（株価純資産倍率）
            pb_ratio = info.get('priceToBook', 0)
            if pb_ratio > 0:
                if pb_ratio < 1:
                    scores.append(80)
                elif pb_ratio > 3:
                    scores.append(20)
                else:
                    scores.append(50)
            
            # ROE（自己資本利益率）
            roe = info.get('returnOnEquity', 0)
            if roe > 0:
                if roe > 0.15:
                    scores.append(80)
                elif roe < 0.05:
                    scores.append(20)
                else:
                    scores.append(50)
            
            # 配当利回り
            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield > 0.03:
                scores.append(70)
            elif dividend_yield > 0:
                scores.append(50)
            else:
                scores.append(30)
            
            return np.mean(scores) if scores else 50.0
            
        except Exception as e:
            logger.warning(f"ファンダメンタル分析エラー {ticker}: {e}")
            return 50.0
    
    def _analyze_sentiment(self, ticker: str, company_name: str) -> Tuple[float, Optional[Dict[str, Any]]]:
        """センチメント分析スコア計算（ModernBERT-ja使用）
        
        Returns:
            (sentiment_score, news_sentiment_details) のタプル
        """
        if not self.enable_news or not self.news_collector or not self.sentiment_analyzer:
            logger.debug("ニュース・センチメント分析が無効です")
            return 0.0, None
        
        try:
            # ニュース収集
            news_items = self.news_collector.collect_all_news(
                ticker, company_name, days_back=3
            )
            
            if not news_items:
                logger.debug(f"ニュースが見つかりません: {ticker}")
                return 0.0, {"message": "ニュースデータなし", "news_count": 0}
            
            # センチメント分析実行
            sentiment_results = self.sentiment_analyzer.analyze_news_batch(news_items)
            
            if not sentiment_results:
                logger.debug(f"センチメント分析結果なし: {ticker}")
                return 0.0, {"message": "センチメント分析失敗", "news_count": len(news_items)}
            
            # 全体センチメント計算
            overall_sentiment, avg_confidence = self.sentiment_analyzer.calculate_overall_sentiment(
                sentiment_results, weight_by_confidence=True
            )
            
            # スコアを-1から1の範囲で返す
            sentiment_score = np.clip(overall_sentiment, -1.0, 1.0)
            
            # 詳細情報
            news_details = {
                "overall_sentiment": overall_sentiment,
                "confidence": avg_confidence,
                "news_count": len(news_items),
                "positive_news": sum(1 for r in sentiment_results if r.sentiment_score > 0.1),
                "negative_news": sum(1 for r in sentiment_results if r.sentiment_score < -0.1),
                "neutral_news": sum(1 for r in sentiment_results if -0.1 <= r.sentiment_score <= 0.1),
                "recent_news_titles": [news.title[:100] for news in news_items[:3]],
                "sentiment_breakdown": {
                    "positive": sum(r.emotion_scores.get("positive", 0) for r in sentiment_results) / len(sentiment_results),
                    "negative": sum(r.emotion_scores.get("negative", 0) for r in sentiment_results) / len(sentiment_results),
                    "neutral": sum(r.emotion_scores.get("neutral", 0) for r in sentiment_results) / len(sentiment_results)
                }
            }
            
            logger.info(f"センチメント分析完了 {ticker}: スコア {sentiment_score:.3f} (信頼度: {avg_confidence:.3f}, ニュース: {len(news_items)}件)")
            
            return sentiment_score, news_details
            
        except Exception as e:
            logger.warning(f"センチメント分析エラー {ticker}: {e}")
            return 0.0, {"error": str(e), "message": "センチメント分析エラー"}
    
    def _get_ai_prediction(self, ticker: str) -> Optional[Dict[str, Any]]:
        """PPOモデルによるAI予測を取得"""
        if not self.enable_ai or not self.ai_model or not self.market_converter:
            return None
        
        if not self.ai_model.is_loaded:
            logger.debug(f"AIモデルが読み込まれていません: {self.ai_model.load_error}")
            return None
        
        try:
            # 市場データを取得してPPO観測値に変換
            observation = self.market_converter.get_observation_for_ticker(ticker)
            
            # PPOモデルで予測実行
            market_data = {
                "ticker": ticker,
                "observation": observation.tolist()
            }
            
            prediction = self.ai_model.predict_action(ticker, market_data)
            
            logger.info(f"AI予測完了 {ticker}: {prediction['action']} (信頼度: {prediction['confidence']:.2f})")
            
            return prediction
            
        except Exception as e:
            logger.warning(f"AI予測エラー {ticker}: {e}")
            return None
    
    def _evaluate_comprehensive_risk(self, ticker: str, company_name: str) -> Optional[RiskAssessment]:
        """包括的リスク評価を実行
        
        Args:
            ticker: ティッカーシンボル
            company_name: 企業名
            
        Returns:
            包括的リスク評価結果
        """
        if not self.enable_risk_evaluation or not self.risk_evaluator:
            logger.debug("リスク評価が無効です")
            return None
        
        try:
            # ポートフォリオ重みを計算（保有中の場合）
            portfolio_weight = 0.0
            if self._is_holding(ticker):
                holdings = self.portfolio.get("holdings", {})
                holding_info = holdings.get(ticker, {})
                total_value = sum(h.get("value", 0) for h in holdings.values())
                holding_value = holding_info.get("value", 0)
                portfolio_weight = holding_value / total_value if total_value > 0 else 0.0
            
            # リスク評価実行
            risk_assessment = self.risk_evaluator.evaluate_comprehensive_risk(
                ticker, company_name, portfolio_weight
            )
            
            logger.info(f"リスク評価完了 {ticker}: {risk_assessment.overall_risk_score:.1f} ({risk_assessment.risk_level.value})")
            
            return risk_assessment
            
        except Exception as e:
            logger.warning(f"リスク評価エラー {ticker}: {e}")
            return None
    
    def _check_ai_health(self) -> bool:
        """AIモデルのヘルスチェック
        
        Returns:
            ヘルスチェック結果
        """
        if not self.ai_model or not self.market_converter:
            logger.warning("AIコンポーネントが初期化されていません")
            return False
        
        # モデル情報を取得
        model_info = self.ai_model.get_model_info()
        
        if not model_info["is_loaded"]:
            logger.warning(f"AIモデルが読み込まれていません: {model_info.get('load_error', 'Unknown error')}")
            return False
        
        logger.info("AIモデルヘルスチェック通過")
        return True
    
    def _check_news_health(self) -> bool:
        """ニュース・センチメント分析システムヘルスチェック
        
        Returns:
            ヘルスチェック結果
        """
        if not self.news_collector or not self.sentiment_analyzer:
            logger.warning("ニュース・センチメント分析コンポーネントが初期化されていません")
            return False
        
        try:
            # センチメント分析モデルの状態確認
            model_status = self.sentiment_analyzer.get_model_status()
            
            if not model_status.get("is_loaded", False):
                logger.warning(f"センチメント分析モデルが読み込まれていません: {model_status.get('load_error')}")
                return False
            
            logger.info("ニュース・センチメント分析システムヘルスチェック通過")
            return True
            
        except Exception as e:
            logger.warning(f"ニュース・センチメント分析ヘルスチェックエラー: {e}")
            return False
    
    def _check_risk_health(self) -> bool:
        """リスク評価システムヘルスチェック
        
        Returns:
            ヘルスチェック結果
        """
        if not self.risk_evaluator:
            logger.warning("リスク評価コンポーネントが初期化されていません")
            return False
        
        try:
            # ベンチマークデータの取得テスト
            benchmark_data = self.risk_evaluator._get_benchmark_data()
            
            if benchmark_data.empty:
                logger.warning("ベンチマークデータが取得できません")
                return False
            
            logger.info("リスク評価システムヘルスチェック通過")
            return True
            
        except Exception as e:
            logger.warning(f"リスク評価ヘルスチェックエラー: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """全システムの現在ステータスを取得
        
        Returns:
            システム統合ステータス情報
        """
        status = {
            "timestamp": datetime.now().isoformat(),
            "ai_system": self._get_ai_status(),
            "news_system": self._get_news_status(),
            "risk_system": self._get_risk_status(),
        }
        
        return status
    
    def _get_ai_status(self) -> Dict[str, Any]:
        """AI機能ステータス"""
        if not self.enable_ai:
            return {"enabled": False, "reason": "AI機能が無効"}
        
        if not self.ai_model:
            return {"enabled": False, "reason": "AIモデル未初期化"}
        
        model_info = self.ai_model.get_model_info()
        
        return {
            "enabled": True,
            "model_loaded": model_info["is_loaded"],
            "model_path": model_info["model_path"],
            "load_error": model_info.get("load_error"),
            "has_converter": self.market_converter is not None,
            "health_check": self._check_ai_health() if self.ai_model else False
        }
    
    def _get_news_status(self) -> Dict[str, Any]:
        """ニュース・センチメント分析ステータス"""
        if not self.enable_news:
            return {"enabled": False, "reason": "ニュース・センチメント分析が無効"}
        
        if not self.news_collector or not self.sentiment_analyzer:
            return {"enabled": False, "reason": "ニュース・センチメント分析未初期化"}
        
        sentiment_status = {}
        try:
            sentiment_status = self.sentiment_analyzer.get_model_status()
        except Exception as e:
            sentiment_status = {"error": str(e)}
        
        return {
            "enabled": True,
            "news_collector_ready": self.news_collector is not None,
            "sentiment_analyzer_status": sentiment_status,
            "health_check": self._check_news_health()
        }
    
    def _get_risk_status(self) -> Dict[str, Any]:
        """リスク評価システムステータス"""
        if not self.enable_risk_evaluation:
            return {"enabled": False, "reason": "リスク評価が無効"}
        
        if not self.risk_evaluator:
            return {"enabled": False, "reason": "リスク評価システム未初期化"}
        
        return {
            "enabled": True,
            "risk_evaluator_ready": self.risk_evaluator is not None,
            "benchmark_symbol": self.risk_evaluator.market_benchmark,
            "health_check": self._check_risk_health()
        }
    
    def get_ai_status(self) -> Dict[str, Any]:
        """AI機能の現在ステータスを取得（後方互換性のため保持）
        
        Returns:
            AIステータス情報
        """
        return self._get_ai_status()
    
    def _calculate_indicators(self, ticker: str) -> Dict[str, float]:
        """各種指標を計算"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            
            if hist.empty:
                return {}
            
            indicators = {}
            
            # RSI
            rsi = self._calculate_rsi(hist['Close'])
            if rsi is not None:
                indicators['RSI'] = rsi
            
            # 移動平均
            indicators['SMA_20'] = hist['Close'].rolling(window=20).mean().iloc[-1]
            indicators['SMA_50'] = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
            
            # ボラティリティ
            indicators['volatility'] = hist['Close'].pct_change().std() * np.sqrt(252)
            
            # ボリューム比率
            indicators['volume_ratio'] = hist['Volume'].iloc[-1] / hist['Volume'].mean()
            
            return indicators
            
        except Exception as e:
            logger.warning(f"指標計算エラー {ticker}: {e}")
            return {}
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[TradingRecommendation]:
        """全銘柄の取引推奨を生成"""
        recommendations = []
        
        for _, row in df.iterrows():
            ticker = row['ticker']
            company_name = row['company_name']
            
            # 分析実行
            analysis = self.analyze_stock(ticker, company_name)
            
            # 保有状況を確認
            is_holding = self._is_holding(ticker)
            
            # 推奨生成（保有状況を考慮）
            recommendation = analysis.to_recommendation(quantity=100)
            
            # 保有していない銘柄のSELL推奨は無視
            if recommendation.action == ActionType.SELL and not is_holding:
                logger.info(f"スキップ: {company_name} - 未保有のためSELL不可")
                continue
            
            if recommendation.should_execute:
                recommendations.append(recommendation)
                logger.info(f"推奨生成: {company_name} - {recommendation.action.value}")
        
        return recommendations
    
    def create_todo_list(self, recommendations: List[TradingRecommendation]) -> List[TodoItem]:
        """取引推奨からTODOリストを作成"""
        todos = []
        
        # 翌営業日を計算
        target_date = self._get_next_business_day()
        
        for rec in recommendations:
            if rec.action in [ActionType.BUY, ActionType.SELL]:
                todo = TodoItem(
                    stock_id=f"stock_{rec.ticker}",
                    ticker=rec.ticker,
                    company_name=rec.company_name,
                    action_type=rec.action,
                    quantity=rec.quantity,
                    target_date=target_date,
                    hold_until=rec.hold_until,
                    status="PENDING",
                    recommendation_details=rec
                )
                todos.append(todo)
                logger.info(f"TODO作成: {todo.company_name} - {todo.action_type.value} x {todo.quantity}")
        
        return todos
    
    def _get_next_business_day(self) -> date:
        """翌営業日を取得"""
        next_day = date.today() + timedelta(days=1)
        
        # 週末の場合は月曜日まで進める
        while next_day.weekday() >= 5:  # 5:土曜, 6:日曜
            next_day += timedelta(days=1)
        
        return next_day
    
    def _load_portfolio(self) -> Dict[str, Any]:
        """ポートフォリオを読み込み"""
        if self.portfolio_path.exists():
            try:
                with open(self.portfolio_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"ポートフォリオ読み込みエラー: {e}")
        return {"holdings": {}}
    
    def _is_holding(self, ticker: str) -> bool:
        """指定銘柄を保有しているか確認"""
        holdings = self.portfolio.get("holdings", {})
        return ticker in holdings and holdings[ticker].get("quantity", 0) > 0
    
    def save_todos(self, todos: List[TodoItem], output_path: str = "data/todos.json"):
        """TODOリストを保存"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        todos_data = [todo.to_dict() for todo in todos]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(todos_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"TODOリストを保存: {output_file} ({len(todos)}件)")
    
    def execute(self) -> List[TodoItem]:
        """分析と推奨生成を実行"""
        logger.info("="*50)
        logger.info("個別銘柄分析・取引推奨生成開始")
        logger.info("="*50)
        
        # target.csv読み込み
        df = self.load_target_companies()
        
        if df.empty:
            logger.warning("分析対象企業がありません")
            return []
        
        # 推奨生成
        recommendations = self.generate_recommendations(df)
        
        # TODOリスト作成
        todos = self.create_todo_list(recommendations)
        
        # 保存
        self.save_todos(todos)
        
        logger.info(f"分析完了: {len(todos)} 件の取引TODO生成")
        
        return todos