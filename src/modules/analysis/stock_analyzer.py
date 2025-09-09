"""
個別銘柄分析実行クラス
"""
import csv
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
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
# from src.infrastructure.ai.agents.trading_agent import TradingAgent
# from src.infrastructure.data_sources.market_data import MarketDataFetcher


logger = create_dual_logger(__name__, console_output=True)


class StockAnalyzer:
    """個別銘柄分析と取引推奨生成"""
    
    def __init__(self, 
                 target_csv_path: str = "data/target.csv",
                 portfolio_path: str = "data/portfolio.json",
                 trading_agent: Optional[Any] = None,
                 data_fetcher: Optional[Any] = None):
        """
        Args:
            target_csv_path: target.csvのパス
            portfolio_path: ポートフォリオファイルのパス
            trading_agent: AIエージェント
            data_fetcher: マーケットデータ取得クラス
        """
        self.target_csv_path = Path(target_csv_path)
        self.portfolio_path = Path(portfolio_path)
        self.trading_agent = trading_agent
        self.data_fetcher = data_fetcher  # or MarketDataFetcher()
        self.portfolio = self._load_portfolio()
        
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
            
            # センチメント分析
            sentiment_score = self._analyze_sentiment(ticker, company_name)
            
            # AI予測（利用可能な場合）
            ai_prediction = self._get_ai_prediction(ticker) if self.trading_agent else None
            
            # 指標計算
            indicators = self._calculate_indicators(ticker)
            
            result = AnalysisResult(
                ticker=ticker,
                company_name=company_name,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                sentiment_score=sentiment_score,
                ai_prediction=ai_prediction,
                indicators=indicators
            )
            
            logger.info(f"分析完了: {company_name} - 総合スコア: {result.total_score:.1f}")
            
            return result
            
        except Exception as e:
            logger.error(f"分析エラー {ticker}: {e}")
            # エラー時はデフォルト値を返す
            return AnalysisResult(
                ticker=ticker,
                company_name=company_name,
                technical_score=50,
                fundamental_score=50,
                sentiment_score=0
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
    
    def _analyze_sentiment(self, ticker: str, company_name: str) -> float:
        """センチメント分析スコア計算"""
        # 実際の実装では、ニュースAPIやSNSデータを分析
        # ここでは簡易的にランダム値を返す
        import random
        
        # -1 to 1 の範囲でランダムなセンチメントスコアを生成
        return random.uniform(-0.5, 0.5)
    
    def _get_ai_prediction(self, ticker: str) -> Optional[Dict[str, Any]]:
        """AI予測を取得"""
        if not self.trading_agent:
            return None
        
        try:
            # AI エージェントによる予測
            prediction = self.trading_agent.predict(ticker)
            return prediction
        except Exception as e:
            logger.warning(f"AI予測エラー {ticker}: {e}")
            return None
    
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