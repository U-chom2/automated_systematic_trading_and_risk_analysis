"""包括的リスク評価システム

市場リスク、ボラティリティリスク、流動性リスク、企業固有リスクなどを
総合的に評価するリスク分析システム
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger_utils import create_dual_logger
from .models import RiskLevel

logger = create_dual_logger(__name__, console_output=True)


class RiskCategory(Enum):
    """リスクカテゴリ"""
    MARKET_RISK = "market_risk"
    VOLATILITY_RISK = "volatility_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    COMPANY_RISK = "company_risk"
    SECTOR_RISK = "sector_risk"
    CONCENTRATION_RISK = "concentration_risk"


@dataclass
class RiskMetrics:
    """リスク指標"""
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    expected_shortfall: float  # 期待ショートフォール
    beta: float  # ベータ値
    alpha: float  # アルファ値
    sharpe_ratio: float  # シャープレシオ
    sortino_ratio: float  # ソルティノレシオ
    max_drawdown: float  # 最大ドローダウン
    volatility: float  # ボラティリティ（年率）
    correlation_with_market: float  # 市場との相関


@dataclass
class LiquidityMetrics:
    """流動性指標"""
    avg_daily_volume: float  # 平均日次出来高
    avg_bid_ask_spread: float  # 平均ビッドアスクスプレッド
    volume_weighted_price: float  # 出来高加重平均価格
    liquidity_score: float  # 流動性スコア（0-100）
    days_to_liquidate: float  # 清算日数


@dataclass
class CompanyRiskMetrics:
    """企業リスク指標"""
    debt_to_equity: float  # 負債比率
    current_ratio: float  # 流動比率
    quick_ratio: float  # 当座比率
    interest_coverage: float  # インタレストカバレッジレシオ
    altman_z_score: float  # アルトマンZスコア
    piotroski_f_score: int  # ピオトロスキーFスコア
    financial_strength_score: float  # 財務健全性スコア


@dataclass
class RiskAssessment:
    """包括的リスク評価結果"""
    ticker: str
    company_name: str
    assessment_date: datetime
    
    # 各リスクカテゴリのスコア（0-100、高いほど危険）
    market_risk_score: float
    volatility_risk_score: float
    liquidity_risk_score: float
    company_risk_score: float
    sector_risk_score: float
    
    # 詳細メトリクス
    risk_metrics: RiskMetrics
    liquidity_metrics: LiquidityMetrics
    company_metrics: CompanyRiskMetrics
    
    # 総合リスク評価
    overall_risk_score: float
    risk_level: RiskLevel
    risk_warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ComprehensiveRiskEvaluator:
    """包括的リスク評価システム"""
    
    def __init__(
        self,
        market_benchmark: str = "^N225",  # 日経平均
        risk_free_rate: float = 0.005,  # リスクフリーレート（0.5%）
        lookback_days: int = 252  # 1年
    ):
        """初期化
        
        Args:
            market_benchmark: 市場ベンチマークティッカー
            risk_free_rate: リスクフリーレート（年率）
            lookback_days: 分析対象期間（日数）
        """
        self.market_benchmark = market_benchmark
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days
        
        # ベンチマークデータキャッシュ
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.benchmark_returns: Optional[pd.Series] = None
        
        logger.info("包括的リスク評価システム初期化完了")
    
    def evaluate_comprehensive_risk(
        self,
        ticker: str,
        company_name: str,
        portfolio_weight: float = 0.0
    ) -> RiskAssessment:
        """包括的リスク評価を実行
        
        Args:
            ticker: ティッカーシンボル
            company_name: 企業名
            portfolio_weight: ポートフォリオ内での重み（0-1）
            
        Returns:
            包括的リスク評価結果
        """
        logger.info(f"包括的リスク評価開始: {company_name} ({ticker})")
        
        try:
            # データ取得
            stock_data = self._get_stock_data(ticker)
            benchmark_data = self._get_benchmark_data()
            
            if stock_data.empty:
                logger.warning(f"株価データが取得できません: {ticker}")
                return self._create_default_assessment(ticker, company_name)
            
            # 各リスク評価を実行
            market_risk_score = self._evaluate_market_risk(stock_data, benchmark_data)
            volatility_risk_score = self._evaluate_volatility_risk(stock_data)
            liquidity_risk_score = self._evaluate_liquidity_risk(stock_data, ticker)
            company_risk_score = self._evaluate_company_risk(ticker)
            sector_risk_score = self._evaluate_sector_risk(ticker)
            
            # 詳細メトリクス計算
            risk_metrics = self._calculate_risk_metrics(stock_data, benchmark_data)
            liquidity_metrics = self._calculate_liquidity_metrics(stock_data, ticker)
            company_metrics = self._calculate_company_metrics(ticker)
            
            # 総合リスクスコア計算（重み付き平均）
            weights = {
                'market': 0.25,
                'volatility': 0.25,
                'liquidity': 0.15,
                'company': 0.25,
                'sector': 0.10
            }
            
            overall_risk_score = (
                market_risk_score * weights['market'] +
                volatility_risk_score * weights['volatility'] +
                liquidity_risk_score * weights['liquidity'] +
                company_risk_score * weights['company'] +
                sector_risk_score * weights['sector']
            )
            
            # リスクレベル判定
            risk_level = self._determine_risk_level(overall_risk_score)
            
            # 警告と推奨事項生成
            warnings = self._generate_risk_warnings(
                market_risk_score, volatility_risk_score, liquidity_risk_score,
                company_risk_score, sector_risk_score, risk_metrics, company_metrics
            )
            
            recommendations = self._generate_recommendations(
                overall_risk_score, risk_level, portfolio_weight, risk_metrics
            )
            
            assessment = RiskAssessment(
                ticker=ticker,
                company_name=company_name,
                assessment_date=datetime.now(),
                market_risk_score=market_risk_score,
                volatility_risk_score=volatility_risk_score,
                liquidity_risk_score=liquidity_risk_score,
                company_risk_score=company_risk_score,
                sector_risk_score=sector_risk_score,
                risk_metrics=risk_metrics,
                liquidity_metrics=liquidity_metrics,
                company_metrics=company_metrics,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                risk_warnings=warnings,
                recommendations=recommendations
            )
            
            logger.info(f"リスク評価完了: {company_name} - 総合リスク: {overall_risk_score:.1f} ({risk_level.value})")
            
            return assessment
            
        except Exception as e:
            logger.error(f"リスク評価エラー {ticker}: {e}")
            return self._create_default_assessment(ticker, company_name)
    
    def _get_stock_data(self, ticker: str) -> pd.DataFrame:
        """株価データを取得"""
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 50)  # バッファを追加
            
            hist = stock.history(start=start_date, end=end_date)
            return hist.tail(self.lookback_days) if len(hist) > self.lookback_days else hist
            
        except Exception as e:
            logger.warning(f"株価データ取得エラー {ticker}: {e}")
            return pd.DataFrame()
    
    def _get_benchmark_data(self) -> pd.DataFrame:
        """ベンチマークデータを取得（キャッシュ付き）"""
        if self.benchmark_data is not None:
            return self.benchmark_data
        
        try:
            benchmark = yf.Ticker(self.market_benchmark)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 50)
            
            self.benchmark_data = benchmark.history(start=start_date, end=end_date)
            self.benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
            
            return self.benchmark_data
            
        except Exception as e:
            logger.warning(f"ベンチマークデータ取得エラー: {e}")
            return pd.DataFrame()
    
    def _evaluate_market_risk(self, stock_data: pd.DataFrame, benchmark_data: pd.DataFrame) -> float:
        """市場リスク評価（0-100、高いほど危険）"""
        try:
            if stock_data.empty or benchmark_data.empty:
                return 50.0
            
            # リターン計算
            stock_returns = stock_data['Close'].pct_change().dropna()
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            
            # 日付を合わせる
            aligned_data = pd.concat([stock_returns, benchmark_returns], axis=1, join='inner')
            aligned_data.columns = ['stock', 'benchmark']
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 30:  # 最低30日のデータが必要
                return 50.0
            
            # ベータ計算
            covariance = aligned_data['stock'].cov(aligned_data['benchmark'])
            benchmark_variance = aligned_data['benchmark'].var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
            
            # 相関係数
            correlation = aligned_data['stock'].corr(aligned_data['benchmark'])
            
            # 市場リスクスコア計算
            # ベータが1より大きいほど、相関が高いほどリスク高
            beta_score = min(abs(beta) * 30, 70)  # ベータによるスコア
            correlation_score = abs(correlation) * 30  # 相関によるスコア
            
            market_risk_score = min(beta_score + correlation_score, 100)
            
            return market_risk_score
            
        except Exception as e:
            logger.debug(f"市場リスク評価エラー: {e}")
            return 50.0
    
    def _evaluate_volatility_risk(self, stock_data: pd.DataFrame) -> float:
        """ボラティリティリスク評価（0-100、高いほど危険）"""
        try:
            if stock_data.empty:
                return 50.0
            
            returns = stock_data['Close'].pct_change().dropna()
            
            if len(returns) < 30:
                return 50.0
            
            # 年率ボラティリティ
            annual_volatility = returns.std() * np.sqrt(252)
            
            # VaR計算（95%）
            var_95 = np.percentile(returns, 5)
            
            # 最大ドローダウン計算
            cumulative = (1 + returns).cumprod()
            max_dd = (cumulative / cumulative.expanding().max() - 1).min()
            
            # ボラティリティスコア
            # 年率30%以上で高リスク、15%以下で低リスク
            volatility_score = min(annual_volatility * 200, 100)  # 30%で60点
            
            # VaRスコア（日次損失5%以上で高リスク）
            var_score = min(abs(var_95) * 500, 40)  # 5%で25点
            
            # ドローダウンスコア（30%以上で高リスク）
            dd_score = min(abs(max_dd) * 100, 30)  # 30%で30点
            
            volatility_risk_score = min(volatility_score + var_score + dd_score, 100)
            
            return volatility_risk_score
            
        except Exception as e:
            logger.debug(f"ボラティリティリスク評価エラー: {e}")
            return 50.0
    
    def _evaluate_liquidity_risk(self, stock_data: pd.DataFrame, ticker: str) -> float:
        """流動性リスク評価（0-100、高いほど危険）"""
        try:
            if stock_data.empty:
                return 50.0
            
            # 出来高分析
            volumes = stock_data['Volume']
            avg_volume = volumes.mean()
            volume_std = volumes.std()
            
            # 最近の出来高トレンド
            recent_volume = volumes.tail(20).mean()
            volume_trend = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 出来高の一貫性（CV: 変動係数）
            volume_cv = volume_std / avg_volume if avg_volume > 0 else 0
            
            # 価格レンジ分析（流動性の代理指標）
            price_ranges = stock_data['High'] - stock_data['Low']
            avg_range = price_ranges.mean()
            range_to_price = avg_range / stock_data['Close'].mean()
            
            # 流動性リスクスコア計算
            volume_score = 50  # デフォルト
            
            if avg_volume < 10000:  # 日次出来高1万未満
                volume_score = 80
            elif avg_volume < 50000:  # 5万未満
                volume_score = 60
            elif avg_volume > 500000:  # 50万以上
                volume_score = 20
            
            # 出来高変動性スコア
            cv_score = min(volume_cv * 100, 30)
            
            # 価格レンジスコア
            range_score = min(range_to_price * 500, 20)
            
            liquidity_risk_score = min(volume_score + cv_score + range_score, 100)
            
            return liquidity_risk_score
            
        except Exception as e:
            logger.debug(f"流動性リスク評価エラー: {e}")
            return 50.0
    
    def _evaluate_company_risk(self, ticker: str) -> float:
        """企業固有リスク評価（0-100、高いほど危険）"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            risk_factors = []
            
            # 財務比率分析
            debt_to_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else None
            current_ratio = info.get('currentRatio', 0)
            quick_ratio = info.get('quickRatio', 0)
            
            # 負債比率リスク
            if debt_to_equity is not None:
                if debt_to_equity > 2.0:  # 200%以上
                    risk_factors.append(30)
                elif debt_to_equity > 1.0:  # 100%以上
                    risk_factors.append(20)
                else:
                    risk_factors.append(10)
            
            # 流動性比率リスク
            if current_ratio > 0:
                if current_ratio < 1.0:  # 流動比率1未満
                    risk_factors.append(25)
                elif current_ratio < 1.5:
                    risk_factors.append(15)
                else:
                    risk_factors.append(5)
            
            # 収益性リスク
            roe = info.get('returnOnEquity', 0)
            if roe is not None and roe != 0:
                if roe < 0:  # 負のROE
                    risk_factors.append(30)
                elif roe < 0.05:  # 5%未満
                    risk_factors.append(20)
                else:
                    risk_factors.append(10)
            
            # 企業規模リスク（時価総額）
            market_cap = info.get('marketCap', 0)
            if market_cap > 0:
                if market_cap < 10_000_000_000:  # 100億未満
                    risk_factors.append(20)
                elif market_cap < 100_000_000_000:  # 1000億未満
                    risk_factors.append(10)
                else:
                    risk_factors.append(5)
            
            company_risk_score = sum(risk_factors) if risk_factors else 50.0
            return min(company_risk_score, 100)
            
        except Exception as e:
            logger.debug(f"企業リスク評価エラー: {e}")
            return 50.0
    
    def _evaluate_sector_risk(self, ticker: str) -> float:
        """セクターリスク評価（0-100、高いほど危険）"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            
            # セクター別リスクスコア（日本市場基準）
            high_risk_sectors = [
                'Biotechnology', 'Technology', 'Real Estate',
                'Energy', 'Materials', 'Airlines'
            ]
            
            medium_risk_sectors = [
                'Industrials', 'Consumer Discretionary', 'Financials',
                'Communication Services'
            ]
            
            low_risk_sectors = [
                'Consumer Staples', 'Healthcare', 'Utilities',
                'Consumer Defensive'
            ]
            
            if any(keyword in sector or keyword in industry for keyword in high_risk_sectors):
                return 70.0
            elif any(keyword in sector or keyword in industry for keyword in medium_risk_sectors):
                return 50.0
            elif any(keyword in sector or keyword in industry for keyword in low_risk_sectors):
                return 30.0
            else:
                return 50.0  # 不明セクター
                
        except Exception as e:
            logger.debug(f"セクターリスク評価エラー: {e}")
            return 50.0
    
    def _calculate_risk_metrics(self, stock_data: pd.DataFrame, benchmark_data: pd.DataFrame) -> RiskMetrics:
        """詳細リスクメトリクス計算"""
        try:
            returns = stock_data['Close'].pct_change().dropna()
            
            if len(returns) < 30:
                return self._default_risk_metrics()
            
            # VaR計算
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall (CVaR)
            es_returns = returns[returns <= var_95]
            expected_shortfall = es_returns.mean() if len(es_returns) > 0 else var_95
            
            # ベータとアルファ
            if not benchmark_data.empty:
                benchmark_returns = benchmark_data['Close'].pct_change().dropna()
                aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
                
                if len(aligned) >= 30:
                    stock_aligned = aligned.iloc[:, 0]
                    bench_aligned = aligned.iloc[:, 1]
                    
                    covariance = stock_aligned.cov(bench_aligned)
                    benchmark_variance = bench_aligned.var()
                    beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
                    
                    # アルファ計算（CAPM）
                    stock_mean_return = stock_aligned.mean()
                    benchmark_mean_return = bench_aligned.mean()
                    alpha = stock_mean_return - (self.risk_free_rate/252 + beta * (benchmark_mean_return - self.risk_free_rate/252))
                    
                    correlation = stock_aligned.corr(bench_aligned)
                else:
                    beta, alpha, correlation = 1.0, 0.0, 0.0
            else:
                beta, alpha, correlation = 1.0, 0.0, 0.0
            
            # シャープレシオ
            excess_returns = returns - self.risk_free_rate/252
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
            
            # ソルティノレシオ
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
            sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std != 0 else 0
            
            # 最大ドローダウン
            cumulative = (1 + returns).cumprod()
            max_drawdown = (cumulative / cumulative.expanding().max() - 1).min()
            
            # ボラティリティ（年率）
            volatility = returns.std() * np.sqrt(252)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                beta=beta,
                alpha=alpha,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                correlation_with_market=correlation
            )
            
        except Exception as e:
            logger.debug(f"リスクメトリクス計算エラー: {e}")
            return self._default_risk_metrics()
    
    def _calculate_liquidity_metrics(self, stock_data: pd.DataFrame, ticker: str) -> LiquidityMetrics:
        """流動性メトリクス計算"""
        try:
            if stock_data.empty:
                return self._default_liquidity_metrics()
            
            # 平均日次出来高
            avg_daily_volume = stock_data['Volume'].mean()
            
            # 出来高加重平均価格（簡易計算）
            vwap = (stock_data['Volume'] * stock_data['Close']).sum() / stock_data['Volume'].sum()
            
            # 流動性スコア（出来高ベース）
            if avg_daily_volume > 1_000_000:
                liquidity_score = 90
            elif avg_daily_volume > 100_000:
                liquidity_score = 70
            elif avg_daily_volume > 10_000:
                liquidity_score = 50
            else:
                liquidity_score = 20
            
            # 清算日数の推定（仮定：1日の出来高の5%まで売却可能）
            days_to_liquidate = 20.0  # デフォルト
            
            return LiquidityMetrics(
                avg_daily_volume=avg_daily_volume,
                avg_bid_ask_spread=0.01,  # 仮想値（実際のスプレッドデータが必要）
                volume_weighted_price=vwap,
                liquidity_score=liquidity_score,
                days_to_liquidate=days_to_liquidate
            )
            
        except Exception as e:
            logger.debug(f"流動性メトリクス計算エラー: {e}")
            return self._default_liquidity_metrics()
    
    def _calculate_company_metrics(self, ticker: str) -> CompanyRiskMetrics:
        """企業リスクメトリクス計算"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # 基本財務比率
            debt_to_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
            current_ratio = info.get('currentRatio', 1.0)
            quick_ratio = info.get('quickRatio', 0.8)
            
            # 利益カバレッジ
            ebit = info.get('ebitda', 0)
            interest_expense = info.get('interestExpense', 0)
            interest_coverage = ebit / abs(interest_expense) if interest_expense and interest_expense != 0 else 10.0
            
            # 簡易アルトマンZスコア（公開データで計算可能な範囲）
            try:
                market_cap = info.get('marketCap', 0)
                total_debt = info.get('totalDebt', 0)
                total_assets = info.get('totalAssets', 1)
                
                working_capital = info.get('totalCurrentAssets', 0) - info.get('totalCurrentLiabilities', 0)
                retained_earnings = info.get('retainedEarnings', 0)
                ebit = info.get('ebit', 0)
                sales = info.get('totalRevenue', 1)
                
                if total_assets > 0:
                    z1 = working_capital / total_assets
                    z2 = retained_earnings / total_assets
                    z3 = ebit / total_assets
                    z4 = market_cap / total_debt if total_debt > 0 else 5.0
                    z5 = sales / total_assets
                    
                    altman_z = 1.2*z1 + 1.4*z2 + 3.3*z3 + 0.6*z4 + 1.0*z5
                else:
                    altman_z = 2.0
            except:
                altman_z = 2.0
            
            # 財務健全性スコア（0-100）
            financial_strength = 50.0  # デフォルト
            
            if current_ratio >= 2.0 and debt_to_equity < 0.5 and interest_coverage > 5:
                financial_strength = 85.0
            elif current_ratio >= 1.5 and debt_to_equity < 1.0 and interest_coverage > 2:
                financial_strength = 70.0
            elif current_ratio < 1.0 or debt_to_equity > 2.0 or interest_coverage < 1:
                financial_strength = 30.0
            
            return CompanyRiskMetrics(
                debt_to_equity=debt_to_equity,
                current_ratio=current_ratio,
                quick_ratio=quick_ratio,
                interest_coverage=interest_coverage,
                altman_z_score=altman_z,
                piotroski_f_score=5,  # 仮想値（詳細分析が必要）
                financial_strength_score=financial_strength
            )
            
        except Exception as e:
            logger.debug(f"企業メトリクス計算エラー: {e}")
            return self._default_company_metrics()
    
    def _determine_risk_level(self, overall_score: float) -> RiskLevel:
        """総合リスクスコアからリスクレベルを判定"""
        if overall_score >= 80:
            return RiskLevel.VERY_HIGH
        elif overall_score >= 60:
            return RiskLevel.HIGH
        elif overall_score >= 40:
            return RiskLevel.MEDIUM
        elif overall_score >= 20:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _generate_risk_warnings(
        self,
        market_risk: float,
        volatility_risk: float,
        liquidity_risk: float,
        company_risk: float,
        sector_risk: float,
        risk_metrics: RiskMetrics,
        company_metrics: CompanyRiskMetrics
    ) -> List[str]:
        """リスク警告を生成"""
        warnings = []
        
        if volatility_risk > 70:
            warnings.append(f"高ボラティリティ: 年率{risk_metrics.volatility:.1%}のボラティリティは平均を大きく上回ります")
        
        if risk_metrics.max_drawdown < -0.3:
            warnings.append(f"大きなドローダウン: 最大{abs(risk_metrics.max_drawdown):.1%}の下落履歴があります")
        
        if liquidity_risk > 70:
            warnings.append("流動性リスク: 出来高が低く、売買が困難になる可能性があります")
        
        if company_metrics.current_ratio < 1.0:
            warnings.append(f"流動性不足: 流動比率{company_metrics.current_ratio:.2f}は財務的な懸念を示します")
        
        if company_metrics.debt_to_equity > 2.0:
            warnings.append(f"高負債: 負債比率{company_metrics.debt_to_equity:.1f}は業界平均を上回ります")
        
        if company_metrics.altman_z_score < 1.8:
            warnings.append("財務危険信号: アルトマンZスコアが破綻リスクを示唆しています")
        
        if abs(risk_metrics.beta) > 1.5:
            warnings.append(f"高い市場感応度: ベータ{risk_metrics.beta:.2f}は市場変動に敏感です")
        
        return warnings
    
    def _generate_recommendations(
        self,
        overall_score: float,
        risk_level: RiskLevel,
        portfolio_weight: float,
        risk_metrics: RiskMetrics
    ) -> List[str]:
        """投資推奨事項を生成"""
        recommendations = []
        
        if risk_level == RiskLevel.VERY_HIGH:
            recommendations.append("投資回避: リスクが極めて高く、投資は推奨しません")
            recommendations.append("既存保有分の売却を検討してください")
        
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("小額投資のみ: ポートフォリオの5%以下に制限することを推奨")
            recommendations.append("頻繁な監視が必要です")
        
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("適度な投資: ポートフォリオの10-15%程度まで投資可能")
            recommendations.append("リスク管理としてストップロス設定を推奨")
        
        elif risk_level in [RiskLevel.LOW, RiskLevel.VERY_LOW]:
            recommendations.append("安全な投資先: ポートフォリオの主要構成銘柄として適用可能")
            if risk_metrics.sharpe_ratio > 1.0:
                recommendations.append("優秀なリスク調整後リターンを示しています")
        
        # ポートフォリオ集中度チェック
        if portfolio_weight > 0.2:
            recommendations.append("集中リスク警告: 単一銘柄への投資比率が20%を超えています")
        
        # ヘッジ推奨
        if abs(risk_metrics.beta) > 1.2:
            recommendations.append("ヘッジ検討: 高ベータ銘柄のため、ポートフォリオヘッジを検討してください")
        
        return recommendations
    
    def _create_default_assessment(self, ticker: str, company_name: str) -> RiskAssessment:
        """デフォルトのリスク評価を作成"""
        return RiskAssessment(
            ticker=ticker,
            company_name=company_name,
            assessment_date=datetime.now(),
            market_risk_score=50.0,
            volatility_risk_score=50.0,
            liquidity_risk_score=50.0,
            company_risk_score=50.0,
            sector_risk_score=50.0,
            risk_metrics=self._default_risk_metrics(),
            liquidity_metrics=self._default_liquidity_metrics(),
            company_metrics=self._default_company_metrics(),
            overall_risk_score=50.0,
            risk_level=RiskLevel.MEDIUM,
            risk_warnings=["データ不足により詳細なリスク分析ができませんでした"],
            recommendations=["より多くのデータが利用可能になるまで投資判断を控えることを推奨"]
        )
    
    def _default_risk_metrics(self) -> RiskMetrics:
        """デフォルトリスクメトリクス"""
        return RiskMetrics(
            var_95=-0.05,
            var_99=-0.08,
            expected_shortfall=-0.06,
            beta=1.0,
            alpha=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=-0.2,
            volatility=0.25,
            correlation_with_market=0.5
        )
    
    def _default_liquidity_metrics(self) -> LiquidityMetrics:
        """デフォルト流動性メトリクス"""
        return LiquidityMetrics(
            avg_daily_volume=100000,
            avg_bid_ask_spread=0.01,
            volume_weighted_price=1000.0,
            liquidity_score=50.0,
            days_to_liquidate=10.0
        )
    
    def _default_company_metrics(self) -> CompanyRiskMetrics:
        """デフォルト企業メトリクス"""
        return CompanyRiskMetrics(
            debt_to_equity=1.0,
            current_ratio=1.5,
            quick_ratio=1.0,
            interest_coverage=5.0,
            altman_z_score=2.0,
            piotroski_f_score=5,
            financial_strength_score=50.0
        )
    
    def evaluate_portfolio_risk(
        self,
        portfolio_assessments: List[RiskAssessment],
        portfolio_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """ポートフォリオレベルのリスク評価
        
        Args:
            portfolio_assessments: 個別銘柄のリスク評価リスト
            portfolio_weights: 各銘柄の重み
            
        Returns:
            ポートフォリオリスク分析結果
        """
        if not portfolio_assessments:
            return {"error": "評価対象がありません"}
        
        # 加重平均リスクスコア計算
        weighted_scores = {}
        total_weight = sum(portfolio_weights.values())
        
        for assessment in portfolio_assessments:
            weight = portfolio_weights.get(assessment.ticker, 0) / total_weight
            
            for risk_type in ['market_risk_score', 'volatility_risk_score', 
                            'liquidity_risk_score', 'company_risk_score', 'sector_risk_score']:
                if risk_type not in weighted_scores:
                    weighted_scores[risk_type] = 0
                weighted_scores[risk_type] += getattr(assessment, risk_type) * weight
        
        # ポートフォリオ全体のリスクレベル
        overall_portfolio_risk = sum(weighted_scores.values()) / len(weighted_scores)
        portfolio_risk_level = self._determine_risk_level(overall_portfolio_risk)
        
        # 集中リスク分析
        concentration_risk = max(portfolio_weights.values()) if portfolio_weights else 0
        
        # セクター分散分析（簡易）
        high_risk_concentration = sum(
            weight for assessment, weight in 
            zip(portfolio_assessments, portfolio_weights.values())
            if assessment.sector_risk_score > 60
        ) / total_weight if total_weight > 0 else 0
        
        return {
            "portfolio_risk_score": overall_portfolio_risk,
            "portfolio_risk_level": portfolio_risk_level.value,
            "risk_breakdown": weighted_scores,
            "concentration_risk": concentration_risk,
            "high_risk_sector_exposure": high_risk_concentration,
            "diversification_score": 100 - concentration_risk * 100,
            "recommendations": self._generate_portfolio_recommendations(
                overall_portfolio_risk, concentration_risk, high_risk_concentration
            )
        }
    
    def _generate_portfolio_recommendations(
        self,
        overall_risk: float,
        concentration_risk: float,
        high_risk_exposure: float
    ) -> List[str]:
        """ポートフォリオ推奨事項を生成"""
        recommendations = []
        
        if concentration_risk > 0.3:
            recommendations.append("集中リスク: 単一銘柄への集中度が高すぎます。分散を図ってください")
        
        if high_risk_exposure > 0.5:
            recommendations.append("高リスク偏重: ポートフォリオが高リスク銘柄に偏っています")
        
        if overall_risk > 70:
            recommendations.append("ポートフォリオ全体のリスクレベルが高すぎます。リスクの低い銘柄の追加を検討してください")
        
        if overall_risk < 30:
            recommendations.append("保守的なポートフォリオです。リターン向上のため、適度なリスク銘柄の追加も検討可能です")
        
        return recommendations