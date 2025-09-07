"""
Investment Analyzer - リファクタリング版総合投資分析システム
クリーンアーキテクチャによる高保守性投資判断AI

このモジュールは、30社の日本株式を対象とした総合的な投資分析を実行し、
テクニカル分析に基づく投資推奨を生成します。
"""

import sys
import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# プロジェクト内モジュールのインポート
from config import config
from data_fetcher import DataFetcher
from technical_analyzer import TechnicalAnalyzer, TechnicalIndicators
from investment_scorer import InvestmentScorer, ScoringResult
from investment_limiter import InvestmentLimiter, LimitedInvestment
from report_generator import ReportGenerator

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 警告フィルターを適用
for filter_rule in config.system.warning_filters:
    warnings.filterwarnings(*filter_rule.split(":"))


class InvestmentAnalyzer:
    """
    総合投資分析クラス
    
    テクニカル分析、スコアリング、リスク管理、レポート生成を統合した
    投資判断システムのメインクラス
    """
    
    def __init__(self, max_investment_per_stock: Optional[float] = None):
        """
        投資分析システムを初期化
        
        Args:
            max_investment_per_stock: 1株当たりの投資上限（デフォルト: 2000.0）
        """
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.investment_scorer = InvestmentScorer()
        self.investment_limiter = InvestmentLimiter()
        self.report_generator = ReportGenerator()
        
        # 投資制限の設定
        if max_investment_per_stock is not None:
            config.investment_limits.max_investment_per_stock = max_investment_per_stock
        
        logger.info("Investment Analyzer initialized successfully")
    
    def analyze_single_stock(
        self, 
        symbol: str, 
        company_name: str, 
        market_cap_millions: float = 1500.0
    ) -> Optional[Dict[str, Any]]:
        """
        単一銘柄の投資分析を実行
        
        Args:
            symbol: 証券シンボル（例: '7203.T'）
            company_name: 企業名
            market_cap_millions: 時価総額（百万円）
        
        Returns:
            分析結果の辞書、失敗時はNone
        """
        try:
            logger.debug(f"Analyzing {symbol} ({company_name})")
            
            # 株価データ取得
            stock_data = self.data_fetcher.get_stock_data(symbol)
            if stock_data is None or stock_data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            current_price = float(stock_data['Close'].iloc[-1])
            
            # テクニカル分析実行
            technical_indicators = self.technical_analyzer.analyze_stock(stock_data)
            
            # 投資スコア計算
            scoring_result = self.investment_scorer.calculate_investment_score(
                technical_indicators, 
                current_price, 
                market_cap_millions
            )
            
            # テクニカルシグナル生成
            technical_signals = self.technical_analyzer.get_technical_signals(
                technical_indicators, 
                current_price
            )
            
            # 小型株ボーナス適用
            final_score = self.investment_scorer.add_small_stock_bonus(
                scoring_result.total_score, 
                market_cap_millions
            )
            
            # 最終的な投資推奨を取得
            final_recommendation = config.get_investment_recommendation(final_score)
            
            # 結果をまとめる
            result = {
                '証券コード': symbol.replace('.T', ''),
                '企業名': company_name,
                'シンボル': symbol,
                '現在株価': current_price,
                '時価総額': market_cap_millions,
                '投資スコア': final_score,
                '投資判断': final_recommendation["judgment"],
                'ポジションサイズ': final_recommendation["position_size"],
                '目標利益率': final_recommendation["target_profit"],
                '損切りライン': final_recommendation["stop_loss"],
                '推奨保有期間': final_recommendation["holding_period"],
                'テクニカルシグナル': ", ".join(technical_signals),
                '1日変化率': technical_indicators.price_change_1d or 0.0,
                '5日変化率': technical_indicators.price_change_5d or 0.0,
                '25日変化率': technical_indicators.price_change_25d or 0.0,
                'RSI': technical_indicators.rsi,
                'ボラティリティ': technical_indicators.volatility,
                # スコア詳細
                'トレンドスコア': scoring_result.component_scores.get("trend", 0),
                'RSIスコア': scoring_result.component_scores.get("rsi", 0),
                'MACDスコア': scoring_result.component_scores.get("macd", 0),
                'モメンタムスコア': scoring_result.component_scores.get("momentum", 0)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)
            return None
    
    def analyze_all_companies(self) -> pd.DataFrame:
        """
        全対象企業の投資分析を実行
        
        Returns:
            分析結果のDataFrame（投資スコア降順でソート済み）
        """
        logger.info("🚀 総合投資判断AI - 30社完全分析システム")
        logger.info("=" * 80)
        
        # ターゲット企業を読み込み
        try:
            companies_df = self.data_fetcher.load_target_companies()
            logger.info(f"🔍 全{len(companies_df)}社の総合分析開始...")
        except Exception as e:
            logger.error(f"Failed to load target companies: {e}")
            raise
        
        # 株価データを一括取得
        symbols = [
            self.data_fetcher.create_symbol_from_code(row['証券コード'])
            for _, row in companies_df.iterrows()
        ]
        
        logger.info("=" * 80)
        logger.info(f"📊 {len(symbols)}銘柄のデータ取得中...")
        
        stock_data_dict = self.data_fetcher.batch_fetch_stock_data(symbols)
        
        logger.info(f"\n✅ {len([v for v in stock_data_dict.values() if v is not None])}/{len(symbols)} 銘柄のデータ取得完了\n")
        
        # 各銘柄を分析
        results = []
        for _, company in companies_df.iterrows():
            symbol = self.data_fetcher.create_symbol_from_code(company['証券コード'])
            company_name = company['企業名']
            market_cap = company.get('時価総額 (百万円)', 1500)
            
            print(f"📈 {company_name} ({symbol}) 分析中...")
            
            # 分析実行
            analysis_result = self.analyze_single_stock(symbol, company_name, market_cap)
            
            if analysis_result:
                results.append(analysis_result)
                score = analysis_result['投資スコア']
                judgment = analysis_result['投資判断']
                print(f"✅ スコア: {score:.0f}点 - {judgment}")
            else:
                print("❌ データ取得失敗")
                # 失敗時の最小限データ
                results.append({
                    '証券コード': company['証券コード'],
                    '企業名': company_name,
                    'シンボル': symbol,
                    '現在株価': 0.0,
                    '時価総額': market_cap,
                    '投資スコア': 0,
                    '投資判断': 'データなし',
                    'ポジションサイズ': 0.0,
                    '目標利益率': 0.0,
                    '損切りライン': 0.0,
                    '推奨保有期間': '分析不可',
                    'テクニカルシグナル': 'データ不足',
                    '1日変化率': 0.0,
                    '5日変化率': 0.0,
                    '25日変化率': 0.0,
                    'RSI': None,
                    'ボラティリティ': None
                })
        
        # DataFrameに変換してソート
        df = pd.DataFrame(results)
        df = df.sort_values('投資スコア', ascending=False)
        
        logger.info(f"\n📊 全{len(df)}社の分析完了")
        
        return df
    
    def generate_investment_recommendations(
        self, 
        analysis_df: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], Dict[str, LimitedInvestment]]:
        """
        投資推奨とポートフォリオ制限を生成
        
        Args:
            analysis_df: 分析結果DataFrame
        
        Returns:
            投資推奨リストとリミッター適用後の投資配分
        """
        # 買い推奨銘柄を抽出（スコア55以上）
        buy_recommendations = analysis_df[
            analysis_df['投資スコア'] >= config.investment_thresholds.buy_threshold
        ].copy()
        
        if buy_recommendations.empty:
            logger.warning("No buy recommendations found")
            return [], {}
        
        # 推奨リストを作成
        recommendations = []
        for _, stock in buy_recommendations.iterrows():
            recommendations.append({
                'symbol': stock['シンボル'],
                'company_name': stock['企業名'],
                'current_price': stock['現在株価'],
                'position_size': abs(stock['ポジションサイズ']),
                'investment_score': stock['投資スコア'],
                'judgment': stock['投資判断']
            })
        
        # 投資制限を適用
        allocations = self.investment_limiter.calculate_portfolio_allocation(recommendations)
        
        logger.info(f"Generated {len(recommendations)} investment recommendations")
        
        return recommendations, allocations
    
    def run_complete_analysis(self) -> pd.DataFrame:
        """
        完全な投資分析を実行（分析 → 制限適用 → レポート生成）
        
        Returns:
            分析結果DataFrame
        """
        try:
            # 1. 全社分析実行
            analysis_df = self.analyze_all_companies()
            
            # 2. 投資推奨生成
            recommendations, allocations = self.generate_investment_recommendations(analysis_df)
            
            # 3. レポート生成
            analysis_results = analysis_df.to_dict('records')
            self.report_generator.generate_console_report(
                analysis_results, 
                allocations,
                config.investment_limits.max_investment_per_stock
            )
            
            # 4. CSV保存
            self.report_generator.save_csv_report(analysis_results)
            
            # 5. 安全性検証
            if allocations:
                safety_validation = self.investment_limiter.validate_investment_safety(allocations)
                warnings_list = self.investment_limiter.generate_risk_warning(safety_validation)
                
                if warnings_list:
                    print("\n🚨 【リスク警告】")
                    print("-" * 50)
                    for warning in warnings_list:
                        print(warning)
                
                # 投資サマリー
                summary = self.investment_limiter.format_investment_summary(allocations, safety_validation)
                print(f"\n{summary}")
            
            logger.info("Complete analysis finished successfully")
            return analysis_df
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}", exc_info=True)
            raise


def main():
    """
    メイン実行関数
    
    リファクタリング版の投資分析システムを実行
    """
    try:
        # 投資分析システム初期化（1株当たり2000円制限）
        analyzer = InvestmentAnalyzer(max_investment_per_stock=2000.0)
        
        # 完全な分析を実行
        results_df = analyzer.run_complete_analysis()
        
        return results_df
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    results = main()