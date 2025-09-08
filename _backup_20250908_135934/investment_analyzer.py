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
from config import config, Config, TradingMode
from data_fetcher import DataFetcher
from technical_analyzer import TechnicalAnalyzer, TechnicalIndicators
from investment_scorer import InvestmentScorer, ScoringResult
from investment_limiter import InvestmentLimiter, LimitedInvestment, DayTradingRisk, StopLossOrder
from report_generator import ReportGenerator

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PPO Model Integration
try:
    from ppo_scoring_adapter import PPOScoringAdapter, create_ppo_adapter
    PPO_AVAILABLE = True
    logger.info("PPO Scoring Adapter available")
except ImportError as e:
    PPO_AVAILABLE = False
    logger.warning(f"PPO Scoring Adapter not available: {e}")

# 警告フィルターを適用
for filter_rule in config.system.warning_filters:
    warnings.filterwarnings(*filter_rule.split(":"))


class InvestmentAnalyzer:
    """
    総合投資分析クラス（デイトレード対応）
    
    Phase1-4の全機能を統合した完全なデイトレード対応投資判断システム
    """
    
    def __init__(self, config_instance: Optional[Config] = None, max_investment_per_stock: Optional[float] = None, use_ppo: bool = True):
        """
        投資分析システムを初期化（取引モード対応）
        
        Args:
            config_instance: 設定インスタンス（Noneの場合はデフォルト）
            max_investment_per_stock: 1株当たりの投資上限（デフォルト: 2000.0）
            use_ppo: PPOモデルを使用するか（デフォルト: True）
        """
        self.config = config_instance or config
        self.trading_mode = getattr(self.config, 'trading_mode', TradingMode.LONG_TERM)
        self.use_ppo = use_ppo and PPO_AVAILABLE
        
        # 各コンポーネントを統一設定で初期化
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.investment_scorer = InvestmentScorer(self.config)
        self.investment_limiter = InvestmentLimiter(self.config)
        self.report_generator = ReportGenerator()
        
        # PPOアダプター初期化
        self.ppo_adapter = None
        if self.use_ppo:
            try:
                self.ppo_adapter = create_ppo_adapter(config_instance=self.config)
                logger.info("PPO Scoring Adapter initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PPO Adapter: {e}")
                self.use_ppo = False
                logger.warning("Falling back to traditional scoring method")
        
        # 投資制限の設定
        if max_investment_per_stock is not None:
            self.config.investment_limits.max_investment_per_stock = max_investment_per_stock
        
        scoring_method = "PPO強化学習" if self.use_ppo else "従来テクニカル"
        mode_text = "デイトレード" if self.trading_mode == TradingMode.DAY_TRADING else "中長期"
        logger.info(f"Investment Analyzer initialized successfully - Mode: {mode_text}, Scoring: {scoring_method}")
    
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
            
            # テクニカル分析実行（取引モード対応）
            if self.trading_mode == TradingMode.DAY_TRADING:
                # デイトレード: 強化版テクニカル分析
                technical_indicators = self.technical_analyzer.analyze_stock_enhanced(stock_data)
                technical_signals = self.technical_analyzer.get_enhanced_technical_signals(
                    technical_indicators, current_price
                )
            else:
                # 中長期: 従来のテクニカル分析
                technical_indicators = self.technical_analyzer.analyze_stock(stock_data)
                technical_signals = self.technical_analyzer.get_technical_signals(
                    technical_indicators, current_price
                )
            
            # 投資スコア計算（PPOまたは従来メソッド）
            if self.use_ppo and self.ppo_adapter:
                # PPOモデルによるスコアリング
                logger.debug(f"Using PPO scoring for {symbol}")
                scoring_result = self.ppo_adapter.calculate_investment_score(
                    technical_indicators,
                    current_price,
                    market_cap_millions,
                    symbol
                )
            else:
                # 従来のテクニカルスコアリング
                logger.debug(f"Using traditional scoring for {symbol}")
                scoring_result = self.investment_scorer.calculate_investment_score(
                    technical_indicators, 
                    current_price, 
                    market_cap_millions
                )
            
            # 小型株ボーナス適用
            final_score = self.investment_scorer.add_small_stock_bonus(
                scoring_result.total_score, 
                market_cap_millions
            )
            
            # 最終的な投資推奨を取得（設定インスタンス使用）
            final_recommendation = self.config.get_investment_recommendation(final_score)
            
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
                'ボラティリティ': getattr(technical_indicators, 'volatility', None) or getattr(technical_indicators, 'short_term_volatility', None),
                # スコア詳細
                'トレンドスコア': scoring_result.component_scores.get("trend", 0),
                'RSIスコア': scoring_result.component_scores.get("rsi", 0),
                'MACDスコア': scoring_result.component_scores.get("macd", 0),
                'モメンタムスコア': scoring_result.component_scores.get("momentum", 0)
            }
            
            # デイトレード固有情報を追加
            if self.trading_mode == TradingMode.DAY_TRADING:
                result.update({
                    '短期ボラティリティ': technical_indicators.short_term_volatility,
                    '3日モメンタム': technical_indicators.momentum_3d,
                    '5日モメンタム': technical_indicators.momentum_5d,
                    'イントラデイ高値比': technical_indicators.intraday_high_ratio,
                    'イントラデイ安値比': technical_indicators.intraday_low_ratio,
                    '5日出来高比率': technical_indicators.volume_ratio_5d,
                    '3日価格変化率': technical_indicators.price_change_3d
                })
            
            # PPOモデル固有情報を追加
            if self.use_ppo and self.ppo_adapter:
                ppo_details = scoring_result.analysis_details
                result.update({
                    'PPOモデル': ppo_details.get('model_path', 'Unknown'),
                    'PPOアクション値': ppo_details.get('ppo_action_value', 0.0),
                    'PPO判断': ppo_details.get('ppo_action_interpretation', 'Unknown'),
                    'PPO信頼度': ppo_details.get('confidence_score', 0.0),
                    '分析手法': 'PPO強化学習モデル'
                })
            else:
                result.update({
                    '分析手法': '従来テクニカル分析'
                })
            
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
        # 買い推奨銘柄を抽出（取引モード対応閾値）
        buy_recommendations = analysis_df[
            analysis_df['投資スコア'] >= self.config.investment_thresholds.buy_threshold
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
                self.config.investment_limits.max_investment_per_stock
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
                
                # 投資サマリー（取引モード対応）
                if self.trading_mode == TradingMode.DAY_TRADING:
                    summary = self.investment_limiter.format_daytrading_summary(allocations, safety_validation)
                    print(f"\n{summary}")
                    
                    # 日次損失状況表示
                    risk_summary = self.investment_limiter.get_risk_management_summary()
                    print(f"\n📈 リスク管理状況:")
                    print(f"  日次利用率: {risk_summary.get('daily_limit_utilization', 0):.1f}%")
                    print(f"  残りポジション枠: {risk_summary.get('remaining_position_slots', 0)}銀柄")
                else:
                    summary = self.investment_limiter.format_investment_summary(allocations, safety_validation)
                    print(f"\n{summary}")
            
                # 6. デイトレードの場合: 損切り情報表示
            if self.trading_mode == TradingMode.DAY_TRADING and allocations:
                # ダミーのポジションで損切りテスト
                test_positions = self._create_test_positions(recommendations)
                if test_positions:
                    stop_loss_orders = self.investment_limiter.calculate_stop_loss_orders(test_positions)
                    stop_summary = self.investment_limiter.format_stop_loss_summary(stop_loss_orders)
                    print(f"\n{stop_summary}")
            
            mode_text = "デイトレード" if self.trading_mode == TradingMode.DAY_TRADING else "中長期"
            logger.info(f"Complete {mode_text} analysis finished successfully")
            return analysis_df
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}", exc_info=True)
            raise


    def _create_test_positions(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """テスト用ポジションを作成（損切りデモ用）"""
        test_positions = {}
        
        for rec in recommendations[:3]:  # 最初の3銀柄でテスト
            symbol = rec['symbol']
            current_price = rec['current_price']
            
            # ランダムなエントリー価格を設定（少し高め）
            entry_price = current_price * np.random.uniform(1.01, 1.05)
            shares = 10  # 固定株数
            
            test_positions[symbol] = {
                'entry_price': entry_price,
                'current_price': current_price,
                'shares': shares
            }
        
        return test_positions
    
    def switch_trading_mode(self, new_mode: TradingMode) -> None:
        """取引モードを切り替える
        
        Args:
            new_mode: 新しい取引モード
        """
        if new_mode == self.trading_mode:
            return  # 既に同じモード
        
        self.config.switch_trading_mode(new_mode)
        self.trading_mode = new_mode
        
        # 各コンポーネントを再初期化
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.investment_scorer = InvestmentScorer(self.config)
        self.investment_limiter = InvestmentLimiter(self.config)
        
        # PPOアダプターも再初期化
        if self.use_ppo:
            try:
                self.ppo_adapter = create_ppo_adapter(config_instance=self.config)
                logger.info("PPO Adapter reinitialized for new trading mode")
            except Exception as e:
                logger.error(f"Failed to reinitialize PPO Adapter: {e}")
                self.use_ppo = False
        
        mode_text = "デイトレード" if new_mode == TradingMode.DAY_TRADING else "中長期"
        logger.info(f"Trading mode switched to: {mode_text}")
    
    def get_trading_info(self) -> Dict[str, Any]:
        """取引情報を取得"""
        info = {
            "trading_mode": self.trading_mode.value,
            "config_info": self.config.get_trading_mode_info(),
            "execution_timing": self.config.get_execution_timing_info(),
            "scoring_method": "PPO強化学習" if self.use_ppo else "従来テクニカル分析",
            "risk_limits": {
                "max_daily_loss": getattr(self.config.investment_limits, 'max_daily_loss', None),
                "max_daily_positions": getattr(self.config.investment_limits, 'max_daily_positions', None),
                "stop_loss_strong": self.config.investment_thresholds.stop_loss_strong,
                "target_profit_strong": self.config.investment_thresholds.target_profit_strong
            }
        }
        
        # PPO固有情報を追加
        if self.use_ppo and self.ppo_adapter:
            ppo_info = self.ppo_adapter.get_model_info()
            info["ppo_model_info"] = {
                "model_name": ppo_info.get("model_name", "Unknown"),
                "device": ppo_info.get("device", "Unknown"),
                "is_loaded": ppo_info.get("is_loaded", False)
            }
        
        return info


def main(trading_mode: TradingMode = TradingMode.LONG_TERM):
    """
    メイン実行関数（デイトレード対応）
    
    Args:
        trading_mode: 取引モード（デフォルト: 中長期）
    """
    try:
        # 設定インスタンス作成
        config_instance = Config(trading_mode)
        
        # 投資分析システム初期化
        analyzer = InvestmentAnalyzer(
            config_instance=config_instance, 
            max_investment_per_stock=2000.0
        )
        
        # 取引情報表示
        trading_info = analyzer.get_trading_info()
        mode_text = "🏃 デイトレード" if trading_mode == TradingMode.DAY_TRADING else "📈 中長期投資"
        
        print(f"\n{mode_text}モード - {trading_info['config_info']['description']}")
        print(f"🕰️ {trading_info['execution_timing']}")
        print(f"🎯 {trading_info['config_info']['target_profit_range']} | 損切り: {trading_info['config_info']['stop_loss_range']}")
        print(f"🤖 分析手法: {trading_info['scoring_method']}")
        
        # PPOモデル情報表示
        if 'ppo_model_info' in trading_info:
            ppo_info = trading_info['ppo_model_info']
            print(f"🧠 PPOモデル: {ppo_info['model_name']} ({ppo_info['device']})")
        
        if trading_mode == TradingMode.DAY_TRADING:
            print(f"🛡️ 日次制限: 最大損失¥{trading_info['risk_limits']['max_daily_loss']:,.0f} | 最大{trading_info['risk_limits']['max_daily_positions']}銀柄")
        
        print("=" * 80)
        
        # 完全な分析を実行
        results_df = analyzer.run_complete_analysis()
        
        return results_df, analyzer
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return None, None
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return None, None


def run_daytrading_analysis():
    """デイトレード専用実行関数"""
    return main(TradingMode.DAY_TRADING)


def run_longterm_analysis():
    """中長期投資専用実行関数"""
    return main(TradingMode.LONG_TERM)


if __name__ == "__main__":
    # デフォルトは中長期モード
    results, analyzer = main()
    
    # コマンドライン引数でデイトレードモードを指定可能
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['day', 'daytrading', 'dt']:
        print("\n" + "=" * 80)
        print("🔄 デイトレードモードで再実行")
        print("=" * 80)
        results, analyzer = run_daytrading_analysis()