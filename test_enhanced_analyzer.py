#!/usr/bin/env python3
"""
Enhanced StockAnalyzer Comprehensive Test Script

このスクリプトは以下の機能強化を包括的にテストします：
1. ModernBERT-ja を使用したセンチメント分析
2. TDNet + Yahoo Finance ニュース収集
3. 包括的リスク評価システム
4. リアルタイム分析トリガー
5. 強化されたStockAnalyzer統合機能
"""

import sys
from pathlib import Path
import json
from typing import Dict, Any
import traceback

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.analyzers.stock_analyzer import StockAnalyzer
from src.analyzers.news_collector import NewsCollector
from src.analyzers.sentiment_analyzer import ModernBERTSentimentAnalyzer
from src.analyzers.risk_evaluator import ComprehensiveRiskEvaluator
from src.analyzers.realtime_triggers import RealtimeAnalysisTriggers
from src.utils.logger_utils import create_dual_logger

# ロガー設定
logger = create_dual_logger(__name__, console_output=True)


def test_news_collection():
    """ニュース収集機能テスト"""
    logger.info("=== ニュース収集機能テスト開始 ===")
    
    try:
        news_collector = NewsCollector()
        
        # テスト銘柄
        test_ticker = "7203.T"  # トヨタ自動車
        test_company = "トヨタ自動車"
        
        logger.info(f"テスト対象: {test_company} ({test_ticker})")
        
        # IR情報収集テスト
        logger.info("TDNet IRニュース収集中...")
        ir_news = news_collector.collect_ir_news(test_ticker, test_company, days_back=7)
        logger.info(f"IR情報: {len(ir_news)}件")
        
        # Yahoo Financeニュース収集テスト
        logger.info("Yahoo Financeニュース収集中...")
        yahoo_news = news_collector.collect_yahoo_finance_news(test_ticker, test_company)
        logger.info(f"Yahoo Financeニュース: {len(yahoo_news)}件")
        
        # 統合ニュース収集テスト
        logger.info("統合ニュース収集中...")
        all_news = news_collector.collect_all_news(test_ticker, test_company, days_back=3)
        logger.info(f"統合ニュース: {len(all_news)}件")
        
        # ニュースサマリー生成
        summary = news_collector.get_news_summary(all_news)
        logger.info(f"ニュースサマリー: {json.dumps(summary, indent=2, ensure_ascii=False)}")
        
        # 最新ニュースタイトル表示
        if all_news:
            logger.info("最新ニュース:")
            for i, news in enumerate(all_news[:3], 1):
                logger.info(f"  {i}. [{news.source}] {news.title[:100]}...")
        
        logger.info("✅ ニュース収集機能テスト完了")
        return True
        
    except Exception as e:
        logger.error(f"❌ ニュース収集テストエラー: {e}")
        logger.debug(traceback.format_exc())
        return False


def test_sentiment_analysis():
    """センチメント分析機能テスト"""
    logger.info("=== センチメント分析機能テスト開始 ===")
    
    try:
        sentiment_analyzer = ModernBERTSentimentAnalyzer()
        
        # モデルステータス確認
        model_status = sentiment_analyzer.get_model_status()
        logger.info(f"モデルステータス: {json.dumps(model_status, indent=2, ensure_ascii=False)}")
        
        # テストテキスト
        test_texts = [
            "トヨタ自動車の業績が大幅に向上し、株価も上昇トレンドを維持している",
            "決算発表で業績悪化が明らかになり、投資家の懸念が高まっている",  
            "新車販売台数の統計が発表された",
        ]
        
        logger.info("テストテキストでセンチメント分析実行...")
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"\nテスト {i}: {text}")
            
            result = sentiment_analyzer.analyze_sentiment(text)
            
            logger.info(f"  センチメントスコア: {result.sentiment_score:.3f}")
            logger.info(f"  信頼度: {result.confidence:.3f}")
            logger.info(f"  感情スコア: {result.emotion_scores}")
            logger.info(f"  重要キーワード: {result.keywords[:5]}")
        
        # ニュース収集 + センチメント分析統合テスト
        logger.info("\n統合テスト: ニュース収集 + センチメント分析")
        news_collector = NewsCollector()
        
        test_ticker = "7203.T"  
        test_company = "トヨタ自動車"
        
        news_items = news_collector.collect_all_news(test_ticker, test_company, days_back=2)
        
        if news_items:
            logger.info(f"{len(news_items)}件のニュースでセンチメント分析実行...")
            
            sentiment_results = sentiment_analyzer.analyze_news_batch(news_items)
            
            if sentiment_results:
                overall_sentiment, avg_confidence = sentiment_analyzer.calculate_overall_sentiment(
                    sentiment_results
                )
                
                logger.info(f"全体センチメント: {overall_sentiment:.3f}")
                logger.info(f"平均信頼度: {avg_confidence:.3f}")
                
                # 個別結果
                for i, result in enumerate(sentiment_results[:3], 1):
                    logger.info(f"  ニュース{i}: {result.sentiment_score:.3f} (信頼度: {result.confidence:.3f})")
            else:
                logger.warning("センチメント分析結果が取得できませんでした")
        else:
            logger.warning("ニュースが取得できませんでした")
        
        logger.info("✅ センチメント分析機能テスト完了")
        return True
        
    except Exception as e:
        logger.error(f"❌ センチメント分析テストエラー: {e}")
        logger.debug(traceback.format_exc())
        return False


def test_risk_evaluation():
    """リスク評価機能テスト"""
    logger.info("=== リスク評価機能テスト開始 ===")
    
    try:
        risk_evaluator = ComprehensiveRiskEvaluator()
        
        # テスト銘柄
        test_cases = [
            ("7203.T", "トヨタ自動車"),
            ("9984.T", "ソフトバンクグループ"),
            ("6758.T", "ソニーグループ")
        ]
        
        logger.info(f"{len(test_cases)}銘柄のリスク評価実行...")
        
        risk_assessments = []
        
        for ticker, company_name in test_cases:
            logger.info(f"\n--- {company_name} ({ticker}) リスク評価 ---")
            
            try:
                assessment = risk_evaluator.evaluate_comprehensive_risk(
                    ticker, company_name, portfolio_weight=0.1
                )
                
                if assessment:
                    logger.info(f"総合リスクスコア: {assessment.overall_risk_score:.1f}")
                    logger.info(f"リスクレベル: {assessment.risk_level.value}")
                    logger.info(f"市場リスク: {assessment.market_risk_score:.1f}")
                    logger.info(f"ボラティリティリスク: {assessment.volatility_risk_score:.1f}")
                    logger.info(f"流動性リスク: {assessment.liquidity_risk_score:.1f}")
                    logger.info(f"企業リスク: {assessment.company_risk_score:.1f}")
                    
                    # リスク指標
                    metrics = assessment.risk_metrics
                    logger.info(f"ベータ: {metrics.beta:.2f}")
                    logger.info(f"ボラティリティ: {metrics.volatility:.1%}")
                    logger.info(f"VaR(95%): {metrics.var_95:.1%}")
                    logger.info(f"最大ドローダウン: {metrics.max_drawdown:.1%}")
                    
                    # 警告と推奨
                    if assessment.risk_warnings:
                        logger.info("リスク警告:")
                        for warning in assessment.risk_warnings[:2]:
                            logger.info(f"  ⚠️ {warning}")
                    
                    if assessment.recommendations:
                        logger.info("推奨事項:")
                        for rec in assessment.recommendations[:2]:
                            logger.info(f"  💡 {rec}")
                    
                    risk_assessments.append(assessment)
                
                else:
                    logger.warning(f"{company_name} のリスク評価に失敗しました")
                    
            except Exception as e:
                logger.error(f"{company_name} のリスク評価エラー: {e}")
        
        # ポートフォリオリスク評価テスト
        if len(risk_assessments) >= 2:
            logger.info("\n--- ポートフォリオリスク評価 ---")
            
            portfolio_weights = {
                assessment.ticker: 1.0 / len(risk_assessments)
                for assessment in risk_assessments
            }
            
            portfolio_risk = risk_evaluator.evaluate_portfolio_risk(
                risk_assessments, portfolio_weights
            )
            
            logger.info(f"ポートフォリオリスクスコア: {portfolio_risk.get('portfolio_risk_score', 0):.1f}")
            logger.info(f"ポートフォリオリスクレベル: {portfolio_risk.get('portfolio_risk_level', 'UNKNOWN')}")
            logger.info(f"集中リスク: {portfolio_risk.get('concentration_risk', 0):.1%}")
            logger.info(f"分散化スコア: {portfolio_risk.get('diversification_score', 0):.1f}")
        
        logger.info("✅ リスク評価機能テスト完了")
        return True
        
    except Exception as e:
        logger.error(f"❌ リスク評価テストエラー: {e}")
        logger.debug(traceback.format_exc())
        return False


def test_realtime_triggers():
    """リアルタイムトリガー機能テスト"""
    logger.info("=== リアルタイムトリガー機能テスト開始 ===")
    
    try:
        # ダミーのStockAnalyzer作成
        stock_analyzer = StockAnalyzer(enable_ai=False, enable_news=False)
        
        # リアルタイムトリガーシステム初期化
        trigger_system = RealtimeAnalysisTriggers(stock_analyzer)
        
        # テスト銘柄
        test_ticker = "7203.T"
        test_company = "トヨタ自動車"
        
        logger.info(f"テスト対象: {test_company} ({test_ticker})")
        
        # 各種トリガーを追加
        logger.info("トリガー設定中...")
        
        # 価格変動トリガー（±3%）
        price_trigger_id = trigger_system.add_price_change_trigger(
            test_ticker, test_company, threshold_percent=3.0, cooldown=300
        )
        
        # 出来高急増トリガー（平均の2倍）
        volume_trigger_id = trigger_system.add_volume_spike_trigger(
            test_ticker, test_company, volume_multiplier=2.0, cooldown=300
        )
        
        # RSI極値トリガー
        rsi_trigger_id = trigger_system.add_rsi_extreme_trigger(
            test_ticker, test_company, rsi_threshold=30, cooldown=600
        )
        
        # ニュースアラートトリガー
        news_trigger_id = trigger_system.add_news_alert_trigger(
            test_ticker, test_company, importance_threshold=0.8, cooldown=600
        )
        
        # コールバック設定
        triggered_events = []
        
        def test_callback(event):
            logger.info(f"🔔 トリガー発火: {event.condition.company_name} - {event.condition.trigger_type.value}")
            logger.info(f"   値: {event.value:.3f}, 時刻: {event.triggered_at}")
            triggered_events.append(event)
        
        trigger_system.add_callback(test_callback)
        
        # システムステータス確認
        status = trigger_system.get_trigger_status()
        logger.info(f"トリガーシステムステータス: {json.dumps(status, indent=2, ensure_ascii=False)}")
        
        # 短時間監視テスト（実際の監視はしない）
        logger.info("トリガー条件チェック（1回のみ）...")
        
        for trigger_id, condition in trigger_system.trigger_conditions.items():
            try:
                trigger_system._check_trigger_condition(trigger_id, condition)
                logger.info(f"✅ {condition.trigger_type.value} チェック完了")
            except Exception as e:
                logger.warning(f"⚠️ {condition.trigger_type.value} チェックエラー: {e}")
        
        logger.info(f"発火したトリガー: {len(triggered_events)}件")
        
        # トリガー無効化テスト
        logger.info("トリガー管理テスト...")
        trigger_system.disable_trigger(price_trigger_id)
        logger.info("価格変動トリガーを無効化")
        
        trigger_system.enable_trigger(price_trigger_id)
        logger.info("価格変動トリガーを再有効化")
        
        trigger_system.remove_trigger(news_trigger_id)
        logger.info("ニュースアラートトリガーを削除")
        
        # 最終ステータス
        final_status = trigger_system.get_trigger_status()
        logger.info(f"最終ステータス: {json.dumps(final_status, indent=2, ensure_ascii=False)}")
        
        # シャットダウン
        trigger_system.shutdown()
        
        logger.info("✅ リアルタイムトリガー機能テスト完了")
        return True
        
    except Exception as e:
        logger.error(f"❌ リアルタイムトリガーテストエラー: {e}")
        logger.debug(traceback.format_exc())
        return False


def test_enhanced_stock_analyzer():
    """強化されたStockAnalyzer統合テスト"""
    logger.info("=== 強化StockAnalyzer統合テスト開始 ===")
    
    try:
        # フル機能有効でStockAnalyzer初期化
        analyzer = StockAnalyzer(
            enable_ai=True,
            enable_news=True, 
            enable_risk_evaluation=True
        )
        
        # システムステータス確認
        logger.info("システムステータス確認...")
        status = analyzer.get_system_status()
        
        logger.info("=== システムステータス ===")
        for system_name, system_status in status.items():
            if system_name != "timestamp":
                logger.info(f"{system_name}:")
                if isinstance(system_status, dict):
                    for key, value in system_status.items():
                        logger.info(f"  {key}: {value}")
        
        # テスト銘柄で包括的分析
        test_cases = [
            ("7203.T", "トヨタ自動車"),
            ("9984.T", "ソフトバンクグループ")
        ]
        
        logger.info(f"\n{len(test_cases)}銘柄で包括的分析実行...")
        
        analysis_results = []
        
        for ticker, company_name in test_cases:
            logger.info(f"\n--- {company_name} ({ticker}) 包括的分析 ---")
            
            try:
                # 包括的分析実行
                result = analyzer.analyze_stock(ticker, company_name)
                
                logger.info(f"テクニカルスコア: {result.technical_score:.1f}")
                logger.info(f"ファンダメンタルスコア: {result.fundamental_score:.1f}")
                logger.info(f"センチメントスコア: {result.sentiment_score:.3f}")
                logger.info(f"リスク調整係数: {result.risk_adjustment_factor:.3f}")
                logger.info(f"総合スコア: {result.total_score:.1f}")
                
                # AI予測結果
                if result.ai_prediction:
                    ai_pred = result.ai_prediction
                    logger.info(f"AI予測: {ai_pred.get('action')} (信頼度: {ai_pred.get('confidence', 0):.2f})")
                else:
                    logger.info("AI予測: なし")
                
                # ニュースセンチメント詳細
                if result.news_sentiment_details:
                    news_details = result.news_sentiment_details
                    logger.info(f"ニュース: {news_details.get('news_count', 0)}件")
                    logger.info(f"  - ポジティブ: {news_details.get('positive_news', 0)}件")
                    logger.info(f"  - ネガティブ: {news_details.get('negative_news', 0)}件")
                    logger.info(f"  - 信頼度: {news_details.get('confidence', 0):.3f}")
                else:
                    logger.info("ニュースセンチメント: なし")
                
                # リスク評価結果
                if result.risk_assessment:
                    risk_assess = result.risk_assessment
                    logger.info(f"リスク評価: {risk_assess.overall_risk_score:.1f} ({risk_assess.risk_level.value})")
                    logger.info(f"  - 市場リスク: {risk_assess.market_risk_score:.1f}")
                    logger.info(f"  - ボラティリティ: {risk_assess.volatility_risk_score:.1f}")
                    logger.info(f"  - 流動性: {risk_assess.liquidity_risk_score:.1f}")
                    logger.info(f"  - 企業: {risk_assess.company_risk_score:.1f}")
                    
                    if risk_assess.risk_warnings:
                        logger.info(f"  主要警告: {risk_assess.risk_warnings[0]}")
                else:
                    logger.info("リスク評価: なし")
                
                # 取引推奨生成
                recommendation = result.to_recommendation(quantity=100)
                logger.info(f"取引推奨: {recommendation.action.value} x {recommendation.quantity}")
                logger.info(f"信頼度: {recommendation.confidence:.2f}")
                logger.info(f"期待リターン: {recommendation.expected_return:.1%}")
                logger.info(f"リスクレベル: {recommendation.risk_level.value}")
                logger.info(f"推奨理由: {recommendation.reasoning}")
                
                analysis_results.append(result)
                
            except Exception as e:
                logger.error(f"{company_name} の分析エラー: {e}")
                logger.debug(traceback.format_exc())
        
        logger.info(f"\n分析完了: {len(analysis_results)}件の分析結果を取得")
        
        # 統計情報
        if analysis_results:
            avg_total_score = sum(r.total_score for r in analysis_results) / len(analysis_results)
            avg_risk_adjustment = sum(r.risk_adjustment_factor for r in analysis_results) / len(analysis_results)
            
            logger.info(f"平均総合スコア: {avg_total_score:.1f}")
            logger.info(f"平均リスク調整係数: {avg_risk_adjustment:.3f}")
            
            buy_recommendations = sum(1 for r in analysis_results if r.to_recommendation().action.value == "BUY")
            sell_recommendations = sum(1 for r in analysis_results if r.to_recommendation().action.value == "SELL")
            hold_recommendations = len(analysis_results) - buy_recommendations - sell_recommendations
            
            logger.info(f"推奨アクション: BUY={buy_recommendations}, SELL={sell_recommendations}, HOLD={hold_recommendations}")
        
        logger.info("✅ 強化StockAnalyzer統合テスト完了")
        return True
        
    except Exception as e:
        logger.error(f"❌ 強化StockAnalyzer統合テストエラー: {e}")
        logger.debug(traceback.format_exc())
        return False


def main():
    """メインテスト実行"""
    logger.info("🚀 Enhanced StockAnalyzer 包括的テスト開始")
    logger.info("=" * 60)
    
    test_results = {}
    
    # テスト実行
    test_functions = [
        ("ニュース収集", test_news_collection),
        ("センチメント分析", test_sentiment_analysis),  
        ("リスク評価", test_risk_evaluation),
        ("リアルタイムトリガー", test_realtime_triggers),
        ("StockAnalyzer統合", test_enhanced_stock_analyzer),
    ]
    
    for test_name, test_func in test_functions:
        logger.info(f"\n{'='*20} {test_name}テスト {'='*20}")
        
        try:
            success = test_func()
            test_results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name}テスト成功")
            else:
                logger.error(f"❌ {test_name}テスト失敗")
                
        except Exception as e:
            logger.error(f"💥 {test_name}テストで予期しないエラー: {e}")
            logger.debug(traceback.format_exc())
            test_results[test_name] = False
    
    # 最終結果サマリー
    logger.info("\n" + "=" * 60)
    logger.info("🏁 テスト結果サマリー")
    logger.info("=" * 60)
    
    successful_tests = sum(1 for success in test_results.values() if success)
    total_tests = len(test_results)
    
    for test_name, success in test_results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n成功率: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    
    if successful_tests == total_tests:
        logger.info("🎉 全テスト成功！Enhanced StockAnalyzer は正常に動作しています。")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - successful_tests}件のテストが失敗しました。")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("テストが中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"テスト実行中に予期しないエラーが発生しました: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)