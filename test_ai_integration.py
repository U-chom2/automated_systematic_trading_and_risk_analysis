#!/usr/bin/env python3
"""AIモデル統合テスト - StockAnalyzerのPPO予測機能テスト"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent))

from src.analyzers.stock_analyzer import StockAnalyzer
from src.utils.logger_utils import create_dual_logger

logger = create_dual_logger(__name__, console_output=True)


def test_ai_status():
    """AI機能のステータス確認テスト"""
    logger.info("=== AI統合ステータステスト ===")
    
    # StockAnalyzerを初期化（AI有効）
    analyzer = StockAnalyzer(enable_ai=True)
    
    # AIステータスを取得
    ai_status = analyzer.get_ai_status()
    logger.info(f"AIステータス: {ai_status}")
    
    return ai_status


def test_ai_prediction():
    """AI予測機能のテスト"""
    logger.info("=== AI予測テスト ===")
    
    # テスト用ティッカー（NTT）
    test_ticker = "9432.T"
    test_company = "NTT"
    
    try:
        # StockAnalyzerを初期化
        analyzer = StockAnalyzer(enable_ai=True)
        
        # 個別銘柄分析実行（AI予測含む）
        logger.info(f"分析開始: {test_company} ({test_ticker})")
        result = analyzer.analyze_stock(test_ticker, test_company)
        
        # 結果確認
        logger.info(f"分析結果:")
        logger.info(f"  技術的スコア: {result.technical_score}")
        logger.info(f"  ファンダメンタル: {result.fundamental_score}")
        logger.info(f"  センチメント: {result.sentiment_score}")
        logger.info(f"  総合スコア: {result.total_score}")
        
        if result.ai_prediction:
            logger.info(f"  AI予測: {result.ai_prediction}")
        else:
            logger.warning("  AI予測が取得できませんでした")
        
        # 取引推奨生成
        recommendation = result.to_recommendation(quantity=100)
        logger.info(f"取引推奨: {recommendation.action.value} - {getattr(recommendation, 'reasoning', 'No reasoning available')}")
        
        return result
        
    except Exception as e:
        logger.error(f"AI予測テストエラー: {e}")
        raise


def test_fallback_behavior():
    """AI無効時のフォールバック動作テスト"""
    logger.info("=== フォールバックテスト ===")
    
    # AI無効でStockAnalyzer初期化
    analyzer = StockAnalyzer(enable_ai=False)
    
    # AIステータス確認
    ai_status = analyzer.get_ai_status()
    logger.info(f"AI無効時ステータス: {ai_status}")
    
    # 分析実行（AI予測なし）
    test_ticker = "9432.T"
    result = analyzer.analyze_stock(test_ticker, "NTT")
    
    logger.info(f"AI無効時結果:")
    logger.info(f"  総合スコア: {result.total_score}")
    logger.info(f"  AI予測: {result.ai_prediction}")  # None であるはず
    
    return result


def test_data_converter():
    """MarketDataConverter単体テスト"""
    logger.info("=== データ変換テスト ===")
    
    try:
        from src.analyzers.market_data_converter import MarketDataConverter
        
        converter = MarketDataConverter()
        
        # 観測値生成テスト
        test_ticker = "9432.T"
        observation = converter.get_observation_for_ticker(test_ticker)
        
        logger.info(f"観測値生成結果:")
        logger.info(f"  ティッカー: {test_ticker}")
        logger.info(f"  観測値サイズ: {observation.shape}")
        logger.info(f"  データ型: {observation.dtype}")
        logger.info(f"  値の範囲: [{observation.min():.3f}, {observation.max():.3f}]")
        
        # キャッシュ統計
        cache_stats = converter.get_cache_stats()
        logger.info(f"キャッシュ統計: {cache_stats}")
        
        return observation
        
    except Exception as e:
        logger.error(f"データ変換テストエラー: {e}")
        raise


def main():
    """メインテスト実行"""
    logger.info("🚀 AIモデル統合テスト開始")
    
    try:
        # 1. AIステータステスト
        ai_status = test_ai_status()
        
        # 2. データ変換テスト
        observation = test_data_converter()
        
        # 3. AI予測テスト（モデル読み込み可能な場合のみ）
        if ai_status.get("model_loaded"):
            result = test_ai_prediction()
            logger.info("✅ AI予測テスト完了")
        else:
            logger.warning("⚠️ AIモデルが読み込めないため、AI予測テストをスキップ")
        
        # 4. フォールバック動作テスト
        fallback_result = test_fallback_behavior()
        logger.info("✅ フォールバックテスト完了")
        
        logger.info("🎉 全テスト完了")
        
    except Exception as e:
        logger.error(f"❌ テスト失敗: {e}")
        raise


if __name__ == "__main__":
    main()