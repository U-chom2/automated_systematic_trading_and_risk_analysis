"""エンドツーエンド統合テスト

起動から投資提案まで完全な流れをテストします。
"""

import unittest
import asyncio
import tempfile
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

import pytest


class TestEndToEndIntegration(unittest.TestCase):
    """エンドツーエンド統合テスト"""

    def setUp(self) -> None:
        """テストセットアップ"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self) -> None:
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_system_startup_to_recommendation(self) -> None:
        """システム起動から投資提案まで完全テスト"""
        # TDDのRed段階 - エンドツーエンドの流れをテスト
        from src.system_integrator import SystemIntegrator
        
        # システム設定
        config = {
            "capital": 100000,
            "risk_per_trade_ratio": 0.01,
            "buy_threshold": 80,
            "data_dir": self.test_dir,
            "mock_mode": True  # モックモードで実行
        }
        
        # ウォッチリスト
        watchlist = ["7203", "6758", "9984"]  # トヨタ、ソニー、SBG
        
        # システム初期化
        integrator = SystemIntegrator(config)
        result = integrator.initialize_system()
        self.assertTrue(result)
        
        # ウォッチリスト設定
        integrator.load_watchlist(watchlist)
        
        # 完全サイクル実行
        recommendations = integrator.run_complete_analysis_cycle()
        
        # 結果の検証
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # 推奨結果の構造確認
        recommendation = recommendations[0]
        required_fields = [
            'symbol', 'action', 'confidence', 'price_target',
            'stop_loss', 'reasoning', 'timestamp'
        ]
        for field in required_fields:
            self.assertIn(field, recommendation)
            
        # アクションが適切か確認
        self.assertIn(recommendation['action'], ['buy', 'sell', 'hold'])
        
        # 信頼度が範囲内か確認
        self.assertGreaterEqual(recommendation['confidence'], 0.0)
        self.assertLessEqual(recommendation['confidence'], 1.0)

    def test_data_collection_pipeline(self) -> None:
        """データ収集パイプラインテスト"""
        from src.system_integrator import SystemIntegrator
        
        config = {
            "data_dir": self.test_dir,
            "mock_mode": True
        }
        
        integrator = SystemIntegrator(config)
        integrator.initialize_system()
        
        # データ収集実行
        collected_data = integrator.collect_market_data(["7203"])
        
        # データの検証
        self.assertIn("7203", collected_data)
        
        stock_data = collected_data["7203"]
        self.assertIn("price_data", stock_data)
        self.assertIn("ir_releases", stock_data)
        self.assertIn("social_sentiment", stock_data)

    def test_ai_analysis_pipeline(self) -> None:
        """AI分析パイプラインテスト"""
        from src.system_integrator import SystemIntegrator
        
        config = {
            "data_dir": self.test_dir,
            "mock_mode": True
        }
        
        integrator = SystemIntegrator(config)
        integrator.initialize_system()
        
        # サンプルデータ
        market_data = {
            "7203": {
                "price_data": {
                    "current_price": 2450.0,
                    "volume": 1000000,
                    "high": 2470.0,
                    "low": 2430.0
                },
                "ir_releases": [
                    {
                        "title": "決算発表のお知らせ",
                        "content": "売上高増加により増益となりました",
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "social_sentiment": {
                    "positive": 0.7,
                    "negative": 0.2,
                    "neutral": 0.1
                }
            }
        }
        
        # AI分析実行
        analysis_results = integrator.perform_ai_analysis(market_data)
        
        # 分析結果の検証
        self.assertIn("7203", analysis_results)
        
        analysis = analysis_results["7203"]
        self.assertIn("catalyst_score", analysis)
        self.assertIn("sentiment_score", analysis)
        self.assertIn("technical_score", analysis)
        self.assertIn("total_score", analysis)

    def test_investment_recommendation_generation(self) -> None:
        """投資提案生成テスト"""
        from src.system_integrator import SystemIntegrator
        
        config = {
            "data_dir": self.test_dir,
            "mock_mode": True,
            "buy_threshold": 70
        }
        
        integrator = SystemIntegrator(config)
        integrator.initialize_system()
        
        # 高スコアの分析結果
        analysis_results = {
            "7203": {
                "catalyst_score": 45,
                "sentiment_score": 25,
                "technical_score": 15,
                "total_score": 85,
                "confidence": 0.85,
                "current_price": 2450.0,
                "risk_assessment": {
                    "stop_loss_percent": 5.0,
                    "profit_target_percent": 15.0
                }
            }
        }
        
        # 投資提案生成
        recommendations = integrator.generate_investment_recommendations(analysis_results)
        
        # 提案の検証
        self.assertEqual(len(recommendations), 1)
        
        recommendation = recommendations[0]
        self.assertEqual(recommendation["symbol"], "7203")
        self.assertEqual(recommendation["action"], "buy")
        self.assertGreater(recommendation["confidence"], 0.8)
        self.assertIn("reasoning", recommendation)

    def test_risk_management_integration(self) -> None:
        """リスク管理統合テスト"""
        from src.system_integrator import SystemIntegrator
        
        config = {
            "data_dir": self.test_dir,
            "mock_mode": True,
            "capital": 100000,
            "risk_per_trade_ratio": 0.02
        }
        
        integrator = SystemIntegrator(config)
        integrator.initialize_system()
        
        # リスク計算実行
        risk_params = integrator.calculate_risk_parameters(
            symbol="7203",
            current_price=2450.0,
            volatility=0.25
        )
        
        # リスク計算結果の検証
        self.assertIn("position_size", risk_params)
        self.assertIn("stop_loss_price", risk_params)
        self.assertIn("max_loss_amount", risk_params)
        
        # ポジションサイズが資本比率に適合しているか
        expected_max_loss = config["capital"] * config["risk_per_trade_ratio"]
        self.assertLessEqual(risk_params["max_loss_amount"], expected_max_loss)

    def test_performance_tracking_integration(self) -> None:
        """パフォーマンス追跡統合テスト"""
        from src.system_integrator import SystemIntegrator
        
        config = {
            "data_dir": self.test_dir,
            "mock_mode": True
        }
        
        integrator = SystemIntegrator(config)
        integrator.initialize_system()
        
        # 模擬取引記録
        integrator.record_simulated_trade({
            "symbol": "7203",
            "action": "buy",
            "price": 2450.0,
            "quantity": 100,
            "timestamp": datetime.now().isoformat()
        })
        
        # パフォーマンス統計取得
        performance = integrator.get_performance_statistics()
        
        # 統計の検証
        self.assertIn("total_trades", performance)
        self.assertIn("win_rate", performance)
        self.assertIn("total_return", performance)
        self.assertGreater(performance["total_trades"], 0)

    def test_error_handling_and_recovery(self) -> None:
        """エラーハンドリングと復旧テスト"""
        from src.system_integrator import SystemIntegrator
        
        config = {
            "data_dir": self.test_dir,
            "mock_mode": True
        }
        
        integrator = SystemIntegrator(config)
        integrator.initialize_system()
        
        # 意図的にエラーを発生させる
        with patch('src.system_integrator.SystemIntegrator.collect_market_data') as mock_collect:
            mock_collect.side_effect = Exception("Data collection failed")
            
            # エラーハンドリングをテスト
            result = integrator.run_complete_analysis_cycle_with_error_handling()
            
            # システムがクラッシュせずに適切にエラーを処理したか確認
            self.assertIsInstance(result, dict)
            self.assertIn("error", result)
            self.assertIn("recovery_action", result)

    def test_configuration_management(self) -> None:
        """設定管理テスト"""
        from src.system_integrator import SystemIntegrator
        
        # カスタム設定
        custom_config = {
            "capital": 500000,
            "risk_per_trade_ratio": 0.005,
            "buy_threshold": 90,
            "data_dir": self.test_dir,
            "mock_mode": True
        }
        
        integrator = SystemIntegrator(custom_config)
        integrator.initialize_system()
        
        # 設定が正しく適用されているか確認
        current_config = integrator.get_current_configuration()
        
        self.assertEqual(current_config["capital"], 500000)
        self.assertEqual(current_config["risk_per_trade_ratio"], 0.005)
        self.assertEqual(current_config["buy_threshold"], 90)

    def test_data_persistence_integration(self) -> None:
        """データ永続化統合テスト"""
        from src.system_integrator import SystemIntegrator
        
        config = {
            "data_dir": self.test_dir,
            "mock_mode": True
        }
        
        integrator = SystemIntegrator(config)
        integrator.initialize_system()
        
        # データ保存
        test_data = {
            "execution_plan": {"id": "plan_001", "strategy": "momentum"},
            "trade_record": {"symbol": "7203", "action": "buy", "price": 2450},
            "system_state": {"status": "active", "uptime": 3600}
        }
        
        # データ永続化
        save_result = integrator.save_all_data(test_data)
        self.assertTrue(save_result)
        
        # データ復元
        loaded_data = integrator.load_all_data()
        
        # データが正しく永続化・復元されているか確認
        self.assertIn("execution_plan", loaded_data)
        self.assertIn("trade_record", loaded_data)
        self.assertIn("system_state", loaded_data)


if __name__ == '__main__':
    unittest.main()