"""実行計画保存・復元のテスト"""

import unittest
import tempfile
import os
from datetime import datetime
from typing import Dict, Any

import pytest


class TestExecutionPlanPersistence(unittest.TestCase):
    """実行計画の永続化テスト"""

    def setUp(self) -> None:
        """テストセットアップ"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self) -> None:
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_save_execution_plan(self) -> None:
        """実行計画の保存テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.execution_plan_manager import ExecutionPlanManager
        
        manager = ExecutionPlanManager(data_dir=self.test_dir)
        
        sample_plan: Dict[str, Any] = {
            'id': 'plan_001',
            'name': 'Sample Trading Plan',
            'strategy': 'momentum_following',
            'parameters': {
                'symbol': 'AAPL',
                'quantity': 100,
                'stop_loss': 0.95,
                'take_profit': 1.10
            },
            'created_at': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        result = manager.save_plan(sample_plan)
        self.assertTrue(result)
        
        # ファイルが作成されていることを確認
        csv_file = os.path.join(self.test_dir, 'execution_plans.csv')
        self.assertTrue(os.path.exists(csv_file))

    def test_load_execution_plan(self) -> None:
        """実行計画の読み込みテスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.execution_plan_manager import ExecutionPlanManager
        
        manager = ExecutionPlanManager(data_dir=self.test_dir)
        
        # 事前にプランを保存
        sample_plan: Dict[str, Any] = {
            'id': 'plan_002',
            'name': 'Load Test Plan',
            'strategy': 'mean_reversion',
            'parameters': {'symbol': 'MSFT', 'quantity': 50},
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        manager.save_plan(sample_plan)
        
        # 保存したプランを読み込み
        loaded_plan = manager.load_plan('plan_002')
        
        self.assertIsNotNone(loaded_plan)
        self.assertEqual(loaded_plan['id'], 'plan_002')
        self.assertEqual(loaded_plan['name'], 'Load Test Plan')
        self.assertEqual(loaded_plan['strategy'], 'mean_reversion')

    def test_list_execution_plans(self) -> None:
        """実行計画一覧取得テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.execution_plan_manager import ExecutionPlanManager
        
        manager = ExecutionPlanManager(data_dir=self.test_dir)
        
        # 複数のプランを保存
        plans = [
            {'id': 'plan_003', 'name': 'Plan A', 'strategy': 'scalping', 
             'created_at': datetime.now().isoformat(), 'status': 'active'},
            {'id': 'plan_004', 'name': 'Plan B', 'strategy': 'swing_trading',
             'created_at': datetime.now().isoformat(), 'status': 'paused'}
        ]
        
        for plan in plans:
            manager.save_plan(plan)
        
        # すべてのプランを取得
        all_plans = manager.list_plans()
        
        self.assertGreaterEqual(len(all_plans), 2)
        
        # 特定の状態のプランのみ取得
        active_plans = manager.list_plans(status='active')
        self.assertEqual(len(active_plans), 1)
        self.assertEqual(active_plans[0]['id'], 'plan_003')

    def test_update_execution_plan(self) -> None:
        """実行計画の更新テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.execution_plan_manager import ExecutionPlanManager
        
        manager = ExecutionPlanManager(data_dir=self.test_dir)
        
        # 初期プランを保存
        original_plan: Dict[str, Any] = {
            'id': 'plan_005',
            'name': 'Update Test Plan',
            'strategy': 'buy_and_hold',
            'parameters': {'symbol': 'GOOGL', 'quantity': 25},
            'created_at': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        manager.save_plan(original_plan)
        
        # プランを更新
        updates: Dict[str, Any] = {
            'status': 'active',
            'parameters': {'symbol': 'GOOGL', 'quantity': 50},
            'updated_at': datetime.now().isoformat()
        }
        
        result = manager.update_plan('plan_005', updates)
        self.assertTrue(result)
        
        # 更新されたプランを確認
        updated_plan = manager.load_plan('plan_005')
        self.assertEqual(updated_plan['status'], 'active')
        self.assertEqual(updated_plan['parameters']['quantity'], 50)

    def test_delete_execution_plan(self) -> None:
        """実行計画の削除テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.execution_plan_manager import ExecutionPlanManager
        
        manager = ExecutionPlanManager(data_dir=self.test_dir)
        
        # プランを保存
        test_plan: Dict[str, Any] = {
            'id': 'plan_006',
            'name': 'Delete Test Plan',
            'strategy': 'arbitrage',
            'created_at': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        manager.save_plan(test_plan)
        
        # プランが存在することを確認
        loaded_plan = manager.load_plan('plan_006')
        self.assertIsNotNone(loaded_plan)
        
        # プランを削除
        result = manager.delete_plan('plan_006')
        self.assertTrue(result)
        
        # プランが削除されたことを確認
        deleted_plan = manager.load_plan('plan_006')
        self.assertIsNone(deleted_plan)


if __name__ == '__main__':
    unittest.main()