"""システム状態管理のテスト"""

import unittest
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, Optional

import pytest


class TestSystemStatePersistence(unittest.TestCase):
    """システム状態の永続化テスト"""

    def setUp(self) -> None:
        """テストセットアップ"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self) -> None:
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_save_system_state(self) -> None:
        """システム状態の保存テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.system_state_manager import SystemStateManager
        
        manager = SystemStateManager(data_dir=self.test_dir)
        
        sample_state: Dict[str, Any] = {
            'state_id': 'state_001',
            'system_status': 'active',
            'last_update': datetime.now().isoformat(),
            'active_plans': ['plan_001', 'plan_002'],
            'portfolio_value': 50000.0,
            'cash_balance': 15000.0,
            'positions': {
                'AAPL': {'quantity': 100, 'avg_price': 150.25},
                'MSFT': {'quantity': 50, 'avg_price': 280.50}
            },
            'risk_metrics': {
                'var_1d': 1250.0,
                'max_drawdown': 0.05,
                'sharpe_ratio': 1.25
            },
            'error_count': 0,
            'last_error': None
        }
        
        result = manager.save_state(sample_state)
        self.assertTrue(result)
        
        # ファイルが作成されていることを確認
        csv_file = os.path.join(self.test_dir, 'system_state.csv')
        self.assertTrue(os.path.exists(csv_file))

    def test_load_latest_system_state(self) -> None:
        """最新システム状態の読み込みテスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.system_state_manager import SystemStateManager
        
        manager = SystemStateManager(data_dir=self.test_dir)
        
        # 複数の状態を保存（異なる時刻で）
        states = [
            {
                'state_id': 'state_002',
                'system_status': 'initializing',
                'last_update': '2025-08-24T10:00:00',
                'portfolio_value': 48000.0,
                'cash_balance': 12000.0
            },
            {
                'state_id': 'state_003',
                'system_status': 'active',
                'last_update': '2025-08-24T12:00:00',
                'portfolio_value': 52000.0,
                'cash_balance': 13000.0
            }
        ]
        
        for state in states:
            manager.save_state(state)
        
        # 最新の状態を取得
        latest_state = manager.load_latest_state()
        
        self.assertIsNotNone(latest_state)
        self.assertEqual(latest_state['state_id'], 'state_003')
        self.assertEqual(latest_state['system_status'], 'active')
        self.assertEqual(latest_state['portfolio_value'], 52000.0)

    def test_load_state_by_id(self) -> None:
        """ID指定でのシステム状態読み込みテスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.system_state_manager import SystemStateManager
        
        manager = SystemStateManager(data_dir=self.test_dir)
        
        # 特定の状態を保存
        test_state: Dict[str, Any] = {
            'state_id': 'state_004',
            'system_status': 'paused',
            'last_update': datetime.now().isoformat(),
            'portfolio_value': 47500.0,
            'cash_balance': 20000.0,
            'maintenance_mode': True,
            'pause_reason': 'scheduled_maintenance'
        }
        
        manager.save_state(test_state)
        
        # IDで状態を取得
        loaded_state = manager.load_state_by_id('state_004')
        
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state['state_id'], 'state_004')
        self.assertEqual(loaded_state['system_status'], 'paused')
        self.assertTrue(loaded_state['maintenance_mode'])

    def test_get_state_history(self) -> None:
        """システム状態履歴取得テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.system_state_manager import SystemStateManager
        
        manager = SystemStateManager(data_dir=self.test_dir)
        
        # 複数の状態を時系列で保存
        states = [
            {
                'state_id': 'state_005',
                'system_status': 'starting',
                'last_update': '2025-08-24T08:00:00',
                'portfolio_value': 45000.0
            },
            {
                'state_id': 'state_006',
                'system_status': 'active',
                'last_update': '2025-08-24T09:00:00',
                'portfolio_value': 46000.0
            },
            {
                'state_id': 'state_007',
                'system_status': 'active',
                'last_update': '2025-08-24T10:00:00',
                'portfolio_value': 47000.0
            }
        ]
        
        for state in states:
            manager.save_state(state)
        
        # 状態履歴を取得（最新から3件）
        history = manager.get_state_history(limit=3)
        
        self.assertEqual(len(history), 3)
        # 最新から順番に並んでいることを確認
        self.assertEqual(history[0]['state_id'], 'state_007')
        self.assertEqual(history[1]['state_id'], 'state_006')
        self.assertEqual(history[2]['state_id'], 'state_005')

    def test_update_system_metrics(self) -> None:
        """システムメトリクス更新テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.system_state_manager import SystemStateManager
        
        manager = SystemStateManager(data_dir=self.test_dir)
        
        # 初期状態を保存
        initial_state: Dict[str, Any] = {
            'state_id': 'state_008',
            'system_status': 'active',
            'last_update': datetime.now().isoformat(),
            'portfolio_value': 50000.0,
            'performance_metrics': {
                'total_trades': 10,
                'win_rate': 0.6,
                'avg_return': 0.02
            }
        }
        
        manager.save_state(initial_state)
        
        # メトリクスを更新
        updated_metrics = {
            'portfolio_value': 52500.0,
            'performance_metrics': {
                'total_trades': 15,
                'win_rate': 0.65,
                'avg_return': 0.025
            },
            'last_update': datetime.now().isoformat()
        }
        
        result = manager.update_metrics('state_008', updated_metrics)
        self.assertTrue(result)
        
        # 更新された状態を確認
        updated_state = manager.load_state_by_id('state_008')
        self.assertEqual(updated_state['portfolio_value'], 52500.0)
        self.assertEqual(updated_state['performance_metrics']['total_trades'], 15)

    def test_record_system_error(self) -> None:
        """システムエラー記録テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.system_state_manager import SystemStateManager
        
        manager = SystemStateManager(data_dir=self.test_dir)
        
        # エラーを記録
        error_info: Dict[str, Any] = {
            'error_id': 'error_001',
            'timestamp': datetime.now().isoformat(),
            'error_type': 'ConnectionError',
            'error_message': 'Failed to connect to trading API',
            'stack_trace': 'Traceback (most recent call last):\n  ...',
            'severity': 'high',
            'component': 'data_collector',
            'resolved': False
        }
        
        result = manager.record_error(error_info)
        self.assertTrue(result)
        
        # エラーログファイルが作成されていることを確認
        error_log_file = os.path.join(self.test_dir, 'system_errors.csv')
        self.assertTrue(os.path.exists(error_log_file))

    def test_get_error_history(self) -> None:
        """エラー履歴取得テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.system_state_manager import SystemStateManager
        
        manager = SystemStateManager(data_dir=self.test_dir)
        
        # 複数のエラーを記録
        errors = [
            {
                'error_id': 'error_002',
                'timestamp': '2025-08-24T10:00:00',
                'error_type': 'APIError',
                'error_message': 'Rate limit exceeded',
                'severity': 'medium',
                'resolved': True
            },
            {
                'error_id': 'error_003',
                'timestamp': '2025-08-24T11:00:00',
                'error_type': 'ValidationError',
                'error_message': 'Invalid order parameters',
                'severity': 'low',
                'resolved': False
            }
        ]
        
        for error in errors:
            manager.record_error(error)
        
        # エラー履歴を取得
        error_history = manager.get_error_history()
        
        self.assertGreaterEqual(len(error_history), 2)
        
        # 未解決エラーのみ取得
        unresolved_errors = manager.get_error_history(resolved_only=False)
        unresolved_count = len([e for e in unresolved_errors if not e['resolved']])
        self.assertGreaterEqual(unresolved_count, 1)

    def test_system_health_check(self) -> None:
        """システムヘルスチェックテスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.system_state_manager import SystemStateManager
        
        manager = SystemStateManager(data_dir=self.test_dir)
        
        # 現在の状態を保存
        current_state: Dict[str, Any] = {
            'state_id': 'state_009',
            'system_status': 'active',
            'last_update': datetime.now().isoformat(),
            'uptime_hours': 24.5,
            'memory_usage_mb': 256,
            'cpu_usage_percent': 15.5,
            'active_connections': 5,
            'last_heartbeat': datetime.now().isoformat()
        }
        
        manager.save_state(current_state)
        
        # ヘルスチェックを実行
        health_status = manager.check_system_health()
        
        self.assertIsNotNone(health_status)
        self.assertIn('overall_status', health_status)
        self.assertIn('uptime_hours', health_status)
        self.assertIn('resource_usage', health_status)
        self.assertIn('last_update_age_minutes', health_status)
        
        # ステータスが正常であることを確認
        self.assertEqual(health_status['overall_status'], 'healthy')


if __name__ == '__main__':
    unittest.main()