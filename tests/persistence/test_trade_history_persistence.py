"""取引履歴管理のテスト"""

import unittest
import tempfile
import os
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List

import pytest


class TestTradeHistoryPersistence(unittest.TestCase):
    """取引履歴の永続化テスト"""

    def setUp(self) -> None:
        """テストセットアップ"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self) -> None:
        """テストクリーンアップ"""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_record_trade(self) -> None:
        """取引記録の保存テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.trade_history_manager import TradeHistoryManager
        
        manager = TradeHistoryManager(data_dir=self.test_dir)
        
        sample_trade: Dict[str, Any] = {
            'trade_id': 'trade_001',
            'symbol': 'AAPL',
            'side': 'buy',  # buy or sell
            'quantity': 100,
            'price': Decimal('150.25'),
            'executed_at': datetime.now().isoformat(),
            'order_type': 'market',
            'commission': Decimal('1.50'),
            'total_amount': Decimal('15026.50'),  # price * quantity + commission
            'plan_id': 'plan_001',  # 関連する実行計画のID
            'status': 'filled'
        }
        
        result = manager.record_trade(sample_trade)
        self.assertTrue(result)
        
        # ファイルが作成されていることを確認
        csv_file = os.path.join(self.test_dir, 'trade_history.csv')
        self.assertTrue(os.path.exists(csv_file))

    def test_get_trades_by_symbol(self) -> None:
        """シンボル別取引履歴取得テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.trade_history_manager import TradeHistoryManager
        
        manager = TradeHistoryManager(data_dir=self.test_dir)
        
        # 複数の取引を記録
        trades = [
            {
                'trade_id': 'trade_002',
                'symbol': 'AAPL',
                'side': 'buy',
                'quantity': 50,
                'price': Decimal('148.75'),
                'executed_at': datetime.now().isoformat(),
                'order_type': 'limit',
                'commission': Decimal('1.25'),
                'total_amount': Decimal('7438.75'),
                'plan_id': 'plan_002',
                'status': 'filled'
            },
            {
                'trade_id': 'trade_003',
                'symbol': 'MSFT',
                'side': 'sell',
                'quantity': 75,
                'price': Decimal('280.50'),
                'executed_at': datetime.now().isoformat(),
                'order_type': 'market',
                'commission': Decimal('2.10'),
                'total_amount': Decimal('21035.40'),
                'plan_id': 'plan_003',
                'status': 'filled'
            },
            {
                'trade_id': 'trade_004',
                'symbol': 'AAPL',
                'side': 'sell',
                'quantity': 25,
                'price': Decimal('152.00'),
                'executed_at': datetime.now().isoformat(),
                'order_type': 'stop',
                'commission': Decimal('1.00'),
                'total_amount': Decimal('3799.00'),
                'plan_id': 'plan_002',
                'status': 'filled'
            }
        ]
        
        for trade in trades:
            manager.record_trade(trade)
        
        # AAPL取引のみ取得
        aapl_trades = manager.get_trades_by_symbol('AAPL')
        
        self.assertEqual(len(aapl_trades), 2)
        for trade in aapl_trades:
            self.assertEqual(trade['symbol'], 'AAPL')

    def test_get_trades_by_date_range(self) -> None:
        """日付範囲別取引履歴取得テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.trade_history_manager import TradeHistoryManager
        from datetime import datetime, timedelta
        
        manager = TradeHistoryManager(data_dir=self.test_dir)
        
        base_date = datetime.now()
        
        # 異なる日付の取引を記録
        trades = [
            {
                'trade_id': 'trade_005',
                'symbol': 'GOOGL',
                'side': 'buy',
                'quantity': 10,
                'price': Decimal('2500.00'),
                'executed_at': (base_date - timedelta(days=2)).isoformat(),
                'order_type': 'market',
                'commission': Decimal('5.00'),
                'total_amount': Decimal('25005.00'),
                'plan_id': 'plan_004',
                'status': 'filled'
            },
            {
                'trade_id': 'trade_006',
                'symbol': 'TSLA',
                'side': 'buy',
                'quantity': 20,
                'price': Decimal('180.25'),
                'executed_at': base_date.isoformat(),
                'order_type': 'limit',
                'commission': Decimal('2.50'),
                'total_amount': Decimal('3607.50'),
                'plan_id': 'plan_005',
                'status': 'filled'
            }
        ]
        
        for trade in trades:
            manager.record_trade(trade)
        
        # 過去1日間の取引を取得
        start_date = base_date - timedelta(days=1)
        end_date = base_date + timedelta(days=1)
        
        recent_trades = manager.get_trades_by_date_range(
            start_date.isoformat(),
            end_date.isoformat()
        )
        
        self.assertEqual(len(recent_trades), 1)
        self.assertEqual(recent_trades[0]['trade_id'], 'trade_006')

    def test_calculate_pnl_by_symbol(self) -> None:
        """シンボル別損益計算テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.trade_history_manager import TradeHistoryManager
        
        manager = TradeHistoryManager(data_dir=self.test_dir)
        
        # 買いと売りのペアを記録
        trades = [
            {
                'trade_id': 'trade_007',
                'symbol': 'NFLX',
                'side': 'buy',
                'quantity': 10,
                'price': Decimal('400.00'),
                'executed_at': datetime.now().isoformat(),
                'order_type': 'market',
                'commission': Decimal('2.00'),
                'total_amount': Decimal('4002.00'),
                'plan_id': 'plan_006',
                'status': 'filled'
            },
            {
                'trade_id': 'trade_008',
                'symbol': 'NFLX',
                'side': 'sell',
                'quantity': 10,
                'price': Decimal('420.00'),
                'executed_at': datetime.now().isoformat(),
                'order_type': 'limit',
                'commission': Decimal('2.10'),
                'total_amount': Decimal('4197.90'),
                'plan_id': 'plan_006',
                'status': 'filled'
            }
        ]
        
        for trade in trades:
            manager.record_trade(trade)
        
        # 損益を計算
        pnl = manager.calculate_pnl_by_symbol('NFLX')
        
        # 期待される損益: (420.00 - 400.00) * 10 - 2.00 - 2.10 = 195.90
        expected_pnl = Decimal('195.90')
        self.assertEqual(pnl, expected_pnl)

    def test_get_trading_summary(self) -> None:
        """取引サマリー取得テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.trade_history_manager import TradeHistoryManager
        
        manager = TradeHistoryManager(data_dir=self.test_dir)
        
        # 複数の取引を記録
        trades = [
            {
                'trade_id': 'trade_009',
                'symbol': 'AMZN',
                'side': 'buy',
                'quantity': 5,
                'price': Decimal('3000.00'),
                'executed_at': datetime.now().isoformat(),
                'commission': Decimal('7.50'),
                'total_amount': Decimal('15007.50'),
                'status': 'filled'
            },
            {
                'trade_id': 'trade_010',
                'symbol': 'AMZN',
                'side': 'sell',
                'quantity': 5,
                'price': Decimal('3100.00'),
                'executed_at': datetime.now().isoformat(),
                'commission': Decimal('7.75'),
                'total_amount': Decimal('15492.25'),
                'status': 'filled'
            }
        ]
        
        for trade in trades:
            manager.record_trade(trade)
        
        # サマリーを取得
        summary = manager.get_trading_summary()
        
        self.assertIn('total_trades', summary)
        self.assertIn('total_buy_volume', summary)
        self.assertIn('total_sell_volume', summary)
        self.assertIn('total_commission', summary)
        self.assertIn('net_pnl', summary)
        self.assertIn('symbols_traded', summary)
        
        self.assertEqual(summary['total_trades'], 2)
        self.assertEqual(len(summary['symbols_traded']), 1)
        self.assertIn('AMZN', summary['symbols_traded'])

    def test_export_trades_to_csv(self) -> None:
        """取引履歴CSV出力テスト"""
        # TDDのRed段階 - まだ実装されていないため失敗
        from src.persistence.trade_history_manager import TradeHistoryManager
        
        manager = TradeHistoryManager(data_dir=self.test_dir)
        
        # 取引を記録
        sample_trade = {
            'trade_id': 'trade_011',
            'symbol': 'NVDA',
            'side': 'buy',
            'quantity': 15,
            'price': Decimal('800.00'),
            'executed_at': datetime.now().isoformat(),
            'commission': Decimal('6.00'),
            'total_amount': Decimal('12006.00'),
            'status': 'filled'
        }
        
        manager.record_trade(sample_trade)
        
        # 特定のファイルにエクスポート
        export_file = os.path.join(self.test_dir, 'exported_trades.csv')
        result = manager.export_to_csv(export_file)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(export_file))
        
        # ファイルの内容を確認（ヘッダーとデータが含まれているか）
        with open(export_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('trade_id', content)
            self.assertIn('trade_011', content)


if __name__ == '__main__':
    unittest.main()