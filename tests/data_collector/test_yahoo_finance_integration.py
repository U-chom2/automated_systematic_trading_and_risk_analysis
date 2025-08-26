"""Yahoo Finance API統合テスト

実際のYahoo Finance APIとの統合をテストします。
"""

import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
import time

import pytest


class TestYahooFinanceIntegration(unittest.TestCase):
    """Yahoo Finance API統合テスト"""

    def setUp(self) -> None:
        """テストセットアップ"""
        # 日本市場の主要銘柄を使用
        self.test_symbols = [
            "7203.T",  # トヨタ自動車
            "6758.T",  # ソニーグループ
            "9984.T",  # ソフトバンクグループ
        ]
        
        # API制限を考慮した待機時間（秒）
        self.api_wait_time = 0.5

    def test_real_time_price_fetch(self) -> None:
        """リアルタイム株価取得テスト"""
        from src.data_collector.yahoo_finance_client import YahooFinanceClient
        
        client = YahooFinanceClient()
        
        for symbol in self.test_symbols:
            # API制限回避のため少し待機
            time.sleep(self.api_wait_time)
            
            # 現在価格を取得
            price_data = client.get_current_price(symbol)
            
            # 基本データの検証
            self.assertIsNotNone(price_data)
            self.assertIn('symbol', price_data)
            self.assertIn('current_price', price_data)
            self.assertIn('volume', price_data)
            self.assertIn('high', price_data)
            self.assertIn('low', price_data)
            self.assertIn('timestamp', price_data)
            
            # 価格が正の値であることを確認
            self.assertGreater(price_data['current_price'], 0)
            
            # 高値・安値の妥当性確認
            self.assertGreaterEqual(price_data['high'], price_data['current_price'])
            self.assertLessEqual(price_data['low'], price_data['current_price'])

    def test_historical_data_fetch(self) -> None:
        """過去データ取得テスト"""
        from src.data_collector.yahoo_finance_client import YahooFinanceClient
        
        client = YahooFinanceClient()
        
        # 過去30日のデータを取得
        symbol = "7203.T"
        historical_data = client.get_historical_data(
            symbol=symbol,
            period_days=30
        )
        
        # データの検証
        self.assertIsNotNone(historical_data)
        self.assertIsInstance(historical_data, list)
        self.assertGreater(len(historical_data), 0)
        
        # 各日のデータ構造を確認
        for daily_data in historical_data:
            self.assertIn('date', daily_data)
            self.assertIn('open', daily_data)
            self.assertIn('high', daily_data)
            self.assertIn('low', daily_data)
            self.assertIn('close', daily_data)
            self.assertIn('volume', daily_data)
            
            # 価格の妥当性確認
            self.assertGreater(daily_data['open'], 0)
            self.assertGreater(daily_data['close'], 0)
            self.assertGreaterEqual(daily_data['high'], daily_data['low'])

    def test_technical_indicators(self) -> None:
        """テクニカル指標計算テスト"""
        from src.data_collector.yahoo_finance_client import YahooFinanceClient
        
        client = YahooFinanceClient()
        
        symbol = "7203.T"
        indicators = client.calculate_technical_indicators(symbol)
        
        # 主要テクニカル指標の検証
        self.assertIsNotNone(indicators)
        self.assertIn('sma_20', indicators)  # 20日移動平均
        self.assertIn('sma_50', indicators)  # 50日移動平均
        self.assertIn('rsi', indicators)      # RSI
        self.assertIn('macd', indicators)     # MACD
        self.assertIn('bollinger_bands', indicators)  # ボリンジャーバンド
        
        # RSIの範囲確認（0-100）
        if indicators['rsi'] is not None:
            self.assertGreaterEqual(indicators['rsi'], 0)
            self.assertLessEqual(indicators['rsi'], 100)

    def test_company_info_fetch(self) -> None:
        """企業情報取得テスト"""
        from src.data_collector.yahoo_finance_client import YahooFinanceClient
        
        client = YahooFinanceClient()
        
        symbol = "7203.T"
        company_info = client.get_company_info(symbol)
        
        # 企業情報の検証
        self.assertIsNotNone(company_info)
        self.assertIn('symbol', company_info)
        self.assertIn('company_name', company_info)
        self.assertIn('market_cap', company_info)
        self.assertIn('pe_ratio', company_info)
        self.assertIn('dividend_yield', company_info)
        self.assertIn('sector', company_info)
        self.assertIn('industry', company_info)

    def test_intraday_data_fetch(self) -> None:
        """日中データ取得テスト"""
        from src.data_collector.yahoo_finance_client import YahooFinanceClient
        
        client = YahooFinanceClient()
        
        symbol = "7203.T"
        intraday_data = client.get_intraday_data(
            symbol=symbol,
            interval='5m'  # 5分足
        )
        
        # データの検証
        self.assertIsNotNone(intraday_data)
        self.assertIsInstance(intraday_data, list)
        
        if len(intraday_data) > 0:
            # 最新のデータポイントを確認
            latest = intraday_data[-1]
            self.assertIn('timestamp', latest)
            self.assertIn('open', latest)
            self.assertIn('high', latest)
            self.assertIn('low', latest)
            self.assertIn('close', latest)
            self.assertIn('volume', latest)

    def test_market_status_check(self) -> None:
        """市場状態確認テスト"""
        from src.data_collector.yahoo_finance_client import YahooFinanceClient
        
        client = YahooFinanceClient()
        
        # 東京証券取引所の状態を確認
        market_status = client.get_market_status('TSE')
        
        self.assertIsNotNone(market_status)
        self.assertIn('market', market_status)
        self.assertIn('is_open', market_status)
        self.assertIn('current_time', market_status)
        self.assertIn('next_open', market_status)
        self.assertIn('next_close', market_status)
        
        # is_openがブール値であることを確認
        self.assertIsInstance(market_status['is_open'], bool)

    def test_batch_price_fetch(self) -> None:
        """複数銘柄一括取得テスト"""
        from src.data_collector.yahoo_finance_client import YahooFinanceClient
        
        client = YahooFinanceClient()
        
        # 複数銘柄を一括取得
        batch_data = client.get_batch_prices(self.test_symbols)
        
        # すべての銘柄のデータが取得できていることを確認
        self.assertEqual(len(batch_data), len(self.test_symbols))
        
        for symbol in self.test_symbols:
            self.assertIn(symbol, batch_data)
            price_data = batch_data[symbol]
            
            if price_data is not None:
                self.assertIn('current_price', price_data)
                self.assertGreater(price_data['current_price'], 0)

    def test_error_handling(self) -> None:
        """エラーハンドリングテスト"""
        from src.data_collector.yahoo_finance_client import YahooFinanceClient
        
        client = YahooFinanceClient()
        
        # 存在しない銘柄コード
        invalid_symbol = "INVALID999.T"
        
        # エラーが適切に処理されることを確認
        price_data = client.get_current_price(invalid_symbol)
        
        # エラー時はNoneまたはエラー情報を含む辞書が返される
        if price_data is not None:
            self.assertIn('error', price_data)

    def test_cache_functionality(self) -> None:
        """キャッシュ機能テスト"""
        from src.data_collector.yahoo_finance_client import YahooFinanceClient
        
        client = YahooFinanceClient(enable_cache=True, cache_ttl=5)
        
        symbol = "7203.T"
        
        # 1回目の取得（キャッシュなし）
        start_time = time.time()
        data1 = client.get_current_price(symbol)
        fetch_time1 = time.time() - start_time
        
        # 2回目の取得（キャッシュあり）
        start_time = time.time()
        data2 = client.get_current_price(symbol)
        fetch_time2 = time.time() - start_time
        
        # キャッシュからの取得の方が高速であることを確認
        self.assertLess(fetch_time2, fetch_time1)
        
        # データが同一であることを確認
        self.assertEqual(data1['current_price'], data2['current_price'])

    def test_rate_limiting(self) -> None:
        """レート制限テスト"""
        from src.data_collector.yahoo_finance_client import YahooFinanceClient
        
        client = YahooFinanceClient(rate_limit=2)  # 秒間2リクエストまで
        
        # 連続リクエスト
        request_times = []
        for i in range(5):
            start_time = time.time()
            client.get_current_price("7203.T")
            request_times.append(time.time() - start_time)
        
        # レート制限により後のリクエストが遅延することを確認
        # （最初のリクエストより後のリクエストの方が時間がかかる）
        self.assertGreater(max(request_times[2:]), min(request_times[:2]))


if __name__ == '__main__':
    unittest.main()