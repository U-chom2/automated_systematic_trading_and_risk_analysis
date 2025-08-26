"""TDnet実データ連携テスト

実際のTDnetサイトからデータを取得する統合テストです。
"""

import unittest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pytest


class TestTDnetRealIntegration(unittest.TestCase):
    """TDnet実データ連携テスト"""

    def setUp(self) -> None:
        """テストセットアップ"""
        # API制限を考慮した待機時間（秒）
        self.api_wait_time = 1.0
        
        # テスト用企業コード（東証上場企業）
        self.test_companies = [
            "7203",  # トヨタ自動車
            "6758",  # ソニーグループ
            "9984",  # ソフトバンクグループ
        ]

    def test_fetch_latest_releases(self) -> None:
        """最新のIR情報取得テスト"""
        from src.data_collector.tdnet_real_scraper import TDnetRealScraper
        
        scraper = TDnetRealScraper()
        
        # 最新のリリースを取得
        releases = scraper.fetch_latest_releases(limit=10)
        
        # データの検証
        self.assertIsNotNone(releases)
        self.assertIsInstance(releases, list)
        
        if len(releases) > 0:
            # 最初のリリースを詳細チェック
            release = releases[0]
            
            # 必須フィールドの確認
            self.assertIn('release_id', release)
            self.assertIn('company_code', release)
            self.assertIn('company_name', release)
            self.assertIn('title', release)
            self.assertIn('release_date', release)
            self.assertIn('release_time', release)
            self.assertIn('pdf_url', release)
            self.assertIn('category', release)
            
            # データの妥当性確認
            self.assertTrue(release['company_code'].isdigit())
            self.assertTrue(len(release['company_code']) == 4)
            self.assertIsNotNone(release['title'])
            self.assertIsNotNone(release['company_name'])

    def test_fetch_company_releases(self) -> None:
        """特定企業のIR情報取得テスト"""
        from src.data_collector.tdnet_real_scraper import TDnetRealScraper
        
        scraper = TDnetRealScraper()
        
        for company_code in self.test_companies[:1]:  # 1社のみテスト
            # API制限回避
            time.sleep(self.api_wait_time)
            
            # 企業のリリースを取得
            releases = scraper.fetch_company_releases(
                company_code=company_code,
                days_back=30
            )
            
            # データの検証
            self.assertIsNotNone(releases)
            self.assertIsInstance(releases, list)
            
            # 取得したリリースが指定企業のものか確認
            for release in releases:
                self.assertEqual(release['company_code'], company_code)

    def test_fetch_by_date_range(self) -> None:
        """日付範囲指定でのIR情報取得テスト"""
        from src.data_collector.tdnet_real_scraper import TDnetRealScraper
        
        scraper = TDnetRealScraper()
        
        # 過去7日間のデータを取得
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        releases = scraper.fetch_by_date_range(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # データの検証
        self.assertIsNotNone(releases)
        self.assertIsInstance(releases, list)
        
        # 日付範囲内のデータか確認
        for release in releases:
            release_date = datetime.strptime(release['release_date'], '%Y-%m-%d')
            self.assertGreaterEqual(release_date, start_date.replace(hour=0, minute=0, second=0, microsecond=0))
            self.assertLessEqual(release_date, end_date)

    def test_search_by_keywords(self) -> None:
        """キーワード検索テスト"""
        from src.data_collector.tdnet_real_scraper import TDnetRealScraper
        
        scraper = TDnetRealScraper()
        
        # キーワードで検索
        keywords = ["決算", "配当", "業績"]
        
        for keyword in keywords[:1]:  # 1つのキーワードのみテスト
            time.sleep(self.api_wait_time)
            
            releases = scraper.search_releases(
                keyword=keyword,
                limit=5
            )
            
            # データの検証
            self.assertIsNotNone(releases)
            self.assertIsInstance(releases, list)
            
            # キーワードが含まれているか確認
            for release in releases:
                title = release.get('title', '')
                # タイトルにキーワードが含まれているか
                self.assertTrue(
                    keyword in title,
                    f"Keyword '{keyword}' not found in title: {title}"
                )

    def test_fetch_important_releases(self) -> None:
        """重要なIR情報の取得テスト"""
        from src.data_collector.tdnet_real_scraper import TDnetRealScraper
        
        scraper = TDnetRealScraper()
        
        # 重要なリリースのみ取得
        important_releases = scraper.fetch_important_releases(
            categories=['決算短信', '業績予想の修正', '配当予想の修正'],
            limit=10
        )
        
        # データの検証
        self.assertIsNotNone(important_releases)
        self.assertIsInstance(important_releases, list)
        
        # カテゴリーが正しいか確認
        valid_categories = ['決算短信', '業績予想の修正', '配当予想の修正']
        for release in important_releases:
            self.assertIn(release['category'], valid_categories)

    def test_parse_release_content(self) -> None:
        """リリース内容の解析テスト"""
        from src.data_collector.tdnet_real_scraper import TDnetRealScraper
        
        scraper = TDnetRealScraper()
        
        # 最新のリリースを1件取得
        releases = scraper.fetch_latest_releases(limit=1)
        
        if releases:
            release = releases[0]
            
            # リリース内容を解析
            parsed_content = scraper.parse_release_content(release)
            
            # 解析結果の検証
            self.assertIsNotNone(parsed_content)
            self.assertIn('release_type', parsed_content)
            self.assertIn('importance_score', parsed_content)
            self.assertIn('keywords', parsed_content)
            
            # 重要度スコアの範囲確認
            self.assertGreaterEqual(parsed_content['importance_score'], 0)
            self.assertLessEqual(parsed_content['importance_score'], 100)

    def test_monitor_real_time_releases(self) -> None:
        """リアルタイムモニタリングテスト"""
        from src.data_collector.tdnet_real_scraper import TDnetRealScraper
        
        scraper = TDnetRealScraper()
        
        # 短時間のモニタリングテスト（5秒間）
        new_releases = []
        
        def callback(release: Dict[str, Any]) -> None:
            """新規リリース検出時のコールバック"""
            new_releases.append(release)
        
        # モニタリング開始（非同期処理のテスト）
        monitor_result = scraper.start_monitoring(
            callback=callback,
            interval_seconds=2,
            duration_seconds=5
        )
        
        # モニタリングが正常に動作したか確認
        self.assertTrue(monitor_result)
        
        # 新規リリースがあれば内容を確認（なくても正常）
        if new_releases:
            for release in new_releases:
                self.assertIn('release_id', release)
                self.assertIn('company_code', release)

    def test_cache_functionality(self) -> None:
        """キャッシュ機能テスト"""
        from src.data_collector.tdnet_real_scraper import TDnetRealScraper
        
        scraper = TDnetRealScraper(enable_cache=True, cache_ttl=10)
        
        # 1回目の取得（キャッシュなし）
        start_time = time.time()
        releases1 = scraper.fetch_latest_releases(limit=5)
        fetch_time1 = time.time() - start_time
        
        # 2回目の取得（キャッシュあり）
        start_time = time.time()
        releases2 = scraper.fetch_latest_releases(limit=5)
        fetch_time2 = time.time() - start_time
        
        # キャッシュからの取得の方が高速であることを確認
        self.assertLess(fetch_time2, fetch_time1)
        
        # データが同一であることを確認
        self.assertEqual(len(releases1), len(releases2))
        if releases1:
            self.assertEqual(releases1[0]['release_id'], releases2[0]['release_id'])

    def test_error_handling(self) -> None:
        """エラーハンドリングテスト"""
        from src.data_collector.tdnet_real_scraper import TDnetRealScraper
        
        scraper = TDnetRealScraper()
        
        # 存在しない企業コード
        invalid_company_code = "0000"
        
        # エラーが適切に処理されることを確認
        releases = scraper.fetch_company_releases(
            company_code=invalid_company_code,
            days_back=7
        )
        
        # 空のリストまたはエラー情報を含む結果が返される
        self.assertIsNotNone(releases)
        self.assertIsInstance(releases, list)
        # 無効な企業コードの場合、結果は空になるはず
        self.assertEqual(len(releases), 0)

    def test_rate_limiting(self) -> None:
        """レート制限テスト"""
        from src.data_collector.tdnet_real_scraper import TDnetRealScraper
        
        scraper = TDnetRealScraper(rate_limit=2)  # 秒間2リクエストまで
        
        # 連続リクエスト
        request_times = []
        for i in range(3):
            start_time = time.time()
            scraper.fetch_latest_releases(limit=1)
            request_times.append(time.time() - start_time)
        
        # レート制限により後のリクエストが遅延することを確認
        # 3回目のリクエストは制限により遅延するはず
        if len(request_times) >= 3:
            avg_first_two = sum(request_times[:2]) / 2
            self.assertGreater(request_times[2], avg_first_two * 0.5)


if __name__ == '__main__':
    unittest.main()