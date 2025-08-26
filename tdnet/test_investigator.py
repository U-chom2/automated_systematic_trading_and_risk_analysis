"""Tests for TDNet disclosure investigation functionality."""

import unittest
from datetime import date
from tdnet.investigator import TDNetInvestigator, check_company_disclosure, check_all_companies_disclosure


class TestTDNetInvestigator(unittest.TestCase):
    """Test cases for TDNet investigator functionality."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.investigator = TDNetInvestigator()
    
    def test_disclosure_august_22_2025_true(self) -> None:
        """Test that キューピーネットHD returns True for 2025-08-22."""
        result = self.investigator.check_disclosure_by_date(
            "キューピーネットHD", 
            date(2025, 8, 22)
        )
        self.assertTrue(result)
    
    def test_disclosure_august_21_2025_false(self) -> None:
        """Test that キューピーネットHD returns False for 2025-08-21."""
        result = self.investigator.check_disclosure_by_date(
            "キューピーネットHD", 
            date(2025, 8, 21)
        )
        self.assertFalse(result)
    
    def test_disclosure_nonexistent_company(self) -> None:
        """Test that non-existent companies always return False."""
        result = self.investigator.check_disclosure_by_date(
            "非存在企業テスト", 
            date(2025, 8, 22)
        )
        self.assertFalse(result)
    
    def test_convenience_function_true_case(self) -> None:
        """Test convenience function with True case."""
        result = check_company_disclosure("キューピーネットHD", date(2025, 8, 22))
        self.assertTrue(result)
    
    def test_convenience_function_false_case(self) -> None:
        """Test convenience function with False case."""
        result = check_company_disclosure("キューピーネットHD", date(2025, 8, 21))
        self.assertFalse(result)
    
    def test_all_companies_august_22_2025(self) -> None:
        """Test checking all companies for 2025-08-22."""
        results = self.investigator.check_all_companies_by_date(date(2025, 8, 22))
        # Should return [True, False] for [キューピーネットHD, 博報堂HD] based on actual TDNet data
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0])   # キューピーネットHD should be True
        self.assertFalse(results[1])  # 博報堂HD should be False
    
    def test_all_companies_august_21_2025(self) -> None:
        """Test checking all companies for 2025-08-21."""
        results = self.investigator.check_all_companies_by_date(date(2025, 8, 21))
        # Should return [False, True] for [キューピーネットHD, 博報堂HD] based on actual TDNet data
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0])  # キューピーネットHD should be False
        self.assertTrue(results[1])   # 博報堂HD should be True (found in actual data)
    
    def test_convenience_function_all_companies_aug22(self) -> None:
        """Test convenience function for checking all companies on 2025-08-22."""
        results = check_all_companies_disclosure(date(2025, 8, 22))
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0])   # キューピーネットHD should be True
        self.assertFalse(results[1])  # 博報堂HD should be False
    
    def test_convenience_function_all_companies_aug21(self) -> None:
        """Test convenience function for checking all companies on 2025-08-21."""
        results = check_all_companies_disclosure(date(2025, 8, 21))
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0])  # キューピーネットHD should be False
        self.assertTrue(results[1])   # 博報堂HD should be True


if __name__ == '__main__':
    unittest.main()