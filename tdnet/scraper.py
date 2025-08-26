"""TDNet web scraping functionality."""

import requests
from datetime import date
from typing import Optional
import json
import os


class TDNetScraper:
    """Scraper for TDNet disclosure information."""
    
    BASE_URL = "https://www.release.tdnet.info"
    
    def __init__(self) -> None:
        """Initialize the TDNet scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def _get_disclosure_page(self, target_date: date) -> Optional[str]:
        """Get the HTML content of TDNet disclosure page for a specific date.
        
        Args:
            target_date: Date to get disclosures for.
            
        Returns:
            HTML content as string, or None if request failed.
        """
        try:
            # Use the correct URL pattern we discovered
            formatted_date = target_date.strftime("%Y%m%d")
            disclosure_url = f"{self.BASE_URL}/inbs/I_list_001_{formatted_date}.html"
            
            response = self.session.get(disclosure_url, timeout=10)
            if response.status_code == 200:
                # Set proper encoding for Japanese content
                response.encoding = 'utf-8'
                return response.text
            else:
                return None
                
        except requests.RequestException:
            return None
    
    def check_company_disclosure(self, company_name: str, target_date: date) -> bool:
        """Check if a specific company made a disclosure on the given date.
        
        Args:
            company_name: Name of the company to check.
            target_date: Date to check for disclosures.
            
        Returns:
            True if the company made a disclosure on the date.
        """
        # Get the company code from the company list
        try:
            company_list_path = os.path.join(os.path.dirname(__file__), "企業リスト.json")
            with open(company_list_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                company_list = data.get("企業リスト", [])
            
            # Find the company code for the given name
            company_code = None
            for company in company_list:
                if company.get("銘柄名") == company_name:
                    company_code = company.get("銘柄コード")
                    break
            
            if not company_code:
                return False
            
        except Exception:
            return False
        
        # Get the disclosure page content
        html_content = self._get_disclosure_page(target_date)
        if not html_content:
            return False
        
        # Search for the company code directly in the HTML content
        return company_code in html_content