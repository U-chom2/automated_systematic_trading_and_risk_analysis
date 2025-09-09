"""
東証グロース企業スクリーニング実行クラス
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any
import yfinance as yf
from src.utils.logger_utils import create_dual_logger
from .models import GrowthCompany, ScreeningResult
# from src.infrastructure.data_sources.market_data import MarketDataFetcher


logger = create_dual_logger(__name__, console_output=True)


class GrowthScreener:
    """東証グロース企業スクリーニング"""
    
    def __init__(self, 
                 market_cap_limit_billion: float = 100,
                 ir_days_within: int = 30,
                 data_fetcher: Optional[Any] = None):
        """
        Args:
            market_cap_limit_billion: 時価総額上限（億円）
            ir_days_within: IR公開日からの日数制限
            data_fetcher: マーケットデータ取得クラス
        """
        self.market_cap_limit_billion = market_cap_limit_billion
        self.ir_days_within = ir_days_within
        self.data_fetcher = data_fetcher  # or MarketDataFetcher()
        
    def get_growth_companies(self) -> List[GrowthCompany]:
        """東証グロース上場企業リストを取得"""
        logger.info("東証グロース企業リストを取得中...")
        
        # 東証グロースの銘柄コードリスト（607社）
        # 実際の実装では、外部APIや公式データソースから取得
        growth_tickers = self._fetch_growth_tickers()
        
        companies = []
        for ticker in growth_tickers:
            try:
                company = self._fetch_company_info(ticker)
                if company:
                    companies.append(company)
            except Exception as e:
                logger.warning(f"企業情報取得エラー {ticker}: {e}")
                continue
                
        logger.info(f"取得完了: {len(companies)}社")
        return companies
    
    def _fetch_growth_tickers(self) -> List[str]:
        """東証グロースの銘柄コードリストを取得"""
        import time
        from typing import Set
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        base_url = "https://moneyworld.jp/stock/stock-list?market=%E6%9D%B1%E8%A8%BC%E3%82%B0%E3%83%AD%E3%83%BC%E3%82%B9"
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        tickers = []
        collected_ids: Set[str] = set()
        
        try:
            logger.info(f"東証グロース企業リストをスクレイピング: {base_url}")
            driver.get(base_url)
            time.sleep(5)
            
            page_num = 1
            consecutive_no_data = 0
            max_consecutive_no_data = 3
            
            while True:
                logger.debug(f"ページ {page_num} を処理中")
                
                # 現在のページから企業情報を取得
                stock_rows = driver.find_elements(By.CLASS_NAME, 'p-stocks_body_row')
                
                new_tickers_count = 0
                for row in stock_rows:
                    try:
                        stock_id = row.find_element(By.CLASS_NAME, 'p-stock_id').text.strip()
                        
                        if stock_id and stock_id not in collected_ids:
                            # 時価総額を取得してフィルタリング
                            try:
                                market_cap_elem = row.find_element(By.CLASS_NAME, 'p-stock_flow')
                                market_cap_text = market_cap_elem.text.strip()
                                
                                # 時価総額を億円単位に変換
                                market_cap_billion = self._parse_market_cap(market_cap_text)
                                
                                # 100億円以下の企業のみ対象
                                if market_cap_billion <= self.market_cap_limit_billion:
                                    ticker = f"{stock_id}.T"
                                    tickers.append(ticker)
                                    collected_ids.add(stock_id)
                                    new_tickers_count += 1
                                    logger.debug(f"追加: {ticker} (時価総額: {market_cap_billion:.1f}億円)")
                                else:
                                    logger.debug(f"スキップ: {stock_id} (時価総額: {market_cap_billion:.1f}億円 > {self.market_cap_limit_billion}億円)")
                            except Exception:
                                # 時価総額が取得できない場合は追加
                                ticker = f"{stock_id}.T"
                                tickers.append(ticker)
                                collected_ids.add(stock_id)
                                new_tickers_count += 1
                                
                    except Exception:
                        continue
                
                if new_tickers_count == 0:
                    consecutive_no_data += 1
                    if consecutive_no_data >= max_consecutive_no_data:
                        logger.info("新規データがないため終了")
                        break
                else:
                    consecutive_no_data = 0
                    logger.info(f"ページ {page_num} から {new_tickers_count} 件取得 (累計: {len(tickers)} 件)")
                
                # nextボタンの確認とクリック
                has_next = False
                try:
                    next_buttons = driver.find_elements(By.CSS_SELECTOR, "span.next a")
                    for button in next_buttons:
                        if button.is_displayed() and button.is_enabled():
                            driver.execute_script("arguments[0].scrollIntoView(true);", button)
                            time.sleep(1)
                            driver.execute_script("arguments[0].click();", button)
                            has_next = True
                            break
                except Exception:
                    pass
                
                if not has_next:
                    logger.info("最終ページに到達")
                    break
                
                time.sleep(3)
                page_num += 1
                
        except Exception as e:
            logger.error(f"スクレイピングエラー: {e}")
            
        finally:
            driver.quit()
        
        logger.info(f"東証グロース企業 {len(tickers)} 社を取得完了")
        return tickers
    
    def _parse_market_cap(self, market_cap_text: str) -> float:
        """時価総額テキストを億円単位の数値に変換"""
        import re
        
        if not market_cap_text:
            return 0.0
        
        # 数値部分を抽出
        num_match = re.search(r'([\d,\.]+)', market_cap_text)
        if not num_match:
            return 0.0
        
        num_str = num_match.group(1).replace(',', '')
        try:
            num = float(num_str)
        except ValueError:
            return 0.0
        
        # 単位を判定して億円に変換
        if '兆' in market_cap_text:
            return num * 10000  # 兆円→億円
        elif '億' in market_cap_text:
            return num  # 億円
        elif '百万' in market_cap_text or 'M' in market_cap_text:
            return num / 100  # 百万円→億円
        else:
            # 単位が不明な場合は円と仮定
            return num / 100000000  # 円→億円
    
    def _fetch_company_info(self, ticker: str) -> Optional[GrowthCompany]:
        """個別企業の情報を取得"""
        try:
            # yfinanceで基本情報取得
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # IR情報は別途取得が必要（実際の実装では専用APIを使用）
            latest_ir_date = self._fetch_latest_ir_date(ticker)
            
            company = GrowthCompany(
                company_code=ticker.replace(".T", ""),
                company_name=info.get('longName', ''),
                ticker=ticker,
                market_cap=info.get('marketCap', 0),
                latest_ir_date=latest_ir_date,
                ir_url=self._get_ir_url(ticker),
                sector=info.get('sector', ''),
                is_target_candidate=False
            )
            
            return company
            
        except Exception as e:
            logger.error(f"企業情報取得失敗 {ticker}: {e}")
            return None
    
    def _fetch_latest_ir_date(self, ticker: str) -> Optional[datetime]:
        """最新のIR公開日を取得"""
        # 実際の実装では、企業のIRページをスクレイピングまたはAPIで取得
        # ここではダミーデータを返す
        from datetime import timedelta
        import random
        
        # ランダムに過去60日以内の日付を生成（テスト用）
        days_ago = random.randint(0, 60)
        return datetime.now() - timedelta(days=days_ago)
    
    def _get_ir_url(self, ticker: str) -> str:
        """IR情報のURLを取得"""
        code = ticker.replace(".T", "")
        return f"https://www.jpx.co.jp/listing/ir-clips/{code}"
    
    def screen_companies(self, companies: List[GrowthCompany]) -> ScreeningResult:
        """企業をスクリーニング"""
        logger.info(f"スクリーニング開始: {len(companies)}社")
        
        passed_companies = []
        
        for company in companies:
            # 時価総額チェック
            if not company.is_within_market_cap_limit(self.market_cap_limit_billion):
                logger.debug(f"{company.company_name}: 時価総額超過 ({company.market_cap_billion:.1f}億円)")
                continue
            
            # IR日付チェック
            if not company.has_recent_ir(self.ir_days_within):
                logger.debug(f"{company.company_name}: IR日付が古い")
                continue
            
            # すべての条件をクリア
            company.is_target_candidate = True
            passed_companies.append(company)
            logger.info(f"✓ {company.company_name} がスクリーニングを通過")
        
        result = ScreeningResult(
            screening_date=datetime.now(),
            total_companies=len(companies),
            filtered_companies=len(passed_companies),
            passed_companies=passed_companies
        )
        
        logger.info(f"スクリーニング完了: {result.filtered_companies}/{result.total_companies}社 通過 (通過率: {result.pass_rate:.1%})")
        
        return result
    
    def save_to_csv(self, result: ScreeningResult, output_path: str = "data/target.csv"):
        """スクリーニング結果をCSVに保存"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        csv_data = result.to_csv_data()
        
        if not csv_data:
            logger.warning("保存するデータがありません")
            return
        
        fieldnames = csv_data[0].keys()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        logger.info(f"スクリーニング結果を保存: {output_file} ({len(csv_data)}社)")
    
    def execute(self) -> ScreeningResult:
        """スクリーニングを実行"""
        logger.info("="*50)
        logger.info("東証グロース企業スクリーニング開始")
        logger.info(f"条件: 時価総額 <= {self.market_cap_limit_billion}億円, IR <= {self.ir_days_within}日以内")
        logger.info("="*50)
        
        # 企業リスト取得
        companies = self.get_growth_companies()
        
        # スクリーニング実行
        result = self.screen_companies(companies)
        
        # CSV保存
        self.save_to_csv(result)
        
        logger.info("スクリーニング処理完了")
        
        return result