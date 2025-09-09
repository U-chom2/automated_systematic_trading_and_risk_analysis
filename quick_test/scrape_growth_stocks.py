"""
東証グロース市場の全企業情報をnextボタンがなくなるまでスクレイピング
"""
import time
from typing import List, Dict, Any, Set
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import re


def setup_driver(headless: bool = True) -> webdriver.Chrome:
    """Chromeドライバーのセットアップ"""
    options = Options()
    if headless:
        options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def scrape_current_page(driver: webdriver.Chrome, collected_ids: Set[str]) -> List[Dict[str, Any]]:
    """現在表示されているページから企業情報を取得"""
    stocks_data = []
    
    try:
        # データが更新されるまで少し待機
        time.sleep(2)
        
        # p-stocks_body_row要素を全て取得
        stock_rows = driver.find_elements(By.CLASS_NAME, 'p-stocks_body_row')
        
        print(f"見つかった銘柄行数: {len(stock_rows)}")
        
        for row in stock_rows:
            try:
                # 企業IDを取得
                try:
                    stock_id = row.find_element(By.CLASS_NAME, 'p-stock_id').text.strip()
                except:
                    stock_id = ''
                
                # すでに取得済みのIDはスキップ
                if stock_id in collected_ids:
                    continue
                
                # 企業名を取得
                try:
                    stock_name = row.find_element(By.CLASS_NAME, 'p-stock_name').text.strip()
                except:
                    stock_name = ''
                
                # 業種を取得
                try:
                    category = row.find_element(By.CLASS_NAME, 'p-stock_category').text.strip()
                except:
                    category = ''
                
                # 株価を取得
                try:
                    price_elem = row.find_element(By.CLASS_NAME, 'p-stock_price')
                    price_text = price_elem.text.strip()
                    price = re.sub(r'[^0-9,.]', '', price_text)
                except:
                    price = ''
                
                # 売買代金を取得
                try:
                    sellprice_elem = row.find_element(By.CLASS_NAME, 'p-stock_sellprice')
                    sellprice = sellprice_elem.text.strip()
                except:
                    sellprice = ''
                
                # 時価総額を取得
                try:
                    market_cap_elem = row.find_element(By.CLASS_NAME, 'p-stock_flow')
                    market_cap = market_cap_elem.text.strip()
                except:
                    market_cap = ''
                
                # データを辞書に格納
                if stock_id or stock_name:
                    stock_info = {
                        '企業ID': stock_id,
                        '企業名': stock_name,
                        '業種': category,
                        '株価': price,
                        '売買代金': sellprice,
                        '時価総額': market_cap,
                        'スクレイピング日時': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    stocks_data.append(stock_info)
                    collected_ids.add(stock_id)
                    print(f"取得: {stock_id} - {stock_name}")
                    
            except Exception as e:
                print(f"行データの取得エラー: {e}")
                continue
                
    except Exception as e:
        print(f"ページスクレイピングエラー: {e}")
    
    return stocks_data


def has_next_button(driver: webdriver.Chrome) -> bool:
    """nextボタンが存在し、クリック可能かを確認"""
    try:
        # 複数の方法でnextボタンを探す
        next_buttons = []
        
        # 方法1: span.next内のリンク
        try:
            next_spans = driver.find_elements(By.CSS_SELECTOR, "span.next a")
            next_buttons.extend(next_spans)
        except:
            pass
        
        # 方法2: rel="next"属性を持つリンク
        try:
            next_links = driver.find_elements(By.CSS_SELECTOR, "a[rel='next']")
            next_buttons.extend(next_links)
        except:
            pass
        
        # 方法3: テキストに"next"を含むリンク
        try:
            all_links = driver.find_elements(By.TAG_NAME, "a")
            for link in all_links:
                text = link.text.lower()
                if 'next' in text or '次' in text:
                    next_buttons.append(link)
        except:
            pass
        
        # クリック可能なnextボタンがあるか確認
        for button in next_buttons:
            try:
                # ボタンが表示されていて、有効な場合
                if button.is_displayed() and button.is_enabled():
                    # 親要素がdisabledクラスを持っていないか確認
                    parent = button.find_element(By.XPATH, "..")
                    parent_class = parent.get_attribute("class") or ""
                    if "disabled" not in parent_class:
                        return True
            except:
                continue
        
        return False
        
    except Exception as e:
        print(f"nextボタンの確認エラー: {e}")
        return False


def click_next_button(driver: webdriver.Chrome) -> bool:
    """nextボタンをクリック"""
    try:
        # 複数の方法でnextボタンを探してクリック
        next_buttons = []
        
        # span.next内のリンクを優先
        try:
            next_spans = driver.find_elements(By.CSS_SELECTOR, "span.next a")
            next_buttons.extend(next_spans)
        except:
            pass
        
        # rel="next"属性を持つリンク
        try:
            next_links = driver.find_elements(By.CSS_SELECTOR, "a[rel='next']")
            next_buttons.extend(next_links)
        except:
            pass
        
        for button in next_buttons:
            try:
                if button.is_displayed() and button.is_enabled():
                    # スクロールして要素を表示
                    driver.execute_script("arguments[0].scrollIntoView(true);", button)
                    time.sleep(1)
                    
                    # JavaScriptでクリック
                    driver.execute_script("arguments[0].click();", button)
                    print("nextボタンをクリックしました")
                    return True
            except:
                continue
                
    except Exception as e:
        print(f"nextボタンのクリックエラー: {e}")
    
    return False


def get_current_page_number(driver: webdriver.Chrome) -> int:
    """現在のページ番号を取得"""
    try:
        current_elem = driver.find_element(By.CSS_SELECTOR, "span.page.current")
        return int(current_elem.text.strip())
    except:
        return 1


def get_total_stock_count(driver: webdriver.Chrome) -> int:
    """総銘柄数を取得"""
    try:
        stock_count_elem = driver.find_element(By.ID, "stock_count")
        count_text = stock_count_elem.text
        # (xxx件)の形式から数値を抽出
        match = re.search(r'\((\d+)件\)', count_text)
        if match:
            return int(match.group(1))
    except:
        pass
    return 0


def main():
    """メイン処理"""
    base_url = "https://moneyworld.jp/stock/stock-list?market=%E6%9D%B1%E8%A8%BC%E3%82%B0%E3%83%AD%E3%83%BC%E3%82%B9"
    
    # 出力ディレクトリの作成
    output_dir = Path(__file__).parent
    output_file = output_dir / f"growth_stocks_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    driver = setup_driver(headless=True)
    all_stocks = []
    collected_ids = set()
    
    try:
        print(f"URL: {base_url} にアクセス中...")
        driver.get(base_url)
        time.sleep(5)
        
        # 総銘柄数を取得
        total_count = get_total_stock_count(driver)
        if total_count > 0:
            print(f"総銘柄数: {total_count} 件")
        
        page_num = 1
        consecutive_no_data = 0
        max_consecutive_no_data = 3
        
        # nextボタンがなくなるまでループ
        while True:
            print(f"\n--- ページ {page_num} の処理 ---")
            
            current_page = get_current_page_number(driver)
            print(f"現在のページ番号: {current_page}")
            
            # 現在のページから企業情報を取得
            page_stocks = scrape_current_page(driver, collected_ids)
            
            if page_stocks:
                all_stocks.extend(page_stocks)
                print(f"ページ {current_page} から {len(page_stocks)} 件の新規企業情報を取得")
                print(f"累計取得数: {len(all_stocks)} 件")
                consecutive_no_data = 0
            else:
                consecutive_no_data += 1
                print(f"新規データなし（{consecutive_no_data}/{max_consecutive_no_data}）")
                
                if consecutive_no_data >= max_consecutive_no_data:
                    print(f"{max_consecutive_no_data}ページ連続で新規データがないため終了します")
                    break
            
            # 総銘柄数に達したかチェック
            if total_count > 0 and len(all_stocks) >= total_count:
                print(f"総銘柄数 {total_count} 件に達しました")
                break
            
            # nextボタンの存在を確認
            if not has_next_button(driver):
                print("nextボタンが見つかりません。最終ページに到達しました。")
                break
            
            # nextボタンをクリック
            if not click_next_button(driver):
                print("nextボタンのクリックに失敗しました")
                break
            
            # ページが更新されるまで待機
            time.sleep(3)
            
            # ページが実際に変わったか確認
            new_page = get_current_page_number(driver)
            if new_page == current_page:
                print("ページが変更されませんでした")
                # 再度nextボタンの確認
                if not has_next_button(driver):
                    print("最終ページに到達しました")
                    break
            else:
                page_num = new_page
                    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nブラウザを閉じています...")
        driver.quit()
    
    # CSVファイルに保存
    if all_stocks:
        df = pd.DataFrame(all_stocks)
        
        # CSVに保存
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n=== 最終結果 ===")
        print(f"合計 {len(df)} 件のユニークな企業情報を取得しました")
        print(f"保存先: {output_file}")
        
        # データの概要を表示
        print("\n--- データ概要 ---")
        print(df.head(10))
        print(f"\n総企業数: {len(df)}")
        
        if total_count > 0:
            print(f"目標銘柄数: {total_count}")
            print(f"取得率: {len(df)/total_count*100:.1f}%")
        
        print(f"カラム: {df.columns.tolist()}")
        
        # 業種別統計
        print("\n--- 業種別統計 ---")
        print(df['業種'].value_counts().head(10))
        
        # データ完全性
        print("\n--- データ完全性 ---")
        print(f"企業IDが入力されている数: {df['企業ID'].notna().sum()}")
        print(f"企業名が入力されている数: {df['企業名'].notna().sum()}")
        print(f"株価が入力されている数: {df['株価'].notna().sum()}")
        print(f"時価総額が入力されている数: {df['時価総額'].notna().sum()}")
        
    else:
        print("取得できた企業情報がありません")
    
    return output_file if all_stocks else None


if __name__ == "__main__":
    result_file = main()
    if result_file:
        print(f"\n処理完了: {result_file}")
    else:
        print("\n処理失敗: データを取得できませんでした")