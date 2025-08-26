#!/usr/bin/env python3
"""
yfinance を使ったNTTの財務データ取得テスト
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any

def test_yfinance_ntt():
    """yfinance でNTTの財務データを詳しく調査"""
    symbol = "9432.T"
    
    print(f"=== yfinance で {symbol} の財務データ調査 ===\n")
    
    try:
        # Tickerオブジェクトを作成
        ticker = yf.Ticker(symbol)
        
        # 1. 基本情報
        print("1. 基本情報:")
        info = ticker.info
        print(f"   会社名: {info.get('longName', 'N/A')}")
        print(f"   業界: {info.get('industry', 'N/A')}")
        print(f"   セクター: {info.get('sector', 'N/A')}")
        print()
        
        # 2. 財務諸表データを確認
        print("2. 利用可能な財務データ:")
        
        # 損益計算書（四半期）
        try:
            quarterly_income = ticker.quarterly_financials
            print(f"   四半期損益計算書: 取得成功")
            print(f"     列数: {quarterly_income.shape[1] if quarterly_income is not None else 'N/A'}")
            print(f"     行数: {quarterly_income.shape[0] if quarterly_income is not None else 'N/A'}")
            if quarterly_income is not None:
                print(f"     主要項目: {list(quarterly_income.index[:10])}")
                
                # 売上高を探す
                revenue_keys = ['Total Revenue', 'Revenue', 'Operating Revenue']
                for key in revenue_keys:
                    if key in quarterly_income.index:
                        revenue_data = quarterly_income.loc[key]
                        print(f"     {key} データ:")
                        print(f"       最新2期: {revenue_data.iloc[:2].values}")
                        
                        if len(revenue_data) >= 2:
                            latest = revenue_data.iloc[0]
                            previous = revenue_data.iloc[1]
                            if pd.notna(latest) and pd.notna(previous) and previous != 0:
                                growth = ((latest - previous) / previous) * 100
                                print(f"       売上高成長率: {growth:.1f}%")
                        break
            print()
        except Exception as e:
            print(f"   四半期損益計算書: エラー - {e}")
            print()
        
        # 損益計算書（年次）
        try:
            annual_income = ticker.financials
            print(f"   年次損益計算書: 取得成功")
            print(f"     列数: {annual_income.shape[1] if annual_income is not None else 'N/A'}")
            print(f"     行数: {annual_income.shape[0] if annual_income is not None else 'N/A'}")
            if annual_income is not None:
                print(f"     主要項目: {list(annual_income.index[:10])}")
            print()
        except Exception as e:
            print(f"   年次損益計算書: エラー - {e}")
            print()
            
        # バランスシート
        try:
            balance_sheet = ticker.balance_sheet
            print(f"   バランスシート: 取得成功")
            print(f"     列数: {balance_sheet.shape[1] if balance_sheet is not None else 'N/A'}")
            print()
        except Exception as e:
            print(f"   バランスシート: エラー - {e}")
            print()
            
        # キャッシュフロー
        try:
            cashflow = ticker.cashflow
            print(f"   キャッシュフロー: 取得成功")
            print(f"     列数: {cashflow.shape[1] if cashflow is not None else 'N/A'}")
            print()
        except Exception as e:
            print(f"   キャッシュフロー: エラー - {e}")
            print()
            
        # 3. 主要統計情報
        print("3. 主要統計情報から売上高関連データ:")
        revenue_related_keys = [
            'totalRevenue', 'revenuePerShare', 'quarterlyRevenueGrowth',
            'revenueQuarterlyGrowth', 'grossMargins', 'operatingMargins'
        ]
        
        for key in revenue_related_keys:
            value = info.get(key)
            if value is not None:
                if key.endswith('Growth'):
                    print(f"   {key}: {value * 100:.1f}%" if isinstance(value, (int, float)) else f"   {key}: {value}")
                else:
                    print(f"   {key}: {value}")
        print()
        
    except Exception as e:
        print(f"全体エラー: {e}")
        import traceback
        traceback.print_exc()

def get_revenue_growth_yfinance(symbol: str = "9432.T") -> Dict[str, Any]:
    """yfinance を使って売上高成長率を取得"""
    result = {
        'revenue_growth': None,
        'is_revenue_up': None,
        'method': None,
        'error': None
    }
    
    try:
        ticker = yf.Ticker(symbol)
        
        # 方法1: 四半期財務データから直接計算
        try:
            quarterly_income = ticker.quarterly_financials
            if quarterly_income is not None and not quarterly_income.empty:
                revenue_keys = ['Total Revenue', 'Revenue', 'Operating Revenue']
                
                for key in revenue_keys:
                    if key in quarterly_income.index:
                        revenue_data = quarterly_income.loc[key]
                        
                        if len(revenue_data) >= 2:
                            latest = revenue_data.iloc[0]
                            previous = revenue_data.iloc[1]
                            
                            if pd.notna(latest) and pd.notna(previous) and previous != 0:
                                growth = ((latest - previous) / previous) * 100
                                result['revenue_growth'] = growth
                                result['is_revenue_up'] = growth > 0
                                result['method'] = f'quarterly_financials_{key}'
                                return result
        except Exception as e:
            pass
        
        # 方法2: info から取得
        try:
            info = ticker.info
            if 'quarterlyRevenueGrowth' in info and info['quarterlyRevenueGrowth'] is not None:
                growth = info['quarterlyRevenueGrowth'] * 100
                result['revenue_growth'] = growth
                result['is_revenue_up'] = growth > 0
                result['method'] = 'info_quarterlyRevenueGrowth'
                return result
        except Exception as e:
            pass
            
        result['error'] = "売上高データを取得できませんでした"
        
    except Exception as e:
        result['error'] = str(e)
        
    return result

if __name__ == "__main__":
    test_yfinance_ntt()
    
    print("\n" + "="*50)
    print("売上高成長率取得テスト:")
    revenue_result = get_revenue_growth_yfinance()
    
    if revenue_result['error']:
        print(f"エラー: {revenue_result['error']}")
    else:
        print(f"売上高成長率: {revenue_result['revenue_growth']:.1f}%")
        print(f"増収か: {'はい' if revenue_result['is_revenue_up'] else 'いいえ'}")
        print(f"取得方法: {revenue_result['method']}")