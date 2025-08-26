#!/usr/bin/env python3
"""
yfinance の詳細デバッグ
"""

import yfinance as yf
import pandas as pd

def debug_yfinance_detail():
    """yfinance の詳細分析"""
    symbol = "9432.T"
    ticker = yf.Ticker(symbol)
    
    print(f"=== yfinance 詳細デバッグ: {symbol} ===\n")
    
    try:
        # 四半期財務データの詳細分析
        print("1. 四半期財務データの詳細:")
        quarterly_income = ticker.quarterly_financials
        
        if quarterly_income is not None:
            print(f"   データ形状: {quarterly_income.shape}")
            print(f"   列（期間）: {list(quarterly_income.columns)}")
            print()
            
            # Total Revenueの詳細
            if 'Total Revenue' in quarterly_income.index:
                revenue_data = quarterly_income.loc['Total Revenue']
                print("   Total Revenue の詳細:")
                print(f"     全データ: {revenue_data.values}")
                print(f"     各期間:")
                for i, (date, value) in enumerate(revenue_data.items()):
                    print(f"       {date}: {value} (NaN: {pd.isna(value)})")
                
                # NaNでない最新2期のデータを取得
                valid_data = revenue_data.dropna()
                print(f"     有効データ: {len(valid_data)}個")
                if len(valid_data) >= 2:
                    latest = valid_data.iloc[0]
                    previous = valid_data.iloc[1] 
                    growth = ((latest - previous) / previous) * 100
                    print(f"     売上高成長率: {growth:.1f}%")
                    print(f"     最新期: {latest:,.0f}")
                    print(f"     前期: {previous:,.0f}")
                print()
            
            # 他の売上関連項目を探す
            print("   売上関連項目の検索:")
            revenue_related = [idx for idx in quarterly_income.index 
                             if any(keyword in idx.lower() for keyword in ['revenue', 'sales', 'income'])]
            
            for item in revenue_related[:10]:  # 上位10項目
                data = quarterly_income.loc[item]
                valid_count = data.dropna().shape[0]
                print(f"     {item}: 有効データ{valid_count}個")
                if valid_count >= 2:
                    valid_data = data.dropna()
                    latest = valid_data.iloc[0]
                    previous = valid_data.iloc[1]
                    if previous != 0:
                        growth = ((latest - previous) / previous) * 100
                        print(f"       成長率: {growth:.1f}%")
        print()
        
        # 年次財務データの確認
        print("2. 年次財務データ:")
        annual_income = ticker.financials
        if annual_income is not None:
            print(f"   データ形状: {annual_income.shape}")
            print(f"   列（年度）: {list(annual_income.columns)}")
            
            if 'Total Revenue' in annual_income.index:
                revenue_data = annual_income.loc['Total Revenue']
                print("   年次Total Revenue:")
                for date, value in revenue_data.items():
                    print(f"     {date}: {value}")
                
                valid_data = revenue_data.dropna()
                if len(valid_data) >= 2:
                    latest = valid_data.iloc[0]
                    previous = valid_data.iloc[1]
                    if previous != 0:
                        growth = ((latest - previous) / previous) * 100
                        print(f"   年次売上高成長率: {growth:.1f}%")
        print()
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_yfinance_detail()