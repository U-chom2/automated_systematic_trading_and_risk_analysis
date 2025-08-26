#!/usr/bin/env python3
"""
yahooqueryの動作確認とデバッグ
"""

from yahooquery import Ticker
from datetime import datetime, timedelta
import pandas as pd

def debug_yahooquery():
    """yahooquery の動作を詳しく調査"""
    symbol = "9432.T"
    ticker = Ticker(symbol)
    
    print(f"=== {symbol} のデバッグ情報 ===\n")
    
    # 1. price データの詳細確認
    print("1. Price データ:")
    try:
        price_data = ticker.price
        print(f"   データ型: {type(price_data)}")
        if isinstance(price_data, dict) and symbol in price_data:
            data = price_data[symbol]
            print(f"   内容: {data}")
            print(f"   キー: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        print()
    except Exception as e:
        print(f"   エラー: {e}")
        print()
    
    # 2. history データの詳細確認
    print("2. History データ:")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        history = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        print(f"   データ型: {type(history)}")
        print(f"   形状: {history.shape if hasattr(history, 'shape') else 'N/A'}")
        
        if isinstance(history, dict):
            print(f"   辞書のキー: {list(history.keys())}")
            if symbol in history:
                data = history[symbol]
                print(f"   {symbol} データ型: {type(data)}")
                if hasattr(data, 'shape'):
                    print(f"   {symbol} 形状: {data.shape}")
                if hasattr(data, 'columns'):
                    print(f"   {symbol} 列名: {list(data.columns)}")
                if hasattr(data, 'index'):
                    print(f"   {symbol} インデックス型: {type(data.index)}")
                    print(f"   {symbol} インデックス最初の5個: {data.index[:5] if len(data.index) >= 5 else data.index}")
        elif isinstance(history, pd.DataFrame):
            print(f"   列名: {list(history.columns)}")
            print(f"   インデックス型: {type(history.index)}")
            print(f"   インデックス最初の5個: {history.index[:5] if len(history.index) >= 5 else history.index}")
            print(f"   最新データ:")
            print(history.tail(3))
        print()
    except Exception as e:
        print(f"   エラー: {e}")
        print()
    
    # 3. 財務データの確認
    print("3. 財務データ:")
    methods_to_try = [
        'financial_data',
        'key_stats', 
        'summary_detail',
        'price',
        'quote_type',
        'summary_profile'
    ]
    
    for method in methods_to_try:
        try:
            data = getattr(ticker, method)
            print(f"   {method}: 成功")
            print(f"     データ型: {type(data)}")
            if isinstance(data, dict) and symbol in data:
                info = data[symbol]
                print(f"     キー数: {len(info) if isinstance(info, dict) else 'N/A'}")
                if isinstance(info, dict):
                    keys = list(info.keys())
                    print(f"     主要キー: {keys[:10]}")  # 最初の10個のキーを表示
        except Exception as e:
            print(f"   {method}: エラー - {e}")
    print()

if __name__ == "__main__":
    debug_yahooquery()