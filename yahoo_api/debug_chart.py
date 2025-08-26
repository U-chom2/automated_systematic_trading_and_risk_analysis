#!/usr/bin/env python3
"""
チャート作成のデバッグ
"""

from yahooquery import Ticker
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

def debug_chart():
    """チャート作成の詳細デバッグ"""
    symbol = "9432.T"
    ticker = Ticker(symbol)
    
    print(f"=== チャートデバッグ: {symbol} ===\n")
    
    try:
        # 過去30日のデータを取得
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        history = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        
        print(f"1. 元データ:")
        print(f"   データ型: {type(history)}")
        print(f"   形状: {history.shape if hasattr(history, 'shape') else 'N/A'}")
        
        # MultiIndexの場合
        if isinstance(history, pd.DataFrame) and isinstance(history.index, pd.MultiIndex):
            print(f"   MultiIndex レベル: {history.index.names}")
            df = history.loc[symbol].copy()
            print(f"   抽出後の形状: {df.shape}")
            print(f"   抽出後のインデックス型: {type(df.index)}")
            print(f"   最初の3行のインデックス: {df.index[:3]}")
            print(f"   インデックスの全て: {list(df.index)}")
            
            # 各インデックス要素の型を確認
            print("   各インデックス要素の型:")
            for i, idx in enumerate(df.index[:5]):  # 最初の5個
                print(f"     {i}: {idx} (型: {type(idx)})")
            
            # インデックスをDateTimeIndexに変換（utc=Trueで統一）
            print("   DateTimeIndexに変換中...")
            try:
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                print(f"   変換後のインデックス型: {type(df.index)}")
            except Exception as e:
                print(f"   UTC変換エラー: {e}")
                # 文字列に変換してから再度試行
                df.index = pd.to_datetime(df.index.astype(str))
                print(f"   文字列経由変換後のインデックス型: {type(df.index)}")
            
            # タイムゾーン情報を確認
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                print(f"   タイムゾーン情報: {df.index.tz}")
                print("   タイムゾーン修正中...")
                df.index = df.index.tz_convert('UTC').tz_localize(None)
                print(f"   修正後のタイムゾーン情報: {df.index.tz}")
            else:
                print("   タイムゾーン情報なし")
            
            # 移動平均を計算
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            
            print(f"   移動平均計算後の列: {list(df.columns)}")
            print(f"   最後の3行:")
            print(df.tail(3))
            
            # 簡単なグラフを作成
            plt.figure(figsize=(10, 6))
            
            # データを直接plot
            dates = df.index
            close_prices = df['close']
            ma5 = df['MA5']
            ma20 = df['MA20']
            
            print(f"   プロット用データ:")
            print(f"     dates型: {type(dates)}, 長さ: {len(dates)}")
            print(f"     close_prices型: {type(close_prices)}, 長さ: {len(close_prices)}")
            
            plt.plot(dates, close_prices, label='Close Price', linewidth=1)
            plt.plot(dates, ma5, label='5-day MA', linewidth=1.5, alpha=0.7)
            plt.plot(dates, ma20, label='20-day MA', linewidth=1.5, alpha=0.7)
            
            plt.title(f'{symbol} Stock Price and Moving Averages')
            plt.xlabel('Date')
            plt.ylabel('Price (JPY)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            output_file = "ntt_moving_average_debug.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   グラフを {output_file} に保存しました")
            
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chart()