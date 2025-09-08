#!/usr/bin/env python
"""クイック市場チェック - 5秒で今の状況を把握"""
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def quick_check():
    print("=" * 50)
    print(f"⚡ Quick Market Check - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)
    
    # 推奨銘柄TOP3の現在価格
    top_stocks = [
        ("4661.T", "オリエンタルランド"),
        ("8058.T", "三菱商事"),
        ("7203.T", "トヨタ自動車"),
    ]
    
    print("\n📊 推奨銘柄の現在価格:")
    for ticker, name in top_stocks:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - prev) / prev * 100) if prev != 0 else 0
                
                # 買いシグナル判定
                signal = "🟢 買い" if change > 0 else "🔴 売り" if change < -1 else "⚪ 様子見"
                
                print(f"  {ticker} {name:15} ¥{current:,.0f} ({change:+.1f}%) {signal}")
        except:
            print(f"  {ticker} {name:15} データ取得エラー")
    
    # 日経平均
    print("\n📈 市場指標:")
    try:
        nikkei = yf.Ticker("^N225")
        nikkei_hist = nikkei.history(period="2d")
        if not nikkei_hist.empty:
            current = nikkei_hist['Close'].iloc[-1]
            prev = nikkei_hist['Close'].iloc[-2] if len(nikkei_hist) > 1 else current
            change = ((current - prev) / prev * 100) if prev != 0 else 0
            trend = "📈 上昇" if change > 0.5 else "📉 下落" if change < -0.5 else "➡️ 横ばい"
            print(f"  日経平均: ¥{current:,.0f} ({change:+.1f}%) {trend}")
    except:
        pass
    
    # USD/JPY
    try:
        usdjpy = yf.Ticker("USDJPY=X")
        usdjpy_hist = usdjpy.history(period="2d")
        if not usdjpy_hist.empty:
            current = usdjpy_hist['Close'].iloc[-1]
            print(f"  USD/JPY: ¥{current:.2f}")
    except:
        pass
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    quick_check()