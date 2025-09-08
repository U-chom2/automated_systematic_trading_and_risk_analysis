"""本日の推奨銘柄分析システム"""
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 日本の主要銘柄リスト
JAPANESE_STOCKS = [
    "7203.T",  # トヨタ自動車
    "9984.T",  # ソフトバンクグループ
    "6758.T",  # ソニーグループ
    "8306.T",  # 三菱UFJフィナンシャル
    "9432.T",  # NTT
    "6861.T",  # キーエンス
    "4063.T",  # 信越化学工業
    "7267.T",  # ホンダ
    "8058.T",  # 三菱商事
    "7974.T",  # 任天堂
    "9433.T",  # KDDI
    "4661.T",  # オリエンタルランド
    "6098.T",  # リクルート
    "3382.T",  # セブン&アイ
    "6501.T",  # 日立製作所
]

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を計算"""
    # 移動平均
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ボリンジャーバンド
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # 出来高の移動平均
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
    return df

def analyze_stock(ticker: str) -> Dict:
    """個別銘柄の分析"""
    try:
        stock = yf.Ticker(ticker)
        
        # 過去90日のデータを取得
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
        
        # テクニカル指標を計算
        df = calculate_technical_indicators(df)
        
        # 最新のデータ
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # スコア計算
        score = 0
        signals = []
        
        # トレンド判定
        if latest['Close'] > latest['SMA_5'] > latest['SMA_20']:
            score += 20
            signals.append("上昇トレンド確認")
        
        # ゴールデンクロス
        if (latest['SMA_5'] > latest['SMA_20'] and 
            prev['SMA_5'] <= prev['SMA_20']):
            score += 30
            signals.append("ゴールデンクロス発生")
        
        # RSI判定
        if 30 < latest['RSI'] < 70:
            score += 10
            if latest['RSI'] < 40:
                score += 10
                signals.append("RSI売られ過ぎ圏から回復")
        
        # MACD判定
        if latest['MACD'] > latest['Signal']:
            score += 15
            signals.append("MACD買いシグナル")
        
        # ボリンジャーバンド判定
        if latest['Close'] < latest['BB_lower']:
            score += 15
            signals.append("BB下限タッチ（反発期待）")
        
        # 出来高判定
        if latest['Volume'] > latest['Volume_MA'] * 1.5:
            score += 10
            signals.append("出来高急増")
        
        # 直近のパフォーマンス
        returns_5d = (latest['Close'] - df.iloc[-5]['Close']) / df.iloc[-5]['Close'] * 100
        returns_20d = (latest['Close'] - df.iloc[-20]['Close']) / df.iloc[-20]['Close'] * 100
        
        # 企業情報を取得
        info = stock.info
        company_name = info.get('longName', ticker)
        market_cap = info.get('marketCap', 0)
        per = info.get('trailingPE', 0)
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'current_price': latest['Close'],
            'score': score,
            'signals': signals,
            'returns_5d': returns_5d,
            'returns_20d': returns_20d,
            'rsi': latest['RSI'],
            'volume_ratio': latest['Volume'] / latest['Volume_MA'],
            'market_cap': market_cap,
            'per': per
        }
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None

def main():
    """メイン分析処理"""
    print("=" * 80)
    print(f"📊 株式分析レポート - {datetime.now().strftime('%Y年%m月%d日')}")
    print("=" * 80)
    print()
    
    print("🔍 分析中...")
    results = []
    
    for ticker in JAPANESE_STOCKS:
        print(f"  {ticker} を分析中...", end='')
        result = analyze_stock(ticker)
        if result:
            results.append(result)
            print(" ✅")
        else:
            print(" ❌")
    
    # スコアでソート
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print()
    print("=" * 80)
    print("🏆 本日の推奨銘柄 TOP 5")
    print("=" * 80)
    print()
    
    for i, stock in enumerate(results[:5], 1):
        print(f"【第{i}位】 {stock['ticker']} - {stock['company_name']}")
        print(f"  📈 総合スコア: {stock['score']}/100点")
        print(f"  💰 現在株価: ¥{stock['current_price']:,.0f}")
        print(f"  📊 5日リターン: {stock['returns_5d']:.2f}%")
        print(f"  📊 20日リターン: {stock['returns_20d']:.2f}%")
        print(f"  📊 RSI: {stock['rsi']:.1f}")
        print(f"  📊 出来高比率: {stock['volume_ratio']:.2f}倍")
        if stock['per'] > 0:
            print(f"  📊 PER: {stock['per']:.2f}")
        
        print("  📝 買いシグナル:")
        for signal in stock['signals']:
            print(f"    ✓ {signal}")
        print()
    
    print("=" * 80)
    print("⚠️  投資判断に関する注意事項")
    print("=" * 80)
    print("• 本分析は参考情報であり、投資判断は自己責任で行ってください")
    print("• 市場環境の急変に注意し、適切なリスク管理を行ってください")
    print("• 分散投資を心がけ、一銘柄への集中投資は避けてください")
    print()
    
    # 市場全体の状況
    print("=" * 80)
    print("📊 市場全体の状況")
    print("=" * 80)
    
    # 日経225の状況を確認
    nikkei = yf.Ticker("^N225")
    nikkei_df = nikkei.history(period="5d")
    if not nikkei_df.empty:
        nikkei_latest = nikkei_df.iloc[-1]['Close']
        nikkei_prev = nikkei_df.iloc[-2]['Close']
        nikkei_change = (nikkei_latest - nikkei_prev) / nikkei_prev * 100
        print(f"日経平均: ¥{nikkei_latest:,.0f} ({nikkei_change:+.2f}%)")
    
    # 為替の確認
    usdjpy = yf.Ticker("USDJPY=X")
    usdjpy_df = usdjpy.history(period="5d")
    if not usdjpy_df.empty:
        usdjpy_latest = usdjpy_df.iloc[-1]['Close']
        print(f"USD/JPY: ¥{usdjpy_latest:.2f}")
    
    print()
    print(f"分析完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()