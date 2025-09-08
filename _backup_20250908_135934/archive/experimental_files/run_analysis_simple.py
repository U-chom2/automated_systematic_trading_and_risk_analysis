#!/usr/bin/env python
"""今日の推奨銘柄を分析・出力する簡易版スクリプト"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis_engine.risk_model import RiskModel
from src.data_collector.target_watchlist_loader import TargetWatchlistLoader


def get_stock_data(symbol: str) -> Dict[str, Any]:
    """Get stock data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get historical data (60 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            return None
        
        # Current price and change
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
        
        # Volume analysis
        avg_volume_20 = hist['Volume'][-20:].mean() if len(hist) >= 20 else hist['Volume'].mean()
        current_volume = hist['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        
        # Technical indicators
        # RSI
        closes = hist['Close']
        delta = closes.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain[-14:].mean() if len(gain) >= 14 else 0
        avg_loss = loss[-14:].mean() if len(loss) >= 14 else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Moving average deviation
        ma_20 = closes[-20:].mean() if len(closes) >= 20 else current_price
        ma_deviation = ((current_price - ma_20) / ma_20 * 100) if ma_20 > 0 else 0
        
        # ATR (Average True Range)
        high_low = hist['High'] - hist['Low']
        atr = high_low[-14:].mean() if len(high_low) >= 14 else 0
        
        # Historical volatility
        returns = closes.pct_change().dropna()
        hv = returns[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0.2
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "change_percent": change_pct,
            "volume": current_volume,
            "volume_ratio": volume_ratio,
            "rsi": rsi,
            "ma_deviation": ma_deviation,
            "atr": atr,
            "historical_volatility": hv,
            "has_data": True
        }
        
    except Exception as e:
        print(f"  ⚠️ データ取得エラー ({symbol}): {e}")
        return None


def calculate_scores(stock_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate investment scores based on available data."""
    scores = {
        "catalyst_score": 0,  # Max 50
        "sentiment_score": 0,  # Max 30
        "technical_score": 0,  # Max 20
        "total_score": 0
    }
    
    if not stock_data or not stock_data.get("has_data"):
        return scores
    
    # Technical scoring (simplified)
    technical_points = 0
    
    # Volume spike bonus (up to 10 points)
    volume_ratio = stock_data.get("volume_ratio", 1.0)
    if volume_ratio > 2.0:
        technical_points += 10
    elif volume_ratio > 1.5:
        technical_points += 7
    elif volume_ratio > 1.2:
        technical_points += 5
    
    # RSI scoring (up to 5 points) - favor moderate RSI
    rsi = stock_data.get("rsi", 50)
    if 40 <= rsi <= 60:
        technical_points += 5
    elif 30 <= rsi <= 70:
        technical_points += 3
    elif rsi < 30:  # Oversold
        technical_points += 4
    
    # Momentum (up to 5 points)
    change_pct = stock_data.get("change_percent", 0)
    if 1 <= change_pct <= 3:
        technical_points += 5
    elif 0 < change_pct < 1:
        technical_points += 3
    elif change_pct > 5:  # Too hot
        technical_points -= 5
    
    scores["technical_score"] = min(20, max(0, technical_points))
    
    # Simulated catalyst score (random for demo)
    # In real system, this would check for IR releases
    np.random.seed(hash(stock_data["symbol"]) % 1000)
    if np.random.random() > 0.7:  # 30% chance of catalyst
        scores["catalyst_score"] = np.random.randint(20, 50)
    
    # Simulated sentiment score
    # In real system, this would analyze SNS/boards
    if volume_ratio > 1.5:  # High volume as proxy for attention
        scores["sentiment_score"] = min(30, int(volume_ratio * 10))
    
    scores["total_score"] = (
        scores["catalyst_score"] + 
        scores["sentiment_score"] + 
        scores["technical_score"]
    )
    
    return scores


def should_buy(stock_data: Dict[str, Any], scores: Dict[str, int]) -> tuple[bool, str]:
    """Determine if stock should be bought."""
    # Check buy threshold (80 points)
    if scores["total_score"] < 80:
        return False, f"スコア不足 ({scores['total_score']}/100)"
    
    # Filter conditions - skip if too hot
    rsi = stock_data.get("rsi", 50)
    ma_deviation = stock_data.get("ma_deviation", 0)
    
    if rsi > 75:
        return False, f"RSI過熱 ({rsi:.1f})"
    
    if ma_deviation > 25:
        return False, f"移動平均乖離過大 ({ma_deviation:.1f}%)"
    
    return True, "条件クリア"


def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print("🚀 AIデイトレードシステム - 本日の銘柄分析（簡易版）")
    print("="*80)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    try:
        # Load watchlist
        print("📌 ウォッチリストを読み込み中...")
        loader = TargetWatchlistLoader()
        companies = loader.load_watchlist()
        print(f"✅ {len(companies)}銘柄を読み込みました\n")
        
        # Initialize risk model
        risk_model = RiskModel()
        if risk_model.is_trained:
            print("✅ リスクモデル（NN）: ロード完了\n")
        else:
            print("⚠️ リスクモデル（NN）: デフォルト値を使用\n")
        
        # Analyze each company
        print("🔍 銘柄分析を開始...")
        print("-" * 40)
        
        results = []
        for i, company in enumerate(companies[:10], 1):  # Limit to 10 for speed
            symbol = company.get("symbol", "")
            # Handle both 'name' and 'company_name' keys
            name = company.get("name") or company.get("company_name") or "不明"
            
            print(f"\n[{i}/{min(10, len(companies))}] {symbol}: {name}")
            
            # Get stock data
            stock_data = get_stock_data(symbol)
            if not stock_data:
                print(f"  ❌ データ取得失敗")
                continue
            
            # Calculate scores
            scores = calculate_scores(stock_data)
            
            # Determine buy decision
            should_buy_flag, reason = should_buy(stock_data, scores)
            
            # Get risk assessment
            risk_assessment = None
            if should_buy_flag and risk_model.is_trained:
                try:
                    indicators = {
                        "historical_volatility": stock_data.get("historical_volatility", 0.2),
                        "atr": stock_data.get("atr", 2.0),
                        "rsi": stock_data.get("rsi", 50),
                        "volume_ratio": stock_data.get("volume_ratio", 1.0),
                        "ma_deviation": stock_data.get("ma_deviation", 0) / 100,  # Convert to decimal
                        "beta": 1.0,  # Default
                        "price_momentum": stock_data.get("change_percent", 0) / 100  # Convert to decimal
                    }
                    stop_loss_pct = risk_model.predict(indicators)
                    risk_assessment = {
                        "stop_loss_percent": stop_loss_pct,
                        "risk_level": "low" if stop_loss_pct < 0.05 else "medium" if stop_loss_pct < 0.10 else "high"
                    }
                except Exception as e:
                    print(f"  ⚠️ リスク評価エラー: {e}")
            
            # Store result
            result = {
                "symbol": symbol,
                "name": name,
                "execute": should_buy_flag,
                "reason": reason,
                "scores": scores,
                "stock_data": stock_data,
                "risk_assessment": risk_assessment
            }
            results.append(result)
            
            # Print summary
            print(f"  現在価格: ¥{stock_data['current_price']:,.0f}")
            print(f"  前日比: {stock_data['change_percent']:.2f}%")
            print(f"  出来高比率: {stock_data['volume_ratio']:.2f}x")
            print(f"  RSI: {stock_data['rsi']:.1f}")
            print(f"  スコア: {scores['total_score']}/100 (C:{scores['catalyst_score']} S:{scores['sentiment_score']} T:{scores['technical_score']})")
            
            if should_buy_flag:
                print(f"  ✅ **買い推奨** - {reason}")
                if risk_assessment:
                    print(f"  損切り幅: {risk_assessment['stop_loss_percent']*100:.2f}%")
            else:
                print(f"  ❌ 見送り - {reason}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Summary
        print("\n" + "="*80)
        print("📊 分析結果サマリー")
        print("="*80)
        
        buy_candidates = [r for r in results if r["execute"]]
        skip_candidates = [r for r in results if not r["execute"]]
        
        print(f"分析銘柄数: {len(results)}")
        print(f"✅ 買い推奨: {len(buy_candidates)}銘柄")
        print(f"❌ 見送り: {len(skip_candidates)}銘柄")
        
        if buy_candidates:
            print("\n🎯 本日の買い推奨銘柄:")
            for i, candidate in enumerate(buy_candidates, 1):
                print(f"  {i}. {candidate['symbol']}: {candidate['name']}")
                print(f"     価格: ¥{candidate['stock_data']['current_price']:,.0f}")
                print(f"     スコア: {candidate['scores']['total_score']}点")
                if candidate.get("risk_assessment"):
                    print(f"     損切り: {candidate['risk_assessment']['stop_loss_percent']*100:.2f}%")
        else:
            print("\n📝 本日は買い推奨銘柄がありません")
            print("（全銘柄が80点未満またはフィルター条件に該当）")
        
        # Save results
        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"analysis_{timestamp}.json"
        
        # Clean for JSON serialization
        clean_results = []
        for r in results:
            clean = r.copy()
            if "stock_data" in clean:
                clean["stock_data"] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                      for k, v in clean["stock_data"].items()}
            clean_results.append(clean)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📁 結果を保存: {filepath}")
        
        print("\n" + "="*80)
        print("⚠️ 免責事項")
        print("="*80)
        print("本分析は参考情報です。投資は自己責任で行ってください。")
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())