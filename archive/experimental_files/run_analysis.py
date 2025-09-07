#!/usr/bin/env python
"""今日の推奨銘柄を分析・出力するスクリプト"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
from typing import Dict, List, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.system_core.target_selector import TargetSelector
from src.analysis_engine.risk_model import RiskModel
from src.execution_manager.order_manager import OrderManager


def print_header():
    """Print analysis header."""
    print("\n" + "="*80)
    print("🚀 AIデイトレードシステム - 本日の銘柄分析")
    print("="*80)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def print_company_analysis(company: Dict[str, Any]):
    """Print individual company analysis."""
    symbol = company.get("symbol", "Unknown")
    name = company.get("name", "Unknown")
    scores = company.get("scores", {})
    
    print(f"\n📊 {symbol} - {name}")
    print("-" * 40)
    
    # スコア詳細
    print("スコア内訳:")
    print(f"  カタリスト重要度: {scores.get('catalyst_score', 0)}/50点")
    print(f"  センチメント: {scores.get('sentiment_score', 0)}/30点")
    print(f"  テクニカル: {scores.get('technical_score', 0)}/20点")
    print(f"  合計スコア: {scores.get('total_score', 0)}/100点")
    
    # 株価情報
    if "price_data" in company:
        price_data = company["price_data"]
        print(f"\n株価情報:")
        print(f"  現在株価: ¥{price_data.get('current_price', 0):,.0f}")
        print(f"  前日比: {price_data.get('change_percent', 0):.2f}%")
        print(f"  出来高: {price_data.get('volume', 0):,}")
    
    # テクニカル指標
    if "technical_indicators" in company:
        tech = company["technical_indicators"]
        print(f"\nテクニカル指標:")
        print(f"  RSI(14): {tech.get('rsi', 0):.1f}")
        print(f"  移動平均乖離率: {tech.get('ma_deviation', 0):.2f}%")
        print(f"  ATR: {tech.get('atr', 0):.2f}")
    
    # 判定
    execute = company.get("execute", False)
    if execute:
        print(f"\n✅ 判定: **買い推奨**")
        
        # リスク情報
        if "risk_assessment" in company:
            risk = company["risk_assessment"]
            print(f"\nリスク評価:")
            print(f"  推奨損切り幅: {risk.get('stop_loss_percent', 0.08)*100:.2f}%")
            print(f"  リスクレベル: {risk.get('risk_level', 'medium')}")
            
            # ポジションサイジング
            capital = 1000000  # 仮の資金100万円
            entry_price = price_data.get('current_price', 100)
            if entry_price > 0:
                position_calc = calculate_position_size(
                    capital, entry_price, 
                    risk.get('stop_loss_percent', 0.08)
                )
                print(f"\nポジションサイジング (資金¥{capital:,}の場合):")
                print(f"  推奨株数: {position_calc['shares']}株")
                print(f"  必要資金: ¥{position_calc['required_capital']:,.0f}")
                print(f"  最大損失額: ¥{position_calc['max_loss']:,.0f}")
                print(f"  損切り価格: ¥{position_calc['stop_loss_price']:,.0f}")
    else:
        reason = company.get("skip_reason", "スコア不足")
        print(f"\n❌ 判定: 見送り（理由: {reason}）")


def calculate_position_size(capital: float, entry_price: float, 
                           stop_loss_percent: float) -> Dict:
    """Calculate position sizing."""
    risk_per_trade = 0.01  # 1% risk per trade
    max_loss = capital * risk_per_trade
    stop_loss_price = entry_price * (1 - stop_loss_percent)
    risk_per_share = entry_price - stop_loss_price
    
    if risk_per_share <= 0:
        return {
            "shares": 0,
            "required_capital": 0,
            "max_loss": 0,
            "stop_loss_price": 0
        }
    
    shares = int(max_loss / risk_per_share)
    # Round to unit lot (100 shares)
    shares = (shares // 100) * 100
    if shares == 0 and max_loss / risk_per_share > 0:
        shares = 100
    
    return {
        "shares": shares,
        "required_capital": shares * entry_price,
        "max_loss": shares * risk_per_share,
        "stop_loss_price": stop_loss_price
    }


def save_results(results: List[Dict[str, Any]], filename: str = None):
    """Save analysis results to file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
    
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    
    # Convert any non-serializable objects
    clean_results = []
    for r in results:
        clean_result = {}
        for key, value in r.items():
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                clean_result[key] = value
            else:
                clean_result[key] = str(value)
        clean_results.append(clean_result)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 分析結果を保存しました: {filepath}")


def main():
    """Main analysis function."""
    print_header()
    
    try:
        # Initialize components
        print("📌 システムを初期化中...")
        selector = TargetSelector()
        risk_model = RiskModel()
        
        # Check if model is loaded
        if risk_model.is_trained:
            print("✅ リスクモデル（NN）: ロード完了")
        else:
            print("⚠️ リスクモデル（NN）: デフォルト値を使用")
        
        # Run analysis
        print("\n🔍 銘柄分析を開始...")
        print("-" * 40)
        
        # Analyze top targets (max 10 for display)
        results = selector.select_top_targets(max_targets=10)
        
        if not results:
            print("⚠️ 分析対象の銘柄が見つかりませんでした。")
            print("ターゲット企業.xlsx ファイルを確認してください。")
            return
        
        # Separate buy candidates and others
        buy_candidates = [r for r in results if r.get("execute", False)]
        skip_candidates = [r for r in results if not r.get("execute", False)]
        
        # Print summary
        print(f"\n📊 分析完了: {len(results)}銘柄を分析")
        print(f"  ✅ 買い推奨: {len(buy_candidates)}銘柄")
        print(f"  ❌ 見送り: {len(skip_candidates)}銘柄")
        
        # Print buy recommendations first
        if buy_candidates:
            print("\n" + "="*80)
            print("🎯 本日の買い推奨銘柄")
            print("="*80)
            
            for i, company in enumerate(buy_candidates, 1):
                print(f"\n【推奨 #{i}】")
                print_company_analysis(company)
        else:
            print("\n" + "="*80)
            print("📝 本日は買い推奨銘柄がありません")
            print("="*80)
            print("理由: 全銘柄が80点未満またはフィルター条件に該当")
        
        # Print top scoring non-buy candidates
        if skip_candidates:
            print("\n" + "="*80)
            print("📋 見送り銘柄（上位3銘柄）")
            print("="*80)
            
            # Sort by score and show top 3
            skip_candidates.sort(
                key=lambda x: x.get("scores", {}).get("total_score", 0), 
                reverse=True
            )
            
            for company in skip_candidates[:3]:
                print_company_analysis(company)
        
        # Save results
        save_results(results)
        
        # Print footer
        print("\n" + "="*80)
        print("📈 分析完了")
        print("="*80)
        
        # Statistics
        if results:
            avg_score = sum(r.get("scores", {}).get("total_score", 0) 
                          for r in results) / len(results)
            print(f"平均スコア: {avg_score:.1f}点")
            
            if buy_candidates:
                print(f"最高スコア: {buy_candidates[0].get('scores', {}).get('total_score', 0)}点")
                print(f"推奨銘柄: {', '.join([c.get('symbol', '') for c in buy_candidates])}")
        
        print(f"\n実行時刻: {datetime.now().strftime('%H:%M:%S')}")
        print("次回実行: 翌営業日 16:15（定時実行予定）")
        
        # Risk disclaimer
        print("\n" + "="*80)
        print("⚠️ 免責事項")
        print("="*80)
        print("本分析は参考情報であり、投資判断は自己責任で行ってください。")
        print("システムは開発中のため、実際の取引前に十分な検証が必要です。")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())