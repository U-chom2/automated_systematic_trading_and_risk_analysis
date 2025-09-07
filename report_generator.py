"""
Report Generator Module
レポート生成モジュール - 投資分析結果のレポートを生成
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

from config import config
from investment_limiter import LimitedInvestment
from investment_scorer import ScoringResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """レポート生成クラス"""
    
    def __init__(self):
        self.config = config.system
        self.limits = config.investment_limits
        self.thresholds = config.investment_thresholds
    
    def generate_console_report(
        self, 
        analysis_results: List[Dict[str, any]],
        allocations: Dict[str, LimitedInvestment] = None,
        max_investment_per_stock: float = None
    ) -> None:
        """コンソール向けレポートを生成"""
        
        if max_investment_per_stock is None:
            max_investment_per_stock = self.limits.max_investment_per_stock
        
        df = pd.DataFrame(analysis_results)
        if df.empty:
            logger.warning("No analysis results to report")
            return
        
        # ヘッダー
        self._print_report_header(len(df), max_investment_per_stock)
        
        # 買い推奨銘柄
        self._print_buy_recommendations(df, allocations)
        
        # ホールド推奨銘柄
        self._print_hold_recommendations(df)
        
        # 売り/回避推奨銘柄
        self._print_sell_recommendations(df)
        
        # 統計サマリー
        self._print_statistics_summary(df, allocations)
        
        # トップピック
        self._print_top_pick(df, allocations, max_investment_per_stock)
        
        # 免責事項
        self._print_disclaimer()
    
    def _print_report_header(self, company_count: int, max_investment: float):
        """レポートヘッダーを出力"""
        print("\n" + "=" * 80)
        print("💰 総合投資推奨レポート")
        print("=" * 80)
        print(f"分析日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
        print(f"分析銘柄数: {company_count}社")
        print(f"🔒 投資制限: 1株当たり最大¥{max_investment:,.0f}の投資リミッター設定")
    
    def _print_buy_recommendations(self, df: pd.DataFrame, allocations: Dict[str, LimitedInvestment]):
        """買い推奨銘柄を出力"""
        buy_stocks = df[df['投資スコア'] >= self.thresholds.buy_threshold].copy()
        
        if buy_stocks.empty:
            print("\n🟢 【買い推奨銘柄】: なし")
            return
        
        print("\n🟢 【買い推奨銘柄ランキング】")
        print("-" * 80)
        
        total_limited_investment = 0
        
        for idx, (_, stock) in enumerate(buy_stocks.iterrows(), 1):
            symbol = stock['シンボル']
            allocation = allocations.get(symbol) if allocations else None
            
            print(f"\n{idx}位: {stock['企業名']} ({symbol})")
            print(f"   📊 投資スコア: {stock['投資スコア']:.0f}点")
            print(f"   💰 現在株価: ¥{stock['現在株価']:.0f}")
            print(f"   📈 投資判断: {stock['投資判断']}")
            
            # 投資額情報
            if allocation:
                if allocation.is_limited:
                    print(f"   🔒 投資制限適用: ¥{allocation.original_amount:,.0f} → ¥{allocation.limited_amount:,.0f}")
                    print(f"   🎯 制限後投資額: ¥{allocation.limited_amount:,.0f} ({allocation.shares:,}株) ⚠️制限適用")
                else:
                    print(f"   🎯 推奨投資額: ¥{allocation.limited_amount:,.0f} ({allocation.shares:,}株)")
                
                total_limited_investment += allocation.limited_amount
            else:
                # フォールバック計算
                original_amount = self.limits.base_investment_amount * abs(stock['ポジションサイズ'])
                max_shares = int(self.limits.max_investment_per_stock / stock['現在株価'])
                original_shares = int(original_amount / stock['現在株価'])
                final_shares = min(max_shares, original_shares)
                final_amount = final_shares * stock['現在株価']
                
                print(f"   🎯 推奨投資額: ¥{final_amount:,.0f} ({final_shares:,}株)")
                total_limited_investment += final_amount
            
            print(f"   📈 目標利益: +{stock['目標利益率']:.0f}% (¥{stock['現在株価'] * (1 + stock['目標利益率']/100):.0f})")
            print(f"   🛑 損切ライン: {stock['損切りライン']:.0f}% (¥{stock['現在株価'] * (1 + stock['損切りライン']/100):.0f})")
            print(f"   ⏰ 推奨保有期間: {stock['推奨保有期間']}")
            print(f"   🔍 テクニカル: {stock['テクニカルシグナル']}")
            
            # パフォーマンス表示
            if all(col in stock.index for col in ['1日変化率', '5日変化率', '25日変化率']):
                print(f"   📊 パフォーマンス: 1日{stock['1日変化率']:.1f}% | 5日{stock['5日変化率']:.1f}% | 25日{stock['25日変化率']:.1f}%")
        
        print(f"\n💰 【投資制限後の合計推奨投資額: ¥{total_limited_investment:,.0f}】")
    
    def _print_hold_recommendations(self, df: pd.DataFrame):
        """ホールド推奨銘柄を出力"""
        hold_stocks = df[
            (df['投資スコア'] >= self.thresholds.hold_threshold) & 
            (df['投資スコア'] < self.thresholds.buy_threshold)
        ]
        
        if hold_stocks.empty:
            return
        
        print("\n🟡 【ホールド推奨銘柄】")
        print("-" * 80)
        for _, stock in hold_stocks.iterrows():
            print(f"• {stock['企業名']} ({stock['シンボル']}) - ¥{stock['現在株価']:.0f} (スコア: {stock['投資スコア']:.0f}点)")
    
    def _print_sell_recommendations(self, df: pd.DataFrame):
        """売り/回避推奨銘柄を出力"""
        sell_stocks = df[df['投資スコア'] < self.thresholds.hold_threshold]
        
        if sell_stocks.empty:
            return
        
        print("\n🔴 【売り/回避推奨銘柄】")
        print("-" * 80)
        for _, stock in sell_stocks.head(3).iterrows():
            print(f"• {stock['企業名']} ({stock['シンボル']}) - ¥{stock['現在株価']:.0f} (スコア: {stock['投資スコア']:.0f}点)")
            if 'テクニカルシグナル' in stock.index:
                print(f"  理由: {stock['投資判断']} - {stock['テクニカルシグナル']}")
    
    def _print_statistics_summary(self, df: pd.DataFrame, allocations: Dict[str, LimitedInvestment]):
        """統計サマリーを出力"""
        buy_count = len(df[df['投資スコア'] >= self.thresholds.buy_threshold])
        hold_count = len(df[
            (df['投資スコア'] >= self.thresholds.hold_threshold) & 
            (df['投資スコア'] < self.thresholds.buy_threshold)
        ])
        sell_count = len(df[df['投資スコア'] < self.thresholds.hold_threshold])
        
        print("\n📊 【分析統計】")
        print("-" * 80)
        print(f"平均投資スコア: {df['投資スコア'].mean():.1f}点")
        print(f"買い推奨: {buy_count}銘柄 ({buy_count/len(df)*100:.1f}%)")
        print(f"ホールド推奨: {hold_count}銘柄 ({hold_count/len(df)*100:.1f}%)")
        print(f"売り/回避推奨: {sell_count}銘柄 ({sell_count/len(df)*100:.1f}%)")
    
    def _print_top_pick(self, df: pd.DataFrame, allocations: Dict[str, LimitedInvestment], max_investment: float):
        """最強推奨銘柄を出力"""
        buy_stocks = df[df['投資スコア'] >= self.thresholds.buy_threshold]
        
        if buy_stocks.empty:
            return
        
        top_pick = buy_stocks.iloc[0]
        symbol = top_pick['シンボル']
        allocation = allocations.get(symbol) if allocations else None
        
        print("\n🌟 【最強推奨銘柄】")
        print("-" * 80)
        print(f"企業: {top_pick['企業名']} ({symbol})")
        print(f"投資スコア: {top_pick['投資スコア']:.0f}点")
        print(f"現在株価: ¥{top_pick['現在株価']:.0f}")
        
        if allocation:
            if allocation.is_limited:
                print(f"制限前投資額: ¥{allocation.original_amount:,.0f}")
                print(f"🔒制限後投資額: ¥{allocation.limited_amount:,.0f} ({allocation.shares:,}株)")
            else:
                print(f"推奨投資額: ¥{allocation.limited_amount:,.0f} ({allocation.shares:,}株)")
        else:
            # フォールバック
            original_amount = self.limits.base_investment_amount * abs(top_pick['ポジションサイズ'])
            max_shares = int(max_investment / top_pick['現在株価'])
            original_shares = int(original_amount / top_pick['現在株価'])
            final_shares = min(max_shares, original_shares)
            final_amount = final_shares * top_pick['現在株価']
            print(f"推奨投資額: ¥{final_amount:,.0f} ({final_shares:,}株)")
        
        print(f"目標株価: ¥{top_pick['現在株価'] * (1 + top_pick['目標利益率']/100):.0f} (+{top_pick['目標利益率']:.0f}%)")
        print(f"保有期間: {top_pick['推奨保有期間']}")
    
    def _print_disclaimer(self):
        """免責事項を出力"""
        print("\n" + "⚠️" * 3 + " 重要な免責事項 " + "⚠️" * 3)
        print("-" * 80)
        print("• この分析はAIによる参考情報であり、投資助言ではありません")
        print("• 投資判断は必ずご自身の責任で行ってください")
        print("• 株式投資にはリスクが伴います。余裕資金で行ってください")
        print("• 損切りラインは必ず設定し、リスク管理を徹底してください")
        print("=" * 80)
    
    def save_csv_report(self, analysis_results: List[Dict[str, any]], output_path: Path = None) -> Path:
        """CSV形式でレポートを保存"""
        if output_path is None:
            output_path = config.output_csv_path
        
        try:
            df = pd.DataFrame(analysis_results)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"CSV report saved to {output_path}")
            print(f"\n💾 分析結果を {output_path} に保存しました")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save CSV report: {e}")
            raise
    
    def generate_markdown_report(self, analysis_results: List[Dict[str, any]]) -> str:
        """Markdown形式のレポートを生成"""
        lines = []
        lines.append("# 投資分析レポート")
        lines.append(f"\n**分析日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
        lines.append(f"**分析銘柄数**: {len(analysis_results)}社")
        
        # 結果データフレーム
        df = pd.DataFrame(analysis_results)
        
        # 買い推奨
        buy_stocks = df[df['投資スコア'] >= self.thresholds.buy_threshold]
        if not buy_stocks.empty:
            lines.append("\n## 🟢 買い推奨銘柄")
            for _, stock in buy_stocks.iterrows():
                lines.append(f"- **{stock['企業名']}** ({stock['シンボル']}) - ¥{stock['現在株価']:.0f}")
                lines.append(f"  - スコア: {stock['投資スコア']:.0f}点")
                lines.append(f"  - 判断: {stock['投資判断']}")
        
        # ホールド推奨
        hold_stocks = df[
            (df['投資スコア'] >= self.thresholds.hold_threshold) & 
            (df['投資スコア'] < self.thresholds.buy_threshold)
        ]
        if not hold_stocks.empty:
            lines.append("\n## 🟡 ホールド推奨銘柄")
            for _, stock in hold_stocks.iterrows():
                lines.append(f"- {stock['企業名']} ({stock['シンボル']}) - ¥{stock['現在株価']:.0f}")
        
        lines.append("\n## 免責事項")
        lines.append("- この分析はAIによる参考情報であり、投資助言ではありません")
        lines.append("- 投資判断は必ずご自身の責任で行ってください")
        
        return "\n".join(lines)