"""
取引シミュレーションシステム
推奨銘柄の自動売買とパフォーマンス追跡
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from vibelogger import create_file_logger
from portfolio_manager import PortfolioManager
from investment_analyzer import InvestmentAnalyzer
from data_fetcher import DataFetcher
from config import Config, TradingMode

logger = create_file_logger(__name__)

@dataclass
class SimulationConfig:
    """シミュレーション設定"""
    initial_capital: float = 100000.0  # 初期資金
    max_positions: int = 5  # 最大同時保有銘柄数
    max_investment_per_stock: float = 30000.0  # 1銘柄あたりの最大投資額
    commission_rate: float = 0.0025  # 手数料率（0.25%）
    trading_mode: TradingMode = TradingMode.DAY_TRADING
    target_profit_pct: float = 2.0  # 目標利益率
    stop_loss_pct: float = -1.5  # 損切りライン
    max_holding_days: int = 3  # 最大保有日数
    
@dataclass
class RecommendedStock:
    """推奨銘柄"""
    symbol: str
    company_name: str
    score: float
    recommendation: str
    current_price: float
    shares: int
    investment_amount: float
    target_price: float
    stop_loss_price: float
    holding_period: str


class TradingSimulator:
    """取引シミュレータークラス"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        初期化
        
        Args:
            config: シミュレーション設定
        """
        self.config = config or SimulationConfig()
        self.portfolio = PortfolioManager(
            initial_cash=self.config.initial_capital,
            commission_rate=self.config.commission_rate
        )
        
        # 分析システム初期化
        trading_config = Config(self.config.trading_mode)
        self.analyzer = InvestmentAnalyzer(trading_config, self.config.max_investment_per_stock, use_ppo=True)
        self.data_fetcher = DataFetcher()
        
        # シミュレーション状態
        self.simulation_date = datetime.now()
        self.recommendations_history = []
        self.daily_performance = []
        
        # 分析システム情報取得
        trading_info = self.analyzer.get_trading_info()
        
        logger.info(f"Trading Simulator initialized")
        logger.info(f"Initial Capital: ¥{self.config.initial_capital:,.0f}")
        logger.info(f"Max Positions: {self.config.max_positions}")
        logger.info(f"Trading Mode: {self.config.trading_mode.value}")
        logger.info(f"Scoring Method: {trading_info['scoring_method']}")
        
        # PPOモデル情報をログ出力
        if 'ppo_model_info' in trading_info:
            ppo_info = trading_info['ppo_model_info']
            logger.info(f"PPO Model: {ppo_info['model_name']} ({ppo_info['device']})")
        
    def get_todays_recommendations(self) -> List[RecommendedStock]:
        """
        今日の推奨銘柄を取得
        
        Returns:
            推奨銘柄リスト
        """
        logger.info("=" * 80)
        logger.info(f"📊 {self.simulation_date.strftime('%Y-%m-%d')} の推奨銘柄分析")
        trading_info = self.analyzer.get_trading_info()
        logger.info(f"🤖 分析手法: {trading_info['scoring_method']}")
        logger.info("=" * 80)
        
        # 全銘柄分析実行
        results_df = self.analyzer.analyze_all_companies()
        
        # 買い推奨銘柄を抽出（スコア75点以上の強い買い推奨のみ）
        buy_recommendations = results_df[
            (results_df['投資判断'] == '強い買い') & 
            (results_df['投資スコア'] >= 75)
        ].sort_values('投資スコア', ascending=False)
        
        recommendations = []
        for _, row in buy_recommendations.head(10).iterrows():
            # 投資額と株数計算
            current_price = row['現在株価']
            
            # 最大投資額と現在のポートフォリオを考慮した投資額決定
            max_invest = min(
                self.config.max_investment_per_stock,
                self.portfolio.cash * 0.3  # 現金の30%まで
            )
            
            # 株数計算（最低1株、最大投資額以内）
            if current_price > max_invest:
                shares = 1  # 高額株でも最低1株は購入
            else:
                shares = int(max_invest / current_price)
                
            if shares <= 0:
                continue
                
            rec = RecommendedStock(
                symbol=row['シンボル'],
                company_name=row['企業名'],
                score=row['投資スコア'],
                recommendation=row['投資判断'],
                current_price=row['現在株価'],
                shares=shares,
                investment_amount=shares * row['現在株価'],
                target_price=row['現在株価'] * (1 + self.config.target_profit_pct / 100),
                stop_loss_price=row['現在株価'] * (1 + self.config.stop_loss_pct / 100),
                holding_period=row.get('推奨保有期間', '1-3日')
            )
            recommendations.append(rec)
            
        logger.info(f"Found {len(recommendations)} buy recommendations")
        return recommendations
        
    def execute_daily_trades(self, recommendations: List[RecommendedStock]) -> Dict:
        """
        日次取引実行
        
        Args:
            recommendations: 推奨銘柄リスト
            
        Returns:
            実行結果
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"💹 取引実行 - {self.simulation_date.strftime('%Y-%m-%d')}")
        logger.info("=" * 80)
        
        executed_buys = []
        skipped_buys = []
        
        # 現在のポジション数確認
        current_positions = len(self.portfolio.positions)
        available_slots = self.config.max_positions - current_positions
        
        logger.info(f"Current positions: {current_positions}/{self.config.max_positions}")
        logger.info(f"Available slots: {available_slots}")
        
        # 推奨銘柄の購入実行
        for rec in recommendations[:available_slots]:
            # すでに保有している銘柄はスキップ
            if rec.symbol in self.portfolio.positions:
                logger.info(f"⏭️ Skipping {rec.symbol} - already in portfolio")
                skipped_buys.append(rec)
                continue
                
            # 購入実行
            success = self.portfolio.buy_stock(
                symbol=rec.symbol,
                company_name=rec.company_name,
                price=rec.current_price,
                shares=rec.shares,
                target_pct=self.config.target_profit_pct,
                stop_loss_pct=self.config.stop_loss_pct,
                max_holding_days=self.config.max_holding_days
            )
            
            if success:
                executed_buys.append(rec)
            else:
                skipped_buys.append(rec)
                
        return {
            "date": self.simulation_date.strftime("%Y-%m-%d"),
            "executed_buys": executed_buys,
            "skipped_buys": skipped_buys,
            "total_recommendations": len(recommendations)
        }
        
    def update_portfolio_prices(self) -> Dict:
        """
        ポートフォリオの価格更新と売却判定
        
        Returns:
            更新結果
        """
        if not self.portfolio.positions:
            logger.info("No active positions to update")
            return {"updated": 0, "sold": []}
            
        logger.info("\n" + "=" * 80)
        logger.info(f"📈 価格更新 - {self.simulation_date.strftime('%Y-%m-%d')}")
        logger.info("=" * 80)
        
        # 保有銘柄の最新価格を取得
        symbols = list(self.portfolio.positions.keys())
        price_data = {}
        
        for symbol in symbols:
            try:
                data = self.data_fetcher.fetch_stock_data(symbol)
                if data is not None and not data.empty:
                    price_data[symbol] = data['Close'].iloc[-1]
                    logger.info(f"  {symbol}: ¥{price_data[symbol]:,.0f}")
            except Exception as e:
                logger.error(f"Failed to fetch price for {symbol}: {e}")
                
        # ポジション更新と自動売却
        sold_symbols = self.portfolio.update_positions(price_data)
        
        return {
            "date": self.simulation_date.strftime("%Y-%m-%d"),
            "updated": len(price_data),
            "sold": sold_symbols
        }
        
    def run_simulation(self, days: int = 1, auto_buy: bool = True) -> Dict:
        """
        シミュレーション実行
        
        Args:
            days: シミュレーション日数
            auto_buy: 自動購入を有効にするか
            
        Returns:
            シミュレーション結果
        """
        logger.info("\n" + "=" * 80)
        logger.info("🚀 シミュレーション開始")
        logger.info("=" * 80)
        logger.info(f"期間: {days}日")
        logger.info(f"自動購入: {'有効' if auto_buy else '無効'}")
        
        results = []
        
        for day in range(days):
            self.simulation_date = datetime.now() + timedelta(days=day)
            logger.info(f"\n📅 Day {day + 1}: {self.simulation_date.strftime('%Y-%m-%d')}")
            
            # 価格更新と売却判定
            if day > 0 or self.portfolio.positions:
                update_result = self.update_portfolio_prices()
            else:
                update_result = {"updated": 0, "sold": []}
                
            # 新規購入（月曜日〜金曜日の16時想定）
            if auto_buy and self.simulation_date.weekday() < 5:
                if len(self.portfolio.positions) < self.config.max_positions:
                    recommendations = self.get_todays_recommendations()
                    trade_result = self.execute_daily_trades(recommendations)
                else:
                    trade_result = {
                        "date": self.simulation_date.strftime("%Y-%m-%d"),
                        "executed_buys": [],
                        "skipped_buys": [],
                        "total_recommendations": 0
                    }
            else:
                trade_result = None
                
            # スナップショット保存
            self.portfolio.save_snapshot()
            
            # 日次結果
            daily_result = {
                "day": day + 1,
                "date": self.simulation_date.strftime("%Y-%m-%d"),
                "portfolio_value": self.portfolio.get_portfolio_value(),
                "positions": len(self.portfolio.positions),
                "update_result": update_result,
                "trade_result": trade_result
            }
            results.append(daily_result)
            
            # ポートフォリオ表示
            self.portfolio.display_portfolio()
            
        # 最終結果
        final_summary = self.portfolio.get_performance_summary()
        
        # データ保存
        self.portfolio.save_to_file()
        
        return {
            "days": days,
            "daily_results": results,
            "final_summary": final_summary
        }
        
    def run_live_simulation(self) -> Dict:
        """
        ライブシミュレーション（今日の推奨を実際に追跡）
        
        Returns:
            実行結果
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔴 ライブシミュレーション開始")
        logger.info("=" * 80)
        logger.info("今日の推奨銘柄を購入し、実際の価格変動を追跡します")
        
        # 今日の推奨取得
        recommendations = self.get_todays_recommendations()
        
        if not recommendations:
            logger.warning("No recommendations found for today")
            return {"status": "no_recommendations"}
            
        # 推奨銘柄の購入実行
        trade_result = self.execute_daily_trades(recommendations)
        
        # ポートフォリオ表示
        self.portfolio.display_portfolio()
        
        # データ保存
        self.portfolio.save_to_file()
        
        # 推奨銘柄の詳細表示
        print("\n" + "=" * 80)
        print("📋 本日の推奨銘柄詳細")
        print("=" * 80)
        
        for i, rec in enumerate(trade_result["executed_buys"], 1):
            print(f"\n{i}. {rec.company_name} ({rec.symbol})")
            print(f"   スコア: {rec.score}点")
            print(f"   購入価格: ¥{rec.current_price:,.0f}")
            print(f"   購入株数: {rec.shares}株")
            print(f"   投資額: ¥{rec.investment_amount:,.0f}")
            print(f"   目標価格: ¥{rec.target_price:,.0f} (+{self.config.target_profit_pct}%)")
            print(f"   損切り価格: ¥{rec.stop_loss_price:,.0f} ({self.config.stop_loss_pct}%)")
            print(f"   推奨保有期間: {rec.holding_period}")
            
        return {
            "status": "success",
            "date": self.simulation_date.strftime("%Y-%m-%d"),
            "recommendations": len(recommendations),
            "executed": len(trade_result["executed_buys"]),
            "portfolio_value": self.portfolio.get_portfolio_value()
        }
        
    def update_live_positions(self) -> Dict:
        """
        ライブポジションの更新（実際の価格で）
        
        Returns:
            更新結果
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 ポジション更新")
        logger.info("=" * 80)
        
        # 価格更新と売却判定
        update_result = self.update_portfolio_prices()
        
        # ポートフォリオ表示
        self.portfolio.display_portfolio()
        
        # データ保存
        self.portfolio.save_to_file()
        
        return update_result
        
    def get_report(self) -> str:
        """
        詳細レポート生成
        
        Returns:
            レポート文字列
        """
        summary = self.portfolio.get_performance_summary()
        
        report = []
        report.append("=" * 80)
        report.append("📊 シミュレーション詳細レポート")
        report.append("=" * 80)
        report.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 資産状況
        report.append("【資産状況】")
        report.append(f"初期資金: ¥{summary['initial_capital']:,.0f}")
        report.append(f"現在資産: ¥{summary['portfolio_value']:,.0f}")
        report.append(f"総損益: ¥{summary['total_pnl']:+,.0f} ({summary['total_return_pct']:+.2f}%)")
        report.append("")
        
        # 取引成績
        report.append("【取引成績】")
        report.append(f"総取引数: {summary['num_closed']}回")
        report.append(f"勝率: {summary['win_rate']:.1f}%")
        report.append(f"勝ち: {summary['winning_trades']}回")
        report.append(f"負け: {summary['losing_trades']}回")
        report.append("")
        
        # 損益詳細
        report.append("【損益詳細】")
        report.append(f"実現損益: ¥{summary['realized_pnl']:+,.0f}")
        report.append(f"含み損益: ¥{summary['unrealized_pnl']:+,.0f}")
        report.append(f"平均利益: ¥{summary['avg_win']:+,.0f}")
        report.append(f"平均損失: ¥{summary['avg_loss']:+,.0f}")
        report.append(f"最大利益: ¥{summary['largest_win']:+,.0f}")
        report.append(f"最大損失: ¥{summary['largest_loss']:+,.0f}")
        report.append("")
        
        # 現在のポジション
        if self.portfolio.positions:
            report.append("【現在のポジション】")
            for symbol, pos in self.portfolio.positions.items():
                report.append(f"{symbol} ({pos.company_name})")
                report.append(f"  保有: {pos.shares}株 @ ¥{pos.entry_price:,.0f}")
                report.append(f"  現在: ¥{pos.current_price:,.0f}")
                report.append(f"  含み損益: ¥{pos.unrealized_pnl:+,.0f} ({pos.unrealized_pnl_pct:+.1f}%)")
                report.append(f"  保有日数: {pos.holding_days}日")
            report.append("")
            
        # 終了した取引
        if self.portfolio.closed_positions:
            report.append("【終了した取引（最新5件）】")
            for pos in self.portfolio.closed_positions[-5:]:
                emoji = "🟢" if pos.realized_pnl > 0 else "🔴"
                report.append(f"{emoji} {pos.symbol} ({pos.company_name})")
                report.append(f"  期間: {pos.entry_date} → {pos.exit_date}")
                report.append(f"  売買: ¥{pos.entry_price:,.0f} → ¥{pos.exit_price:,.0f}")
                report.append(f"  損益: ¥{pos.realized_pnl:+,.0f} ({pos.realized_pnl_pct:+.1f}%)")
                report.append(f"  理由: {pos.status}")
            report.append("")
            
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """メイン実行関数"""
    import sys
    
    # コマンドライン引数解析
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"
    
    # シミュレーター初期化
    config = SimulationConfig(
        initial_capital=100000.0,  # 10万円
        max_positions=5,
        max_investment_per_stock=30000.0,
        trading_mode=TradingMode.DAY_TRADING
    )
    
    simulator = TradingSimulator(config)
    
    if mode == "live":
        # ライブシミュレーション（今日の推奨を購入）
        print("🔴 ライブシミュレーションモード")
        result = simulator.run_live_simulation()
        print(f"\n実行結果: {result['status']}")
        
    elif mode == "update":
        # ポジション更新
        print("📊 ポジション更新モード")
        result = simulator.update_live_positions()
        print(f"\n更新数: {result['updated']}, 売却数: {len(result['sold'])}")
        
    elif mode == "test":
        # テストシミュレーション（数日分）
        print("🧪 テストシミュレーションモード")
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        result = simulator.run_simulation(days=days, auto_buy=True)
        print(f"\n{days}日間のシミュレーション完了")
        
    elif mode == "report":
        # レポート生成
        print("📋 レポート生成モード")
        # 最新のデータを読み込み
        files = list(Path("simulation_data").glob("positions_*.json"))
        if files:
            latest = sorted(files)[-1]
            timestamp = latest.stem.replace("positions_", "")
            simulator.portfolio.load_from_file(timestamp)
            print(simulator.get_report())
        else:
            print("No simulation data found")
            
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python trading_simulator.py [live|update|test|report] [days]")
        

if __name__ == "__main__":
    main()