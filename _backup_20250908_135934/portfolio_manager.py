"""
ポートフォリオ管理システム
シミュレーション取引のポートフォリオを管理
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from vibelogger import create_file_logger

logger = create_file_logger(__name__)

@dataclass
class Position:
    """保有ポジション"""
    symbol: str
    company_name: str
    entry_date: str
    entry_price: float
    shares: int
    current_price: float
    target_price: float
    stop_loss_price: float
    holding_days: int
    max_holding_days: int
    unrealized_pnl: float
    unrealized_pnl_pct: float
    status: str  # active, target_reached, stop_loss, expired
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    
@dataclass
class Trade:
    """取引記録"""
    trade_id: str
    symbol: str
    company_name: str
    trade_type: str  # buy, sell
    trade_date: str
    price: float
    shares: int
    amount: float
    commission: float
    reason: str  # entry, target, stop_loss, expired
    
@dataclass
class PortfolioSnapshot:
    """ポートフォリオスナップショット"""
    date: str
    cash: float
    total_value: float
    position_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    num_positions: int
    winning_trades: int
    losing_trades: int
    win_rate: float


class PortfolioManager:
    """ポートフォリオ管理クラス"""
    
    def __init__(self, initial_cash: float = 100000.0, commission_rate: float = 0.0025):
        """
        初期化
        
        Args:
            initial_cash: 初期資金（デフォルト10万円）
            commission_rate: 手数料率（デフォルト0.25%）
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        
        # ポジション管理
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # 取引履歴
        self.trades: List[Trade] = []
        
        # スナップショット履歴
        self.snapshots: List[PortfolioSnapshot] = []
        
        # データ保存パス
        self.data_dir = Path("simulation_data")
        self.data_dir.mkdir(exist_ok=True)
        
        logger.info(f"Portfolio initialized with ¥{initial_cash:,.0f}")
        
    def buy_stock(self, symbol: str, company_name: str, price: float, 
                  shares: int, target_pct: float = 2.0, 
                  stop_loss_pct: float = -1.5, max_holding_days: int = 3) -> bool:
        """
        株式購入
        
        Args:
            symbol: 銘柄コード
            company_name: 企業名
            price: 購入価格
            shares: 株数
            target_pct: 目標利益率（%）
            stop_loss_pct: 損切りライン（%）
            max_holding_days: 最大保有日数
            
        Returns:
            購入成功/失敗
        """
        # 資金チェック
        amount = price * shares
        commission = amount * self.commission_rate
        total_cost = amount + commission
        
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for {symbol}: ¥{total_cost:,.0f} > ¥{self.cash:,.0f}")
            return False
            
        # ポジション作成
        position = Position(
            symbol=symbol,
            company_name=company_name,
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            entry_price=price,
            shares=shares,
            current_price=price,
            target_price=price * (1 + target_pct / 100),
            stop_loss_price=price * (1 + stop_loss_pct / 100),
            holding_days=0,
            max_holding_days=max_holding_days,
            unrealized_pnl=0,
            unrealized_pnl_pct=0,
            status="active"
        )
        
        # 取引記録
        trade = Trade(
            trade_id=f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            symbol=symbol,
            company_name=company_name,
            trade_type="buy",
            trade_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            price=price,
            shares=shares,
            amount=amount,
            commission=commission,
            reason="entry"
        )
        
        # 更新
        self.positions[symbol] = position
        self.trades.append(trade)
        self.cash -= total_cost
        
        logger.info(f"🛒 Bought {shares} shares of {symbol} @ ¥{price:,.0f}")
        logger.info(f"   Target: ¥{position.target_price:,.0f} (+{target_pct}%)")
        logger.info(f"   Stop Loss: ¥{position.stop_loss_price:,.0f} ({stop_loss_pct}%)")
        logger.info(f"   Remaining cash: ¥{self.cash:,.0f}")
        
        return True
        
    def sell_stock(self, symbol: str, price: float, reason: str = "manual") -> bool:
        """
        株式売却
        
        Args:
            symbol: 銘柄コード
            price: 売却価格
            reason: 売却理由（target, stop_loss, expired, manual）
            
        Returns:
            売却成功/失敗
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return False
            
        position = self.positions[symbol]
        
        # 売却処理
        amount = price * position.shares
        commission = amount * self.commission_rate
        net_amount = amount - commission
        
        # 実現損益計算
        realized_pnl = net_amount - (position.entry_price * position.shares * (1 + self.commission_rate))
        realized_pnl_pct = (realized_pnl / (position.entry_price * position.shares)) * 100
        
        # ポジション更新
        position.exit_date = datetime.now().strftime("%Y-%m-%d")
        position.exit_price = price
        position.realized_pnl = realized_pnl
        position.realized_pnl_pct = realized_pnl_pct
        position.status = reason
        
        # 取引記録
        trade = Trade(
            trade_id=f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            symbol=symbol,
            company_name=position.company_name,
            trade_type="sell",
            trade_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            price=price,
            shares=position.shares,
            amount=amount,
            commission=commission,
            reason=reason
        )
        
        # 更新
        self.closed_positions.append(position)
        del self.positions[symbol]
        self.trades.append(trade)
        self.cash += net_amount
        
        # ログ出力
        emoji = "🎯" if realized_pnl > 0 else "😢"
        logger.info(f"{emoji} Sold {position.shares} shares of {symbol} @ ¥{price:,.0f}")
        logger.info(f"   Reason: {reason}")
        logger.info(f"   P&L: ¥{realized_pnl:,.0f} ({realized_pnl_pct:+.1f}%)")
        logger.info(f"   Cash balance: ¥{self.cash:,.0f}")
        
        return True
        
    def update_positions(self, price_data: Dict[str, float]) -> List[str]:
        """
        ポジション更新と自動売却判定
        
        Args:
            price_data: 銘柄コード -> 現在価格のマッピング
            
        Returns:
            売却した銘柄リスト
        """
        sold_symbols = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in price_data:
                logger.warning(f"No price data for {symbol}")
                continue
                
            current_price = price_data[symbol]
            position.current_price = current_price
            position.holding_days += 1
            
            # 含み損益更新
            position.unrealized_pnl = (current_price - position.entry_price) * position.shares
            position.unrealized_pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            
            # 売却判定
            sell_reason = None
            
            if current_price >= position.target_price:
                sell_reason = "target_reached"
                logger.info(f"🎯 Target reached for {symbol}: ¥{current_price:,.0f} >= ¥{position.target_price:,.0f}")
            elif current_price <= position.stop_loss_price:
                sell_reason = "stop_loss"
                logger.info(f"🛑 Stop loss triggered for {symbol}: ¥{current_price:,.0f} <= ¥{position.stop_loss_price:,.0f}")
            elif position.holding_days >= position.max_holding_days:
                sell_reason = "expired"
                logger.info(f"⏰ Holding period expired for {symbol}: {position.holding_days} days")
                
            # 売却実行
            if sell_reason:
                if self.sell_stock(symbol, current_price, sell_reason):
                    sold_symbols.append(symbol)
                    
        return sold_symbols
        
    def get_portfolio_value(self) -> float:
        """ポートフォリオ総価値を取得"""
        position_value = sum(p.current_price * p.shares for p in self.positions.values())
        return self.cash + position_value
        
    def get_performance_summary(self) -> Dict:
        """パフォーマンスサマリーを取得"""
        # 実現損益
        realized_pnl = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl)
        
        # 含み損益
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        # 総損益
        total_pnl = realized_pnl + unrealized_pnl
        
        # 勝率計算
        winning_trades = [p for p in self.closed_positions if p.realized_pnl and p.realized_pnl > 0]
        losing_trades = [p for p in self.closed_positions if p.realized_pnl and p.realized_pnl <= 0]
        total_trades = len(winning_trades) + len(losing_trades)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        # 平均損益
        avg_win = np.mean([p.realized_pnl for p in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([p.realized_pnl for p in losing_trades]) if losing_trades else 0
        
        # リターン率
        total_return_pct = (total_pnl / self.initial_cash) * 100
        
        return {
            "portfolio_value": self.get_portfolio_value(),
            "cash": self.cash,
            "position_value": self.get_portfolio_value() - self.cash,
            "initial_capital": self.initial_cash,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "num_positions": len(self.positions),
            "num_closed": len(self.closed_positions),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": max([p.realized_pnl for p in winning_trades], default=0),
            "largest_loss": min([p.realized_pnl for p in losing_trades], default=0)
        }
        
    def save_snapshot(self):
        """現在のスナップショットを保存"""
        summary = self.get_performance_summary()
        
        snapshot = PortfolioSnapshot(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cash=summary["cash"],
            total_value=summary["portfolio_value"],
            position_value=summary["position_value"],
            unrealized_pnl=summary["unrealized_pnl"],
            realized_pnl=summary["realized_pnl"],
            total_pnl=summary["total_pnl"],
            num_positions=summary["num_positions"],
            winning_trades=summary["winning_trades"],
            losing_trades=summary["losing_trades"],
            win_rate=summary["win_rate"]
        )
        
        self.snapshots.append(snapshot)
        
    def save_to_file(self):
        """データをファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ポジションデータ
        positions_data = {
            "active": [asdict(p) for p in self.positions.values()],
            "closed": [asdict(p) for p in self.closed_positions]
        }
        
        with open(self.data_dir / f"positions_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(positions_data, f, ensure_ascii=False, indent=2)
            
        # 取引履歴
        trades_data = [asdict(t) for t in self.trades]
        with open(self.data_dir / f"trades_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(trades_data, f, ensure_ascii=False, indent=2)
            
        # スナップショット
        snapshots_data = [asdict(s) for s in self.snapshots]
        with open(self.data_dir / f"snapshots_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(snapshots_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Portfolio data saved to {self.data_dir}")
        
    def load_from_file(self, timestamp: str):
        """ファイルからデータを読み込み"""
        # ポジションデータ
        with open(self.data_dir / f"positions_{timestamp}.json", "r", encoding="utf-8") as f:
            positions_data = json.load(f)
            
        self.positions = {p["symbol"]: Position(**p) for p in positions_data["active"]}
        self.closed_positions = [Position(**p) for p in positions_data["closed"]]
        
        # 取引履歴
        with open(self.data_dir / f"trades_{timestamp}.json", "r", encoding="utf-8") as f:
            trades_data = json.load(f)
        self.trades = [Trade(**t) for t in trades_data]
        
        # スナップショット
        with open(self.data_dir / f"snapshots_{timestamp}.json", "r", encoding="utf-8") as f:
            snapshots_data = json.load(f)
        self.snapshots = [PortfolioSnapshot(**s) for s in snapshots_data]
        
        logger.info(f"Portfolio data loaded from {self.data_dir}")
        
    def display_portfolio(self):
        """ポートフォリオの表示"""
        print("\n" + "=" * 80)
        print("📊 ポートフォリオ状況")
        print("=" * 80)
        
        summary = self.get_performance_summary()
        
        # サマリー表示
        print(f"💰 総資産: ¥{summary['portfolio_value']:,.0f}")
        print(f"   現金: ¥{summary['cash']:,.0f}")
        print(f"   ポジション価値: ¥{summary['position_value']:,.0f}")
        print(f"📈 総損益: ¥{summary['total_pnl']:+,.0f} ({summary['total_return_pct']:+.2f}%)")
        print(f"   実現損益: ¥{summary['realized_pnl']:+,.0f}")
        print(f"   含み損益: ¥{summary['unrealized_pnl']:+,.0f}")
        
        # アクティブポジション
        if self.positions:
            print("\n🔄 アクティブポジション:")
            print("-" * 80)
            for symbol, pos in self.positions.items():
                status_emoji = "🟢" if pos.unrealized_pnl > 0 else "🔴"
                print(f"{status_emoji} {symbol} ({pos.company_name})")
                print(f"   購入: ¥{pos.entry_price:,.0f} × {pos.shares}株")
                print(f"   現在: ¥{pos.current_price:,.0f} ({pos.unrealized_pnl_pct:+.1f}%)")
                print(f"   含み損益: ¥{pos.unrealized_pnl:+,.0f}")
                print(f"   保有日数: {pos.holding_days}/{pos.max_holding_days}日")
                
        # 成績統計
        if summary['num_closed'] > 0:
            print("\n📊 取引成績:")
            print("-" * 80)
            print(f"勝率: {summary['win_rate']:.1f}% ({summary['winning_trades']}勝/{summary['losing_trades']}敗)")
            print(f"平均利益: ¥{summary['avg_win']:+,.0f}")
            print(f"平均損失: ¥{summary['avg_loss']:+,.0f}")
            print(f"最大利益: ¥{summary['largest_win']:+,.0f}")
            print(f"最大損失: ¥{summary['largest_loss']:+,.0f}")
            
        print("=" * 80)