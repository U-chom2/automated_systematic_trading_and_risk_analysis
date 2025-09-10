"""
取引記録モジュールのデータモデル
"""
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Dict, Any
from decimal import Decimal


@dataclass
class TradeRecord:
    """取引記録"""
    id: str
    todo_id: Optional[str]
    portfolio_id: str
    stock_id: str
    ticker: str
    company_name: str
    trade_date: date
    trade_type: str  # BUY or SELL
    quantity: int
    execution_price: Decimal
    closing_price: Decimal
    commission: Decimal = Decimal("0")
    created_at: datetime = None
    trade_details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        
        # Decimal型変換
        if not isinstance(self.execution_price, Decimal):
            self.execution_price = Decimal(str(self.execution_price))
        if not isinstance(self.closing_price, Decimal):
            self.closing_price = Decimal(str(self.closing_price))
        if not isinstance(self.commission, Decimal):
            self.commission = Decimal(str(self.commission))
    
    @property
    def total_amount(self) -> Decimal:
        """取引総額（手数料込み）"""
        base_amount = self.execution_price * self.quantity
        if self.trade_type == "BUY":
            return base_amount + self.commission
        else:
            return base_amount - self.commission
    
    @property
    def price_change(self) -> Decimal:
        """約定価格と終値の差"""
        return self.closing_price - self.execution_price
    
    @property
    def price_change_percent(self) -> float:
        """価格変動率"""
        if self.execution_price == 0:
            return 0.0
        return float((self.price_change / self.execution_price) * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "id": self.id,
            "todo_id": self.todo_id,
            "portfolio_id": self.portfolio_id,
            "stock_id": self.stock_id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "trade_date": self.trade_date.isoformat(),
            "trade_type": self.trade_type,
            "quantity": self.quantity,
            "execution_price": str(self.execution_price),
            "closing_price": str(self.closing_price),
            "commission": str(self.commission),
            "total_amount": str(self.total_amount),
            "price_change": str(self.price_change),
            "price_change_percent": self.price_change_percent,
            "created_at": self.created_at.isoformat(),
            "trade_details": self.trade_details
        }


@dataclass
class DailySettlement:
    """日次決済情報"""
    settlement_date: date
    total_trades: int
    buy_trades: int
    sell_trades: int
    total_buy_amount: Decimal
    total_sell_amount: Decimal
    total_commission: Decimal
    net_cash_flow: Decimal
    trade_records: list[TradeRecord]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        
        # Decimal型変換
        if not isinstance(self.total_buy_amount, Decimal):
            self.total_buy_amount = Decimal(str(self.total_buy_amount))
        if not isinstance(self.total_sell_amount, Decimal):
            self.total_sell_amount = Decimal(str(self.total_sell_amount))
        if not isinstance(self.total_commission, Decimal):
            self.total_commission = Decimal(str(self.total_commission))
        if not isinstance(self.net_cash_flow, Decimal):
            self.net_cash_flow = Decimal(str(self.net_cash_flow))
    
    @classmethod
    def from_records(cls, settlement_date: date, records: list[TradeRecord]) -> "DailySettlement":
        """取引記録から日次決済を作成"""
        buy_records = [r for r in records if r.trade_type == "BUY"]
        sell_records = [r for r in records if r.trade_type == "SELL"]
        
        total_buy_amount = sum((r.execution_price * r.quantity for r in buy_records), Decimal("0"))
        total_sell_amount = sum((r.execution_price * r.quantity for r in sell_records), Decimal("0"))
        total_commission = sum((r.commission for r in records), Decimal("0"))
        
        # 純キャッシュフロー（売却額 - 購入額 - 手数料）
        net_cash_flow = total_sell_amount - total_buy_amount - total_commission
        
        return cls(
            settlement_date=settlement_date,
            total_trades=len(records),
            buy_trades=len(buy_records),
            sell_trades=len(sell_records),
            total_buy_amount=total_buy_amount,
            total_sell_amount=total_sell_amount,
            total_commission=total_commission,
            net_cash_flow=net_cash_flow,
            trade_records=records
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "settlement_date": self.settlement_date.isoformat(),
            "total_trades": self.total_trades,
            "buy_trades": self.buy_trades,
            "sell_trades": self.sell_trades,
            "total_buy_amount": str(self.total_buy_amount),
            "total_sell_amount": str(self.total_sell_amount),
            "total_commission": str(self.total_commission),
            "net_cash_flow": str(self.net_cash_flow),
            "trade_records": [r.to_dict() for r in self.trade_records],
            "created_at": self.created_at.isoformat()
        }
    
    def get_summary(self) -> str:
        """決済サマリーを取得"""
        return (
            f"=== 日次決済サマリー ({self.settlement_date}) ===\n"
            f"総取引数: {self.total_trades}件 (買い: {self.buy_trades}件, 売り: {self.sell_trades}件)\n"
            f"買付総額: ¥{self.total_buy_amount:,.0f}\n"
            f"売却総額: ¥{self.total_sell_amount:,.0f}\n"
            f"手数料総額: ¥{self.total_commission:,.0f}\n"
            f"純キャッシュフロー: ¥{self.net_cash_flow:+,.0f}\n"
        )


@dataclass
class Portfolio:
    """ポートフォリオ"""
    id: str
    name: str
    initial_capital: Decimal
    current_cash: Decimal
    positions: Dict[str, int]  # ticker -> quantity
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        
        # Decimal型変換
        if not isinstance(self.initial_capital, Decimal):
            self.initial_capital = Decimal(str(self.initial_capital))
        if not isinstance(self.current_cash, Decimal):
            self.current_cash = Decimal(str(self.current_cash))
    
    def execute_trade(self, record: TradeRecord):
        """取引を実行"""
        if record.trade_type == "BUY":
            # 買い注文
            if self.current_cash < record.total_amount:
                raise ValueError(f"資金不足: 必要額 {record.total_amount}, 残高 {self.current_cash}")
            
            self.current_cash -= record.total_amount
            self.positions[record.ticker] = self.positions.get(record.ticker, 0) + record.quantity
            
        elif record.trade_type == "SELL":
            # 売り注文
            current_quantity = self.positions.get(record.ticker, 0)
            if current_quantity < record.quantity:
                raise ValueError(f"保有数量不足: {record.ticker} 必要数 {record.quantity}, 保有数 {current_quantity}")
            
            self.current_cash += record.total_amount
            self.positions[record.ticker] -= record.quantity
            
            # 0株になったら削除
            if self.positions[record.ticker] == 0:
                del self.positions[record.ticker]
        
        self.updated_at = datetime.now()
    
    def get_position_value(self, prices: Dict[str, Decimal]) -> Decimal:
        """ポジションの評価額を計算"""
        total_value = Decimal("0")
        
        for ticker, quantity in self.positions.items():
            if ticker in prices:
                total_value += prices[ticker] * quantity
        
        return total_value
    
    def get_total_value(self, prices: Dict[str, Decimal]) -> Decimal:
        """総資産を計算"""
        return self.current_cash + self.get_position_value(prices)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "id": self.id,
            "name": self.name,
            "initial_capital": str(self.initial_capital),
            "current_cash": str(self.current_cash),
            "positions": self.positions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }