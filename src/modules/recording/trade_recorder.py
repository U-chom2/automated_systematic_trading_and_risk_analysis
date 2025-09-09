"""
取引記録実行クラス
"""
import json
import uuid
from datetime import datetime, date, time
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Dict, Any
import yfinance as yf
from src.utils.logger_utils import create_dual_logger

from .models import TradeRecord, DailySettlement, Portfolio
from src.modules.analysis.models import TodoItem


logger = create_dual_logger(__name__, console_output=True)


class TradeRecorder:
    """取引記録管理"""
    
    def __init__(self,
                 todos_path: str = "data/todos.json",
                 records_path: str = "data/trade_records.json",
                 portfolio_path: str = "data/portfolio.json",
                 commission_rate: float = 0.001):  # 0.1%の手数料
        """
        Args:
            todos_path: TODOリストのパス
            records_path: 取引記録の保存パス
            portfolio_path: ポートフォリオの保存パス
            commission_rate: 手数料率
        """
        self.todos_path = Path(todos_path)
        self.records_path = Path(records_path)
        self.portfolio_path = Path(portfolio_path)
        self.commission_rate = Decimal(str(commission_rate))
        
        # ポートフォリオ読み込みまたは初期化
        self.portfolio = self._load_or_create_portfolio()
    
    def _load_or_create_portfolio(self) -> Portfolio:
        """ポートフォリオを読み込みまたは作成"""
        if self.portfolio_path.exists():
            try:
                with open(self.portfolio_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                portfolio = Portfolio(
                    id=data['id'],
                    name=data['name'],
                    initial_capital=Decimal(data['initial_capital']),
                    current_cash=Decimal(data['current_cash']),
                    positions=data['positions'],
                    created_at=datetime.fromisoformat(data['created_at']),
                    updated_at=datetime.fromisoformat(data['updated_at'])
                )
                logger.info(f"ポートフォリオを読み込みました: {portfolio.name}")
                return portfolio
                
            except Exception as e:
                logger.error(f"ポートフォリオ読み込みエラー: {e}")
        
        # 新規作成
        portfolio = Portfolio(
            id=str(uuid.uuid4()),
            name="メインポートフォリオ",
            initial_capital=Decimal("10000000"),  # 1000万円
            current_cash=Decimal("10000000"),
            positions={}
        )
        logger.info("新規ポートフォリオを作成しました")
        return portfolio
    
    def load_todos(self) -> List[Dict[str, Any]]:
        """TODOリストを読み込み"""
        if not self.todos_path.exists():
            logger.warning(f"TODOファイルが存在しません: {self.todos_path}")
            return []
        
        try:
            with open(self.todos_path, 'r', encoding='utf-8') as f:
                todos = json.load(f)
                logger.info(f"TODOリストを読み込みました: {len(todos)}件")
                return todos
        except Exception as e:
            logger.error(f"TODO読み込みエラー: {e}")
            return []
    
    def get_closing_price(self, ticker: str, target_date: Optional[date] = None) -> Decimal:
        """終値を取得"""
        try:
            stock = yf.Ticker(ticker)
            
            if target_date:
                # 特定日の終値
                hist = stock.history(start=target_date, end=target_date)
                if not hist.empty:
                    return Decimal(str(hist['Close'].iloc[0]))
            
            # 最新の終値
            info = stock.info
            if 'previousClose' in info:
                return Decimal(str(info['previousClose']))
            
            # 履歴から取得
            hist = stock.history(period="1d")
            if not hist.empty:
                return Decimal(str(hist['Close'].iloc[-1]))
            
            logger.warning(f"終値取得失敗: {ticker}")
            return Decimal("0")
            
        except Exception as e:
            logger.error(f"終値取得エラー {ticker}: {e}")
            return Decimal("0")
    
    def calculate_commission(self, amount: Decimal) -> Decimal:
        """手数料を計算"""
        commission = amount * self.commission_rate
        # 最低手数料100円
        return max(commission, Decimal("100"))
    
    def create_trade_record(self, todo: Dict[str, Any], closing_price: Decimal) -> TradeRecord:
        """TODOから取引記録を作成"""
        execution_price = closing_price  # 終値で約定
        quantity = todo['quantity']
        amount = execution_price * quantity
        commission = self.calculate_commission(amount)
        
        record = TradeRecord(
            id=str(uuid.uuid4()),
            todo_id=todo.get('id'),
            portfolio_id=self.portfolio.id,
            stock_id=todo['stock_id'],
            ticker=todo['ticker'],
            company_name=todo['company_name'],
            trade_date=date.today(),
            trade_type=todo['action_type'],
            quantity=quantity,
            execution_price=execution_price,
            closing_price=closing_price,
            commission=commission,
            trade_details={
                "execution_time": "16:00:00",
                "market_condition": "CLOSE",
                "todo_created_at": todo.get('created_at')
            }
        )
        
        return record
    
    def execute_todos(self, target_date: Optional[date] = None) -> List[TradeRecord]:
        """TODOリストを実行して取引記録を作成"""
        if target_date is None:
            target_date = date.today()
        
        logger.info(f"="*50)
        logger.info(f"取引実行開始: {target_date}")
        logger.info(f"="*50)
        
        # TODOリスト読み込み
        todos = self.load_todos()
        
        if not todos:
            logger.info("実行するTODOがありません")
            return []
        
        # 対象日のTODOをフィルタ
        target_todos = [
            todo for todo in todos
            if todo.get('target_date') == target_date.isoformat()
            and todo.get('status') == 'PENDING'
        ]
        
        if not target_todos:
            logger.info(f"{target_date} に実行するTODOがありません")
            return []
        
        logger.info(f"実行対象TODO: {len(target_todos)}件")
        
        records = []
        
        for todo in target_todos:
            try:
                ticker = todo['ticker']
                company_name = todo['company_name']
                action = todo['action_type']
                
                logger.info(f"処理中: {company_name} ({ticker}) - {action}")
                
                # 終値取得
                closing_price = self.get_closing_price(ticker, target_date)
                
                if closing_price == 0:
                    logger.warning(f"終値取得失敗のためスキップ: {ticker}")
                    continue
                
                # 取引記録作成
                record = self.create_trade_record(todo, closing_price)
                
                # ポートフォリオ更新
                try:
                    self.portfolio.execute_trade(record)
                    records.append(record)
                    
                    # TODO のステータス更新
                    todo['status'] = 'EXECUTED'
                    todo['executed_at'] = datetime.now().isoformat()
                    
                    logger.info(
                        f"✓ 取引実行: {company_name} {action} "
                        f"{record.quantity}株 @ ¥{record.execution_price:,.0f}"
                    )
                    
                except ValueError as e:
                    logger.error(f"取引実行エラー: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"TODO処理エラー: {e}")
                continue
        
        # TODOリスト更新を保存
        self._save_todos(todos)
        
        # ポートフォリオ保存
        self._save_portfolio()
        
        logger.info(f"取引実行完了: {len(records)}件")
        
        return records
    
    def _save_todos(self, todos: List[Dict[str, Any]]):
        """TODOリストを保存"""
        with open(self.todos_path, 'w', encoding='utf-8') as f:
            json.dump(todos, f, ensure_ascii=False, indent=2, default=str)
    
    def _save_portfolio(self):
        """ポートフォリオを保存"""
        self.portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.portfolio_path, 'w', encoding='utf-8') as f:
            json.dump(self.portfolio.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"ポートフォリオを保存: 現金 ¥{self.portfolio.current_cash:,.0f}")
    
    def save_records(self, records: List[TradeRecord]):
        """取引記録を保存"""
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 既存の記録を読み込み
        existing_records = []
        if self.records_path.exists():
            try:
                with open(self.records_path, 'r', encoding='utf-8') as f:
                    existing_records = json.load(f)
            except:
                pass
        
        # 新規記録を追加
        all_records = existing_records + [r.to_dict() for r in records]
        
        with open(self.records_path, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"取引記録を保存: {self.records_path} ({len(records)}件追加)")
    
    def create_daily_settlement(self, records: List[TradeRecord], settlement_date: Optional[date] = None) -> DailySettlement:
        """日次決済を作成"""
        if settlement_date is None:
            settlement_date = date.today()
        
        settlement = DailySettlement.from_records(settlement_date, records)
        
        logger.info(settlement.get_summary())
        
        return settlement
    
    def save_settlement(self, settlement: DailySettlement, output_path: str = "data/settlements.json"):
        """決済情報を保存"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 既存の決済情報を読み込み
        existing_settlements = []
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_settlements = json.load(f)
            except:
                pass
        
        # 新規決済を追加
        existing_settlements.append(settlement.to_dict())
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_settlements, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"決済情報を保存: {output_file}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """ポートフォリオの現在状態を取得"""
        # 現在の価格を取得
        prices = {}
        for ticker in self.portfolio.positions.keys():
            prices[ticker] = self.get_closing_price(ticker)
        
        position_value = self.portfolio.get_position_value(prices)
        total_value = self.portfolio.get_total_value(prices)
        
        return {
            "portfolio_id": self.portfolio.id,
            "name": self.portfolio.name,
            "initial_capital": str(self.portfolio.initial_capital),
            "current_cash": str(self.portfolio.current_cash),
            "position_value": str(position_value),
            "total_value": str(total_value),
            "return": str(total_value - self.portfolio.initial_capital),
            "return_rate": float((total_value - self.portfolio.initial_capital) / self.portfolio.initial_capital),
            "positions": [
                {
                    "ticker": ticker,
                    "quantity": quantity,
                    "current_price": str(prices.get(ticker, 0)),
                    "market_value": str(prices.get(ticker, 0) * quantity)
                }
                for ticker, quantity in self.portfolio.positions.items()
            ],
            "updated_at": self.portfolio.updated_at.isoformat()
        }
    
    def execute_daily_settlement(self, target_date: Optional[date] = None) -> DailySettlement:
        """日次決済処理を実行"""
        logger.info("="*50)
        logger.info("日次決済処理開始")
        logger.info("="*50)
        
        # TODOを実行
        records = self.execute_todos(target_date)
        
        if records:
            # 記録保存
            self.save_records(records)
            
            # 決済作成
            settlement = self.create_daily_settlement(records, target_date)
            
            # 決済保存
            self.save_settlement(settlement)
        else:
            # 取引なしの決済
            settlement = DailySettlement(
                settlement_date=target_date or date.today(),
                total_trades=0,
                buy_trades=0,
                sell_trades=0,
                total_buy_amount=Decimal("0"),
                total_sell_amount=Decimal("0"),
                total_commission=Decimal("0"),
                net_cash_flow=Decimal("0"),
                trade_records=[]
            )
            logger.info("取引なし")
        
        # ポートフォリオ状態表示
        status = self.get_portfolio_status()
        logger.info(f"ポートフォリオ状態:")
        logger.info(f"  総資産: ¥{Decimal(status['total_value']):,.0f}")
        logger.info(f"  現金: ¥{Decimal(status['current_cash']):,.0f}")
        logger.info(f"  評価額: ¥{Decimal(status['position_value']):,.0f}")
        logger.info(f"  損益: ¥{Decimal(status['return']):+,.0f} ({status['return_rate']:+.2%})")
        
        logger.info("日次決済処理完了")
        
        return settlement