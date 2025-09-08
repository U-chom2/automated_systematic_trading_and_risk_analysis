"""取引履歴の永続化管理"""

import os
import shutil
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List, Optional
from .base_csv_manager import BaseCSVManager


class TradeHistoryManager(BaseCSVManager):
    """取引履歴の保存・管理を行うクラス
    
    取引履歴をCSVファイルで永続化し、分析機能を提供します。
    """
    
    CSV_FILENAME = 'trade_history.csv'
    
    def get_csv_headers(self) -> List[str]:
        """CSV ヘッダーの定義
        
        Returns:
            取引履歴CSVのヘッダー列名リスト
        """
        return [
            'trade_id', 'symbol', 'side', 'quantity', 'price',
            'executed_at', 'order_type', 'commission', 'total_amount',
            'plan_id', 'status'
        ]
    
    def record_trade(self, trade_data: Dict[str, Any]) -> bool:
        """取引記録を保存
        
        Args:
            trade_data: 保存する取引データ
            
        Returns:
            保存成功時True、失敗時False
        """
        csv_path = self._get_csv_path(self.CSV_FILENAME)
        headers = self.get_csv_headers()
        
        # ヘッダーが存在しない場合は作成
        self._write_csv_header(csv_path, headers)
        
        # Decimal型を文字列に変換
        processed_data = {}
        for key, value in trade_data.items():
            if isinstance(value, Decimal):
                processed_data[key] = str(value)
            else:
                processed_data[key] = value
        
        # JSON化してCSVに追加
        json_data = self._convert_dict_to_json_strings(processed_data)
        row_data = [json_data.get(header, '') for header in headers]
        
        return self._append_csv_row(csv_path, row_data)
    
    def get_trades_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """指定シンボルの取引履歴を取得
        
        Args:
            symbol: 銘柄シンボル
            
        Returns:
            該当する取引データのリスト
        """
        csv_path = self._get_csv_path(self.CSV_FILENAME)
        trades = self._read_csv_rows(csv_path)
        
        result = []
        for trade in trades:
            converted_trade = self._convert_json_strings_to_dict(trade)
            if converted_trade.get('symbol') == symbol:
                # Decimal型に戻す
                if 'price' in converted_trade and converted_trade['price']:
                    converted_trade['price'] = Decimal(converted_trade['price'])
                if 'commission' in converted_trade and converted_trade['commission']:
                    converted_trade['commission'] = Decimal(converted_trade['commission'])
                if 'total_amount' in converted_trade and converted_trade['total_amount']:
                    converted_trade['total_amount'] = Decimal(converted_trade['total_amount'])
                    
                result.append(converted_trade)
        
        return result
    
    def get_trades_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """指定日付範囲の取引履歴を取得
        
        Args:
            start_date: 開始日時（ISO形式）
            end_date: 終了日時（ISO形式）
            
        Returns:
            該当する取引データのリスト
        """
        csv_path = self._get_csv_path(self.CSV_FILENAME)
        trades = self._read_csv_rows(csv_path)
        
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        result = []
        for trade in trades:
            converted_trade = self._convert_json_strings_to_dict(trade)
            executed_at_str = converted_trade.get('executed_at', '')
            
            if executed_at_str:
                try:
                    executed_dt = datetime.fromisoformat(
                        executed_at_str.replace('Z', '+00:00')
                    )
                    if start_dt <= executed_dt <= end_dt:
                        result.append(converted_trade)
                except ValueError:
                    continue
        
        return result
    
    def calculate_pnl_by_symbol(self, symbol: str) -> Decimal:
        """指定シンボルの損益を計算
        
        Args:
            symbol: 銘柄シンボル
            
        Returns:
            計算された損益（Decimal）
        """
        trades = self.get_trades_by_symbol(symbol)
        
        total_buy_cost = Decimal('0')
        total_sell_proceeds = Decimal('0')
        total_commission = Decimal('0')
        
        for trade in trades:
            price = trade.get('price', Decimal('0'))
            quantity = int(trade.get('quantity', 0))
            commission = trade.get('commission', Decimal('0'))
            side = trade.get('side', '')
            
            total_commission += commission
            
            if side == 'buy':
                total_buy_cost += price * quantity
            elif side == 'sell':
                total_sell_proceeds += price * quantity
        
        return total_sell_proceeds - total_buy_cost - total_commission
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """取引サマリーを取得
        
        Returns:
            取引サマリー情報
        """
        csv_path = self._get_csv_path(self.CSV_FILENAME)
        trades = self._read_csv_rows(csv_path)
        
        total_trades = len(trades)
        total_buy_volume = Decimal('0')
        total_sell_volume = Decimal('0')
        total_commission = Decimal('0')
        symbols_traded = set()
        
        for trade in trades:
            converted_trade = self._convert_json_strings_to_dict(trade)
            
            price = Decimal(converted_trade.get('price', '0'))
            quantity = int(converted_trade.get('quantity', 0))
            commission = Decimal(converted_trade.get('commission', '0'))
            side = converted_trade.get('side', '')
            symbol = converted_trade.get('symbol', '')
            
            total_commission += commission
            symbols_traded.add(symbol)
            
            if side == 'buy':
                total_buy_volume += price * quantity
            elif side == 'sell':
                total_sell_volume += price * quantity
        
        # 簡単な損益計算（売上 - 購入額 - 手数料）
        net_pnl = total_sell_volume - total_buy_volume - total_commission
        
        return {
            'total_trades': total_trades,
            'total_buy_volume': total_buy_volume,
            'total_sell_volume': total_sell_volume,
            'total_commission': total_commission,
            'net_pnl': net_pnl,
            'symbols_traded': list(symbols_traded)
        }
    
    def export_to_csv(self, export_path: str) -> bool:
        """取引履歴を指定パスにエクスポート
        
        Args:
            export_path: エクスポート先ファイルパス
            
        Returns:
            エクスポート成功時True、失敗時False
        """
        csv_path = self._get_csv_path(self.CSV_FILENAME)
        
        if not csv_path.exists():
            return False
        
        try:
            shutil.copy2(csv_path, export_path)
            return True
        except (IOError, OSError):
            return False