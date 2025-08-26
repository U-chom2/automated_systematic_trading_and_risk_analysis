#!/usr/bin/env python3
"""
日本電信電話(株)9432.Tの株価情報取得・分析スクリプト

yahooquery を使用して以下の情報を取得：
1. 一日の総流動金額
2. 一日の総流動株数  
3. 過去20日間の株価の最高値,最低値
4. 株価の移動平均線（png）
5. 直近の四半期決算で増収か増益か
"""

import sys
from typing import Dict, Tuple, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from yahooquery import Ticker
import yfinance as yf


class NTTStockAnalyzer:
    """NTT株式の分析を行うクラス"""
    
    def __init__(self, symbol: str = "9432.T") -> None:
        """
        初期化
        
        Args:
            symbol: 株式銘柄コード（デフォルト: 9432.T）
        """
        self.symbol = symbol
        self.ticker = Ticker(symbol)
        self.yf_ticker = yf.Ticker(symbol)
        
    def get_daily_trading_volume(self) -> Tuple[Optional[float], Optional[int]]:
        """
        一日の総流動金額と総流動株数を取得
        
        Returns:
            Tuple[Optional[float], Optional[int]]: (総流動金額, 総流動株数)
        """
        try:
            # 最新の価格情報を取得
            price_data = self.ticker.price
            if self.symbol in price_data:
                data = price_data[self.symbol]
                volume = data.get('regularMarketVolume')
                price = data.get('regularMarketPrice')
                
                if volume is not None and price is not None:
                    total_amount = volume * price
                    return total_amount, volume
                    
        except Exception as e:
            print(f"取引量データの取得エラー: {e}")
            
        return None, None
    
    def get_20day_high_low(self) -> Tuple[Optional[float], Optional[float]]:
        """
        過去20日間の株価の最高値・最低値を取得
        
        Returns:
            Tuple[Optional[float], Optional[float]]: (最高値, 最低値)
        """
        try:
            # 過去30日のデータを取得（土日祝日を考慮して余裕を持たせる）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            history = self.ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            # DataFrameの処理（MultiIndexを考慮）
            if isinstance(history, pd.DataFrame) and not history.empty:
                # MultiIndexの場合、指定シンボルのデータのみ抽出
                if isinstance(history.index, pd.MultiIndex):
                    df = history.loc[self.symbol].copy()
                else:
                    df = history.copy()
                    
                # 過去20営業日分のデータを取得
                recent_data = df.tail(20)
                high_price = recent_data['high'].max()
                low_price = recent_data['low'].min()
                return high_price, low_price
                
        except Exception as e:
            print(f"価格データの取得エラー: {e}")
            
        return None, None
    
    def create_moving_average_chart(self, output_file: str = "ntt_moving_average.png") -> bool:
        """
        移動平均線のグラフを作成してPNGファイルとして保存
        
        Args:
            output_file: 出力ファイル名
            
        Returns:
            bool: 成功した場合True
        """
        try:
            # 過去3ヶ月のデータを取得
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            history = self.ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            # DataFrameの場合の処理（MultiIndexを考慮）
            if isinstance(history, pd.DataFrame) and not history.empty:
                # MultiIndexの場合、指定シンボルのデータのみ抽出
                if isinstance(history.index, pd.MultiIndex):
                    df = history.loc[self.symbol].copy()
                else:
                    df = history.copy()
            # dictの場合の処理
            elif isinstance(history, dict) and self.symbol in history:
                df = pd.DataFrame(history[self.symbol])
            else:
                print("履歴データの形式が不正です")
                return False
            
            if not df.empty:
                # インデックスがdatetimeでない場合は変換
                if not isinstance(df.index, pd.DatetimeIndex):
                    # タイムゾーンの混在問題を解決するためutc=Trueで統一してから削除
                    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                
                # 移動平均線を計算
                df['MA5'] = df['close'].rolling(window=5).mean()
                df['MA20'] = df['close'].rolling(window=20).mean()
                
                # グラフを作成
                plt.figure(figsize=(12, 8))
                plt.plot(df.index, df['close'], label='Close Price', linewidth=1)
                plt.plot(df.index, df['MA5'], label='5-day MA', linewidth=1.5)
                plt.plot(df.index, df['MA20'], label='20-day MA', linewidth=1.5)
                
                plt.title(f'{self.symbol} Stock Price and Moving Averages', fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Price (JPY)', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"移動平均線グラフを {output_file} に保存しました")
                return True
                
        except Exception as e:
            print(f"グラフ作成エラー: {e}")
            
        return False
    
    def get_revenue_growth_yfinance(self) -> Dict[str, Any]:
        """
        yfinance を使って売上高成長率を取得
        
        Returns:
            Dict[str, Any]: 売上高情報
        """
        result = {
            'revenue_growth': None,
            'is_revenue_up': None,
            'method': None,
            'error': None
        }
        
        try:
            # 四半期財務データから取得
            quarterly_income = self.yf_ticker.quarterly_financials
            
            if quarterly_income is not None and not quarterly_income.empty:
                if 'Total Revenue' in quarterly_income.index:
                    revenue_data = quarterly_income.loc['Total Revenue']
                    
                    # NaNでない有効データを取得
                    valid_data = revenue_data.dropna()
                    
                    if len(valid_data) >= 2:
                        latest = valid_data.iloc[0]
                        previous = valid_data.iloc[1]
                        
                        if previous != 0:
                            growth = ((latest - previous) / previous) * 100
                            result['revenue_growth'] = growth
                            result['is_revenue_up'] = growth > 0
                            result['method'] = 'yfinance_quarterly_Total_Revenue'
                            return result
                            
            # 年次データからも試行
            annual_income = self.yf_ticker.financials
            if annual_income is not None and not annual_income.empty:
                if 'Total Revenue' in annual_income.index:
                    revenue_data = annual_income.loc['Total Revenue']
                    valid_data = revenue_data.dropna()
                    
                    if len(valid_data) >= 2:
                        latest = valid_data.iloc[0]
                        previous = valid_data.iloc[1]
                        
                        if previous != 0:
                            growth = ((latest - previous) / previous) * 100
                            result['revenue_growth'] = growth
                            result['is_revenue_up'] = growth > 0
                            result['method'] = 'yfinance_annual_Total_Revenue'
                            return result
            
            result['error'] = "yfinanceでも売上高データを取得できませんでした"
                    
        except Exception as e:
            result['error'] = f"yfinanceエラー: {str(e)}"
            
        return result
    
    def get_quarterly_earnings_info(self) -> Dict[str, Any]:
        """
        直近の四半期決算情報を取得
        
        Returns:
            Dict[str, Any]: 決算情報
        """
        result = {
            'revenue_growth': None,
            'profit_growth': None,
            'is_revenue_up': None,
            'is_profit_up': None,
            'error': None
        }
        
        try:
            # yahooquery の API を確認して適切なメソッドを使用
            # 複数の方法を試行
            financial_data = None
            
            # 方法1: quarterly_financials を試行
            try:
                financial_data = self.ticker.quarterly_financials
            except AttributeError:
                pass
            
            # 方法2: financial_data を試行
            if financial_data is None:
                try:
                    financial_data = self.ticker.financial_data
                except AttributeError:
                    pass
            
            # 方法3: income_statement を試行
            if financial_data is None:
                try:
                    financial_data = self.ticker.income_statement
                except AttributeError:
                    pass
                    
            # 方法4: key_stats を試行（代替情報）
            try:
                key_stats = self.ticker.key_stats
                if isinstance(key_stats, dict) and self.symbol in key_stats:
                    stats = key_stats[self.symbol]
                    
                    # 利益成長率などの情報があれば使用
                    growth_keys = ['quarterlyEarningsGrowth', 'earningsQuarterlyGrowth']
                    for key in growth_keys:
                        if key in stats and stats[key] is not None:
                            result['profit_growth'] = float(stats[key]) * 100  # パーセンテージに変換
                            result['is_profit_up'] = stats[key] > 0
                            break
                    
                    revenue_keys = ['quarterlyRevenueGrowth', 'revenueQuarterlyGrowth']
                    for key in revenue_keys:
                        if key in stats and stats[key] is not None:
                            result['revenue_growth'] = float(stats[key]) * 100  # パーセンテージに変換
                            result['is_revenue_up'] = stats[key] > 0
                            break
                    
                    # 何かデータが取得できた場合はここで返す
                    if result['profit_growth'] is not None or result['revenue_growth'] is not None:
                        return result
            except (AttributeError, KeyError, TypeError):
                pass
            
            if financial_data and isinstance(financial_data, dict) and self.symbol in financial_data:
                data = financial_data[self.symbol]
                
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # 最新の2四半期分のデータを取得
                    recent_quarters = data.iloc[:, :2] if data.shape[1] >= 2 else data
                    
                    # 売上高の確認（複数のキー名を試行）
                    revenue_keys = ['TotalRevenue', 'Revenue', 'totalRevenue', 'revenue']
                    for key in revenue_keys:
                        if key in recent_quarters.index:
                            revenue_data = recent_quarters.loc[key]
                            if len(revenue_data) >= 2:
                                latest_revenue = revenue_data.iloc[0]
                                previous_revenue = revenue_data.iloc[1]
                                
                                if pd.notna(latest_revenue) and pd.notna(previous_revenue) and previous_revenue != 0:
                                    revenue_growth = ((latest_revenue - previous_revenue) / previous_revenue) * 100
                                    result['revenue_growth'] = revenue_growth
                                    result['is_revenue_up'] = revenue_growth > 0
                                    break
                    
                    # 純利益の確認（複数のキー名を試行）
                    profit_keys = ['NetIncome', 'NetIncomeFromContinuingOps', 'netIncome']
                    for key in profit_keys:
                        if key in recent_quarters.index:
                            profit_data = recent_quarters.loc[key]
                            if len(profit_data) >= 2:
                                latest_profit = profit_data.iloc[0]
                                previous_profit = profit_data.iloc[1]
                                
                                if pd.notna(latest_profit) and pd.notna(previous_profit) and previous_profit != 0:
                                    profit_growth = ((latest_profit - previous_profit) / previous_profit) * 100
                                    result['profit_growth'] = profit_growth
                                    result['is_profit_up'] = profit_growth > 0
                                    break
            else:
                result['error'] = "財務データにアクセスできませんでした。データが利用できない可能性があります。"
                    
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def run_full_analysis(self) -> None:
        """全ての分析を実行して結果を表示"""
        print(f"=== {self.symbol} 株式分析結果 ===\n")
        
        # 1. 一日の総流動金額と総流動株数
        total_amount, volume = self.get_daily_trading_volume()
        print("1. 一日の取引情報:")
        if total_amount is not None and volume is not None:
            print(f"   総流動金額: {total_amount:,.0f} 円")
            print(f"   総流動株数: {volume:,} 株")
        else:
            print("   データを取得できませんでした")
        print()
        
        # 2. 過去20日間の最高値・最低値
        high, low = self.get_20day_high_low()
        print("2. 過去20日間の価格:")
        if high is not None and low is not None:
            print(f"   最高値: {high:,.0f} 円")
            print(f"   最低値: {low:,.0f} 円")
        else:
            print("   データを取得できませんでした")
        print()
        
        # 3. 移動平均線グラフ作成
        print("3. 移動平均線グラフ:")
        success = self.create_moving_average_chart()
        if not success:
            print("   グラフの作成に失敗しました")
        print()
        
        # 4. 四半期決算情報
        earnings = self.get_quarterly_earnings_info()
        print("4. 直近四半期決算:")
        
        # まずyfinanceで売上高を試行
        revenue_yf = self.get_revenue_growth_yfinance()
        
        if revenue_yf['error'] is None and revenue_yf['revenue_growth'] is not None:
            status = "増収" if revenue_yf['is_revenue_up'] else "減収"
            print(f"   売上高: {status} ({revenue_yf['revenue_growth']:+.1f}%) [yfinance]")
        elif earnings['revenue_growth'] is not None:
            status = "増収" if earnings['is_revenue_up'] else "減収"
            print(f"   売上高: {status} ({earnings['revenue_growth']:+.1f}%) [yahooquery]")
        else:
            print("   売上高データ: 取得できませんでした")
            
        if earnings['profit_growth'] is not None:
            status = "増益" if earnings['is_profit_up'] else "減益"
            print(f"   純利益: {status} ({earnings['profit_growth']:+.1f}%) [yahooquery]")
        else:
            print("   純利益データ: 取得できませんでした")


def main() -> None:
    """メイン実行関数"""
    try:
        analyzer = NTTStockAnalyzer("9432.T")
        analyzer.run_full_analysis()
        
    except Exception as e:
        print(f"分析実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()