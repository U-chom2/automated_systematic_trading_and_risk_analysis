import yfinance as yf
import pandas

#TikerでAPLL指定。
STOCK = yf.Ticker("AAPL") 

# 損益計算書P/L (.income_stmt/.quarterly_income_stmt)------------------------
STOCK_income_stmt = STOCK.income_stmt
print(f'{type(STOCK_income_stmt)=}\n{STOCK_income_stmt}')

# 四半期ごとの場合は
STOCK_quarterly_income_stmt = STOCK.quarterly_income_stmt
print(f'{type(STOCK_quarterly_income_stmt)=}\n{STOCK_quarterly_income_stmt}')

# csvへ出力
STOCK_income_stmt.to_csv(STOCK_info['underlyingSymbol']+"_損益計算書.csv")


# 貸借対照表B/S (.balance_sheet/.quarterly_balance_sheet)--------------------
STOCK_balance_sheet = STOCK.balance_sheet
print(f'{type(STOCK_balance_sheet)=}\n{STOCK_balance_sheet}')

# 四半期ごとの場合は
STOCK_quarterly_balance_sheet = STOCK.quarterly_balance_sheet
print(f'{type(STOCK_quarterly_balance_sheet)=}\n{STOCK_quarterly_balance_sheet}')

# csvへ出力
STOCK_balance_sheet.to_csv(STOCK_info['underlyingSymbol']+"_貸借対照表.csv")


# キャッシュフロー計算書C/F (.cashflow /.quarterly_cashflow )----------------
STOCK_cashflow = STOCK.cashflow
print(f'{type(STOCK_cashflow)=}\n{STOCK_cashflow}')

# 四半期ごとの場合は
STOCK_quarterly_cashflow = STOCK.quarterly_cashflow
print(f'{type(STOCK_quarterly_cashflow)=}\n{STOCK_quarterly_cashflow}')

# csvへ出力
STOCK_cashflow.to_csv(STOCK_info['underlyingSymbol']+"_キャッシュフロー計算書.csv")

