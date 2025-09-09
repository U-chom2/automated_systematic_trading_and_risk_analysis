import pandas as pd
from datetime import datetime, timedelta
import jquantsapi
# -------------------------------------------------
# 1. J-Quants APIクライアントの初期化
# -------------------------------------------------
# ご自身のメールアドレスとパスワードを設定してください
MAIL_ADDRESS = "oo.u.chom.oo@gmail.com"
PASSWORD = "YUKI080911astray"

try:
    cli = jquantsapi.Client(mail_address=MAIL_ADDRESS, password=PASSWORD)
    print("J-Quants APIへのログインに成功しました。")
except Exception as e:
    print(f"エラー: J-Quants APIへのログインに失敗しました。{e}")
    exit()

# -------------------------------------------------
# 2. 上場銘柄一覧の取得とグロース市場での絞り込み
# -------------------------------------------------
try:
    df_listed_info = cli.get_listed_info()
    if df_listed_info is None or df_listed_info.empty:
        print("上場銘柄一覧の取得に失敗しました。")
        exit()
        
    # 市場区分が「グロース」の企業のみを抽出
    df_growth = df_listed_info[df_listed_info['MarketCodeName'] == 'グロース'].copy()
    print(f"東証グロース市場の上場企業数: {len(df_growth)}社")

except Exception as e:
    print(f"エラー: 上場銘柄一覧の取得中にエラーが発生しました。 {e}")
    exit()

# -------------------------------------------------
# 3. 株価データの取得 (無料プランの遅延を考慮)
# -------------------------------------------------
# 無料プランでは12週間の遅延があるため、取得可能な最新の日付を計算
# J-Quants APIは土日祝日のデータを返さないため、直近の平日まで遡る
latest_available_date = datetime.now() - timedelta(weeks=12)
while True:
    try:
        date_str = latest_available_date.strftime("%Y-%m-%d")
        print(f"{date_str} の株価データを取得します...")
        df_prices = cli.get_prices_daily_quotes(date=date_str)
        
        if df_prices is not None and not df_prices.empty:
            print("株価データの取得に成功しました。")
            break
        else:
            # データがなければ1日前の日付で再試行
            print(f"{date_str} のデータが見つかりませんでした。1日前のデータを探します。")
            latest_available_date -= timedelta(days=1)
            if (datetime.now() - latest_available_date).days > 100: # 無限ループ防止
                 print("エラー: 直近の有効な株価データが見つかりませんでした。")
                 exit()

    except Exception as e:
        print(f"エラー: 株価データの取得中にエラーが発生しました。 {e}")
        exit()

# -------------------------------------------------
# 4. データの結合と表示
# -------------------------------------------------
# 銘柄コードをキーにして、グロース企業のデータと株価データを結合
# 株価データに'Code'列が存在することを確認
if 'Code' in df_prices.columns:
    # データ型を合わせる
    df_growth['Code'] = df_growth['Code'].astype(str)
    df_prices['Code'] = df_prices['Code'].astype(str)
    
    df_merged = pd.merge(df_growth, df_prices, on='Code', how='inner')

    # 必要な列を抽出して表示
    result_columns = {
        'Code': '企業ID（銘柄コード）',
        'CompanyName': '企業名',
        'Close': '株価（終値）'
    }
    df_result = df_merged[result_columns.keys()].rename(columns=result_columns)
    
    # 時価総額は計算できない旨を列として追加
    df_result['時価総額'] = '別途「発行済株式総数」が必要'
    
    print("\n--- 東証グロース市場 企業情報一覧 ---")
    print(f"データ基準日: {date_str}（注：無料プランのため約12週間の遅延があります）")
    print(df_result.to_string())

else:
    print("エラー: 取得した株価データに 'Code' 列が含まれていません。")