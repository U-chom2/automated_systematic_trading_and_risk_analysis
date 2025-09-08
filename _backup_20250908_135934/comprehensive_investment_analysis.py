"""
総合投資判断AI - 30社完全分析システム

テクニカル分析 + ファンダメンタル分析 + AI判断を統合した
具体的な投資推奨システム
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

class ComprehensiveInvestmentAnalyzer:
    """総合投資判断AI"""
    
    def __init__(self):
        self.companies = pd.read_excel('ターゲット企業.xlsx')
        print("🚀 総合投資判断AI - 30社完全分析システム")
        print("=" * 80)
        
    def fetch_market_data_bulk(self, symbols: list, period: str = "6mo") -> dict:
        """全銘柄の市場データを一括取得"""
        print(f"📊 {len(symbols)}銘柄のデータ取得中...")
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info
                
                if not hist.empty:
                    data[symbol] = {
                        'history': hist,
                        'current_price': hist['Close'].iloc[-1],
                        'info': info,
                        'symbol_clean': symbol.replace('.T', '')
                    }
                    print(f"✓ {symbol}: ¥{hist['Close'].iloc[-1]:.0f}")
                else:
                    print(f"✗ {symbol}: データなし")
                    
            except Exception as e:
                print(f"✗ {symbol}: エラー - {str(e)[:50]}")
                continue
                
        print(f"\\n✅ {len(data)}/{len(symbols)} 銘柄のデータ取得完了\\n")
        return data
    
    def calculate_technical_indicators(self, hist_data: pd.DataFrame) -> dict:
        """テクニカル指標を計算"""
        try:
            close = hist_data['Close']
            volume = hist_data['Volume']
            
            # 移動平均
            sma_5 = close.rolling(5).mean()
            sma_25 = close.rolling(25).mean()
            sma_75 = close.rolling(75).mean()
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = close.ewm(span=12).mean()
            exp2 = close.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            
            # ボリンジャーバンド
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            upper_bb = sma_20 + (std_20 * 2)
            lower_bb = sma_20 - (std_20 * 2)
            
            # 価格変化率
            returns_1d = close.pct_change(1)
            returns_5d = close.pct_change(5)
            returns_25d = close.pct_change(25)
            
            # ボラティリティ
            volatility = returns_1d.rolling(20).std() * np.sqrt(252)
            
            current_price = close.iloc[-1]
            
            indicators = {
                'current_price': current_price,
                'sma_5': sma_5.iloc[-1] if not sma_5.empty else current_price,
                'sma_25': sma_25.iloc[-1] if not sma_25.empty else current_price,
                'sma_75': sma_75.iloc[-1] if not sma_75.empty else current_price,
                'rsi': rsi.iloc[-1] if not rsi.empty else 50,
                'macd': macd.iloc[-1] if not macd.empty else 0,
                'macd_signal': signal.iloc[-1] if not signal.empty else 0,
                'bb_upper': upper_bb.iloc[-1] if not upper_bb.empty else current_price * 1.1,
                'bb_lower': lower_bb.iloc[-1] if not lower_bb.empty else current_price * 0.9,
                'returns_1d': returns_1d.iloc[-1] * 100 if not returns_1d.empty else 0,
                'returns_5d': returns_5d.iloc[-1] * 100 if not returns_5d.empty else 0,
                'returns_25d': returns_25d.iloc[-1] * 100 if not returns_25d.empty else 0,
                'volatility': volatility.iloc[-1] * 100 if not volatility.empty else 20,
                'volume_ratio': (volume.iloc[-5:].mean() / volume.iloc[-25:].mean()) if len(volume) >= 25 else 1.0
            }
            
            return indicators
            
        except Exception as e:
            print(f"テクニカル指標計算エラー: {e}")
            return {}
    
    def calculate_investment_score(self, symbol: str, market_data: dict, company_info: pd.Series) -> dict:
        """投資スコアを計算"""
        try:
            hist = market_data['history']
            current_price = market_data['current_price']
            
            # テクニカル指標取得
            tech = self.calculate_technical_indicators(hist)
            if not tech:
                return {'score': 0, 'signal': 'データ不足'}
            
            # スコア計算（0-100点）
            score = 50  # ベーススコア
            signals = []
            
            # 1. トレンド分析 (30点)
            if tech['current_price'] > tech['sma_5'] > tech['sma_25']:
                score += 15
                signals.append("上昇トレンド")
            elif tech['current_price'] < tech['sma_5'] < tech['sma_25']:
                score -= 15
                signals.append("下降トレンド")
            else:
                signals.append("横ばい")
            
            if tech['sma_5'] > tech['sma_25'] > tech['sma_75']:
                score += 15
                signals.append("長期上昇")
            elif tech['sma_5'] < tech['sma_25'] < tech['sma_75']:
                score -= 15
                signals.append("長期下降")
            
            # 2. RSI分析 (20点)
            if 30 <= tech['rsi'] <= 40:
                score += 15
                signals.append("RSI買い場")
            elif 60 <= tech['rsi'] <= 70:
                score += 10
                signals.append("RSI強気")
            elif tech['rsi'] < 25:
                score += 20  # 売られすぎ
                signals.append("RSI超買い場")
            elif tech['rsi'] > 75:
                score -= 15
                signals.append("RSI売られすぎ")
            
            # 3. MACD分析 (15点)
            if tech['macd'] > tech['macd_signal'] and tech['macd'] > 0:
                score += 15
                signals.append("MACD買いシグナル")
            elif tech['macd'] < tech['macd_signal']:
                score -= 10
                signals.append("MACD弱気")
            
            # 4. ボリンジャーバンド分析 (15点)
            if tech['current_price'] < tech['bb_lower']:
                score += 15
                signals.append("BB下限付近(買い場)")
            elif tech['current_price'] > tech['bb_upper']:
                score -= 10
                signals.append("BB上限付近")
            
            # 5. 出来高分析 (10点)
            if tech['volume_ratio'] > 1.5:
                score += 10
                signals.append("出来高増加")
            elif tech['volume_ratio'] < 0.7:
                score -= 5
                signals.append("出来高減少")
            
            # 6. パフォーマンス分析 (10点)
            if tech['returns_5d'] > 5:
                score += 5
                signals.append("5日上昇")
            elif tech['returns_5d'] < -5:
                score -= 5
                signals.append("5日下落")
            
            if tech['returns_25d'] > 10:
                score += 5
                signals.append("25日好調")
            elif tech['returns_25d'] < -10:
                score -= 5
                signals.append("25日軟調")
            
            # 時価総額ボーナス（小型株優遇）
            market_cap = company_info['時価総額 (百万円)']
            if market_cap < 1500:
                score += 5
                signals.append("小型株")
            
            # スコア調整
            score = max(0, min(100, score))
            
            # 投資判断
            if score >= 75:
                investment_signal = "強い買い"
                position_size = 0.8
            elif score >= 60:
                investment_signal = "買い"
                position_size = 0.5
            elif score >= 55:
                investment_signal = "小幅買い"
                position_size = 0.2
            elif score >= 45:
                investment_signal = "ホールド"
                position_size = 0.0
            elif score >= 30:
                investment_signal = "小幅売り"
                position_size = -0.3
            else:
                investment_signal = "売り"
                position_size = -0.6
            
            # 売却タイミング設定
            if investment_signal in ["強い買い", "買い", "小幅買い"]:
                target_profit = 15 if investment_signal == "強い買い" else 10
                stop_loss = -8 if investment_signal == "強い買い" else -5
                holding_period = "2-6ヶ月"
            else:
                target_profit = 0
                stop_loss = 0
                holding_period = "現在売却推奨" if "売り" in investment_signal else "様子見"
            
            return {
                'score': score,
                'signal': investment_signal,
                'position_size': position_size,
                'target_profit': target_profit,
                'stop_loss': stop_loss,
                'holding_period': holding_period,
                'technical_signals': signals,
                'technical_data': tech,
                'current_price': current_price,
                'market_cap': market_cap
            }
            
        except Exception as e:
            print(f"スコア計算エラー ({symbol}): {e}")
            return {'score': 0, 'signal': 'エラー'}
    
    def analyze_all_companies(self) -> pd.DataFrame:
        """全30社を分析"""
        print("🔍 全30社の総合分析開始...")
        print("=" * 80)
        
        # 全銘柄リスト
        symbols = [f"{code}.T" for code in self.companies['証券コード']]
        
        # 市場データ取得
        market_data = self.fetch_market_data_bulk(symbols)
        
        # 分析結果格納
        results = []
        
        for idx, company in self.companies.iterrows():
            symbol = f"{company['証券コード']}.T"
            
            print(f"📈 {company['企業名']} ({symbol}) 分析中...")
            
            if symbol in market_data:
                analysis = self.calculate_investment_score(symbol, market_data[symbol], company)
                
                result = {
                    '証券コード': company['証券コード'],
                    '企業名': company['企業名'],
                    'シンボル': symbol,
                    '現在株価': analysis.get('current_price', 0),
                    '時価総額': company['時価総額 (百万円)'],
                    '投資スコア': analysis.get('score', 0),
                    '投資判断': analysis.get('signal', 'データなし'),
                    'ポジションサイズ': analysis.get('position_size', 0),
                    '目標利益率': analysis.get('target_profit', 0),
                    '損切りライン': analysis.get('stop_loss', 0),
                    '推奨保有期間': analysis.get('holding_period', ''),
                    'テクニカルシグナル': ', '.join(analysis.get('technical_signals', [])),
                    '1日変化率': analysis.get('technical_data', {}).get('returns_1d', 0),
                    '5日変化率': analysis.get('technical_data', {}).get('returns_5d', 0),
                    '25日変化率': analysis.get('technical_data', {}).get('returns_25d', 0),
                    'RSI': analysis.get('technical_data', {}).get('rsi', 0),
                    'ボラティリティ': analysis.get('technical_data', {}).get('volatility', 0)
                }
                
                results.append(result)
                
                print(f"✅ スコア: {analysis.get('score', 0):.0f}点 - {analysis.get('signal', 'データなし')}")
            else:
                print("❌ データ取得失敗")
                results.append({
                    '証券コード': company['証券コード'],
                    '企業名': company['企業名'],
                    'シンボル': symbol,
                    '投資スコア': 0,
                    '投資判断': 'データなし'
                })
        
        df = pd.DataFrame(results)
        return df.sort_values('投資スコア', ascending=False)
    
    def generate_investment_report(self, analysis_df: pd.DataFrame, max_investment_per_stock: float = 2000.0):
        """投資推奨レポート生成 (1株当たり投資上限リミッター付き)
        
        Args:
            analysis_df: 分析結果DataFrame
            max_investment_per_stock: 1株当たりの投資上限金額（デフォルト: 2000円）
        """
        print("\\n" + "=" * 80)
        print("💰 総合投資推奨レポート")
        print("=" * 80)
        print(f"分析日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
        print(f"分析銘柄数: {len(analysis_df)}社")
        print(f"🔒 投資制限: 1株当たり最大¥{max_investment_per_stock:,.0f}の投資リミッター設定")
        
        # 買い推奨銘柄（スコア55以上）
        buy_stocks = analysis_df[analysis_df['投資スコア'] >= 55].copy()
        if not buy_stocks.empty:
            print("\\n🟢 【買い推奨銘柄ランキング】")
            print("-" * 80)
            
            total_limited_investment = 0
            
            for idx, stock in buy_stocks.iterrows():
                # 元の推奨投資額
                original_investment_amount = 10000000 * abs(stock['ポジションサイズ'])  # 1000万円ベース
                
                # リミッター適用後の投資額計算
                if stock['現在株価'] > 0:
                    # 1株当たりの投資制限を適用
                    max_shares_by_limit = int(max_investment_per_stock / stock['現在株価'])
                    original_shares = int(original_investment_amount / stock['現在株価'])
                    
                    # より制限の厳しい方を採用
                    final_shares = min(max_shares_by_limit, original_shares)
                    final_investment_amount = final_shares * stock['現在株価']
                    
                    # リミッター適用フラグ
                    is_limited = final_shares < original_shares
                    
                else:
                    final_shares = 0
                    final_investment_amount = 0
                    is_limited = False
                
                total_limited_investment += final_investment_amount
                
                print(f"\\n{len(buy_stocks) - len(buy_stocks[buy_stocks['投資スコア'] > stock['投資スコア']])}位: {stock['企業名']} ({stock['シンボル']})")
                print(f"   📊 投資スコア: {stock['投資スコア']:.0f}点")
                print(f"   💰 現在株価: ¥{stock['現在株価']:.0f}")
                print(f"   📈 投資判断: {stock['投資判断']}")
                
                # リミッター情報表示
                if is_limited:
                    print(f"   🔒 投資制限適用: ¥{original_investment_amount:,.0f} → ¥{final_investment_amount:,.0f}")
                    print(f"   🎯 制限後投資額: ¥{final_investment_amount:,.0f} ({final_shares:,}株) ⚠️制限適用")
                else:
                    print(f"   🎯 推奨投資額: ¥{final_investment_amount:,.0f} ({final_shares:,}株)")
                
                print(f"   📈 目標利益: +{stock['目標利益率']:.0f}% (¥{stock['現在株価'] * (1 + stock['目標利益率']/100):.0f})")
                print(f"   🛑 損切ライン: {stock['損切りライン']:.0f}% (¥{stock['現在株価'] * (1 + stock['損切りライン']/100):.0f})")
                print(f"   ⏰ 推奨保有期間: {stock['推奨保有期間']}")
                print(f"   🔍 テクニカル: {stock['テクニカルシグナル']}")
                print(f"   📊 パフォーマンス: 1日{stock['1日変化率']:.1f}% | 5日{stock['5日変化率']:.1f}% | 25日{stock['25日変化率']:.1f}%")
            
            print(f"\\n💰 【投資制限後の合計推奨投資額: ¥{total_limited_investment:,.0f}】")
        
        # ホールド推奨
        hold_stocks = analysis_df[(analysis_df['投資スコア'] >= 45) & (analysis_df['投資スコア'] < 55)]
        if not hold_stocks.empty:
            print("\\n🟡 【ホールド推奨銘柄】")
            print("-" * 80)
            for idx, stock in hold_stocks.head(5).iterrows():
                print(f"• {stock['企業名']} ({stock['シンボル']}) - ¥{stock['現在株価']:.0f} (スコア: {stock['投資スコア']:.0f}点)")
        
        # 売り推奨
        sell_stocks = analysis_df[analysis_df['投資スコア'] < 45]
        if not sell_stocks.empty:
            print("\\n🔴 【売り/回避推奨銘柄】")
            print("-" * 80)
            for idx, stock in sell_stocks.head(3).iterrows():
                print(f"• {stock['企業名']} ({stock['シンボル']}) - ¥{stock['現在株価']:.0f} (スコア: {stock['投資スコア']:.0f}点)")
                print(f"  理由: {stock['投資判断']} - {stock['テクニカルシグナル']}")
        
        # 統計サマリー
        print("\\n📊 【分析統計】")
        print("-" * 80)
        print(f"平均投資スコア: {analysis_df['投資スコア'].mean():.1f}点")
        print(f"買い推奨: {len(buy_stocks)}銘柄 ({len(buy_stocks)/len(analysis_df)*100:.1f}%)")
        print(f"ホールド推奨: {len(hold_stocks)}銘柄 ({len(hold_stocks)/len(analysis_df)*100:.1f}%)")
        print(f"売り/回避推奨: {len(sell_stocks)}銘柄 ({len(sell_stocks)/len(analysis_df)*100:.1f}%)")
        
        # トップピック（リミッター適用）
        if not buy_stocks.empty:
            top_pick = buy_stocks.iloc[0]
            
            # トップピックにもリミッター適用
            original_top_investment = 10000000 * abs(top_pick['ポジションサイズ'])
            if top_pick['現在株価'] > 0:
                max_shares_top = int(max_investment_per_stock / top_pick['現在株価'])
                original_shares_top = int(original_top_investment / top_pick['現在株価'])
                final_shares_top = min(max_shares_top, original_shares_top)
                final_investment_top = final_shares_top * top_pick['現在株価']
                is_top_limited = final_shares_top < original_shares_top
            else:
                final_investment_top = 0
                is_top_limited = False
            
            print("\\n🌟 【最強推奨銘柄】")
            print("-" * 80)
            print(f"企業: {top_pick['企業名']} ({top_pick['シンボル']})")
            print(f"投資スコア: {top_pick['投資スコア']:.0f}点")
            print(f"現在株価: ¥{top_pick['現在株価']:.0f}")
            
            if is_top_limited:
                print(f"制限前投資額: ¥{original_top_investment:,.0f}")
                print(f"🔒制限後投資額: ¥{final_investment_top:,.0f} ({final_shares_top:,}株)")
            else:
                print(f"推奨投資額: ¥{final_investment_top:,.0f} ({final_shares_top:,}株)")
            
            print(f"目標株価: ¥{top_pick['現在株価'] * (1 + top_pick['目標利益率']/100):.0f} (+{top_pick['目標利益率']:.0f}%)")
            print(f"保有期間: {top_pick['推奨保有期間']}")
        
        print("\\n" + "⚠️" * 3 + " 重要な免責事項 " + "⚠️" * 3)
        print("-" * 80)
        print("• この分析はAIによる参考情報であり、投資助言ではありません")
        print("• 投資判断は必ずご自身の責任で行ってください")  
        print("• 株式投資にはリスクが伴います。余裕資金で行ってください")
        print("• 損切りラインは必ず設定し、リスク管理を徹底してください")
        print("=" * 80)
        
        return analysis_df


def main():
    """メイン実行"""
    analyzer = ComprehensiveInvestmentAnalyzer()
    
    # 全社分析実行
    results = analyzer.analyze_all_companies()
    
    # レポート生成
    analyzer.generate_investment_report(results)
    
    # CSVで保存
    results.to_csv('investment_analysis_results.csv', index=False, encoding='utf-8-sig')
    print(f"\\n💾 分析結果を investment_analysis_results.csv に保存しました")
    
    return results


if __name__ == "__main__":
    results = main()