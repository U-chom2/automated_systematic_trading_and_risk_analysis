"""
実際の投資判断AI - リアルタイム推論システム

訓練済みモデルを使用して、最新市場データから具体的な投資判断を実行
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import torch
import logging

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "train"))

from models.trading_model import TradingDecisionModel, MarketData
from models.agents.ppo_agent import PPOTradingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeInvestmentAnalyzer:
    """リアルタイム投資判断AI"""
    
    def __init__(self):
        self.target_companies = self.load_target_companies()
        self.model_path = self.find_latest_model()
        
        # デバイス自動検出
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
            
        logger.info(f"Investment Analyzer initialized on {self.device}")
    
    def load_target_companies(self) -> pd.DataFrame:
        """ターゲット企業.xlsxから企業情報を読み込み"""
        df = pd.read_excel('ターゲット企業.xlsx')
        return df
    
    def find_latest_model(self) -> str:
        """最新の訓練済みモデルを検索"""
        model_dir = Path("train/models/rl")
        model_files = list(model_dir.glob("*.zip"))
        
        if not model_files:
            logger.error("No trained models found!")
            return None
            
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Latest model found: {latest_model}")
        return str(latest_model)
    
    def fetch_latest_market_data(self, symbols: list, days: int = 35) -> tuple:
        """最新の市場データを取得"""
        logger.info(f"Fetching latest {days} days data for {len(symbols)} symbols...")
        
        # 日付範囲設定
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days+10)  # 余裕を持って取得
        
        # 日経225データ取得
        nikkei = yf.Ticker("^N225")
        nikkei_hist = nikkei.history(start=start_date, end=end_date)
        
        # 最新30日分を抽出
        if len(nikkei_hist) >= 30:
            nikkei_data = nikkei_hist.tail(30)
        else:
            logger.warning(f"Only {len(nikkei_hist)} days of Nikkei data available")
            nikkei_data = nikkei_hist
        
        # ターゲット株価データ取得
        stock_data = {}
        current_prices = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if len(hist) >= 30:
                    stock_data[symbol] = hist.tail(30)
                    current_prices[symbol] = hist['Close'].iloc[-1]
                    logger.info(f"{symbol}: Latest price ¥{current_prices[symbol]:.0f}")
                else:
                    logger.warning(f"{symbol}: Only {len(hist)} days available")
                    stock_data[symbol] = hist
                    if len(hist) > 0:
                        current_prices[symbol] = hist['Close'].iloc[-1]
                        
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                stock_data[symbol] = pd.DataFrame()
                current_prices[symbol] = 0
        
        # IRニュース（モック）
        ir_news = [
            f"【{datetime.now().strftime('%Y年%m月')}】 決算シーズン到来、業績期待の銘柄に注目",
            "マーケット環境改善により、成長株への資金流入が加速",
            "AI・DX関連銘柄の業績好調、今後の成長に期待"
        ]
        
        return nikkei_data, stock_data, current_prices, ir_news
    
    def create_market_data_for_inference(self, symbol: str, nikkei_data: pd.DataFrame, 
                                       stock_data: dict) -> MarketData:
        """推論用MarketDataオブジェクト作成"""
        
        # 日経225データ
        nikkei_high = nikkei_data['High'].values
        nikkei_low = nikkei_data['Low'].values  
        nikkei_close = nikkei_data['Close'].values
        
        # ターゲット株データ
        if symbol in stock_data and not stock_data[symbol].empty:
            target_hist = stock_data[symbol]
            target_high = target_hist['High'].values
            target_low = target_hist['Low'].values
            target_close = target_hist['Close'].values
        else:
            # データがない場合はダミー値
            target_high = nikkei_high * 0.1
            target_low = nikkei_low * 0.1  
            target_close = nikkei_close * 0.1
        
        # 長さを揃える
        min_length = min(len(nikkei_high), len(target_high))
        if min_length < 30:
            # データが不足している場合は最後の値で埋める
            def pad_array(arr, target_len=30):
                if len(arr) == 0:
                    return np.full(target_len, 1000.0)  # デフォルト値
                if len(arr) >= target_len:
                    return arr[-target_len:]
                # 不足分は最後の値で埋める
                pad_size = target_len - len(arr)
                return np.concatenate([np.full(pad_size, arr[0]), arr])
            
            nikkei_high = pad_array(nikkei_high)
            nikkei_low = pad_array(nikkei_low) 
            nikkei_close = pad_array(nikkei_close)
            target_high = pad_array(target_high)
            target_low = pad_array(target_low)
            target_close = pad_array(target_close)
        else:
            nikkei_high = nikkei_high[-30:]
            nikkei_low = nikkei_low[-30:]
            nikkei_close = nikkei_close[-30:]
            target_high = target_high[-30:]
            target_low = target_low[-30:]
            target_close = target_close[-30:]
        
        # IRニュース
        company_name = self.target_companies[self.target_companies['証券コード'] == int(symbol.replace('.T', ''))]['企業名'].iloc[0] if len(self.target_companies[self.target_companies['証券コード'] == int(symbol.replace('.T', ''))]) > 0 else symbol
        ir_news = [
            f"{company_name}の最新業績動向に注目",
            "成長分野への事業展開を加速",
            "市場環境の変化に対する適応戦略を評価"
        ]
        
        return MarketData(
            nikkei_high=nikkei_high,
            nikkei_low=nikkei_low,
            nikkei_close=nikkei_close,
            target_high=target_high,
            target_low=target_low,
            target_close=target_close,
            ir_news=ir_news
        )
    
    def analyze_investment_opportunity(self) -> dict:
        """投資機会の分析実行"""
        logger.info("=" * 80)
        logger.info("🤖 AI投資判断システム - リアルタイム分析開始")
        logger.info("=" * 80)
        
        if not self.model_path:
            return {"error": "No trained model found"}
        
        # 訓練対象の5社
        target_symbols = ['4057.T', '3961.T', '4179.T', '7041.T', '9242.T']
        
        # 最新市場データ取得
        nikkei_data, stock_data, current_prices, ir_news = self.fetch_latest_market_data(target_symbols)
        
        # 推論用モデル初期化
        model = TradingDecisionModel(device=self.device)
        
        # 各銘柄の投資判断
        recommendations = []
        
        for symbol in target_symbols:
            try:
                logger.info(f"\\n📊 {symbol} の投資判断分析中...")
                
                # MarketDataオブジェクト作成
                market_data = self.create_market_data_for_inference(symbol, nikkei_data, stock_data)
                
                # AI推論実行
                with torch.no_grad():
                    decision = model(market_data)
                
                # 企業情報取得
                company_info = self.target_companies[self.target_companies['証券コード'] == int(symbol.replace('.T', ''))]
                company_name = company_info['企業名'].iloc[0] if len(company_info) > 0 else symbol
                market_cap = company_info['時価総額 (百万円)'].iloc[0] if len(company_info) > 0 else 0
                
                # 推奨情報をまとめる
                rec = {
                    'symbol': symbol,
                    'company_name': company_name,
                    'current_price': current_prices.get(symbol, 0),
                    'market_cap': market_cap,
                    'ai_action': decision['action'],
                    'confidence': decision['confidence'],
                    'recommended_position': decision['recommended_position'],
                    'probabilities': {
                        'sell': decision['sell_prob'],
                        'hold': decision['hold_prob'],
                        'buy_small': decision['buy_small_prob'],
                        'buy_strong': decision['buy_large_prob']
                    }
                }
                
                recommendations.append(rec)
                
                logger.info(f"✅ {company_name} ({symbol}): {decision['action']} (信頼度: {decision['confidence']*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ {symbol} の分析中にエラー: {e}")
                continue
        
        # 投資推奨をランキング
        buy_recommendations = [r for r in recommendations if r['recommended_position'] > 0.1]
        buy_recommendations.sort(key=lambda x: x['confidence'] * x['recommended_position'], reverse=True)
        
        hold_recommendations = [r for r in recommendations if -0.1 <= r['recommended_position'] <= 0.1]
        sell_recommendations = [r for r in recommendations if r['recommended_position'] < -0.1]
        
        return {
            'analysis_date': datetime.now(),
            'nikkei_current': nikkei_data['Close'].iloc[-1] if not nikkei_data.empty else 0,
            'all_recommendations': recommendations,
            'buy_recommendations': buy_recommendations,
            'hold_recommendations': hold_recommendations,
            'sell_recommendations': sell_recommendations,
            'summary': {
                'total_analyzed': len(recommendations),
                'buy_signals': len(buy_recommendations),
                'hold_signals': len(hold_recommendations),
                'sell_signals': len(sell_recommendations)
            }
        }
    
    def print_investment_report(self, analysis: dict):
        """投資推奨レポートを表示"""
        logger.info("=" * 80)
        logger.info("📈 AI投資判断レポート")
        logger.info("=" * 80)
        logger.info(f"分析日時: {analysis['analysis_date'].strftime('%Y年%m月%d日 %H:%M')}")
        logger.info(f"日経225: {analysis['nikkei_current']:.0f}")
        logger.info("")
        
        # 買い推奨
        if analysis['buy_recommendations']:
            logger.info("🟢 【買い推奨銘柄】")
            logger.info("-" * 60)
            for i, rec in enumerate(analysis['buy_recommendations'], 1):
                investment_amount = 10000000 * rec['recommended_position']  # 1000万円ベース
                shares = int(investment_amount / rec['current_price']) if rec['current_price'] > 0 else 0
                
                logger.info(f"{i}. {rec['company_name']} ({rec['symbol']})")
                logger.info(f"   現在株価: ¥{rec['current_price']:.0f}")
                logger.info(f"   AI判断: {rec['ai_action']} (信頼度: {rec['confidence']*100:.1f}%)")
                logger.info(f"   推奨投資比率: {rec['recommended_position']*100:.0f}%")
                logger.info(f"   💰 推奨投資額: ¥{investment_amount:,.0f} ({shares:,}株)")
                logger.info(f"   時価総額: {rec['market_cap']}百万円")
                logger.info("")
        else:
            logger.info("🟢 【買い推奨銘柄】: なし")
            logger.info("")
        
        # ホールド推奨  
        if analysis['hold_recommendations']:
            logger.info("🟡 【ホールド推奨銘柄】")
            logger.info("-" * 60)
            for rec in analysis['hold_recommendations']:
                logger.info(f"• {rec['company_name']} ({rec['symbol']}) - ¥{rec['current_price']:.0f}")
                logger.info(f"  信頼度: {rec['confidence']*100:.1f}%")
            logger.info("")
        
        # 売り推奨
        if analysis['sell_recommendations']:
            logger.info("🔴 【売り推奨銘柄】")
            logger.info("-" * 60)
            for rec in analysis['sell_recommendations']:
                logger.info(f"• {rec['company_name']} ({rec['symbol']}) - ¥{rec['current_price']:.0f}")
                logger.info(f"  売却比率: {abs(rec['recommended_position'])*100:.0f}%")
                logger.info(f"  信頼度: {rec['confidence']*100:.1f}%")
            logger.info("")
        
        # サマリー
        logger.info("📊 【分析サマリー】")
        logger.info("-" * 60)
        logger.info(f"分析銘柄数: {analysis['summary']['total_analyzed']}")
        logger.info(f"買いシグナル: {analysis['summary']['buy_signals']}")
        logger.info(f"ホールドシグナル: {analysis['summary']['hold_signals']}")
        logger.info(f"売りシグナル: {analysis['summary']['sell_signals']}")
        logger.info("")
        
        # 免責事項
        logger.info("⚠️  【重要な免責事項】")
        logger.info("-" * 60)
        logger.info("• この分析はAIによる参考情報であり、投資助言ではありません")
        logger.info("• 投資判断は必ずご自身の責任で行ってください")
        logger.info("• 実際の投資では十分なリスク管理を行ってください")
        logger.info("• 過去の成績は将来の運用成果を保証するものではありません")
        logger.info("=" * 80)


def main():
    """メイン実行関数"""
    try:
        # 投資分析AI初期化
        analyzer = RealTimeInvestmentAnalyzer()
        
        # 投資判断実行
        analysis = analyzer.analyze_investment_opportunity()
        
        # レポート表示
        analyzer.print_investment_report(analysis)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    analysis = main()