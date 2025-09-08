"""バックテストAPIエンドポイント"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import date
from decimal import Decimal

from ....application.use_cases.backtesting import (
    RunBacktestUseCase,
    AnalyzeBacktestResultsUseCase,
    BacktestConfig,
    BacktestResult,
)
from ....domain.services.trading_strategy import TradingStrategy, StrategyType
from ....domain.services.signal_generator import SignalGenerator
from ....infrastructure.repositories.market_data_repository_impl import MarketDataRepositoryImpl
from ....infrastructure.repositories.stock_repository_impl import StockRepositoryImpl


router = APIRouter()


def get_market_data_repository():
    """市場データリポジトリを取得"""
    return MarketDataRepositoryImpl()


def get_stock_repository():
    """株式リポジトリを取得"""
    return StockRepositoryImpl()


def get_trading_strategy(strategy_type: StrategyType):
    """取引戦略を取得"""
    return TradingStrategy(strategy_type)


def get_signal_generator():
    """シグナルジェネレーターを取得"""
    return SignalGenerator()


@router.post("/run")
async def run_backtest(
    tickers: List[str],
    start_date: date,
    end_date: date,
    initial_capital: float,
    strategy_type: str = "AI_DRIVEN",
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    max_positions: int = 20,
    rebalance_frequency: str = "weekly",
    market_repo: MarketDataRepositoryImpl = Depends(get_market_data_repository),
    stock_repo: StockRepositoryImpl = Depends(get_stock_repository),
) -> dict:
    """バックテストを実行
    
    Args:
        tickers: ティッカーリスト
        start_date: 開始日
        end_date: 終了日
        initial_capital: 初期資金
        strategy_type: 戦略タイプ
        commission_rate: 手数料率
        slippage_rate: スリッページ率
        max_positions: 最大ポジション数
        rebalance_frequency: リバランス頻度
    
    Returns:
        バックテスト結果
    """
    try:
        # 戦略タイプを変換
        try:
            strategy_enum = StrategyType[strategy_type]
        except KeyError:
            raise ValueError(f"Invalid strategy type: {strategy_type}")
        
        # バックテスト設定を作成
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=Decimal(str(initial_capital)),
            strategy_type=strategy_enum,
            tickers=tickers,
            commission_rate=Decimal(str(commission_rate)),
            slippage_rate=Decimal(str(slippage_rate)),
            max_positions=max_positions,
            rebalance_frequency=rebalance_frequency,
        )
        
        config.validate()
        
        # バックテストを実行
        strategy = get_trading_strategy(strategy_enum)
        signal_gen = get_signal_generator()
        
        use_case = RunBacktestUseCase(
            market_repo, stock_repo, strategy, signal_gen
        )
        
        result = await use_case.execute(config)
        
        return result.to_dict()
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to run backtest")


@router.post("/compare")
async def compare_strategies(
    tickers: List[str],
    start_date: date,
    end_date: date,
    initial_capital: float,
    strategies: List[str],
    market_repo: MarketDataRepositoryImpl = Depends(get_market_data_repository),
    stock_repo: StockRepositoryImpl = Depends(get_stock_repository),
) -> dict:
    """複数の戦略を比較
    
    Args:
        tickers: ティッカーリスト
        start_date: 開始日
        end_date: 終了日
        initial_capital: 初期資金
        strategies: 戦略タイプのリスト
    
    Returns:
        比較結果
    """
    try:
        results = []
        
        for strategy_type in strategies:
            try:
                strategy_enum = StrategyType[strategy_type]
            except KeyError:
                continue
            
            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=Decimal(str(initial_capital)),
                strategy_type=strategy_enum,
                tickers=tickers,
            )
            
            strategy = get_trading_strategy(strategy_enum)
            signal_gen = get_signal_generator()
            
            use_case = RunBacktestUseCase(
                market_repo, stock_repo, strategy, signal_gen
            )
            
            result = await use_case.execute(config)
            results.append(result)
        
        # 結果を分析
        analyze_use_case = AnalyzeBacktestResultsUseCase()
        comparison = await analyze_use_case.execute(results)
        
        return comparison
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to compare strategies")


@router.post("/optimize")
async def optimize_parameters(
    tickers: List[str],
    start_date: date,
    end_date: date,
    initial_capital: float,
    strategy_type: str = "AI_DRIVEN",
    market_repo: MarketDataRepositoryImpl = Depends(get_market_data_repository),
    stock_repo: StockRepositoryImpl = Depends(get_stock_repository),
) -> dict:
    """パラメータを最適化
    
    Args:
        tickers: ティッカーリスト
        start_date: 開始日
        end_date: 終了日
        initial_capital: 初期資金
        strategy_type: 戦略タイプ
    
    Returns:
        最適化結果
    """
    try:
        # パラメータグリッド
        param_grid = {
            "max_positions": [10, 20, 30],
            "rebalance_frequency": ["daily", "weekly", "monthly"],
            "commission_rate": [0.0005, 0.001, 0.002],
        }
        
        best_result = None
        best_params = None
        best_sharpe = float("-inf")
        
        # グリッドサーチ
        for max_pos in param_grid["max_positions"]:
            for freq in param_grid["rebalance_frequency"]:
                for comm_rate in param_grid["commission_rate"]:
                    try:
                        strategy_enum = StrategyType[strategy_type]
                        
                        config = BacktestConfig(
                            start_date=start_date,
                            end_date=end_date,
                            initial_capital=Decimal(str(initial_capital)),
                            strategy_type=strategy_enum,
                            tickers=tickers,
                            commission_rate=Decimal(str(comm_rate)),
                            max_positions=max_pos,
                            rebalance_frequency=freq,
                        )
                        
                        strategy = get_trading_strategy(strategy_enum)
                        signal_gen = get_signal_generator()
                        
                        use_case = RunBacktestUseCase(
                            market_repo, stock_repo, strategy, signal_gen
                        )
                        
                        result = await use_case.execute(config)
                        
                        if float(result.sharpe_ratio) > best_sharpe:
                            best_sharpe = float(result.sharpe_ratio)
                            best_result = result
                            best_params = {
                                "max_positions": max_pos,
                                "rebalance_frequency": freq,
                                "commission_rate": comm_rate,
                            }
                    except Exception:
                        continue
        
        if best_result:
            return {
                "best_parameters": best_params,
                "best_sharpe_ratio": best_sharpe,
                "result": best_result.to_dict(),
            }
        else:
            raise ValueError("Optimization failed")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to optimize parameters")