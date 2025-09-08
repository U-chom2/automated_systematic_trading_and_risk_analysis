"""シグナルAPIエンドポイント"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from uuid import UUID

from ....application.use_cases.signal_processing import (
    GenerateSignalsUseCase,
    ProcessSignalUseCase,
    GetActiveSignalsUseCase,
)
from ....application.dto.signal_dto import (
    SignalDTO,
    SignalResponseDTO,
    SignalBatchDTO,
)
from ....domain.services.signal_generator import SignalGenerator
from ....infrastructure.repositories.signal_repository_impl import SignalRepositoryImpl
from ....infrastructure.repositories.stock_repository_impl import StockRepositoryImpl
from ....infrastructure.repositories.market_data_repository_impl import MarketDataRepositoryImpl
from ....infrastructure.repositories.portfolio_repository_impl import PortfolioRepositoryImpl
from ....domain.services.trading_strategy import TradingStrategy, StrategyType


router = APIRouter()


def get_signal_repository():
    """シグナルリポジトリを取得"""
    return SignalRepositoryImpl()


def get_stock_repository():
    """株式リポジトリを取得"""
    return StockRepositoryImpl()


def get_market_data_repository():
    """市場データリポジトリを取得"""
    return MarketDataRepositoryImpl()


def get_portfolio_repository():
    """ポートフォリオリポジトリを取得"""
    return PortfolioRepositoryImpl()


def get_signal_generator():
    """シグナルジェネレーターを取得"""
    return SignalGenerator()


def get_trading_strategy():
    """取引戦略を取得"""
    return TradingStrategy(StrategyType.AI_DRIVEN)


@router.post("/generate", response_model=SignalBatchDTO)
async def generate_signals(
    tickers: Optional[List[str]] = None,
    signal_types: Optional[List[str]] = None,
    lookback_days: int = Query(30, ge=1, le=365),
    signal_gen: SignalGenerator = Depends(get_signal_generator),
    signal_repo: SignalRepositoryImpl = Depends(get_signal_repository),
    stock_repo: StockRepositoryImpl = Depends(get_stock_repository),
    market_repo: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> SignalBatchDTO:
    """シグナルを生成
    
    Args:
        tickers: 対象ティッカーリスト
        signal_types: シグナルタイプ
        lookback_days: 過去データ参照期間
    
    Returns:
        生成されたシグナル
    """
    try:
        use_case = GenerateSignalsUseCase(
            signal_gen, signal_repo, stock_repo, market_repo
        )
        return await use_case.execute(tickers, signal_types, lookback_days)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate signals")


@router.get("/active", response_model=List[SignalDTO])
async def get_active_signals(
    ticker: Optional[str] = Query(None, description="ティッカーシンボル"),
    signal_type: Optional[str] = Query(None, description="シグナルタイプ"),
    min_confidence: float = Query(0.0, ge=0.0, le=100.0),
    repository: SignalRepositoryImpl = Depends(get_signal_repository),
) -> List[SignalDTO]:
    """アクティブなシグナルを取得
    
    Args:
        ticker: ティッカーシンボル
        signal_type: シグナルタイプ
        min_confidence: 最小信頼度
    
    Returns:
        アクティブなシグナル
    """
    try:
        use_case = GetActiveSignalsUseCase(repository)
        return await use_case.execute(ticker, signal_type, min_confidence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get active signals")


@router.post("/{signal_id}/process", response_model=SignalResponseDTO)
async def process_signal(
    signal_id: UUID,
    portfolio_id: UUID,
    signal_repo: SignalRepositoryImpl = Depends(get_signal_repository),
    portfolio_repo: PortfolioRepositoryImpl = Depends(get_portfolio_repository),
    strategy: TradingStrategy = Depends(get_trading_strategy),
) -> SignalResponseDTO:
    """シグナルを処理
    
    Args:
        signal_id: シグナルID
        portfolio_id: ポートフォリオID
    
    Returns:
        処理結果
    """
    try:
        use_case = ProcessSignalUseCase(signal_repo, portfolio_repo, strategy)
        return await use_case.execute(signal_id, portfolio_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process signal")


@router.get("/{signal_id}", response_model=SignalDTO)
async def get_signal(
    signal_id: UUID,
    repository: SignalRepositoryImpl = Depends(get_signal_repository),
) -> SignalDTO:
    """シグナルを取得
    
    Args:
        signal_id: シグナルID
    
    Returns:
        シグナル
    """
    try:
        signal = await repository.find_by_id(signal_id)
        if not signal:
            raise HTTPException(status_code=404, detail="Signal not found")
        return SignalDTO.from_entity(signal)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get signal")


@router.delete("/{signal_id}")
async def expire_signal(
    signal_id: UUID,
    repository: SignalRepositoryImpl = Depends(get_signal_repository),
) -> dict:
    """シグナルを期限切れにする
    
    Args:
        signal_id: シグナルID
    
    Returns:
        結果
    """
    try:
        success = await repository.expire(signal_id)
        if success:
            return {"message": "Signal expired successfully"}
        else:
            raise HTTPException(status_code=404, detail="Signal not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to expire signal")