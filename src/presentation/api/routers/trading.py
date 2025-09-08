"""取引APIエンドポイント"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from ....application.use_cases.trading import (
    ExecuteTradeUseCase,
    CancelTradeUseCase,
    GetTradeHistoryUseCase,
    GetPendingTradesUseCase,
    GetTradeStatisticsUseCase,
)
from ....application.dto.trade_dto import (
    TradeDTO,
    CreateTradeDTO,
    TradeSummaryDTO,
)
from ....infrastructure.repositories.trade_repository_impl import TradeRepositoryImpl
from ....infrastructure.repositories.portfolio_repository_impl import PortfolioRepositoryImpl
from ....infrastructure.repositories.market_data_repository_impl import MarketDataRepositoryImpl


router = APIRouter()


def get_trade_repository():
    """取引リポジトリを取得"""
    return TradeRepositoryImpl()


def get_portfolio_repository():
    """ポートフォリオリポジトリを取得"""
    return PortfolioRepositoryImpl()


def get_market_data_repository():
    """市場データリポジトリを取得"""
    return MarketDataRepositoryImpl()


@router.post("/", response_model=TradeDTO)
async def execute_trade(
    dto: CreateTradeDTO,
    trade_repo: TradeRepositoryImpl = Depends(get_trade_repository),
    portfolio_repo: PortfolioRepositoryImpl = Depends(get_portfolio_repository),
    market_repo: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> TradeDTO:
    """取引を実行
    
    Args:
        dto: 取引作成DTO
    
    Returns:
        実行された取引
    """
    try:
        dto.validate()
        use_case = ExecuteTradeUseCase(trade_repo, portfolio_repo, market_repo)
        return await use_case.execute(dto)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to execute trade")


@router.get("/", response_model=List[TradeDTO])
async def get_trade_history(
    portfolio_id: Optional[UUID] = Query(None, description="ポートフォリオID"),
    ticker: Optional[str] = Query(None, description="ティッカーシンボル"),
    start_date: Optional[datetime] = Query(None, description="開始日"),
    end_date: Optional[datetime] = Query(None, description="終了日"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    repository: TradeRepositoryImpl = Depends(get_trade_repository),
) -> List[TradeDTO]:
    """取引履歴を取得
    
    Args:
        portfolio_id: ポートフォリオID
        ticker: ティッカーシンボル
        start_date: 開始日
        end_date: 終了日
        limit: 取得件数上限
        offset: オフセット
    
    Returns:
        取引履歴
    """
    try:
        use_case = GetTradeHistoryUseCase(repository)
        return await use_case.execute(
            portfolio_id, ticker, start_date, end_date, limit, offset
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get trade history")


@router.get("/pending", response_model=List[TradeDTO])
async def get_pending_trades(
    portfolio_id: Optional[UUID] = Query(None, description="ポートフォリオID"),
    repository: TradeRepositoryImpl = Depends(get_trade_repository),
) -> List[TradeDTO]:
    """待機中の取引を取得
    
    Args:
        portfolio_id: ポートフォリオID
    
    Returns:
        待機中の取引
    """
    try:
        use_case = GetPendingTradesUseCase(repository)
        return await use_case.execute(portfolio_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get pending trades")


@router.get("/{trade_id}", response_model=TradeDTO)
async def get_trade(
    trade_id: UUID,
    repository: TradeRepositoryImpl = Depends(get_trade_repository),
) -> TradeDTO:
    """取引を取得
    
    Args:
        trade_id: 取引ID
    
    Returns:
        取引
    """
    try:
        trade = await repository.find_by_id(trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        return TradeDTO.from_entity(trade)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get trade")


@router.post("/{trade_id}/cancel")
async def cancel_trade(
    trade_id: UUID,
    repository: TradeRepositoryImpl = Depends(get_trade_repository),
) -> dict:
    """取引をキャンセル
    
    Args:
        trade_id: 取引ID
    
    Returns:
        キャンセル結果
    """
    try:
        use_case = CancelTradeUseCase(repository)
        success = await use_case.execute(trade_id)
        
        if success:
            return {"message": "Trade cancelled successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to cancel trade")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to cancel trade")


@router.get("/statistics/{portfolio_id}", response_model=TradeSummaryDTO)
async def get_trade_statistics(
    portfolio_id: UUID,
    start_date: Optional[datetime] = Query(None, description="開始日"),
    end_date: Optional[datetime] = Query(None, description="終了日"),
    repository: TradeRepositoryImpl = Depends(get_trade_repository),
) -> TradeSummaryDTO:
    """取引統計を取得
    
    Args:
        portfolio_id: ポートフォリオID
        start_date: 開始日
        end_date: 終了日
    
    Returns:
        取引統計
    """
    try:
        use_case = GetTradeStatisticsUseCase(repository)
        return await use_case.execute(portfolio_id, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@router.post("/batch")
async def execute_batch_trades(
    trades: List[CreateTradeDTO],
    trade_repo: TradeRepositoryImpl = Depends(get_trade_repository),
    portfolio_repo: PortfolioRepositoryImpl = Depends(get_portfolio_repository),
    market_repo: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> dict:
    """複数の取引を一括実行
    
    Args:
        trades: 取引作成DTOのリスト
    
    Returns:
        実行結果
    """
    try:
        # 各取引をバリデーション
        for trade in trades:
            trade.validate()
        
        # TODO: BatchExecuteTradesUseCaseを実装して使用
        executed = []
        failed = []
        
        use_case = ExecuteTradeUseCase(trade_repo, portfolio_repo, market_repo)
        
        for trade_dto in trades:
            try:
                result = await use_case.execute(trade_dto)
                executed.append(result)
            except Exception as e:
                failed.append({
                    "trade": trade_dto.dict(),
                    "error": str(e)
                })
        
        return {
            "executed": len(executed),
            "failed": len(failed),
            "details": {
                "executed_trades": executed,
                "failed_trades": failed,
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to execute batch trades")