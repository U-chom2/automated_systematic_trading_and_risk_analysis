"""ポートフォリオAPIエンドポイント"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from ....application.use_cases.portfolio_management import (
    CreatePortfolioUseCase,
    UpdatePortfolioUseCase,
    GetPortfolioUseCase,
    ListPortfoliosUseCase,
    DeletePortfolioUseCase,
    GetPortfolioPerformanceUseCase,
)
from ....application.dto.portfolio_dto import (
    PortfolioDTO,
    CreatePortfolioDTO,
    UpdatePortfolioDTO,
    PortfolioSummaryDTO,
)
from ....infrastructure.repositories.portfolio_repository_impl import PortfolioRepositoryImpl
from ....infrastructure.repositories.market_data_repository_impl import MarketDataRepositoryImpl


router = APIRouter()


def get_portfolio_repository():
    """ポートフォリオリポジトリを取得"""
    return PortfolioRepositoryImpl()


def get_market_data_repository():
    """市場データリポジトリを取得"""
    return MarketDataRepositoryImpl()


@router.post("/", response_model=PortfolioDTO)
async def create_portfolio(
    dto: CreatePortfolioDTO,
    repository: PortfolioRepositoryImpl = Depends(get_portfolio_repository),
) -> PortfolioDTO:
    """ポートフォリオを作成
    
    Args:
        dto: ポートフォリオ作成DTO
    
    Returns:
        作成されたポートフォリオ
    """
    try:
        dto.validate()
        use_case = CreatePortfolioUseCase(repository)
        return await use_case.execute(dto)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create portfolio")


@router.get("/", response_model=List[PortfolioSummaryDTO])
async def list_portfolios(
    active_only: bool = Query(False, description="アクティブなもののみ"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    repository: PortfolioRepositoryImpl = Depends(get_portfolio_repository),
) -> List[PortfolioSummaryDTO]:
    """ポートフォリオ一覧を取得
    
    Args:
        active_only: アクティブなもののみ
        limit: 取得件数上限
        offset: オフセット
    
    Returns:
        ポートフォリオ一覧
    """
    try:
        use_case = ListPortfoliosUseCase(repository)
        portfolios = await use_case.execute(active_only, limit, offset)
        return [PortfolioSummaryDTO.from_portfolio_dto(p) for p in portfolios]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list portfolios")


@router.get("/{portfolio_id}", response_model=PortfolioDTO)
async def get_portfolio(
    portfolio_id: UUID,
    include_positions: bool = Query(True, description="ポジション情報を含める"),
    update_prices: bool = Query(False, description="最新価格で更新"),
    portfolio_repo: PortfolioRepositoryImpl = Depends(get_portfolio_repository),
    market_repo: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> PortfolioDTO:
    """ポートフォリオを取得
    
    Args:
        portfolio_id: ポートフォリオID
        include_positions: ポジション情報を含める
        update_prices: 最新価格で更新
    
    Returns:
        ポートフォリオ
    """
    try:
        use_case = GetPortfolioUseCase(portfolio_repo, market_repo)
        portfolio = await use_case.execute(portfolio_id, include_positions, update_prices)
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return portfolio
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get portfolio")


@router.put("/{portfolio_id}", response_model=PortfolioDTO)
async def update_portfolio(
    portfolio_id: UUID,
    dto: UpdatePortfolioDTO,
    repository: PortfolioRepositoryImpl = Depends(get_portfolio_repository),
) -> PortfolioDTO:
    """ポートフォリオを更新
    
    Args:
        portfolio_id: ポートフォリオID
        dto: 更新DTO
    
    Returns:
        更新されたポートフォリオ
    """
    try:
        dto.validate()
        use_case = UpdatePortfolioUseCase(repository)
        return await use_case.execute(portfolio_id, dto)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update portfolio")


@router.delete("/{portfolio_id}")
async def delete_portfolio(
    portfolio_id: UUID,
    archive: bool = Query(True, description="アーカイブするか"),
    repository: PortfolioRepositoryImpl = Depends(get_portfolio_repository),
) -> dict:
    """ポートフォリオを削除
    
    Args:
        portfolio_id: ポートフォリオID
        archive: アーカイブするか（Falseの場合は完全削除）
    
    Returns:
        削除結果
    """
    try:
        use_case = DeletePortfolioUseCase(repository)
        success = await use_case.execute(portfolio_id, archive)
        
        if success:
            return {"message": "Portfolio deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete portfolio")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete portfolio")


@router.get("/{portfolio_id}/performance")
async def get_portfolio_performance(
    portfolio_id: UUID,
    start_date: datetime = Query(..., description="開始日"),
    end_date: datetime = Query(..., description="終了日"),
    portfolio_repo: PortfolioRepositoryImpl = Depends(get_portfolio_repository),
    market_repo: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> dict:
    """ポートフォリオのパフォーマンスを取得
    
    Args:
        portfolio_id: ポートフォリオID
        start_date: 開始日
        end_date: 終了日
    
    Returns:
        パフォーマンスデータ
    """
    try:
        use_case = GetPortfolioPerformanceUseCase(portfolio_repo, market_repo)
        return await use_case.execute(portfolio_id, start_date, end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get performance")


@router.post("/{portfolio_id}/rebalance")
async def rebalance_portfolio(
    portfolio_id: UUID,
    target_allocations: dict,
    portfolio_repo: PortfolioRepositoryImpl = Depends(get_portfolio_repository),
) -> dict:
    """ポートフォリオをリバランス
    
    Args:
        portfolio_id: ポートフォリオID
        target_allocations: 目標アロケーション
    
    Returns:
        リバランス結果
    """
    # TODO: リバランスユースケースを実装
    return {
        "message": "Rebalancing scheduled",
        "portfolio_id": str(portfolio_id),
        "target_allocations": target_allocations,
    }