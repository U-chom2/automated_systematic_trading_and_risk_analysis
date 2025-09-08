"""市場データAPIエンドポイント"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Optional
from datetime import date

from ....application.dto.market_data_dto import (
    MarketDataRequestDTO,
    MarketDataResponseDTO,
    PriceDTO,
    OHLCVDTO,
    MarketStatusDTO,
)
from ....infrastructure.repositories.market_data_repository_impl import MarketDataRepositoryImpl


router = APIRouter()


def get_market_data_repository():
    """市場データリポジトリを取得"""
    return MarketDataRepositoryImpl()


@router.get("/price/{ticker}", response_model=PriceDTO)
async def get_latest_price(
    ticker: str,
    repository: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> PriceDTO:
    """最新価格を取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        最新価格
    """
    try:
        price = await repository.get_latest_price(ticker)
        if not price:
            raise HTTPException(status_code=404, detail="Price not found")
        return PriceDTO.from_entity(ticker, price)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get price")


@router.post("/prices", response_model=Dict[str, PriceDTO])
async def get_latest_prices(
    tickers: List[str],
    repository: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> Dict[str, PriceDTO]:
    """複数銘柄の最新価格を取得
    
    Args:
        tickers: ティッカーシンボルのリスト
    
    Returns:
        最新価格のマップ
    """
    try:
        prices = await repository.get_latest_prices(tickers)
        return {
            ticker: PriceDTO.from_entity(ticker, price)
            for ticker, price in prices.items()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get prices")


@router.get("/ohlcv/{ticker}", response_model=List[OHLCVDTO])
async def get_ohlcv(
    ticker: str,
    start_date: date = Query(..., description="開始日"),
    end_date: date = Query(..., description="終了日"),
    interval: str = Query("1d", description="インターバル"),
    repository: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> List[OHLCVDTO]:
    """OHLCVデータを取得
    
    Args:
        ticker: ティッカーシンボル
        start_date: 開始日
        end_date: 終了日
        interval: インターバル
    
    Returns:
        OHLCVデータ
    """
    try:
        ohlcv_list = await repository.get_ohlcv(ticker, start_date, end_date, interval)
        return [OHLCVDTO.from_entity(ticker, ohlcv) for ohlcv in ohlcv_list]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get OHLCV data")


@router.post("/ohlcv/batch", response_model=MarketDataResponseDTO)
async def get_multiple_ohlcv(
    request: MarketDataRequestDTO,
    repository: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> MarketDataResponseDTO:
    """複数銘柄のOHLCVデータを取得
    
    Args:
        request: 市場データリクエスト
    
    Returns:
        市場データレスポンス
    """
    try:
        request.validate()
        
        start_date = date.fromisoformat(request.start_date)
        end_date = date.fromisoformat(request.end_date)
        
        ohlcv_data = await repository.get_multiple_ohlcv(
            request.tickers, start_date, end_date, request.interval
        )
        
        return MarketDataResponseDTO.create(request, ohlcv_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get OHLCV data")


@router.get("/status", response_model=Dict[str, str])
async def get_market_status(
    repository: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> Dict[str, str]:
    """市場ステータスを取得
    
    Returns:
        市場ステータス
    """
    try:
        return await repository.get_market_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get market status")


@router.get("/hours/{exchange}")
async def get_market_hours(
    exchange: str,
    repository: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> Dict[str, str]:
    """市場時間を取得
    
    Args:
        exchange: 取引所コード
    
    Returns:
        市場時間
    """
    try:
        return await repository.get_market_hours(exchange)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get market hours")


@router.get("/indicators/{ticker}")
async def get_technical_indicators(
    ticker: str,
    indicators: List[str] = Query(..., description="指標名リスト"),
    period: int = Query(20, ge=1, le=200),
    repository: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> dict:
    """テクニカル指標を取得
    
    Args:
        ticker: ティッカーシンボル
        indicators: 指標名リスト
        period: 計算期間
    
    Returns:
        テクニカル指標
    """
    try:
        result = await repository.get_technical_indicators(ticker, indicators, period)
        return {
            "ticker": ticker,
            "period": period,
            "indicators": {
                name: [float(v) for v in values]
                for name, values in result.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get indicators")


@router.get("/company/{ticker}")
async def get_company_info(
    ticker: str,
    repository: MarketDataRepositoryImpl = Depends(get_market_data_repository),
) -> dict:
    """企業情報を取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        企業情報
    """
    try:
        info = await repository.get_company_info(ticker)
        if not info:
            raise HTTPException(status_code=404, detail="Company not found")
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get company info")