"""ヘルスチェックエンドポイント"""
from fastapi import APIRouter, Depends
from typing import Dict, Any

from ....infrastructure.database.connection import get_db_connection


router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """ヘルスチェック
    
    Returns:
        ヘルスステータス
    """
    return {
        "status": "healthy",
        "service": "trading-api",
    }


@router.get("/health/db")
async def database_health() -> Dict[str, Any]:
    """データベースヘルスチェック
    
    Returns:
        データベースステータス
    """
    try:
        db = get_db_connection()
        is_healthy = await db.health_check()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "database": {
                "postgres": "connected" if is_healthy else "disconnected",
                "redis": "connected" if db.redis else "not configured",
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/health/ready")
async def readiness_check() -> Dict[str, str]:
    """レディネスチェック
    
    Returns:
        レディネスステータス
    """
    # すべてのサービスが準備完了かチェック
    try:
        db = get_db_connection()
        is_ready = await db.health_check()
        
        if is_ready:
            return {"status": "ready"}
        else:
            return {"status": "not ready"}
    except Exception:
        return {"status": "not ready"}