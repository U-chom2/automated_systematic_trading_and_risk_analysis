"""FastAPIアプリケーション"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from ...infrastructure.database.connection import initialize_database, close_database
from .routers import health


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    # 起動時の処理
    logger.info("Starting application...")
    
    # データベース接続を初期化
    try:
        # 環境変数から接続情報を取得（実際には設定ファイルから）
        postgres_url = "postgresql+asyncpg://user:password@localhost/trading_db"
        redis_url = "redis://localhost:6379"
        
        await initialize_database(postgres_url, redis_url)
        logger.info("Database connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    yield
    
    # 終了時の処理
    logger.info("Shutting down application...")
    await close_database()
    logger.info("Database connection closed")


def create_app() -> FastAPI:
    """FastAPIアプリケーションを作成
    
    Returns:
        FastAPIアプリケーション
    """
    app = FastAPI(
        title="Automated Trading System API",
        description="AI駆動の自動売買システムAPI",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 本番環境では制限する
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # エラーハンドラー
    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    # ルーターを登録
    app.include_router(health.router, tags=["health"])
    
    @app.get("/")
    async def root():
        """ルートエンドポイント"""
        return {
            "message": "Automated Trading System API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
        }
    
    return app


# アプリケーションインスタンス
app = create_app()