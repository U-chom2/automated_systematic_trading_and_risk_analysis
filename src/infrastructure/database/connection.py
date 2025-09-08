"""データベース接続管理"""
from typing import Optional
from contextlib import asynccontextmanager
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import redis.asyncio as redis


class DatabaseConnection:
    """データベース接続管理クラス"""
    
    def __init__(
        self,
        postgres_url: str,
        redis_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
    ):
        """初期化
        
        Args:
            postgres_url: PostgreSQL接続URL
            redis_url: Redis接続URL（オプション）
            pool_size: コネクションプールサイズ
            max_overflow: 最大オーバーフロー数
        """
        self.postgres_url = postgres_url
        self.redis_url = redis_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._redis_client: Optional[redis.Redis] = None
        self._asyncpg_pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """データベースに接続"""
        # SQLAlchemyエンジンを作成
        self._engine = create_async_engine(
            self.postgres_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_pre_ping=True,
            echo=False,  # 本番環境ではFalse
        )
        
        # セッションファクトリーを作成
        self._session_factory = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # asyncpgプールを作成（高速な直接クエリ用）
        self._asyncpg_pool = await asyncpg.create_pool(
            self.postgres_url.replace("postgresql+asyncpg://", "postgresql://"),
            min_size=5,
            max_size=self.pool_size,
        )
        
        # Redis接続を作成
        if self.redis_url:
            self._redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
    
    async def disconnect(self) -> None:
        """データベース接続を閉じる"""
        if self._engine:
            await self._engine.dispose()
        
        if self._asyncpg_pool:
            await self._asyncpg_pool.close()
        
        if self._redis_client:
            await self._redis_client.close()
    
    @asynccontextmanager
    async def get_session(self):
        """セッションを取得"""
        if not self._session_factory:
            raise RuntimeError("Database not connected")
        
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def execute_raw(self, query: str, *args) -> list:
        """生のSQLクエリを実行"""
        if not self._asyncpg_pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._asyncpg_pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def execute_many(self, query: str, args_list: list) -> None:
        """複数の引数で同じクエリを実行"""
        if not self._asyncpg_pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._asyncpg_pool.acquire() as connection:
            await connection.executemany(query, args_list)
    
    @property
    def redis(self) -> Optional[redis.Redis]:
        """Redisクライアントを取得"""
        return self._redis_client
    
    @property
    def engine(self) -> Optional[AsyncEngine]:
        """SQLAlchemyエンジンを取得"""
        return self._engine
    
    async def create_tables(self) -> None:
        """テーブルを作成"""
        if not self._engine:
            raise RuntimeError("Database not connected")
        
        from .models import Base
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self) -> None:
        """テーブルを削除"""
        if not self._engine:
            raise RuntimeError("Database not connected")
        
        from .models import Base
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def health_check(self) -> bool:
        """ヘルスチェック"""
        try:
            # PostgreSQLのチェック
            if self._asyncpg_pool:
                async with self._asyncpg_pool.acquire() as connection:
                    await connection.fetchval("SELECT 1")
            
            # Redisのチェック
            if self._redis_client:
                await self._redis_client.ping()
            
            return True
        except Exception:
            return False


# シングルトンインスタンス
_db_connection: Optional[DatabaseConnection] = None


def get_db_connection() -> DatabaseConnection:
    """データベース接続を取得"""
    global _db_connection
    if _db_connection is None:
        raise RuntimeError("Database connection not initialized")
    return _db_connection


async def initialize_database(
    postgres_url: str,
    redis_url: Optional[str] = None,
    **kwargs
) -> DatabaseConnection:
    """データベースを初期化"""
    global _db_connection
    
    if _db_connection is not None:
        await _db_connection.disconnect()
    
    _db_connection = DatabaseConnection(postgres_url, redis_url, **kwargs)
    await _db_connection.connect()
    
    return _db_connection


async def close_database() -> None:
    """データベース接続を閉じる"""
    global _db_connection
    
    if _db_connection is not None:
        await _db_connection.disconnect()
        _db_connection = None