"""データベースセッション管理"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from .connection import get_db_connection


class SessionManager:
    """セッション管理クラス"""
    
    def __init__(self):
        """初期化"""
        self._connection = get_db_connection()
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """セッションコンテキストマネージャー"""
        async with self._connection.get_session() as session:
            yield session
    
    async def get_session(self) -> AsyncSession:
        """セッションを取得（非推奨）"""
        # このメソッドは直接使用せず、session()コンテキストマネージャーを使用することを推奨
        if not self._connection._session_factory:
            raise RuntimeError("Database not connected")
        
        return self._connection._session_factory()


# シングルトンインスタンス
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """セッションマネージャーを取得"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """セッションを取得（依存性注入用）"""
    manager = get_session_manager()
    async with manager.session() as session:
        yield session