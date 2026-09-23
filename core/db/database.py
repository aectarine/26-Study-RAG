from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.cfg.settings import Settings


class Database:
    """SQLAlchemy 엔진과 세션 팩토리를 보유하는 애플리케이션 단위 객체입니다."""

    def __init__(self, settings: Settings):
        self.engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,  # 기존 연결 확인 → 끊어졌으면 새 연결 시도
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
        )
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def close(self) -> None:
        await self.engine.dispose()
