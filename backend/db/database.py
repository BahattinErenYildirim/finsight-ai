"""
Veritabanı Bağlantısı — SQLAlchemy async engine + session factory.

Desteklenen DB: PostgreSQL (production), SQLite (test/geliştirme)
Bağlantı dizisi .env'den okunur:
  DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/finsight
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from dotenv import load_dotenv

from backend.db.models import Base

load_dotenv()
logger = logging.getLogger("finsight.db")

# ── Bağlantı dizisi ───────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./finsight_dev.db",   # Geliştirme varsayılanı
)

# ── Async engine ──────────────────────────────────────────────────────────────
_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DB_ECHO", "false").lower() == "true",
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    connect_args=_connect_args,
)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Tabloları oluşturur (ilk çalıştırma veya migration yoksa)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Veritabanı tabloları hazır.")


async def drop_db() -> None:
    """Tüm tabloları siler — sadece test ortamında kullanın."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("Tüm tablolar silindi.")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager — manuel kullanım için.

    Örnek:
        async with get_db_session() as session:
            result = await session.execute(select(StockInfo))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency injection için.

    Örnek:
        @router.get("/stocks")
        async def get_stocks(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
