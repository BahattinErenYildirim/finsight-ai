"""
Redis Cache — Hızlı erişim katmanı.

Kullanım alanları:
  - Hisse fiyatları     : 5 dakika TTL
  - AI analiz raporları : 10 dakika TTL
  - Sektör verileri     : 15 dakika TTL
  - Rate limit sayaçları: 1 dakika pencere

Redis yoksa (geliştirme ortamı) otomatik olarak
bellek içi sözlük kullanan InMemoryCache'e düşer.
"""
import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("finsight.cache")

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── TTL sabitleri (saniye) ────────────────────────────────────────────────────
TTL_PRICE       = 300   # 5 dk  — canlı fiyat
TTL_REPORT      = 600   # 10 dk — AI raporu
TTL_SECTOR      = 900   # 15 dk — sektör rotasyonu
TTL_STOCK_INFO  = 300   # 5 dk  — temel bilgiler
TTL_RATE_LIMIT  = 60    # 1 dk  — rate limit penceresi


class _InMemoryCache:
    """Redis yokken kullanılan basit TTL cache (thread-safe değil, tek işlem için)."""

    def __init__(self):
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        self._store[key] = (time.time() + ttl, value)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def incr(self, key: str, ttl: int = 60) -> int:
        entry = self._store.get(key)
        if entry is None or time.time() > entry[0]:
            self._store[key] = (time.time() + ttl, 1)
            return 1
        expires_at, val = entry
        new_val = val + 1
        self._store[key] = (expires_at, new_val)
        return new_val

    def clear(self) -> None:
        self._store.clear()


# ── Redis veya InMemory seçimi ────────────────────────────────────────────────
try:
    import redis.asyncio as aioredis
    _redis_client: Optional[aioredis.Redis] = None

    async def _get_redis() -> aioredis.Redis:
        global _redis_client
        if _redis_client is None:
            _redis_client = aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
            )
        return _redis_client

    _USE_REDIS = True
    logger.info("Redis modülü bulundu — %s", REDIS_URL)
except ImportError:
    _USE_REDIS = False
    logger.warning("redis paketi yok, InMemoryCache kullanılıyor.")

_mem_cache = _InMemoryCache()


# ── Genel API ─────────────────────────────────────────────────────────────────
async def cache_get(key: str) -> Optional[Any]:
    """Cache'den değer okur (JSON decode eder). Yoksa None döner."""
    if _USE_REDIS:
        try:
            r = await _get_redis()
            raw = await r.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception as e:
            logger.warning("Redis get hatası (%s): %s — fallback", key, e)

    raw = _mem_cache.get(key)
    return raw


async def cache_set(key: str, value: Any, ttl: int = TTL_PRICE) -> None:
    """Cache'e JSON serileştirerek yazar."""
    if _USE_REDIS:
        try:
            r = await _get_redis()
            await r.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
            return
        except Exception as e:
            logger.warning("Redis set hatası (%s): %s — fallback", key, e)

    _mem_cache.set(key, value, ttl=ttl)


async def cache_delete(key: str) -> None:
    """Cache'den siler."""
    if _USE_REDIS:
        try:
            r = await _get_redis()
            await r.delete(key)
            return
        except Exception as e:
            logger.warning("Redis delete hatası (%s): %s", key, e)

    _mem_cache.delete(key)


async def rate_limit_check(identifier: str, max_requests: int = 60) -> tuple[bool, int]:
    """
    Basit sliding window rate limiter.

    Args:
        identifier: Kullanıcı IP veya user_id bazlı anahtar
        max_requests: Pencere başına maksimum istek sayısı

    Returns:
        (allowed: bool, current_count: int)
    """
    key = f"rl:{identifier}"

    if _USE_REDIS:
        try:
            r = await _get_redis()
            count = await r.incr(key)
            if count == 1:
                await r.expire(key, TTL_RATE_LIMIT)
            return count <= max_requests, count
        except Exception as e:
            logger.warning("Rate limit Redis hatası: %s", e)

    count = _mem_cache.incr(key, ttl=TTL_RATE_LIMIT)
    return count <= max_requests, count


# ── Yardımcı key şablonları ───────────────────────────────────────────────────
def key_stock_info(ticker: str) -> str:
    return f"info:{ticker.upper()}"

def key_price_df(ticker: str, period: str) -> str:
    return f"price:{ticker.upper()}:{period}"

def key_report(ticker: str) -> str:
    return f"report:{ticker.upper()}"

def key_sector(period: str) -> str:
    return f"sector:{period}"

def key_technicals(ticker: str) -> str:
    return f"ta:{ticker.upper()}"
