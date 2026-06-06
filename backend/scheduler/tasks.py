"""
Zamanlanmış Görevler — APScheduler ile otomatik veri güncelleme.

Görevler:
  fetch_bist_prices   : Borsa saatlerinde her 5 dk OHLCV çekimi
  refresh_stock_infos : Her 15 dk temel bilgi güncelleme
  fetch_sector_data   : Her 30 dk sektör endeksi güncelleme
  check_price_alerts  : Her 5 dk alert kontrol ve bildirim

Çalıştır (ayrı process):
  python -m backend.scheduler.tasks

veya API ile birlikte (lifespan içinden):
  from backend.scheduler.tasks import start_scheduler
  start_scheduler()
"""
import asyncio
import logging
import os
from datetime import datetime, time as dtime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger("finsight.scheduler")

# BIST'te takip edilecek varsayılan hisseler
DEFAULT_TICKERS = os.getenv(
    "TRACKED_TICKERS",
    "THYAO,ASELS,GARAN,SISE,EREGL,KCHOL,AKBNK,YKBNK,TOASO,FROTO",
).split(",")

# BIST saatleri (Türkiye saati = UTC+3)
BIST_OPEN_UTC  = dtime(7, 0)   # 10:00 TR = 07:00 UTC
BIST_CLOSE_UTC = dtime(15, 0)  # 18:00 TR = 15:00 UTC


def _is_bist_open() -> bool:
    """Şu an BIST açık mı? (UTC saatine göre, hafta içi)"""
    now = datetime.utcnow()
    if now.weekday() >= 5:   # Cumartesi=5, Pazar=6
        return False
    return BIST_OPEN_UTC <= now.time() <= BIST_CLOSE_UTC


# ── Görev fonksiyonları ───────────────────────────────────────────────────────
async def fetch_bist_prices():
    """
    Takip listesindeki hisselerin OHLCV verisini çekip veritabanına yazar.
    Sadece BIST açık saatlerinde çalışır.
    """
    if not _is_bist_open():
        logger.debug("BIST kapalı, fiyat çekimi atlandı.")
        return

    from data_fetcher import get_price_history
    from backend.db.database import get_db_session
    from backend.db.repositories import StockPriceRepo
    from backend.cache.redis_client import cache_delete, key_stock_info

    logger.info("Toplu fiyat güncellemesi başlıyor (%d hisse)...", len(DEFAULT_TICKERS))

    async with get_db_session() as session:
        repo = StockPriceRepo(session)
        success, fail = 0, 0

        for ticker in DEFAULT_TICKERS:
            try:
                df = get_price_history(ticker.strip(), period="5d")
                count = await repo.upsert_from_df(ticker.strip(), df)
                await cache_delete(key_stock_info(ticker.strip()))
                logger.debug("✓ %s — %d satır güncellendi", ticker, count)
                success += 1
            except Exception as e:
                logger.warning("✗ %s fiyat hatası: %s", ticker, e)
                fail += 1

    logger.info("Fiyat güncellemesi tamamlandı: %d başarılı, %d hatalı", success, fail)


async def refresh_stock_infos():
    """Temel şirket bilgilerini (F/K, PD/DD, piyasa değeri) günceller."""
    from data_fetcher import get_stock_info
    from backend.db.database import get_db_session
    from backend.db.repositories import StockInfoRepo

    logger.info("Temel bilgi güncellemesi başlıyor...")

    async with get_db_session() as session:
        repo = StockInfoRepo(session)
        for ticker in DEFAULT_TICKERS:
            try:
                data = get_stock_info(ticker.strip())
                await repo.upsert(data)
                logger.debug("✓ %s bilgisi güncellendi", ticker)
                await asyncio.sleep(0.5)   # yfinance rate limit koruması
            except Exception as e:
                logger.warning("✗ %s bilgi hatası: %s", ticker, e)

    logger.info("Temel bilgi güncellemesi tamamlandı.")


async def fetch_sector_data():
    """Sektör endeks verilerini çekip cache'i günceller."""
    from sector_analysis import get_sector_momentum
    from backend.cache.redis_client import cache_set, key_sector, TTL_SECTOR

    logger.info("Sektör verisi güncelleniyor...")
    try:
        df = get_sector_momentum(period="3mo")
        if not df.empty:
            records = df.to_dict(orient="records")
            await cache_set(key_sector("3mo"), records, ttl=TTL_SECTOR)
            logger.info("Sektör verisi güncellendi (%d sektör)", len(records))
    except Exception as e:
        logger.error("Sektör verisi güncelleme hatası: %s", e)


async def check_price_alerts():
    """
    Aktif fiyat alertlerini kontrol eder.
    Tetiklenen alertler için Telegram/email bildirimi gönderir.
    """
    from backend.db.database import get_db_session
    from backend.db.repositories import AlertRepo, StockPriceRepo

    logger.debug("Alert kontrolü çalışıyor...")

    async with get_db_session() as session:
        alert_repo = AlertRepo(session)
        price_repo = StockPriceRepo(session)

        for ticker in DEFAULT_TICKERS:
            try:
                current_price = await price_repo.get_latest_price(ticker.strip())
                if current_price is None:
                    continue

                alerts = await alert_repo.get_active_by_ticker(ticker.strip())
                for alert in alerts:
                    triggered = False

                    if alert.alert_type == "above" and current_price >= alert.threshold:
                        triggered = True
                        msg = f"📈 {ticker} fiyatı {current_price:.2f} TL — hedef {alert.threshold:.2f} TL üstüne geçti!"
                    elif alert.alert_type == "below" and current_price <= alert.threshold:
                        triggered = True
                        msg = f"📉 {ticker} fiyatı {current_price:.2f} TL — {alert.threshold:.2f} TL altına düştü!"
                    else:
                        continue

                    if triggered:
                        logger.info("ALERT tetiklendi: %s", msg)
                        await _send_notification(alert.user_id, msg)
                        await alert_repo.mark_triggered(alert.id)

            except Exception as e:
                logger.warning("Alert kontrolü hatası (%s): %s", ticker, e)


async def _send_notification(user_id: int, message: str):
    """
    Bildirim gönderici — şu an sadece log.
    Faz 2'de Telegram Bot + Email entegre edilecek.
    """
    logger.info("BİLDİRİM → user_id=%d: %s", user_id, message)
    # TODO Faz 2: telegram_bot.send(user_id, message)
    # TODO Faz 2: email_client.send(user_id, message)


# ── Scheduler kurulum ─────────────────────────────────────────────────────────
def create_scheduler() -> AsyncIOScheduler:
    """Görevleri tanımlanmış scheduler nesnesi döner."""
    scheduler = AsyncIOScheduler(timezone="UTC")

    # Fiyatları her 5 dakikada güncelle
    scheduler.add_job(
        fetch_bist_prices,
        trigger=IntervalTrigger(minutes=5),
        id="fetch_prices",
        name="BIST Fiyat Güncellemesi",
        replace_existing=True,
        misfire_grace_time=60,
    )

    # Temel bilgileri her 15 dakikada güncelle
    scheduler.add_job(
        refresh_stock_infos,
        trigger=IntervalTrigger(minutes=15),
        id="refresh_infos",
        name="Temel Bilgi Güncellemesi",
        replace_existing=True,
        misfire_grace_time=120,
    )

    # Sektör verisini her 30 dakikada güncelle
    scheduler.add_job(
        fetch_sector_data,
        trigger=IntervalTrigger(minutes=30),
        id="sector_data",
        name="Sektör Verisi Güncellemesi",
        replace_existing=True,
    )

    # Alertleri her 5 dakikada kontrol et
    scheduler.add_job(
        check_price_alerts,
        trigger=IntervalTrigger(minutes=5),
        id="check_alerts",
        name="Fiyat Alert Kontrolü",
        replace_existing=True,
        misfire_grace_time=60,
    )

    logger.info("Scheduler %d görevle yapılandırıldı.", len(scheduler.get_jobs()))
    return scheduler


_scheduler: AsyncIOScheduler | None = None


def start_scheduler() -> AsyncIOScheduler:
    """Scheduler'ı başlatır ve global referansı döner."""
    global _scheduler
    if _scheduler is None or not _scheduler.running:
        _scheduler = create_scheduler()
        _scheduler.start()
        logger.info("Scheduler başlatıldı.")
    return _scheduler


def stop_scheduler():
    """Scheduler'ı durdurur."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler durduruldu.")


# ── Standalone çalıştırma ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    )

    async def main():
        scheduler = create_scheduler()
        scheduler.start()
        logger.info("Scheduler çalışıyor. Durdurmak için Ctrl+C.")
        try:
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            scheduler.shutdown()
            logger.info("Scheduler kapatıldı.")

    asyncio.run(main())
