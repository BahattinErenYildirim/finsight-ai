"""
Portfolio Persistence — finsight_dev.db (SQLite) ile watchlist yonetimi.
Streamlit session_state ile senkronize calisir; uygulama yeniden baslasa
bile portfoy verisi korunur.

Mevcut 'watchlist' tablosuna uyumlu: added_at opsiyonel (varsa kullanilir).
"""
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger("finsight.db")

DB_PATH = Path(__file__).parent / "finsight_dev.db"


def _get_conn() -> sqlite3.Connection:
    """Thread-safe SQLite baglantisi dondurur."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Tabloda belirtilen sutunun var olup olmadigini kontrol eder."""
    try:
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row["name"] == column for row in info)
    except Exception:
        return False


def init_db() -> None:
    """Watchlist tablosunu olusturur veya eksik sutunlari ekler."""
    try:
        conn = _get_conn()
        # Tablo yoksa olustur
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker   TEXT    UNIQUE NOT NULL
            )
        """)
        # added_at sutunu yoksa ekle (migration)
        if not _has_column(conn, "watchlist", "added_at"):
            conn.execute("ALTER TABLE watchlist ADD COLUMN added_at REAL DEFAULT 0")
        conn.commit()
        conn.close()
        logger.debug("DB init OK: %s", DB_PATH)
    except Exception as e:
        logger.error("init_db hatasi: %s", e)


def get_watchlist() -> list[str]:
    """Kayitli tum ticker'lari ekleme sirasiyla dondurur."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT ticker FROM watchlist ORDER BY id ASC"
        ).fetchall()
        conn.close()
        return [r["ticker"] for r in rows]
    except Exception as e:
        logger.warning("get_watchlist hatasi: %s", e)
        return []


def add_ticker(ticker: str) -> bool:
    """
    Watchlist'e ticker ekler.
    Zaten mevcutsa sessizce atlar (INSERT OR IGNORE), True doner.
    """
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)",
            (ticker.strip().upper(),),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error("add_ticker hatasi (%s): %s", ticker, e)
        return False


def remove_ticker(ticker: str) -> None:
    """Ticker'i watchlist'ten siler."""
    try:
        conn = _get_conn()
        conn.execute(
            "DELETE FROM watchlist WHERE ticker = ?",
            (ticker.strip().upper(),),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("remove_ticker hatasi (%s): %s", ticker, e)


def clear_watchlist() -> None:
    """Tum watchlist kayitlarini siler."""
    try:
        conn = _get_conn()
        conn.execute("DELETE FROM watchlist")
        conn.commit()
        conn.close()
        logger.info("Watchlist temizlendi.")
    except Exception as e:
        logger.error("clear_watchlist hatasi: %s", e)
