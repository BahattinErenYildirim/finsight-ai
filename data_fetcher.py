"""
Data fetcher — BIST equities via yfinance.
Returns fundamentals (P/E, P/B, etc.).

Caching: functools.lru_cache per ticker.
Errors: external calls wrapped in try/except.
Retry: automatic retries on network errors.
"""
import sys
import time
import logging
import functools
import threading
import yfinance as yf
import pandas as pd
from config import BIST_SUFFIX

logger = logging.getLogger("finsight.data")


def _safe_exc_msg(exc: BaseException) -> str:
    """Log satırlarında Windows charmap hatasını önler."""
    msg = str(exc)
    try:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        msg.encode(enc)
        return msg
    except (UnicodeEncodeError, LookupError, AttributeError):
        return msg.encode("ascii", errors="replace").decode("ascii")


def _normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """yf.download çoklu sütun formatını history() ile uyumlu hale getirir."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.droplevel(1)
    out.columns = [str(c).title() for c in out.columns]
    return out


def _fetch_yf_info(stock: yf.Ticker, symbol: str, ticker: str) -> dict:
    """
    yfinance .info — Windows + Türkçe klasör yolunda charmap hatası verebilir.
    Bu durumda fast_info veya kısa fiyat geçmişi ile yedeklenir.
    """
    try:
        info = stock.info or {}
        if info:
            return info
    except UnicodeEncodeError:
        logger.warning(
            "yfinance info encoding hatasi (%s), yedek yontem deneniyor.", ticker
        )
    except Exception as e:
        logger.error("yfinance info hatasi (%s): %s", ticker, _safe_exc_msg(e))

    # Yedek 1: fast_info
    try:
        fast = getattr(stock, "fast_info", None)
        if fast:
            last = getattr(fast, "last_price", None) or getattr(fast, "lastPrice", None)
            if last is not None:
                return {
                    "longName": ticker,
                    "currentPrice": float(last),
                    "regularMarketPrice": float(last),
                }
    except Exception:
        pass

    # Yedek 2: son kapanış fiyatı
    try:
        hist = yf.download(
            symbol,
            period="5d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        hist = _normalize_history_df(hist)
        if hist is not None and not hist.empty and "Close" in hist.columns:
            last_close = float(hist["Close"].iloc[-1])
            return {
                "longName": ticker,
                "currentPrice": last_close,
                "regularMarketPrice": last_close,
            }
    except Exception as e:
        logger.debug("Fiyat yedegi basarisiz (%s): %s", ticker, _safe_exc_msg(e))

    return {}


def _fetch_yf_history(stock: yf.Ticker, symbol: str, period: str, ticker: str) -> pd.DataFrame:
    """OHLCV — history() bos donerse download() ile dener."""
    try:
        df = stock.history(period=period)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.debug("history() hatasi (%s): %s", ticker, _safe_exc_msg(e))

    try:
        df = yf.download(
            symbol,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        df = _normalize_history_df(df)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.error("download() hatasi (%s): %s", ticker, _safe_exc_msg(e))

    return pd.DataFrame()

# ── Retry decorator ──────────────────────────────────────────────────────────
def _retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Ağ hatalarında otomatik tekrar deneme decorator'ı."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError, OSError) as e:
                    last_exc = e
                    wait = delay * (backoff ** (attempt - 1))
                    logger.warning(
                        "%s — Deneme %d/%d başarısız (%s). %.1fs sonra tekrar...",
                        func.__name__, attempt, max_retries, e, wait,
                    )
                    time.sleep(wait)
                except Exception:
                    # Ağ dışı hatalar doğrudan yükseltilir
                    raise
            raise last_exc  # type: ignore
        return wrapper
    return decorator


# ── In-memory cache (TTL 5 dk) ───────────────────────────────────────────────
_info_cache: dict[str, tuple[float, dict]] = {}
_history_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 300  # 5 dakika


@_retry(max_retries=3, delay=1.0)
def get_stock_info(ticker: str) -> dict:
    """
    Hisse senedinin temel bilgilerini ve finansal göstergelerini döndürür.

    Args:
        ticker: BIST hisse kodu (Örn: "THYAO", "SISE")

    Returns:
        Hisse bilgilerini içeren sözlük

    Raises:
        ValueError: Hisse bulunamazsa
    """
    ticker = ticker.strip().upper()
    cache_key = ticker

    # Cache kontrolü
    with _cache_lock:
        if cache_key in _info_cache:
            ts, cached = _info_cache[cache_key]
            if time.time() - ts < _CACHE_TTL:
                logger.debug("Cache hit: %s stock info", ticker)
                return cached

    symbol = f"{ticker}{BIST_SUFFIX}"
    stock = yf.Ticker(symbol)
    info = _fetch_yf_info(stock, symbol, ticker)

    # Boş/geçersiz yanıt kontrolü
    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        logger.warning("Boş veri döndü: %s — yfinance info boş olabilir.", ticker)

    # ── Demo Sektör Verileri (Ortalama F/K) ──
    # Not: yfinance sektör isimleri İngilizcedir.
    sector_pe_means = {
        "Financial Services": 6.5,
        "Industrials": 14.2,
        "Basic Materials": 10.8,
        "Consumer Cyclical": 12.0,
        "Technology": 18.5,
        "Healthcare": 22.1,
        "Energy": 8.4,
        "Communication Services": 11.2,
        "Utilities": 9.5,
        "Real Estate": 5.2,
        "Consumer Defensive": 15.3,
    }
    
    sector = info.get("sector", "Bilinmiyor")

    result = {
        "hisse_kodu": ticker,
        "sirket_adi": info.get("longName") or info.get("shortName", "Bilinmiyor"),
        "son_fiyat": _safe_round(info.get("currentPrice") or info.get("regularMarketPrice")),
        "sektor": sector,
        "sektor_fk_ort": sector_pe_means.get(sector, "Bilinmiyor"),
        "piyasa_degeri": info.get("marketCap"),
        "fk_orani": _safe_round(info.get("trailingPE") or info.get("forwardPE")),
        "pddd_orani": _safe_round(info.get("priceToBook")),
        "net_kar_buyumesi": _safe_round(
            info.get("earningsGrowth") and info.get("earningsGrowth") * 100
        ),
        "borc_favok": _format_debt_ebitda(info),
    }

    # Cache'e yaz
    with _cache_lock:
        _info_cache[cache_key] = (time.time(), result)
    return result


@_retry(max_retries=3, delay=1.0)
def get_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Hisse fiyat geçmişini DataFrame olarak döndürür.

    Args:
        ticker: BIST hisse kodu
        period: Zaman dilimi (1mo, 3mo, 6mo, 1y, vb.)

    Returns:
        OHLCV verileri içeren DataFrame

    Raises:
        ValueError: Fiyat verisi bulunamazsa
    """
    ticker = ticker.strip().upper()
    cache_key = f"{ticker}_{period}"

    # Cache kontrolü
    with _cache_lock:
        if cache_key in _history_cache:
            ts, cached = _history_cache[cache_key]
            if time.time() - ts < _CACHE_TTL:
                logger.debug("Cache hit: %s price history", ticker)
                return cached

    symbol = f"{ticker}{BIST_SUFFIX}"
    stock = yf.Ticker(symbol)
    df = _fetch_yf_history(stock, symbol, period, ticker)

    if df is None or df.empty:
        raise ValueError(
            f"{ticker} için fiyat verisi bulunamadı. "
            "Hisse kodu doğru mu? (Örn: THYAO, ASELS, GARAN)"
        )

    # Cache'e yaz
    with _cache_lock:
        _history_cache[cache_key] = (time.time(), df)
    return df


def clear_cache():
    """Tüm in-memory cache'i temizler."""
    with _cache_lock:
        _info_cache.clear()
        _history_cache.clear()
    logger.info("Data cache temizlendi.")


def _safe_round(value, decimals: int = 2):
    """Değeri güvenli biçimde yuvarlar, None ise 'Yetersiz Veri' döner."""
    if value is None:
        return "Yetersiz Veri"
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return "Yetersiz Veri"


def _format_debt_ebitda(info: dict) -> str:
    """Borç/FAVÖK oranını hesaplar."""
    total_debt = info.get("totalDebt")
    ebitda = info.get("ebitda")
    if total_debt and ebitda and ebitda != 0:
        return str(round(total_debt / ebitda, 2))
    return "Yetersiz Veri"
