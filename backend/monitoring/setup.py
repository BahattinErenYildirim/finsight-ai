"""
Monitoring Kurulumu — Sentry hata izleme + Prometheus metrikler.

Özel metrikler:
  finsight_analysis_requests_total    : Analiz isteği sayacı (ticker, status)
  finsight_analysis_duration_seconds  : Analiz süresi histogramı
  finsight_gemini_requests_total      : Gemini API çağrı sayacı
  finsight_cache_hits_total           : Cache hit/miss sayacı
  finsight_active_users               : Anlık aktif kullanıcı sayısı (gauge)
"""
import logging
import time
from functools import wraps
from typing import Callable

logger = logging.getLogger("finsight.monitoring")

# ── Prometheus ────────────────────────────────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram, Gauge

    analysis_requests = Counter(
        "finsight_analysis_requests_total",
        "Hisse analizi isteği sayısı",
        ["ticker", "status"],           # status: success | error | cached
    )

    analysis_duration = Histogram(
        "finsight_analysis_duration_seconds",
        "Hisse analizi toplam süresi (saniye)",
        ["ticker"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )

    gemini_requests = Counter(
        "finsight_gemini_requests_total",
        "Gemini API çağrı sayısı",
        ["model", "status"],            # status: success | error | rate_limit
    )

    cache_operations = Counter(
        "finsight_cache_hits_total",
        "Cache işlemi sayısı",
        ["operation", "result"],        # operation: get|set, result: hit|miss
    )

    active_users = Gauge(
        "finsight_active_users",
        "Anlık aktif kullanıcı sayısı (oturum açmış)",
    )

    scheduler_runs = Counter(
        "finsight_scheduler_runs_total",
        "Zamanlanmış görev çalışma sayısı",
        ["job", "status"],
    )

    _PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus metrikleri tanımlandı.")

except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client yok — metrikler devre dışı.")

    # Noop sınıflar — kod her ortamda çalışsın
    class _Noop:
        def labels(self, **kwargs): return self
        def inc(self, *a, **kw): pass
        def observe(self, *a, **kw): pass
        def set(self, *a, **kw): pass
        def time(self): return _NoopCtx()

    class _NoopCtx:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    analysis_requests = _Noop()
    analysis_duration = _Noop()
    gemini_requests   = _Noop()
    cache_operations  = _Noop()
    active_users      = _Noop()
    scheduler_runs    = _Noop()


# ── Dekoratörler ─────────────────────────────────────────────────────────────
def track_analysis(ticker_param: str = "ticker"):
    """
    Analiz endpoint'leri için metrik dekoratörü.

    Kullanım:
        @router.get("/{ticker}")
        @track_analysis()
        async def get_stock(ticker: str): ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ticker = kwargs.get(ticker_param, "unknown").upper()
            start  = time.perf_counter()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.perf_counter() - start
                analysis_requests.labels(ticker=ticker, status=status).inc()
                analysis_duration.labels(ticker=ticker).observe(duration)
        return wrapper
    return decorator


def track_gemini(model: str = "gemini-2.0-flash"):
    """Gemini API çağrılarını izlemek için dekoratör."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            status = "success"
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err = str(e).lower()
                status = "rate_limit" if "429" in err else "error"
                raise
            finally:
                gemini_requests.labels(model=model, status=status).inc()
        return wrapper
    return decorator


def track_scheduler_job(job_name: str):
    """Scheduler görevlerini izlemek için dekoratör."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            status = "success"
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                status = "error"
                logger.error("Scheduler görevi başarısız (%s): %s", job_name, e)
                raise
            finally:
                scheduler_runs.labels(job=job_name, status=status).inc()
        return wrapper
    return decorator


# ── Cache metrik yardımcısı ───────────────────────────────────────────────────
def record_cache_hit(operation: str = "get"):
    cache_operations.labels(operation=operation, result="hit").inc()

def record_cache_miss(operation: str = "get"):
    cache_operations.labels(operation=operation, result="miss").inc()
