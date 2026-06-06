"""
Hisse Analiz API Rotaları — /api/v1/stocks/*

Endpoint'ler:
  GET  /stocks/{ticker}          — Temel bilgiler + teknik + AI raporu
  GET  /stocks/{ticker}/price    — Fiyat geçmişi (JSON)
  GET  /stocks/{ticker}/report   — Sadece AI raporu (DB cache öncelikli)
  GET  /stocks/{ticker}/news     — Haber sentiment listesi
  GET  /stocks/compare           — Çoklu hisse karşılaştırma
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.db.repositories import StockInfoRepo, StockPriceRepo, AnalysisRepo
from backend.cache.redis_client import (
    cache_get, cache_set, cache_delete,
    key_stock_info, TTL_STOCK_INFO,
)
from backend.api.auth import get_current_user

from data_fetcher import get_stock_info, get_price_history
from technical_analysis import compute_indicators
from news_sentiment import get_news_with_sentiment, format_news_for_prompt
from llm_analyzer import analyze_stock

logger = logging.getLogger("finsight.api.stocks")
router = APIRouter(prefix="/stocks", tags=["Stocks"])


# ── GET /stocks/{ticker} ──────────────────────────────────────────────────────
@router.get("/{ticker}")
async def get_stock_analysis(
    ticker: str,
    period: str = Query("1y", pattern="^(1mo|3mo|6mo|1y|2y)$"),
    force_refresh: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Tam analiz paketi: temel bilgiler + teknik göstergeler + AI raporu.
    Redis/DB cache'i önce kontrol eder, yoksa yfinance + Gemini'den çeker.
    """
    ticker = ticker.upper()

    # 1. Cache kontrolü
    if not force_refresh:
        cached = await cache_get(key_stock_info(ticker))
        if cached:
            logger.debug("Cache hit: %s stock info", ticker)
            return cached

    # 2. Temel veri
    try:
        stock_data = await asyncio.to_thread(get_stock_info, ticker)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{ticker} verisi alınamadı: {e}")

    # 3. Fiyat + teknik analiz
    try:
        df = await asyncio.to_thread(get_price_history, ticker, period)
        technicals = await asyncio.to_thread(compute_indicators, df)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fiyat verisi alınamadı: {e}")

    # 4. Haberleri çek
    news = []
    try:
        news = await asyncio.to_thread(get_news_with_sentiment, ticker, max_items=8)
    except Exception:
        pass

    # 5. AI Raporu — önce DB cache'e bak
    repo = AnalysisRepo(db)
    report = await repo.get_latest(ticker, max_age_minutes=10)
    if report is None:
        try:
            news_text = format_news_for_prompt(news)
            report = await asyncio.to_thread(
                analyze_stock, stock_data, technicals, news_text
            )
            await repo.save(ticker, report, user_id=current_user.id)
        except Exception as e:
            logger.warning("AI analizi başarısız (%s): %s", ticker, e)
            report = {}

    # 6. StockInfo DB'ye yaz
    info_repo = StockInfoRepo(db)
    await info_repo.upsert(stock_data)

    # 7. Cevabı hazırla ve cache'e yaz
    response = {
        "ticker":      ticker,
        "stock_data":  stock_data,
        "technicals":  {k: v for k, v in technicals.items() if isinstance(v, (str, int, float, type(None)))},
        "news":        news,
        "report":      report,
    }
    await cache_set(key_stock_info(ticker), response, ttl=TTL_STOCK_INFO)
    return response


# ── GET /stocks/{ticker}/price ────────────────────────────────────────────────
@router.get("/{ticker}/price")
async def get_price_history_endpoint(
    ticker: str,
    period: str = Query("1y", pattern="^(1mo|3mo|6mo|1y|2y)$"),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """OHLCV fiyat geçmişini döner."""
    ticker = ticker.upper()

    # DB'den dene
    price_repo = StockPriceRepo(db)
    days_map   = {"1mo": 35, "3mo": 95, "6mo": 185, "1y": 370, "2y": 740}
    df = await price_repo.get_price_df(ticker, days=days_map.get(period, 370))

    if df is None:
        # yfinance'ten çek, DB'ye kaydet
        try:
            df = await asyncio.to_thread(get_price_history, ticker, period)
            await price_repo.upsert_from_df(ticker, df)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    records = df.reset_index().to_dict(orient="records")
    for r in records:
        if hasattr(r.get("Date"), "isoformat"):
            r["Date"] = r["Date"].isoformat()

    return {"ticker": ticker, "period": period, "count": len(records), "prices": records}


# ── GET /stocks/{ticker}/report ───────────────────────────────────────────────
@router.get("/{ticker}/report")
async def get_ai_report(
    ticker: str,
    force_new: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Sadece AI raporunu döner. force_new=True ile yeniden oluşturur."""
    ticker = ticker.upper()

    if not force_new:
        repo = AnalysisRepo(db)
        report = await repo.get_latest(ticker, max_age_minutes=10)
        if report:
            return {"ticker": ticker, "report": report, "from_cache": True}

    # Yeniden üret
    try:
        stock_data = await asyncio.to_thread(get_stock_info, ticker)
        df         = await asyncio.to_thread(get_price_history, ticker)
        technicals = await asyncio.to_thread(compute_indicators, df)
        news       = await asyncio.to_thread(get_news_with_sentiment, ticker, max_items=8)
        news_text  = format_news_for_prompt(news)
        report     = await asyncio.to_thread(
            analyze_stock, stock_data, technicals, news_text
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Rapor oluşturulamadı: {e}")

    repo = AnalysisRepo(db)
    await repo.save(ticker, report, user_id=current_user.id)
    await cache_delete(key_stock_info(ticker))

    return {"ticker": ticker, "report": report, "from_cache": False}


# ── GET /stocks/{ticker}/news ─────────────────────────────────────────────────
@router.get("/{ticker}/news")
async def get_news(
    ticker: str,
    limit: int = Query(8, ge=1, le=20),
    current_user=Depends(get_current_user),
):
    """Haber listesini sentiment etiketleriyle döner."""
    ticker = ticker.upper()
    try:
        news = await asyncio.to_thread(get_news_with_sentiment, ticker, max_items=limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"ticker": ticker, "count": len(news), "news": news}


async def _compare_single(t: str) -> tuple[str, dict]:
    """Tek hisse karşılaştırma verisi (thread pool)."""
    try:
        sd = await asyncio.to_thread(get_stock_info, t)
        df = await asyncio.to_thread(get_price_history, t, "3mo")
        ta = await asyncio.to_thread(compute_indicators, df)
        return t, {
            "fiyat":   sd.get("son_fiyat"),
            "fk":      sd.get("fk_orani"),
            "rsi":     ta.get("rsi_degeri"),
            "macd":    ta.get("macd_durumu"),
            "sektor":  sd.get("sektor"),
        }
    except Exception as e:
        return t, {"hata": str(e)}


# ── GET /stocks/compare ───────────────────────────────────────────────────────
@router.get("/compare/multi")
async def compare_stocks(
    tickers: str = Query(..., description="Virgülle ayrılmış hisse kodları: THYAO,ASELS"),
    current_user=Depends(get_current_user),
):
    """Çoklu hisse özet karşılaştırma."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(ticker_list) > 10:
        raise HTTPException(status_code=400, detail="En fazla 10 hisse karşılaştırılabilir.")

    pairs = await asyncio.gather(*[_compare_single(t) for t in ticker_list])
    results = dict(pairs)

    return {"tickers": ticker_list, "comparison": results}
