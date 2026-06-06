"""
pytest fixtures — Tüm test dosyaları tarafından paylaşılır.
"""
import asyncio
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from backend.db.models import Base
from backend.api.main import app


# ── Async test desteği ────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── In-memory SQLite veritabanı (testler için) ────────────────────────────────
@pytest_asyncio.fixture(scope="function")
async def test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        yield session

    await engine.dispose()


# ── FastAPI test client ───────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    from httpx import AsyncClient
    return AsyncClient(app=app, base_url="http://test")


# ── Örnek fiyat DataFrame'i ───────────────────────────────────────────────────
@pytest.fixture
def sample_price_df():
    """250 günlük sahte OHLCV verisi (teknik analiz testleri için)."""
    import numpy as np

    n = 250
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    close = 100.0
    closes = []
    for _ in range(n):
        close += np.random.randn() * 2
        close = max(close, 10)
        closes.append(close)

    closes = pd.Series(closes)
    highs  = closes + abs(pd.Series(np.random.randn(n))) * 1.5
    lows   = closes - abs(pd.Series(np.random.randn(n))) * 1.5
    opens  = closes.shift(1).fillna(closes.iloc[0])

    return pd.DataFrame({
        "Open":   opens.values,
        "High":   highs.values,
        "Low":    lows.values,
        "Close":  closes.values,
        "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)


# ── Örnek hisse bilgisi sözlüğü ───────────────────────────────────────────────
@pytest.fixture
def sample_stock_data():
    return {
        "hisse_kodu":        "THYAO",
        "sirket_adi":        "Türk Hava Yolları A.O.",
        "son_fiyat":         298.50,
        "sektor":            "Industrials",
        "sektor_fk_ort":     14.2,
        "piyasa_degeri":     130_000_000_000,
        "fk_orani":          8.5,
        "pddd_orani":        3.1,
        "net_kar_buyumesi":  18.5,
        "borc_favok":        "2.8",
    }


# ── Örnek AI raporu ───────────────────────────────────────────────────────────
@pytest.fixture
def sample_report():
    return {
        "analiz_ozeti": {
            "genel_gorus":           "Nötr - Temkinli - Test raporu",
            "temel_analiz_yorumu":   "F/K oranı sektör ortalamasının altında.",
            "teknik_analiz_yorumu":  "RSI nötr bölgede.",
            "haber_sentiment_yorumu":"Haberler genel olarak nötr.",
            "bollinger_yorumu":      "Fiyat orta bant civarında.",
        },
        "risk_analizi": {
            "risk_skoru":  42,
            "risk_gerekcesi": "Orta seviye risk.",
        },
    }


# ── Mock Gemini yanıtı ────────────────────────────────────────────────────────
@pytest.fixture
def mock_gemini_response(sample_report):
    mock = MagicMock()
    mock.text = json.dumps(sample_report, ensure_ascii=False)
    return mock
