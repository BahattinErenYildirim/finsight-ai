"""
Repository Katmanı — Tüm DB sorguları buradan geçer.

Her entity için ayrı repository sınıfı:
  StockPriceRepo   : OHLCV fiyat kayıt/sorgulama
  StockInfoRepo    : Temel şirket verisi
  AnalysisRepo     : AI rapor kayıt/sorgulama
  UserRepo         : Kullanıcı CRUD
  AlertRepo        : Alert yönetimi
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import select, delete, update, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.db.models import (
    StockPrice, StockInfo, AnalysisReport, User, Alert,
)

logger = logging.getLogger("finsight.repo")


# ── StockPrice ────────────────────────────────────────────────────────────────
class StockPriceRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_from_df(self, ticker: str, df: pd.DataFrame) -> int:
        """
        DataFrame'den toplu fiyat kaydı yapar (varsa günceller, yoksa ekler).
        Returns: eklenen/güncellenen kayıt sayısı
        """
        rows = []
        for date, row in df.iterrows():
            rows.append({
                "ticker": ticker.upper(),
                "date": date.to_pydatetime() if hasattr(date, "to_pydatetime") else date,
                "open":   float(row.get("Open",  0) or 0),
                "high":   float(row.get("High",  0) or 0),
                "low":    float(row.get("Low",   0) or 0),
                "close":  float(row.get("Close", 0) or 0),
                "volume": float(row.get("Volume",0) or 0),
            })

        if not rows:
            return 0

        # PostgreSQL UPSERT (SQLite için fallback)
        try:
            stmt = pg_insert(StockPrice).values(rows)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_ticker_date",
                set_={"close": stmt.excluded.close, "volume": stmt.excluded.volume},
            )
            await self.session.execute(stmt)
        except Exception:
            # SQLite fallback: sil + ekle
            await self.session.execute(
                delete(StockPrice).where(StockPrice.ticker == ticker.upper())
            )
            self.session.add_all([StockPrice(**r) for r in rows])

        logger.debug("Upsert %d satır — %s", len(rows), ticker)
        return len(rows)

    async def get_price_df(
        self,
        ticker: str,
        days: int = 365,
    ) -> Optional[pd.DataFrame]:
        """Son N günlük fiyat verisini DataFrame olarak döner."""
        since = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(StockPrice)
            .where(
                and_(
                    StockPrice.ticker == ticker.upper(),
                    StockPrice.date >= since,
                )
            )
            .order_by(StockPrice.date)
        )
        rows = result.scalars().all()
        if not rows:
            return None

        df = pd.DataFrame(
            [{
                "Date":   r.date,
                "Open":   r.open,
                "High":   r.high,
                "Low":    r.low,
                "Close":  r.close,
                "Volume": r.volume,
            } for r in rows]
        ).set_index("Date")
        return df

    async def get_latest_price(self, ticker: str) -> Optional[float]:
        """En son kapanış fiyatını döner."""
        result = await self.session.execute(
            select(StockPrice.close)
            .where(StockPrice.ticker == ticker.upper())
            .order_by(desc(StockPrice.date))
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return row


# ── StockInfo ─────────────────────────────────────────────────────────────────
class StockInfoRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, data: dict) -> StockInfo:
        """Şirket temel bilgilerini günceller veya ekler."""
        ticker = data["hisse_kodu"].upper()
        result = await self.session.execute(
            select(StockInfo).where(StockInfo.ticker == ticker)
        )
        obj = result.scalar_one_or_none()

        if obj is None:
            obj = StockInfo(ticker=ticker)
            self.session.add(obj)

        obj.sirket_adi       = data.get("sirket_adi")
        obj.sektor           = data.get("sektor")
        obj.son_fiyat        = _safe_float(data.get("son_fiyat"))
        obj.piyasa_degeri    = _safe_float(data.get("piyasa_degeri"))
        obj.fk_orani         = _safe_float(data.get("fk_orani"))
        obj.pddd_orani       = _safe_float(data.get("pddd_orani"))
        obj.net_kar_buyumesi = _safe_float(data.get("net_kar_buyumesi"))
        obj.borc_favok       = str(data.get("borc_favok", ""))
        obj.sektor_fk_ort    = _safe_float(data.get("sektor_fk_ort"))
        obj.updated_at       = datetime.utcnow()

        return obj

    async def get(self, ticker: str) -> Optional[StockInfo]:
        result = await self.session.execute(
            select(StockInfo).where(StockInfo.ticker == ticker.upper())
        )
        return result.scalar_one_or_none()

    async def is_stale(self, ticker: str, max_age_minutes: int = 5) -> bool:
        """Kaydın belirtilen süreden eski olup olmadığını kontrol eder."""
        obj = await self.get(ticker)
        if obj is None or obj.updated_at is None:
            return True
        age = datetime.utcnow() - obj.updated_at
        return age.total_seconds() > max_age_minutes * 60


# ── AnalysisReport ────────────────────────────────────────────────────────────
class AnalysisRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(
        self,
        ticker: str,
        report: dict,
        model: str = "gemini-2.0-flash",
        user_id: Optional[int] = None,
    ) -> AnalysisReport:
        """AI raporunu veritabanına kaydeder."""
        risk = report.get("risk_analizi", {}).get("risk_skoru")
        obj = AnalysisReport(
            ticker      = ticker.upper(),
            report_json = json.dumps(report, ensure_ascii=False),
            risk_score  = int(risk) if isinstance(risk, (int, float)) else None,
            model_used  = model,
            user_id     = user_id,
        )
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def get_latest(self, ticker: str, max_age_minutes: int = 10) -> Optional[dict]:
        """Cache amaçlı: son N dakika içindeki raporu getirir."""
        since = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        result = await self.session.execute(
            select(AnalysisReport)
            .where(
                and_(
                    AnalysisReport.ticker == ticker.upper(),
                    AnalysisReport.created_at >= since,
                )
            )
            .order_by(desc(AnalysisReport.created_at))
            .limit(1)
        )
        obj = result.scalar_one_or_none()
        if obj is None:
            return None
        return json.loads(obj.report_json)

    async def get_history(self, ticker: str, limit: int = 20) -> list[dict]:
        """Bir hissenin geçmiş raporlarını döner."""
        result = await self.session.execute(
            select(AnalysisReport)
            .where(AnalysisReport.ticker == ticker.upper())
            .order_by(desc(AnalysisReport.created_at))
            .limit(limit)
        )
        rows = result.scalars().all()
        return [
            {
                "id":         r.id,
                "ticker":     r.ticker,
                "risk_score": r.risk_score,
                "model":      r.model_used,
                "created_at": r.created_at.isoformat(),
                "report":     json.loads(r.report_json),
            }
            for r in rows
        ]


# ── UserRepo ──────────────────────────────────────────────────────────────────
class UserRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, email: str, hashed_pw: str, full_name: str = "") -> User:
        user = User(email=email.lower(), hashed_pw=hashed_pw, full_name=full_name)
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: int) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def update_last_login(self, user_id: int) -> None:
        await self.session.execute(
            update(User)
            .where(User.id == user_id)
            .values(last_login=datetime.utcnow())
        )


# ── AlertRepo ─────────────────────────────────────────────────────────────────
class AlertRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        user_id: int,
        ticker: str,
        alert_type: str,
        threshold: float,
    ) -> Alert:
        alert = Alert(
            user_id=user_id,
            ticker=ticker.upper(),
            alert_type=alert_type,
            threshold=threshold,
        )
        self.session.add(alert)
        await self.session.flush()
        return alert

    async def get_active_by_ticker(self, ticker: str) -> list[Alert]:
        result = await self.session.execute(
            select(Alert)
            .where(
                and_(
                    Alert.ticker == ticker.upper(),
                    Alert.is_active.is_(True),
                )
            )
        )
        return list(result.scalars().all())

    async def mark_triggered(self, alert_id: int) -> None:
        await self.session.execute(
            update(Alert)
            .where(Alert.id == alert_id)
            .values(is_active=False, triggered_at=datetime.utcnow())
        )


# ── Yardımcı ──────────────────────────────────────────────────────────────────
def _safe_float(value) -> Optional[float]:
    if value is None or value == "Yetersiz Veri":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
