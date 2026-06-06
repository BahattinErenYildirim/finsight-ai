
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, UniqueConstraint, Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Hisse Fiyat Verisi ────────────────────────────────────────────────────────
class StockPrice(Base):
    __tablename__ = "stock_price"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    ticker     = Column(String(12), nullable=False, index=True)
    date       = Column(DateTime, nullable=False)
    open       = Column(Float)
    high       = Column(Float)
    low        = Column(Float)
    close      = Column(Float, nullable=False)
    volume     = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_ticker_date"),
        Index("ix_stock_price_ticker_date", "ticker", "date"),
    )

    def __repr__(self):
        return f"<StockPrice {self.ticker} {self.date:%Y-%m-%d} close={self.close}>"


# ── Şirket Temel Bilgileri ────────────────────────────────────────────────────
class StockInfo(Base):
    __tablename__ = "stock_info"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    ticker           = Column(String(12), unique=True, nullable=False, index=True)
    sirket_adi       = Column(String(200))
    sektor           = Column(String(100))
    son_fiyat        = Column(Float)
    piyasa_degeri    = Column(Float)
    fk_orani         = Column(Float)
    pddd_orani       = Column(Float)
    net_kar_buyumesi = Column(Float)
    borc_favok       = Column(String(20))
    sektor_fk_ort    = Column(Float)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<StockInfo {self.ticker} {self.sirket_adi}>"


# ── Teknik Analiz Sonuçları ───────────────────────────────────────────────────
class TechnicalSnapshot(Base):
    __tablename__ = "technical_snapshot"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    ticker          = Column(String(12), nullable=False, index=True)
    date            = Column(DateTime, nullable=False)
    rsi             = Column(Float)
    macd_line       = Column(Float)
    macd_signal     = Column(Float)
    macd_hist       = Column(Float)
    sma50           = Column(Float)
    sma200          = Column(Float)
    bb_upper        = Column(Float)
    bb_lower        = Column(Float)
    bb_mid          = Column(Float)
    pivot           = Column(Float)
    destek_1        = Column(Float)
    direnc_1        = Column(Float)
    created_at      = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_technical_ticker_date"),
    )


# ── AI Analiz Raporu ──────────────────────────────────────────────────────────
class AnalysisReport(Base):
    __tablename__ = "analysis_report"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticker      = Column(String(12), nullable=False, index=True)
    report_json = Column(Text, nullable=False)   # JSON string
    risk_score  = Column(Integer)
    model_used  = Column(String(50))
    created_at  = Column(DateTime, default=datetime.utcnow, index=True)

    # İlişki
    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    user    = relationship("User", back_populates="reports")


# ── Kullanıcı ─────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "user"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    email        = Column(String(255), unique=True, nullable=False, index=True)
    hashed_pw    = Column(String(255), nullable=False)
    full_name    = Column(String(200))
    is_active    = Column(Boolean, default=True)
    is_premium   = Column(Boolean, default=False)
    created_at   = Column(DateTime, default=datetime.utcnow)
    last_login   = Column(DateTime)

    # İlişkiler
    watchlists = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    alerts     = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    reports    = relationship("AnalysisReport", back_populates="user")


# ── Takip Listesi ─────────────────────────────────────────────────────────────
class Watchlist(Base):
    __tablename__ = "watchlist"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("user.id"), nullable=False)
    ticker     = Column(String(12), nullable=False)
    added_at   = Column(DateTime, default=datetime.utcnow)
    note       = Column(String(500))

    user = relationship("User", back_populates="watchlists")

    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_watchlist_user_ticker"),
    )


# ── Fiyat Alerti ─────────────────────────────────────────────────────────────
class Alert(Base):
    __tablename__ = "alert"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    user_id      = Column(Integer, ForeignKey("user.id"), nullable=False)
    ticker       = Column(String(12), nullable=False)
    alert_type   = Column(String(20), nullable=False)   # "above" | "below" | "rsi_low" | "rsi_high"
    threshold    = Column(Float, nullable=False)
    is_active    = Column(Boolean, default=True)
    triggered_at = Column(DateTime, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="alerts")
