"""
BIST Hisse Tarayıcısı (Screener) — Paralel veri çekimi ile BIST100'ün
en likit hisselerini teknik ve temel göstergelere göre tarar.

Kullanım:
    df = run_screener()           # Tüm evren
    df = run_screener(["THYAO"])  # Belirli hisseler
    df = apply_filters(df, rsi_filter="Aşırı Satım (RSI<35)")
"""
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import get_stock_info, get_price_history
from technical_analysis import compute_indicators

logger = logging.getLogger("finsight.screener")

# ── BIST Evren (BIST100 en likit 35 hisse) ───────────────────────────────────
BIST_UNIVERSE: list[str] = [
    "THYAO", "ASELS", "SISE",  "GARAN", "EREGL", "KCHOL", "AKBNK",
    "YKBNK", "TOASO", "FROTO", "BIMAS", "MGROS", "TUPRS", "ARCLK",
    "VESTL", "SAHOL", "KOZAL", "PETKM", "PGSUS", "TCELL", "ENKAI",
    "TKFEN", "HALKB", "VAKBN", "ISCTR", "EKGYO", "TTRAK", "OTKAR",
    "AYGAZ", "CCOLA", "SODA",  "TSKB",  "ULKER", "KRDMD", "TAVHL",
]


def _fetch_screener_data(ticker: str) -> dict | None:
    """
    Tek hisse için screener metriklerini çeker.
    Herhangi bir hata olursa None döner (executor güvenle atlayabilir).
    """
    try:
        sd = get_stock_info(ticker)
        df = get_price_history(ticker, period="3mo")
        ta = compute_indicators(df)

        close = df["Close"]

        # ── Günlük değişim (1G %) ──────────────────────────────────────────
        day_change: float | None = None
        if len(close) >= 2:
            prev, curr = float(close.iloc[-2]), float(close.iloc[-1])
            if prev != 0:
                day_change = round(((curr - prev) / prev) * 100, 2)

        # ── Aylık değişim (1A %, ~21 iş günü) ────────────────────────────
        month_change: float | None = None
        if len(close) >= 22:
            prev_m = float(close.iloc[-22])
            if prev_m != 0:
                month_change = round(((float(close.iloc[-1]) - prev_m) / prev_m) * 100, 2)

        # ── RSI ───────────────────────────────────────────────────────────
        rsi_raw = ta.get("rsi_degeri")
        rsi: float | None = float(rsi_raw) if isinstance(rsi_raw, (int, float)) else None

        # ── F/K ───────────────────────────────────────────────────────────
        fk_raw = sd.get("fk_orani")
        fk: float | None = float(fk_raw) if isinstance(fk_raw, (int, float)) else None

        # ── PD/DD ─────────────────────────────────────────────────────────
        pddd_raw = sd.get("pddd_orani")
        pddd: float | None = float(pddd_raw) if isinstance(pddd_raw, (int, float)) else None

        # ── Hacim (son gün / 20 günlük ort.) ─────────────────────────────
        vol_today: float | None = None
        vol_ratio: float | None = None
        if "Volume" in df.columns and len(df) >= 2:
            vol_today = float(df["Volume"].iloc[-1])
            if len(df) >= 21:
                avg_vol = float(df["Volume"].iloc[-21:-1].mean())
                if avg_vol > 0:
                    vol_ratio = round(vol_today / avg_vol, 2)

        # ── Sinyal rozetleri ──────────────────────────────────────────────
        signals: list[str] = []

        if rsi is not None:
            if rsi < 35:
                signals.append("🟢 Aşırı Satım")
            elif rsi > 65:
                signals.append("🔴 Aşırı Alım")

        macd_str = ta.get("macd_durumu", "")
        if "Golden" in macd_str:
            signals.append("💛 Golden Cross")
        elif "Dead" in macd_str:
            signals.append("☠️ Dead Cross")

        bb_str = ta.get("bollinger_durumu", "")
        if "Alt Bandın Altında" in bb_str:
            signals.append("📉 BB Alt Bant")
        elif "Üst Bandın Üstünde" in bb_str:
            signals.append("📈 BB Üst Bant")

        sektor_fk_raw = sd.get("sektor_fk_ort")
        if (
            fk is not None
            and isinstance(sektor_fk_raw, (int, float))
            and fk < sektor_fk_raw * 0.8
        ):
            signals.append("⭐ Değer Hissesi")

        return {
            "Ticker":      ticker,
            "Şirket":      (sd.get("sirket_adi") or ticker)[:28],
            "Sektör":      (sd.get("sektor") or "—")[:22],
            "Fiyat (TL)":  sd.get("son_fiyat"),
            "1G %":        day_change,
            "1A %":        month_change,
            "RSI":         rsi,
            "F/K":         fk,
            "PD/DD":       pddd,
            "Hacim":       vol_today,
            "Hacim/Ort":   vol_ratio,
            "Sinyaller":   " | ".join(signals) if signals else "—",
        }
    except Exception as e:
        logger.debug("Screener veri hatası (%s): %s", ticker, e)
        return None


def run_screener(
    tickers: list[str] | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Paralel ThreadPoolExecutor ile BIST screener çalıştırır.

    Args:
        tickers:     Taranacak hisse listesi. None → BIST_UNIVERSE.
        max_workers: Paralel thread sayısı (varsayılan 8).

    Returns:
        RSI'ya göre artan sırada screener DataFrame'i.
        Hata/boş tickers atlanır.
    """
    universe = tickers or BIST_UNIVERSE
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_screener_data, t): t for t in universe}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                ticker = futures.get(future, "?")
                logger.warning("Screener thread hatası (%s): %s", ticker, e)
                continue
            if result is not None:
                results.append(result)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # RSI artan sıra — Aşırı Satım önce (alım fırsatı)
    if "RSI" in df.columns:
        df = df.sort_values("RSI", ascending=True, na_position="last")
    return df.reset_index(drop=True)


def apply_filters(
    df: pd.DataFrame,
    rsi_filter: str = "Tümü",
    fk_max: float | None = None,
    pddd_max: float | None = None,
    vol_ratio_min: float | None = None,
    sektor_filter: list[str] | None = None,
    day_change_filter: str = "Tümü",
) -> pd.DataFrame:
    """
    Screener DataFrame'ine filtre uygular.

    Args:
        df:                run_screener() çıktısı
        rsi_filter:        "Tümü" | "Aşırı Satım (RSI<35)" |
                           "Nötr (35–65)" | "Aşırı Alım (RSI>65)"
        fk_max:            Maksimum F/K değeri (None → filtre yok)
        pddd_max:          Maksimum PD/DD (None → filtre yok)
        vol_ratio_min:     Min. hacim/20g ort. oranı (None → filtre yok)
        sektor_filter:     Dahil edilecek sektörler listesi (None → tümü)
        day_change_filter: "Tümü" | "Pozitif" | "Negatif"

    Returns:
        Filtrelenmiş DataFrame.
    """
    out = df.copy()

    # RSI filtresi
    if "RSI" in out.columns:
        if rsi_filter == "Aşırı Satım (RSI<35)":
            out = out[out["RSI"].notna() & (out["RSI"] < 35)]
        elif rsi_filter == "Nötr (35–65)":
            out = out[out["RSI"].notna() & (out["RSI"] >= 35) & (out["RSI"] <= 65)]
        elif rsi_filter == "Aşırı Alım (RSI>65)":
            out = out[out["RSI"].notna() & (out["RSI"] > 65)]

    # F/K filtresi
    if fk_max is not None and "F/K" in out.columns:
        out = out[out["F/K"].apply(
            lambda x: isinstance(x, (int, float)) and x <= fk_max
        )]

    # PD/DD filtresi
    if pddd_max is not None and "PD/DD" in out.columns:
        out = out[out["PD/DD"].apply(
            lambda x: isinstance(x, (int, float)) and x <= pddd_max
        )]

    # Hacim filtresi (son gün / 20g ortalama)
    if vol_ratio_min is not None and "Hacim/Ort" in out.columns:
        out = out[out["Hacim/Ort"].apply(
            lambda x: isinstance(x, (int, float)) and x >= vol_ratio_min
        )]

    # Sektör filtresi
    if sektor_filter and "Sektör" in out.columns:
        out = out[out["Sektör"].isin(sektor_filter)]

    # Günlük değişim filtresi
    if day_change_filter == "Pozitif" and "1G %" in out.columns:
        out = out[out["1G %"].notna() & (out["1G %"] > 0)]
    elif day_change_filter == "Negatif" and "1G %" in out.columns:
        out = out[out["1G %"].notna() & (out["1G %"] < 0)]

    return out.reset_index(drop=True)
