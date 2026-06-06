"""
Sektörel Analiz ve Rotasyon (Top-Down Approach)
═══════════════════════════════════════════════════════════════
BIST Ana Sektör endekslerini BIST100'e göre kıyaslayarak paranın
hangi sektöre girdiğini (Relative Strength & Momentum) tespit eder.
"""
import config  # noqa: F401 — SSL / UTF-8 (Türkçe klasör yolu)

import logging
import time

import pandas as pd

from data_fetcher import _fetch_yf_history
import yfinance as yf

logger = logging.getLogger("finsight.sector")

# Yahoo Finance'te doğrulanmış BIST sektör endeksleri (XUHAT.IS Yahoo'da yok)
SECTOR_INDICES = {
    "XU100.IS": "BIST 100 (Ana Endeks)",
    "XBANK.IS": "Bankacılık",
    "XUSIN.IS": "Sınai (Sanayi)",
    "XTKJS.IS": "Teknoloji",
    "XGIDA.IS": "Gıda & İçecek",
    "XULAS.IS": "Ulaştırma (Havacılık vb.)",
    "XHOLD.IS": "Holding & Yatırım",
    "XMESY.IS": "Metal Ana Sanayi",
    "XELKT.IS": "Elektrik",
    "XTRZM.IS": "Turizm",
    "XSGRT.IS": "Sigorta",
    "XMANA.IS": "Ana Metal",
    "XKMYA.IS": "Kimya, Petrol, Plastik",
    "XTEKS.IS": "Tekstil, Deri",
    "XILTM.IS": "İletişim",
    "XUTEK.IS": "Bilişim",
    "XSPOR.IS": "Spor",
    "XGMYO.IS": "Gayrimenkul Yat. Ort.",
}

_sector_cache = {}
_CACHE_TTL = 900  # 15 dakika cache


def get_sector_momentum(period: str = "3mo") -> pd.DataFrame:
    """
    Sektör endekslerinin belirtilen periyottaki getirilerini hesaplar
    ve BIST 100'e göre 'Görece Güç' (Relative Strength) oranını bulur.
    """
    cache_key = f"sectors_{period}"
    if cache_key in _sector_cache:
        ts, cached_df = _sector_cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached_df

    data = []

    for symbol, name in SECTOR_INDICES.items():
        try:
            stock = yf.Ticker(symbol)
            df = _fetch_yf_history(stock, symbol, period, symbol.replace(".IS", ""))
            if df is None or df.empty or len(df) < 5:
                logger.debug("Sektör atlandı (veri yok): %s", symbol)
                continue

            start_price = df["Close"].iloc[0]
            end_price = df["Close"].iloc[-1]

            month_start_idx = -21 if len(df) >= 21 else 0
            month_start_price = df["Close"].iloc[month_start_idx]

            total_return = ((end_price - start_price) / start_price) * 100
            monthly_return = ((end_price - month_start_price) / month_start_price) * 100

            data.append({
                "Sembol": symbol,
                "Sektör": name,
                "Aylık Getiri (%)": round(monthly_return, 2),
                f"Toplam Getiri ({period} %):": round(total_return, 2),
                "Son Fiyat": round(end_price, 2),
            })

        except Exception as e:
            logger.warning("Sektör verisi çekilemedi (%s): %s", symbol, e)

    df_sectors = pd.DataFrame(data)

    if not df_sectors.empty:
        bist100_row = df_sectors[df_sectors["Sembol"] == "XU100.IS"]
        bist100_monthly = bist100_row["Aylık Getiri (%)"].values[0] if not bist100_row.empty else 0
        df_sectors["Görece Güç (Alpha %)"] = round(
            df_sectors["Aylık Getiri (%)"] - bist100_monthly, 2
        )
        df_sectors = df_sectors.sort_values(by="Aylık Getiri (%)", ascending=False).reset_index(drop=True)
        _sector_cache[cache_key] = (time.time(), df_sectors)

    return df_sectors
