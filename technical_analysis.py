"""
Teknik Analiz — RSI, MACD, SMA ve Bollinger Bands göstergelerini hesaplar.
ta (Technical Analysis) kütüphanesini kullanır.
"""
import pandas as pd
import ta
from typing import Optional


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Fiyat DataFrame'inden teknik göstergeleri hesaplar.

    Args:
        df: OHLCV verileri (en az 'Close' sütunu gerekli)

    Returns:
        Teknik göstergeleri içeren sözlük
    """
    close = df["Close"]

    # ── RSI (14 günlük) ──────────────────────────────────────────────────────
    rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    rsi_value = round(rsi_series.iloc[-1], 2) if not rsi_series.empty else None
    rsi_sinyal = _rsi_signal(rsi_value)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_indicator = ta.trend.MACD(close=close)
    macd_durumu = _macd_signal(
        macd_indicator.macd(),
        macd_indicator.macd_signal(),
    )

    # ── SMA 50 / SMA 200 ─────────────────────────────────────────────────────
    sma50 = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    sma_durumu = _sma_signal(sma50, sma200, close)

    sma50_son = round(sma50.iloc[-1], 2) if not sma50.empty and pd.notna(sma50.iloc[-1]) else None
    sma200_son = round(sma200.iloc[-1], 2) if not sma200.empty and pd.notna(sma200.iloc[-1]) else None

    # ── Bollinger Bands (20, 2σ) ──────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_durumu = _bollinger_signal(
        bb.bollinger_hband(),
        bb.bollinger_lband(),
        bb.bollinger_mavg(),
        close,
    )

    # ── Destek / Direnç Seviyeleri ────────────────────────────────────────────
    sr = compute_support_resistance(df)

    # ── Fibonacci geri çekilme ────────────────────────────────────────────────
    fib = compute_fibonacci_retracement(df)

    # ── Hacim göstergeleri (OBV, VWAP) ───────────────────────────────────────
    vol = compute_volume_indicators(df)

    # ── Sinyal Başarı Oranları (Backtest) ─────────────────────────────────────
    backtest = compute_signal_accuracy(df)

    return {
        "rsi_degeri": rsi_value if rsi_value is not None else "Yetersiz Veri",
        "rsi_sinyal": rsi_sinyal,
        "macd_durumu": macd_durumu,
        "sma_durumu": sma_durumu,
        "sma50_son": sma50_son,
        "sma200_son": sma200_son,
        "bollinger_durumu": bb_durumu,
        "destek_1": sr["destek_1"],
        "direnc_1": sr["direnc_1"],
        "pivot": sr["pivot"],
        "destek_2": sr["destek_2"],
        "direnc_2": sr["direnc_2"],
        "fibonacci": fib,
        "obv_son": vol["obv_son"],
        "obv_trend": vol["obv_trend"],
        "vwap_son": vol["vwap_son"],
        "vwap_durumu": vol["vwap_durumu"],
        "backtest": backtest,
    }


def compute_fibonacci_retracement(df: pd.DataFrame, window: int = 60) -> dict:
    """
    Son swing high/low üzerinden Fibonacci geri çekilme seviyeleri.
    """
    try:
        recent = df.tail(window)
        swing_high = recent["High"].max()
        swing_low = recent["Low"].min()
        diff = swing_high - swing_low
        if diff <= 0:
            raise ValueError("Geçersiz swing aralığı")

        levels = {
            "0.0%": round(swing_high, 2),
            "23.6%": round(swing_high - diff * 0.236, 2),
            "38.2%": round(swing_high - diff * 0.382, 2),
            "50.0%": round(swing_high - diff * 0.5, 2),
            "61.8%": round(swing_high - diff * 0.618, 2),
            "78.6%": round(swing_high - diff * 0.786, 2),
            "100.0%": round(swing_low, 2),
        }
        close_last = float(df["Close"].iloc[-1])
        if close_last >= levels["38.2%"]:
            zone = "Üst bölge — güçlü trend"
        elif close_last >= levels["61.8%"]:
            zone = "Orta bölge — düzeltme / konsolidasyon"
        else:
            zone = "Alt bölge — zayıflama riski"

        return {"seviyeler": levels, "bolge": zone, "swing_high": swing_high, "swing_low": swing_low}
    except Exception:
        return {
            "seviyeler": {},
            "bolge": "Yetersiz Veri",
            "swing_high": "Yetersiz Veri",
            "swing_low": "Yetersiz Veri",
        }


def compute_volume_indicators(df: pd.DataFrame) -> dict:
    """OBV ve VWAP hesaplar."""
    try:
        if "Volume" not in df.columns or len(df) < 5:
            raise ValueError("Hacim verisi yok")

        close = df["Close"]
        volume = df["Volume"]

        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        obv_last = obv.iloc[-1]
        obv_prev = obv.iloc[-5] if len(obv) >= 5 else obv.iloc[0]
        if pd.isna(obv_last) or pd.isna(obv_prev):
            obv_trend = "Yetersiz Veri"
        elif obv_last > obv_prev:
            obv_trend = "Yükselen OBV — alım baskısı"
        elif obv_last < obv_prev:
            obv_trend = "Düşen OBV — satım baskısı"
        else:
            obv_trend = "Yatay OBV"

        vwap = ta.volume.VolumeWeightedAveragePrice(
            high=df["High"], low=df["Low"], close=close, volume=volume,
        ).volume_weighted_average_price()
        vwap_last = vwap.iloc[-1]
        price = close.iloc[-1]
        if pd.isna(vwap_last):
            vwap_durumu = "Yetersiz Veri"
        elif price > vwap_last * 1.01:
            vwap_durumu = f"Fiyat VWAP üstünde ({vwap_last:.2f})"
        elif price < vwap_last * 0.99:
            vwap_durumu = f"Fiyat VWAP altında ({vwap_last:.2f})"
        else:
            vwap_durumu = f"Fiyat VWAP civarında ({vwap_last:.2f})"

        return {
            "obv_son": round(float(obv_last), 0) if pd.notna(obv_last) else "Yetersiz Veri",
            "obv_trend": obv_trend,
            "vwap_son": round(float(vwap_last), 2) if pd.notna(vwap_last) else "Yetersiz Veri",
            "vwap_durumu": vwap_durumu,
        }
    except Exception:
        return {
            "obv_son": "Yetersiz Veri",
            "obv_trend": "Yetersiz Veri",
            "vwap_son": "Yetersiz Veri",
            "vwap_durumu": "Yetersiz Veri",
        }


def compute_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Pivot Point yöntemiyle destek ve direnç seviyeleri hesaplar.
    Klasik Floor Pivot: P = (H + L + C) / 3

    Args:
        df: OHLCV DataFrame
        window: Bakış penceresi (gün)

    Returns:
        {pivot, destek_1, destek_2, direnc_1, direnc_2}
    """
    try:
        recent = df.tail(window)
        high = recent["High"].max()
        low = recent["Low"].min()
        close_last = df["Close"].iloc[-1]

        pivot = (high + low + close_last) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        return {
            "pivot": round(pivot, 2),
            "destek_1": round(s1, 2),
            "destek_2": round(s2, 2),
            "direnc_1": round(r1, 2),
            "direnc_2": round(r2, 2),
        }
    except Exception:
        return {
            "pivot": "Yetersiz Veri",
            "destek_1": "Yetersiz Veri",
            "destek_2": "Yetersiz Veri",
            "direnc_1": "Yetersiz Veri",
            "direnc_2": "Yetersiz Veri",
        }


def compute_signal_accuracy(df: pd.DataFrame, forward_days: int = 5) -> dict:
    """
    Teknik sinyallerin tarihsel başarı oranlarını hesaplar (mini backtest).

    RSI < 30 sonrası yükseliş, RSI > 70 sonrası düşüş,
    MACD golden/dead cross sonrası yön doğruluğu.

    Args:
        df: OHLCV DataFrame (en az 60 gün)
        forward_days: Sinyal sonrası bakılacak gün sayısı

    Returns:
        {rsi_oversold_accuracy, rsi_overbought_accuracy,
         macd_cross_accuracy, toplam_sinyal, basarili_sinyal}
    """
    if len(df) < 60:
        return {"yeterli_veri": False}

    close = df["Close"]
    results = {
        "rsi_asiri_satim_basari": None,
        "rsi_asiri_alim_basari": None,
        "toplam_sinyal": 0,
        "basarili_sinyal": 0,
        "yeterli_veri": True,
    }

    try:
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # RSI < 30 → sonraki N günde fiyat arttı mı?
        oversold_signals = rsi[rsi < 30].index
        oversold_wins = 0
        oversold_total = 0
        for date in oversold_signals:
            idx = df.index.get_loc(date)
            future = idx + forward_days
            if future < len(df):
                oversold_total += 1
                if close.iloc[future] > close.iloc[idx]:
                    oversold_wins += 1

        # RSI > 70 → sonraki N günde fiyat düştü mü?
        overbought_signals = rsi[rsi > 70].index
        overbought_wins = 0
        overbought_total = 0
        for date in overbought_signals:
            idx = df.index.get_loc(date)
            future = idx + forward_days
            if future < len(df):
                overbought_total += 1
                if close.iloc[future] < close.iloc[idx]:
                    overbought_wins += 1

        results["rsi_asiri_satim_basari"] = (
            round(oversold_wins / oversold_total * 100, 1) if oversold_total >= 3 else None
        )
        results["rsi_asiri_alim_basari"] = (
            round(overbought_wins / overbought_total * 100, 1) if overbought_total >= 3 else None
        )
        results["toplam_sinyal"] = oversold_total + overbought_total
        results["basarili_sinyal"] = oversold_wins + overbought_wins

    except Exception:
        pass

    return results


def _rsi_signal(rsi: Optional[float]) -> str:
    """RSI değerine göre sinyal belirler."""
    if rsi is None:
        return "Yetersiz Veri"
    if rsi >= 70:
        return "Aşırı Alım Bölgesi (Dikkat!)"
    elif rsi <= 30:
        return "Aşırı Satım Bölgesi (Fırsat?)"
    elif rsi >= 60:
        return "Alım Baskısı Güçlü"
    elif rsi <= 40:
        return "Satım Baskısı Var"
    return "Nötr Bölge"


def _macd_signal(macd_line: pd.Series, signal_line: pd.Series) -> str:
    """MACD ve sinyal çizgisi ilişkisine göre durum belirler."""
    if macd_line.empty or signal_line.empty:
        return "Yetersiz Veri"

    macd_last = macd_line.iloc[-1]
    signal_last = signal_line.iloc[-1]

    if pd.isna(macd_last) or pd.isna(signal_last):
        return "Yetersiz Veri"

    diff = macd_line.tail(5) - signal_line.tail(5)
    cross_up = any(diff.iloc[i - 1] < 0 and diff.iloc[i] > 0 for i in range(1, len(diff)))
    cross_down = any(diff.iloc[i - 1] > 0 and diff.iloc[i] < 0 for i in range(1, len(diff)))

    if cross_up:
        return "Golden Cross (Yukarı Kesişim) — Yükseliş Sinyali"
    elif cross_down:
        return "Dead Cross (Aşağı Kesişim) — Düşüş Sinyali"
    elif macd_last > signal_last:
        return "Pozitif Bölge — MACD sinyal çizgisinin üstünde"
    else:
        return "Negatif Bölge — MACD sinyal çizgisinin altında"


def _sma_signal(sma50: pd.Series, sma200: pd.Series, close: pd.Series) -> str:
    """SMA 50/200 ve fiyat ilişkisine göre durum belirler."""
    if sma50.empty or sma200.empty:
        return "Yetersiz Veri"

    s50 = sma50.iloc[-1]
    s200 = sma200.iloc[-1]
    price = close.iloc[-1]

    if pd.isna(s50) or pd.isna(s200):
        return "Yetersiz Veri (200 günlük veri yetersiz)"

    if s50 > s200 and price > s50:
        return f"Güçlü Yükseliş Trendi — Fiyat({price:.2f}) > SMA50({s50:.2f}) > SMA200({s200:.2f})"
    elif s50 > s200:
        return f"Yükseliş Trendi — SMA50({s50:.2f}) > SMA200({s200:.2f}), Fiyat: {price:.2f}"
    elif s50 < s200 and price < s50:
        return f"Güçlü Düşüş Trendi — Fiyat({price:.2f}) < SMA50({s50:.2f}) < SMA200({s200:.2f})"
    elif s50 < s200:
        return f"Düşüş Trendi — SMA50({s50:.2f}) < SMA200({s200:.2f}), Fiyat: {price:.2f}"
    else:
        return f"Yatay Seyir — SMA50 ≈ SMA200 ({s50:.2f}), Fiyat: {price:.2f}"


def _bollinger_signal(
    upper: pd.Series,
    lower: pd.Series,
    mid: pd.Series,
    close: pd.Series,
) -> str:
    """Bollinger Bantlarına göre volatilite ve fiyat konumunu belirler."""
    if upper.empty or lower.empty:
        return "Yetersiz Veri"

    u = upper.iloc[-1]
    l_band = lower.iloc[-1]
    m = mid.iloc[-1]
    price = close.iloc[-1]

    if pd.isna(u) or pd.isna(l_band):
        return "Yetersiz Veri"

    band_width = round(((u - l_band) / m) * 100, 2) if m != 0 else 0

    if price > u:
        return f"Üst Bandın Üstünde — Aşırı Alım / Yüksek Volatilite (Bant Genişliği: %{band_width})"
    elif price < l_band:
        return f"Alt Bandın Altında — Aşırı Satım / Yüksek Volatilite (Bant Genişliği: %{band_width})"
    elif price > m:
        return f"Üst Yarıda — Güçlü Görünüm (Fiyat: {price:.2f}, Orta: {m:.2f}, Genişlik: %{band_width})"
    else:
        return f"Alt Yarıda — Zayıf Görünüm (Fiyat: {price:.2f}, Orta: {m:.2f}, Genişlik: %{band_width})"
