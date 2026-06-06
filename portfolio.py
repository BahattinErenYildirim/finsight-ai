"""
Portföy Analizi — Çoklu hisse karşılaştırma, korelasyon matrisi,
ağırlıklı risk skoru ve diversifikasyon analizi.
"""
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import get_stock_info, get_price_history
from technical_analysis import compute_indicators
from news_sentiment import get_news_with_sentiment, format_news_for_prompt


def _load_single_ticker(t: str, period: str) -> tuple[str, dict]:
    """Tek hisse watchlist verisi."""
    t = t.strip().upper()
    if not t:
        return t, {}
    try:
        sd = get_stock_info(t)
        df = get_price_history(t, period=period)
        ta_ = compute_indicators(df)
        nw = get_news_with_sentiment(t, max_items=3)
        return t, {
            "stock_data": sd,
            "df": df,
            "technicals": ta_,
            "news": nw,
        }
    except Exception as e:
        return t, {"error": str(e)}


def load_watchlist(tickers: list[str], period: str = "1y", max_workers: int = 6) -> dict:
    """
    Birden fazla hissenin verilerini paralel çeker.

    Returns:
        {ticker: {"stock_data": ..., "df": ..., "technicals": ..., "news": ...}}
    """
    valid = [t.strip().upper() for t in tickers if t and t.strip()]
    if not valid:
        return {}

    results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_load_single_ticker, t, period): t for t in valid}
        for future in as_completed(futures):
            try:
                ticker, data = future.result()
                if ticker:
                    results[ticker] = data
            except Exception as e:
                ticker = futures.get(future, "?")
                results[ticker] = {"error": str(e)}
    return results


def compute_max_drawdown(prices: pd.Series) -> float | None:
    """Zirve-dip bazlı maksimum düşüş yüzdesi (negatif değer)."""
    if prices is None or len(prices) < 2:
        return None
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return round(float(drawdown.min() * 100), 2)


def compute_sortino_ratio(daily_returns: pd.Series, risk_free: float = 0.0) -> float | None:
    """Yıllıklandırılmış Sortino oranı."""
    if daily_returns is None or len(daily_returns) < 10:
        return None
    excess = daily_returns - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) == 0:
        return None
    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std):
        return None
    sortino = (excess.mean() / downside_std) * np.sqrt(252)
    return round(float(sortino), 2)


def build_comparison_table(watchlist: dict) -> pd.DataFrame:
    """
    Watchlist verilerinden karşılaştırma tablosu oluşturur.

    Returns:
        Hisse | Fiyat | F/K | PD/DD | RSI | MACD | Bollinger | ...
    """
    rows = []
    for ticker, data in watchlist.items():
        if "error" in data:
            rows.append({"Hisse": ticker, "Durum": f"Hata: {data['error']}"})
            continue

        sd = data["stock_data"]
        ta_ = data["technicals"]
        rows.append({
            "Hisse": ticker,
            "Fiyat (TL)": sd.get("son_fiyat", "-"),
            "F/K": sd.get("fk_orani", "-"),
            "PD/DD": sd.get("pddd_orani", "-"),
            "Borç/FAVÖK": sd.get("borc_favok", "-"),
            "RSI": ta_.get("rsi_degeri", "-"),
            "MACD": _short_signal(ta_.get("macd_durumu", "-")),
            "SMA Trend": _short_signal(ta_.get("sma_durumu", "-")),
            "Bollinger": _short_signal(ta_.get("bollinger_durumu", "-")),
        })

    return pd.DataFrame(rows)


def compute_correlation_matrix(watchlist: dict) -> pd.DataFrame | None:
    """
    Hisseler arası günlük getiri korelasyon matrisini hesaplar.

    Returns:
        NxN korelasyon DataFrame'i veya None
    """
    close_data = {}
    for ticker, data in watchlist.items():
        if "error" in data:
            continue
        df = data["df"]
        if "Close" in df.columns and len(df) >= 20:
            close_data[ticker] = df["Close"]

    if len(close_data) < 2:
        return None

    # Ortak tarihlerde birleştir
    combined = pd.DataFrame(close_data)
    combined = combined.dropna()

    if len(combined) < 10:
        return None

    # Günlük getiri
    returns = combined.pct_change().dropna()
    return returns.corr().round(3)


def simulate_portfolio(
    watchlist: dict,
    weights: dict[str, float],
    risk_scores: dict[str, int] | None = None,
) -> dict:
    """
    Portföy simülasyonu: ağırlıklı risk skoru, diversifikasyon metrikleri.

    Args:
        watchlist: load_watchlist çıktısı
        weights: {ticker: ağırlık_yüzde} (toplam 100)
        risk_scores: {ticker: 0-100 risk skoru} — AI'dan gelir

    Returns:
        Portföy analiz sonuçları dict
    """
    valid_tickers = [t for t in weights if t in watchlist and "error" not in watchlist[t]]

    if not valid_tickers:
        return {"error": "Geçerli hisse bulunamadı"}

    # Ağırlıkları normalize et
    total_w = sum(weights[t] for t in valid_tickers)
    if total_w == 0:
        return {"error": "Ağırlık toplamı sıfır"}
    norm_weights = {t: weights[t] / total_w for t in valid_tickers}

    # 1) Ağırlıklı risk skoru
    if risk_scores:
        weighted_risk = sum(
            norm_weights[t] * risk_scores.get(t, 50)
            for t in valid_tickers
        )
    else:
        weighted_risk = None

    # 2) Getiri istatistikleri
    returns_data = {}
    for t in valid_tickers:
        df = watchlist[t]["df"]
        if "Close" in df.columns and len(df) >= 2:
            daily_ret = df["Close"].pct_change().dropna()
            returns_data[t] = {
                "ortalama_gunluk": round(daily_ret.mean() * 100, 4),
                "volatilite": round(daily_ret.std() * 100, 4),
                "toplam_getiri": round(
                    ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100, 2
                ),
            }

    # 3) Portföy düzeyinde getiri & volatilite
    combined_close = {}
    for t in valid_tickers:
        df = watchlist[t]["df"]
        if "Close" in df.columns:
            combined_close[t] = df["Close"]

    sortino = max_dd = None
    if len(combined_close) >= 2:
        combined_df = pd.DataFrame(combined_close).dropna()
        daily_returns = combined_df.pct_change().dropna()

        w_array = np.array([norm_weights[t] for t in combined_df.columns])
        port_returns_series = daily_returns.values @ w_array
        port_return = float(port_returns_series.mean() * 252 * 100)
        cov_matrix = daily_returns.cov().values * 252
        port_vol = float(np.sqrt(w_array @ cov_matrix @ w_array) * 100)
        sharpe = port_return / port_vol if port_vol > 0 else 0
        port_daily = pd.Series(port_returns_series, index=daily_returns.index)
        sortino = compute_sortino_ratio(port_daily)
        port_prices = (1 + port_daily).cumprod()
        max_dd = compute_max_drawdown(port_prices)
    else:
        port_return = port_vol = sharpe = None

    # 4) Diversifikasyon skoru (0-100)
    n = len(valid_tickers)
    if n >= 2:
        hhi = sum(v ** 2 for v in norm_weights.values())
        max_hhi = 1.0  # tek hisse
        min_hhi = 1.0 / n  # eşit dağılım
        if max_hhi != min_hhi:
            diversification = round((1 - (hhi - min_hhi) / (max_hhi - min_hhi)) * 100, 1)
        else:
            diversification = 100.0
    else:
        diversification = 0.0

    # 5) Sektör dağılımı
    sectors = {}
    for t in valid_tickers:
        sector = watchlist[t]["stock_data"].get("sektor", "Bilinmiyor")
        sectors[sector] = sectors.get(sector, 0) + norm_weights[t] * 100

    return {
        "hisse_sayisi": n,
        "agirliklar": {t: round(norm_weights[t] * 100, 1) for t in valid_tickers},
        "agirlikli_risk": round(weighted_risk, 1) if weighted_risk is not None else None,
        "portfoy_yillik_getiri": round(port_return, 2) if port_return is not None else None,
        "portfoy_volatilite": round(port_vol, 2) if port_vol is not None else None,
        "sharpe_orani": round(sharpe, 2) if sharpe is not None else None,
        "sortino_orani": sortino,
        "max_drawdown_pct": max_dd,
        "diversifikasyon_skoru": diversification,
        "sektor_dagilimi": sectors,
        "hisse_getirileri": returns_data,
    }


def build_portfolio_prompt(watchlist: dict, portfolio_result: dict) -> str:
    """
    Toplu AI analizi için LLM prompt'u oluşturur.
    """
    lines = ["Aşağıdaki BIST portföyünü analiz et.\n"]

    for ticker, data in watchlist.items():
        if "error" in data:
            continue
        sd = data["stock_data"]
        ta_ = data["technicals"]
        lines.append(f"--- {ticker} ({sd.get('sirket_adi', '')}) ---")
        lines.append(f"Fiyat: {sd.get('son_fiyat', '-')} TL | F/K: {sd.get('fk_orani', '-')} | RSI: {ta_.get('rsi_degeri', '-')}")
        lines.append(f"MACD: {ta_.get('macd_durumu', '-')}")
        lines.append(f"Bollinger: {ta_.get('bollinger_durumu', '-')}")
        news_text = format_news_for_prompt(data.get("news", []))
        lines.append(f"Haberler: {news_text}\n")

    pr = portfolio_result
    lines.append("--- PORTFÖY METRİKLERİ ---")
    lines.append(f"Hisse Sayısı: {pr.get('hisse_sayisi', '-')}")
    lines.append(f"Ağırlıklar: {pr.get('agirliklar', {})}")
    lines.append(f"Diversifikasyon Skoru: {pr.get('diversifikasyon_skoru', '-')}/100")
    lines.append(f"Portföy Volatilite: %{pr.get('portfoy_volatilite', '-')}")
    lines.append(f"Sharpe Oranı: {pr.get('sharpe_orani', '-')}")

    lines.append("\nJSON formatında yanıt ver:")
    lines.append('{"portfoy_ozeti":"Genel portföy değerlendirmesi 3-4 cümle",'
                 '"guclu_yonler":"Portföyün güçlü yönleri",'
                 '"zayif_yonler":"Riskler ve iyileştirme önerileri",'
                 '"diversifikasyon_yorumu":"Sektör ve ağırlık dağılımı yorumu",'
                 '"portfoy_risk_skoru":65}')

    return "\n".join(lines)


def _short_signal(text: str) -> str:
    """Uzun sinyal metnini kısa badge'e çevirir."""
    t = text.lower()
    if any(k in t for k in ["yükseliş", "golden", "pozitif", "güçlü", "fırsat", "üst yarı"]):
        return "🟢 Pozitif"
    if any(k in t for k in ["düşüş", "dead", "negatif", "aşırı alım", "dikkat", "alt yarı", "zayıf"]):
        return "🔴 Negatif"
    return "🟡 Nötr"
