"""
FinSight AI — Streamlit dashboard for BIST equity analysis.
"""
import config  # noqa: F401 — Windows UTF-8 side effect on import

import html as html_lib
import json
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import ta
from pdf_report import generate_report
from portfolio import load_watchlist, build_comparison_table, compute_correlation_matrix, simulate_portfolio, build_portfolio_prompt
import plotly.express as px

from data_fetcher import get_stock_info, get_price_history
from technical_analysis import compute_indicators
from news_sentiment import get_news_with_sentiment, format_news_for_prompt, summarize_news
from llm_analyzer import analyze_stock
from db import (
    init_db, get_watchlist, add_ticker as db_add_ticker,
    remove_ticker as db_remove_ticker, clear_watchlist as db_clear_watchlist,
)
from screener import run_screener, apply_filters, BIST_UNIVERSE

# Veritabanini baslat (ilk calistirmada tablolari olusturur)
init_db()

# Page config
st.set_page_config(
    page_title="FinSight AI — BIST Analiz",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""<style>
.main .block-container{padding-top:1.5rem;padding-bottom:1rem}
.metric-box{background:#1a1d2e;border:1px solid #2a2d42;border-radius:10px;padding:14px 18px;text-align:center}
.metric-label{font-size:12px;color:#6b7280;margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em}
.metric-value{font-size:22px;font-weight:600;color:#e2e8f0}
.metric-sub{font-size:11px;color:#6b7280;margin-top:3px}
.analysis-card{background:#1a1d2e;border:1px solid #2a2d42;border-radius:10px;padding:16px 18px;margin-bottom:10px;height:100%}
.card-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;color:#6b7280}
.card-content{font-size:13px;line-height:1.7;color:#cbd5e1}
.news-card{background:#1a1d2e;border:1px solid #2a2d42;border-radius:8px;padding:12px 16px;margin-bottom:8px;display:flex;align-items:flex-start;gap:12px}
.news-title{font-size:13px;color:#cbd5e1;line-height:1.5}
.news-meta{font-size:11px;color:#6b7280;margin-top:4px}
.news-summary{font-size:12px;color:#94a3b8;margin-top:6px;line-height:1.5}
.sentiment-badge{font-size:11px;font-weight:600;padding:3px 10px;border-radius:12px;white-space:nowrap;flex-shrink:0;margin-top:2px}
.badge-pozitif{background:#052e16;color:#4ade80;border:1px solid #166534}
.badge-negatif{background:#450a0a;color:#f87171;border:1px solid #991b1b}
.badge-notr{background:#1a1d2e;color:#6b7280;border:1px solid #374151}
div[data-testid="stTabs"] button{font-size:14px}
</style>""", unsafe_allow_html=True)

_PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d0f1a",
    font=dict(color="#6b7280", size=11),
    margin=dict(l=8, r=8, t=28, b=8),
    hovermode="x unified",
)


# Cached loaders
@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(ticker: str, period: str):
    """Load ticker data with 5-minute TTL cache."""
    try:
        stock_data = get_stock_info(ticker)
    except Exception as e:
        raise ValueError(f"{ticker} fundamentals failed: {e}") from e

    try:
        df = get_price_history(ticker, period=period)
    except Exception as e:
        raise ValueError(f"{ticker} price history failed: {e}") from e

    try:
        technicals = compute_indicators(df)
    except Exception:
        technicals = {"rsi_degeri": "Hata", "rsi_sinyal": "N/A",
                      "macd_durumu": "N/A", "sma_durumu": "N/A",
                      "sma50_son": None, "sma200_son": None, "bollinger_durumu": "N/A"}

    try:
        news = get_news_with_sentiment(ticker, max_items=10)
    except Exception:
        news = []  # news optional — do not fail the app

    return stock_data, df, technicals, news


@st.cache_data(ttl=86400, show_spinner=False)
def run_llm(ticker: str, sd_json: str, ta_json: str, news_text: str) -> dict:
    """Run LLM analysis with 24-hour cache per inputs."""
    return analyze_stock(json.loads(sd_json), json.loads(ta_json), news_text)


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator series to the chart DataFrame."""
    df = df.copy()
    close = df["Close"]
    try:
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["sma50"]    = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        df["sma200"]   = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
        df["rsi"]      = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        macd           = ta.trend.MACD(close=close)
        df["macd_line"]   = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"]   = macd.macd_diff()
    except Exception:
        pass
    return df


# ── Grafik fonksiyonları ──────────────────────────────────────────────────────
def build_main_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.15, 0.20],
        vertical_spacing=0.02,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    # SMA çizgileri
    for col_name, color, label in [("sma50", "#f59e0b", "SMA 50"), ("sma200", "#8b5cf6", "SMA 200")]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=label,
                line=dict(color=color, width=1.3),
            ), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"], name="BB upper",
            line=dict(color="#60a5fa", width=0.8, dash="dot"), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"], name="BB Alt",
            line=dict(color="#60a5fa", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(96,165,250,0.07)", showlegend=False,
        ), row=1, col=1)

    # Hacim (Volume)
    if "Volume" in df.columns:
        colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Hacim",
            marker_color=colors, showlegend=False,
        ), row=2, col=1)

    # RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"], name="RSI",
            line=dict(color="#a78bfa", width=1.5), showlegend=False,
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", line_width=0.7, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", line_width=0.7, row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)", line_width=0, row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.06)", line_width=0, row=3, col=1)

    # MACD
    if "macd_hist" in df.columns:
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["macd_hist"],
            marker_color=colors, showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_line"], name="MACD",
            line=dict(color="#60a5fa", width=1.2), showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_signal"], name="Sinyal",
            line=dict(color="#f59e0b", width=1.2), showlegend=False,
        ), row=4, col=1)

    fig.update_layout(
        height=750, xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        **_PLOT_THEME,
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="#1a1d2e",
                     showspikes=True, spikethickness=1, spikecolor="#374151")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#1a1d2e")
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


def _safe_risk_score(raw) -> int:
    """LLM çıktısı string/float olabilir; 0–100 arası tamsayıya çevirir."""
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = 50
    return max(0, min(100, value))


def build_risk_gauge(score) -> go.Figure:
    score = _safe_risk_score(score)
    color = "#26a69a" if score <= 30 else ("#fbbf24" if score <= 60 else "#ef5350")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": color}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#374151"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#1a1d2e", "borderwidth": 0,
            "steps": [
                {"range": [0, 30],   "color": "rgba(38,166,154,0.18)"},
                {"range": [30, 60],  "color": "rgba(251,191,36,0.18)"},
                {"range": [60, 100], "color": "rgba(239,83,80,0.18)"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.82, "value": score},
        },
    ))
    gauge_layout = {
        **_PLOT_THEME,
        "height": 230,
        "margin": dict(l=20, r=20, t=20, b=10),
    }
    gauge_layout.pop("hovermode", None)  # gösterge grafiklerinde geçersiz
    fig.update_layout(**gauge_layout)
    return fig


# ── Yardımcılar ───────────────────────────────────────────────────────────────
def _badge(sentiment: str) -> str:
    cls = {"Pozitif": "badge-pozitif", "Negatif": "badge-negatif"}.get(sentiment, "badge-notr")
    return f'<span class="sentiment-badge {cls}">{sentiment}</span>'


def _fmt_cap(v) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    if v >= 1e12:
        return f"₺{v/1e12:.2f}T"
    if v >= 1e9:
        return f"₺{v/1e9:.1f}Mn"
    if v >= 1e6:
        return f"₺{v/1e6:.0f}M"
    return "—"


def _signal_color(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["yükseliş", "golden", "pozitif", "güçlü", "fırsat"]):
        return "#26a69a"
    if any(k in t for k in ["düşüş", "dead", "negatif", "aşırı alım", "dikkat"]):
        return "#ef5350"
    return "#cbd5e1"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 FinSight AI")
    st.caption("BIST Yapay Zeka Analiz Platformu")
    st.divider()

    ticker_input = st.text_input(
        "Hisse Kodu",
        value=st.session_state.get("last_ticker", "THYAO"),
        placeholder="THYAO, ASELS, SISE...",
    )
    period_map = {"1 Ay": "1mo", "3 Ay": "3mo", "6 Ay": "6mo", "1 Yıl": "1y", "2 Yıl": "2y"}
    period_label = st.selectbox("Periyot", list(period_map.keys()), index=3)
    analyze_clicked = st.button("🔍 Analiz Et", type="primary", width="stretch")

    st.divider()
    st.caption("Hızlı erişim")
    quick_tickers = ["THYAO", "ASELS", "SISE", "GARAN", "EREGL", "KCHOL"]
    cols = st.columns(3)
    for i, q in enumerate(quick_tickers):
        if cols[i % 3].button(q, key=f"q_{q}", width="stretch"):
            st.session_state.update({"last_ticker": q, "last_period": "1y"})
            st.session_state.pop("report", None)
            st.rerun()

    st.divider()
    st.markdown("#### 📊 Portföy izleme listesi")
    if "portfolio_tickers" not in st.session_state:
        st.session_state["portfolio_tickers"] = get_watchlist()

    port_input = st.text_input(
        "Hisse ekle", placeholder="örn. SISE", key="port_add_input",
    )
    if st.button("➕ Portföye ekle", width="stretch", key="port_add_btn"):
        t = port_input.strip().upper()
        if t and t not in st.session_state["portfolio_tickers"]:
            db_add_ticker(t)
            st.session_state["portfolio_tickers"].append(t)
            st.rerun()

    if st.session_state["portfolio_tickers"]:
        st.caption(f"Portföy: {', '.join(st.session_state['portfolio_tickers'])}")
        for _pt in list(st.session_state["portfolio_tickers"]):
            _pcol_a, _pcol_b = st.columns([3, 1])
            _pcol_a.caption(_pt)
            if _pcol_b.button("✕", key=f"port_rm_{_pt}", help=f"{_pt} portföyden sil"):
                db_remove_ticker(_pt)
                st.session_state["portfolio_tickers"].remove(_pt)
                st.session_state.pop("portfolio_data", None)
                st.rerun()
        if st.button("🗑️ Portföyü temizle", width="stretch", key="port_clear"):
            db_clear_watchlist()
            st.session_state["portfolio_tickers"] = []
            st.session_state.pop("portfolio_data", None)
            st.rerun()

    st.divider()
    st.caption("⚠️ Yatırım tavsiyesi değildir. Tüm çıktılar yalnızca bilgilendirme amaçlıdır.")


# ── Tetikleyici ───────────────────────────────────────────────────────────────
if analyze_clicked:
    if not ticker_input.strip():
        st.error("Hisse kodu boş olamaz.")
        st.stop()
    st.session_state.update({
        "last_ticker": ticker_input.strip().upper(),
        "last_period": period_map[period_label],
    })
    st.session_state.pop("report", None)
    st.rerun()


# ── Karşılama ekranı ──────────────────────────────────────────────────────────
if "last_ticker" not in st.session_state:
    st.markdown("# 📈 FinSight AI")
    st.markdown("### BIST Hisseleri için Yapay Zeka Destekli Finansal Analiz")
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Kenar çubuktan bir hisse kodu girin ve **Analiz Et**'e tıklayın.")
    st.markdown("#### Örnek hisseler")
    demo_cols = st.columns(6)
    for i, q in enumerate(quick_tickers):
        if demo_cols[i].button(q, key=f"d_{q}", width="stretch"):
            st.session_state.update({"last_ticker": q, "last_period": "1y"})
            st.session_state.pop("report", None)
            st.rerun()
    st.stop()


# ── Veri yükleme ──────────────────────────────────────────────────────────────
current_ticker = st.session_state["last_ticker"]
current_period = st.session_state.get("last_period", "1y")

status_container = st.empty()
with status_container.status(f"📡 {current_ticker} data loading...", expanded=False) as status:
    try:
        status.update(label=f"📊 {current_ticker} fundamentals loading...")
        stock_data, df, technicals, news = load_market_data(current_ticker, current_period)

        status.update(label="📈 Preparing chart data...")
        df_chart = enrich_df(df)

        status.update(label="✅ Data ready!", state="complete")
    except ValueError as e:
        status.update(label="❌ Error", state="error")
        st.error(
            f"**{current_ticker}** data could not be loaded.\n\n"
            f"**Details:** {e}\n\n"
            "**Check:**\n"
            "- Is the ticker valid on BIST? (e.g. THYAO, ASELS)\n"
            "- Is your internet connection active?"
        )
        st.stop()
    except Exception as e:
        status.update(label="❌ Connection error", state="error")
        st.error(
            f"An unexpected error occurred.\n\n"
            f"**Hata:** `{type(e).__name__}: {e}`\n\n"
            "Please wait a few seconds and try again."
        )
        st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
price   = stock_data.get("son_fiyat", "—")
change_pct = None
if len(df) >= 2:
    prev, curr = df["Close"].iloc[-2], df["Close"].iloc[-1]
    if prev and prev != 0:
        change_pct = ((curr - prev) / prev) * 100

chg_str   = f"{change_pct:+.2f}%" if change_pct is not None else "—"
chg_color = "#26a69a" if (change_pct or 0) >= 0 else "#ef5350"

badges = []
try:
    rsi_val = technicals.get("rsi_degeri")
    if isinstance(rsi_val, (int, float)):
        if rsi_val < 30:
            badges.append("<span style='background:rgba(16, 185, 129, 0.2);color:#10b981;border:1px solid #10b981;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:500;'>Oversold</span>")
        elif rsi_val > 70:
            badges.append("<span style='background:rgba(239, 68, 68, 0.2);color:#ef4444;border:1px solid #ef4444;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:500;'>Overbought</span>")
    
    fk = stock_data.get("fk_orani")
    sekt_fk = stock_data.get("sektor_fk_ort")
    if isinstance(fk, (int, float)) and isinstance(sekt_fk, (int, float)) and fk < sekt_fk:
        badges.append("<span style='background:rgba(59, 130, 246, 0.2);color:#3b82f6;border:1px solid #3b82f6;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:500;'>Discounted</span>")
except Exception:
    pass

badge_html = " ".join(badges)

st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;flex-wrap:wrap;">
  <span style="font-size:26px;font-weight:600;color:#e2e8f0">{stock_data.get('sirket_adi', current_ticker)}</span>
  <span style="font-size:15px;color:#6b7280">{current_ticker} &nbsp;·&nbsp; {stock_data.get('sektor','')}</span>
  {badge_html}
</div>
<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:20px">
  <span style="font-size:36px;font-weight:700;color:#e2e8f0">{price} TL</span>
  <span style="font-size:18px;color:{chg_color}">{chg_str}</span>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
for col, label, value, sub in [
    (m1, "P/E Ratio",      stock_data.get("fk_orani", "—"),   f"Sector avg: {stock_data.get('sektor_fk_ort', '—')}"),
    (m2, "PD/DD",          stock_data.get("pddd_orani", "—"), ""),
    (m3, "Market cap",  _fmt_cap(stock_data.get("piyasa_degeri")), ""),
    (m4, "RSI (14)",       technicals.get("rsi_degeri", "—"), technicals.get("rsi_sinyal", "")),
    (m5, "Debt/EBITDA",     stock_data.get("borc_favok", "—"), ""),
]:
    col.markdown(f"""<div class="metric-box">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub[:40]}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
has_portfolio = len(st.session_state.get("portfolio_tickers", [])) >= 2
if has_portfolio:
    tabs = st.tabs(["🌍 Sektör Rotasyonu", "📈 Teknik Analiz", "🤖 AI Analiz", "📰 Haberler", "🔍 Tarayıcı", "📊 Portföy"])
    tab0, tab1, tab2, tab3, tab_screener, tab4 = tabs
else:
    tabs = st.tabs(["🌍 Sektör Rotasyonu", "📈 Teknik Analiz", "🤖 AI Analiz", "📰 Haberler", "🔍 Tarayıcı"])
    tab0, tab1, tab2, tab3, tab_screener = tabs
    tab4 = None

# ══ Tab 0 (Sektör Rotasyonu) ══════════════════════════════════════════════════
with tab0:
    st.markdown("### 🌍 Sektör Rotasyonu (Top-Down Analiz)")
    st.markdown("BIST Ana sektörlerinin XU100'e (BIST 100) göre göreceli momentumu.")
    
    with st.spinner("Sektör verileri hesaplanıyor..."):
        try:
            from sector_analysis import get_sector_momentum
            df_sectors = get_sector_momentum(period="3mo")
            if not df_sectors.empty:
                st.dataframe(
                    df_sectors,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Aylık Getiri (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Toplam Getiri (3mo %):": st.column_config.NumberColumn(format="%.2f%%"),
                        "Son Fiyat": st.column_config.NumberColumn(format="%.2f TL"),
                        "Görece Güç (Alpha %)": st.column_config.ProgressColumn(
                            "Görece Güç (Alpha %)",
                            help="BIST 100 getirisine göre fark",
                            format="%.2f%%",
                            min_value=-20,
                            max_value=20,
                        ),
                    }
                )
            else:
                st.warning("Sektör verileri yüklenemedi.")
        except Exception as e:
            st.error(f"Sector module error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)


# ══ Tab 1 ═════════════════════════════════════════════════════════════════════
with tab1:
    st.plotly_chart(build_main_chart(df_chart), width="stretch",
                    config={"displayModeBar": False})

    st.markdown("#### Sinyal özeti")
    s1, s2, s3, s4 = st.columns(4)
    for col, title, key in [
        (s1, "RSI",              "rsi_sinyal"),
        (s2, "MACD",             "macd_durumu"),
        (s3, "Hareketli Ort.",   "sma_durumu"),
        (s4, "Bollinger",        "bollinger_durumu"),
    ]:
        val = technicals.get(key, "—")
        color = _signal_color(val)
        col.markdown(f"""<div class="analysis-card">
          <div class="card-title">{title}</div>
          <div class="card-content" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tcol1, tcol2 = st.columns(2)

    with tcol1:
        st.markdown("#### 🎯 Destek & direnç (pivot)")
        # HTML satırları sütun girintisi ile başlamamalı — aksi halde Markdown kod bloğu sayar (ham metin).
        st.markdown(
            f"""<div style="background:#1e2235; padding:16px; border-radius:12px; border:1px solid #2d3748;">
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#ef5350">Resistance 2</span> <span style="font-weight:bold">{technicals.get("direnc_2", "-")}</span>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#fbbf24">Resistance 1</span> <span style="font-weight:bold">{technicals.get("direnc_1", "-")}</span>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px; border-top:1px solid #475569; border-bottom:1px solid #475569; padding:4px 0;">
<span style="color:#94a3b8">Pivot</span> <span style="font-weight:bold;color:#e2e8f0">{technicals.get("pivot", "-")}</span>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#60a5fa">Support 1</span> <span style="font-weight:bold">{technicals.get("destek_1", "-")}</span>
</div>
<div style="display:flex; justify-content:space-between;">
<span style="color:#26a69a">Support 2</span> <span style="font-weight:bold">{technicals.get("destek_2", "-")}</span>
</div>
</div>""",
            unsafe_allow_html=True,
        )

    with tcol2:
        st.markdown("#### 🧪 Sinyal isabet oranları (backtest)")
        bt = technicals.get("backtest", {})
        if bt.get("yeterli_veri"):
            rsi_buy = bt.get("rsi_asiri_satim_basari")
            rsi_sell = bt.get("rsi_asiri_alim_basari")

            st.markdown(
                f"""<div style="background:#1e2235; padding:16px; border-radius:12px; border:1px solid #2d3748;">
<div style="font-size:13px; color:#94a3b8; margin-bottom:16px;">
Historical {bt.get("toplam_sinyal", 0)} signals (last 5 days) hit rate.
</div>
<div style="margin-bottom:12px;">
<div style="display:flex; justify-content:space-between; margin-bottom:4px;">
<span style="color:#26a69a; font-size:14px;">RSI Oversold (Alım Fırsatı)</span>
<span style="font-weight:bold;">{f"%{rsi_buy}" if rsi_buy is not None else "Veri Yok"}</span>
</div>
<div style="width:100%; background:#2d3748; border-radius:4px; height:6px;">
<div style="width:{rsi_buy or 0}%; background:#26a69a; height:100%; border-radius:4px;"></div>
</div>
</div>
<div>
<div style="display:flex; justify-content:space-between; margin-bottom:4px;">
<span style="color:#ef5350; font-size:14px;">RSI Overbought (Satış Baskısı)</span>
<span style="font-weight:bold;">{f"%{rsi_sell}" if rsi_sell is not None else "Veri Yok"}</span>
</div>
<div style="width:100%; background:#2d3748; border-radius:4px; height:6px;">
<div style="width:{rsi_sell or 0}%; background:#ef5350; height:100%; border-radius:4px;"></div>
</div>
</div>
</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Backtest için yeterli geçmiş yok (min. 60 gün).")

    st.markdown("<br>", unsafe_allow_html=True)
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown("#### 📐 Fibonacci geri çekilme")
        fib = technicals.get("fibonacci", {})
        if fib.get("seviyeler"):
            for lvl, val in fib["seviyeler"].items():
                st.markdown(f"**{lvl}:** {val}")
            st.caption(fib.get("bolge", ""))
        else:
            st.caption("Yetersiz veri")
    with fcol2:
        st.markdown("#### 📊 Hacim göstergeleri")
        st.markdown(f"**OBV:** {technicals.get('obv_son', '—')}")
        st.caption(technicals.get("obv_trend", ""))
        st.markdown(f"**VWAP:** {technicals.get('vwap_son', '—')}")
        st.caption(technicals.get("vwap_durumu", ""))


# ══ Tab 2 ═════════════════════════════════════════════════════════════════════
with tab2:
    # Kota tasarrufu: Streamlit her etkileşimde tüm sekmeleri çalıştırır. Otomatik
    # Gemini çağrısı yapılmaz — yalnızca aşağıdaki butona basınca çalışır.
    from config import is_llm_configured, is_ollama_provider, llm_provider_label

    _report_ok = (
        "report" in st.session_state
        and st.session_state.get("report_ticker") == current_ticker
    )

    if not _report_ok:
        _quota_hint = (
            "**Local Ollama:** Runs locally; no API quota. "
            "First report may take 1–3 minutes."
            if is_ollama_provider()
            else "**Gemini quota:** We do not run AI on every refresh. "
            "Click once to generate; same ticker cached for 24 hours "
            "cache is used."
        )
        st.info(_quota_hint)
        if not is_llm_configured():
            st.error(
                "⚠️ **LLM not configured**\n\n"
                "**Ollama:** `.env` → `LLM_PROVIDER=ollama`, `ollama serve` çalışsın\n"
                "**Gemini:** [API key](https://aistudio.google.com/apikey) → `LLM_PROVIDER=gemini`"
            )
        elif st.button(
            f"🤖 Generate AI report ({llm_provider_label()})",
            type="primary",
            key="btn_run_llm",
        ):
            with st.status("🤖 AI is thinking...", expanded=True) as status:
                import time as _time
                st.write("Sending data and news to the model...")
                _time.sleep(0.2)
                try:
                    news_text = format_news_for_prompt(news)
                    ta_safe = {
                        k: v for k, v in technicals.items()
                        if isinstance(v, (str, int, float, type(None)))
                    }
                    # sort_keys: Streamlit @st.cache_data anahtarı stabil olsun
                    sd_json = json.dumps(stock_data, ensure_ascii=False, sort_keys=True)
                    ta_json = json.dumps(ta_safe, ensure_ascii=False, sort_keys=True)
                    report = run_llm(current_ticker, sd_json, ta_json, news_text)
                    st.session_state["report"] = report
                    st.session_state["report_ticker"] = current_ticker
                    status.update(label="✅ Analysis complete", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label="❌ Analysis failed", state="error", expanded=False)
                    st.error(f"AI analysis failed: {e}")
        st.caption(
            f"Provider: **{llm_provider_label()}**. "
            "For another ticker use **Analyze** in the sidebar — open the AI tab "
            "only when you need the report."
        )
    else:
        if st.button("🔄 Regenerate AI report", key="btn_refresh_llm"):
            st.session_state.pop("report", None)
            st.session_state.pop("report_ticker", None)
            st.rerun()

    if _report_ok:
        report     = st.session_state["report"]
        ozet       = report.get("analiz_ozeti", {})
        risk       = report.get("risk_analizi", {})
        risk_score = _safe_risk_score(risk.get("risk_skoru", 50))
        risk_label = "Low risk" if risk_score <= 30 else ("Medium risk" if risk_score <= 60 else "High risk")
        risk_color = "#26a69a"    if risk_score <= 30 else ("#fbbf24" if risk_score <= 60 else "#ef5350")

        g_col, r_col = st.columns([1, 2])
        with g_col:
            st.plotly_chart(build_risk_gauge(risk_score), width="stretch",
                            config={"displayModeBar": False})
            st.markdown(f"<div style='text-align:center;font-size:16px;font-weight:600;"
                        f"color:{risk_color};margin-top:-6px'>{risk_label}</div>",
                        unsafe_allow_html=True)
        with r_col:
            for title, key in [("Overall view", "genel_gorus"), ("Risk rationale", None)]:
                val = ozet.get(key, "") if key else risk.get("risk_gerekcesi", "—")
                st.markdown(f"""<div class="analysis-card">
                  <div class="card-title">{title}</div>
                  <div class="card-content">{val or '—'}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("#### Detailed analysis")
        a1, a2 = st.columns(2)
        a3, a4 = st.columns(2)
        for col, title, key in [
            (a1, "Temel Analiz",    "temel_analiz_yorumu"),
            (a2, "Teknik Analiz",   "teknik_analiz_yorumu"),
            (a3, "Haber Sentiment", "haber_sentiment_yorumu"),
            (a4, "Bollinger",       "bollinger_yorumu"),
        ]:
            col.markdown(f"""<div class="analysis-card">
              <div class="card-title">{title}</div>
              <div class="card-content">{ozet.get(key, '—')}</div>
            </div>""", unsafe_allow_html=True)


# ══ Tab 3 ═════════════════════════════════════════════════════════════════════
with tab3:
    st.caption("Source: Yahoo Finance · Sentiment: LLM or keywords · Cache ~5 min")

    if not news:
        st.info(
            "No recent news for this ticker. Yahoo Finance bazen BIST için sınırlı "
            "sonuç döndürür; farklı bir hisse deneyin veya birkaç dakika sonra yenileyin."
        )
    else:
        ns = summarize_news(news)
        skor_color = "#26a69a" if ns["skor"] >= 20 else ("#ef5350" if ns["skor"] <= -20 else "#fbbf24")

        h1, h2 = st.columns([2, 1])
        with h1:
            st.markdown(
                f"**News mood:** {ns['etiket']} "
                f"(skor **{ns['skor']:+d}** / 100)"
            )
            st.progress((ns["skor"] + 100) / 200)
        with h2:
            st.markdown(
                f"<div style='text-align:right;font-size:28px;font-weight:700;color:{skor_color}'>"
                f"{ns['skor']:+d}</div>",
                unsafe_allow_html=True,
            )

        nc1, nc2, nc3, nc4 = st.columns(4)
        nc1.metric("Toplam", ns["count"])
        nc2.metric("Pozitif", ns["pozitif"])
        nc3.metric("Negatif", ns["negatif"])
        nc4.metric("Nötr", ns["notr"])

        chart_df = pd.DataFrame({
            "Duygu": ["Pozitif", "Negatif", "Nötr"],
            "Adet": [ns["pozitif"], ns["negatif"], ns["notr"]],
        })
        fig_news = px.bar(
            chart_df, x="Duygu", y="Adet", color="Duygu",
            color_discrete_map={"Pozitif": "#26a69a", "Negatif": "#ef5350", "Nötr": "#6b7280"},
        )
        fig_news.update_layout(
            height=220, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0f1a",
            font=dict(color="#6b7280", size=11),
        )
        st.plotly_chart(fig_news, width="stretch", config={"displayModeBar": False})

        filt = st.selectbox(
            "Filtrele",
            ["All", "Pozitif", "Negatif", "Nötr"],
            key="news_sentiment_filter",
        )
        shown = news if filt == "All" else [n for n in news if n.get("sentiment") == filt]

        st.markdown("<br>", unsafe_allow_html=True)
        for item in shown:
            title = html_lib.escape(item.get("title", ""))
            summary = html_lib.escape(item.get("summary", "") or "")
            publisher = html_lib.escape(item.get("publisher", "") or "")
            when = html_lib.escape(item.get("published_display", "") or "")
            link = item.get("link", "").strip()
            safe_link = link if link.startswith("http") else ""

            meta_parts = [p for p in (when, publisher) if p]
            meta_html = " · ".join(meta_parts)
            meta_block = f'<div class="news-meta">{meta_html}</div>' if meta_html else ""

            summary_block = (
                f'<div class="news-summary">{summary}</div>' if summary else ""
            )
            link_block = (
                f'<a href="{html_lib.escape(safe_link)}" target="_blank" rel="noopener" '
                f'style="font-size:11px;color:#60a5fa;text-decoration:none">Habere git →</a>'
                if safe_link else '<span style="font-size:11px;color:#6b7280">No link</span>'
            )

            st.markdown(
                f"""<div class="news-card">
              {_badge(item.get("sentiment", "Nötr"))}
              <div style="flex:1">
                <div class="news-title">{title}</div>
                {meta_block}
                {summary_block}
                <div style="margin-top:6px">{link_block}</div>
              </div>
            </div>""",
                unsafe_allow_html=True,
            )


# ══ Tab Screener ══════════════════════════════════════════════════════════════
with tab_screener:
    st.markdown("### 🔍 BIST Hisse Tarayıcısı")
    st.markdown(
        f"BIST100'ün en likit **{len(BIST_UNIVERSE)} hissesini** teknik ve temel "
        "göstergelere göre paralel olarak tarar. RSI, F/K ve momentum sinyallerini "
        "tek ekranda gösterir."
    )

    # ── Filtre kontrolleri ────────────────────────────────────────────────────
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns([2, 2, 2, 2, 2, 1])
    with sc1:
        scr_rsi = st.selectbox(
            "RSI Filtresi",
            ["Tümü", "Aşırı Satım (RSI<35)", "Nötr (35–65)", "Aşırı Alım (RSI>65)"],
            key="scr_rsi_f",
        )
    with sc2:
        scr_fk = st.selectbox(
            "F/K Maks.",
            ["Tümü", "< 10", "< 15", "< 20"],
            key="scr_fk_f",
        )
    with sc3:
        scr_pddd = st.selectbox(
            "PD/DD Maks.",
            ["Tümü", "< 1.5", "< 2.5", "< 4"],
            key="scr_pddd_f",
        )
    with sc4:
        scr_vol = st.selectbox(
            "Hacim/Ort Min.",
            ["Tümü", "> 1.0x", "> 1.5x", "> 2.0x"],
            key="scr_vol_f",
        )
    with sc5:
        scr_day = st.selectbox(
            "Günlük Değişim",
            ["Tümü", "Pozitif", "Negatif"],
            key="scr_day_f",
        )
    with sc6:
        scr_run_btn = st.button(
            "🔍 Tara", type="primary", key="scr_run_btn", use_container_width=True
        )

    # ── Tarama ───────────────────────────────────────────────────────────────
    if scr_run_btn:
        st.session_state.pop("screener_df", None)
        with st.status(
            f"🔍 {len(BIST_UNIVERSE)} hisse paralel taranıyor...", expanded=True
        ) as scr_status:
            st.write("📡 8 paralel thread ile veri çekiliyor...")
            _scr_df = run_screener()
            st.session_state["screener_df"] = _scr_df
            scr_status.update(
                label=f"✅ {len(_scr_df)} hisse tarandı",
                state="complete",
                expanded=False,
            )

    if "screener_df" in st.session_state:
        df_raw_scr = st.session_state["screener_df"]

        if df_raw_scr.empty:
            st.warning("Veri çekilemedi. İnternet bağlantısını kontrol edin.")
        else:
            # Filtre uygula
            _fk_max_map = {"< 10": 10.0, "< 15": 15.0, "< 20": 20.0}
            _pddd_max_map = {"< 1.5": 1.5, "< 2.5": 2.5, "< 4": 4.0}
            _vol_min_map = {"> 1.0x": 1.0, "> 1.5x": 1.5, "> 2.0x": 2.0}
            df_filtered_scr = apply_filters(
                df_raw_scr,
                rsi_filter=scr_rsi,
                fk_max=_fk_max_map.get(scr_fk),
                pddd_max=_pddd_max_map.get(scr_pddd),
                vol_ratio_min=_vol_min_map.get(scr_vol),
                day_change_filter=scr_day,
            )

            # ── Özet Metrikler ──────────────────────────────────────────────
            _oversold = int((df_raw_scr["RSI"].notna() & (df_raw_scr["RSI"] < 35)).sum())
            _overbought = int((df_raw_scr["RSI"].notna() & (df_raw_scr["RSI"] > 65)).sum())
            _value = int(df_raw_scr["Sinyaller"].str.contains("Değer", na=False).sum())
            _golden = int(df_raw_scr["Sinyaller"].str.contains("Golden", na=False).sum())

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Taranan Hisse", f"{len(df_raw_scr)}")
            sm2.metric("🟢 Aşırı Satım", f"{_oversold}")
            sm3.metric("⭐ Değer Hissesi", f"{_value}")
            sm4.metric("💛 Golden Cross", f"{_golden}")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Fırsat Radarı ────────────────────────────────────────────────
            _opps = df_raw_scr[
                df_raw_scr["Sinyaller"].str.contains(
                    "Aşırı Satım|Golden Cross|Değer", na=False, regex=True
                )
            ].head(5)

            if not _opps.empty:
                st.markdown("#### 🎯 Fırsat Radarı")
                st.caption("En güçlü sinyale sahip hisseler")
                _opp_cols = st.columns(len(_opps))
                for _i, (_, _row) in enumerate(_opps.iterrows()):
                    _rsi_v = _row.get("RSI")
                    _rsi_color = (
                        "#26a69a" if (isinstance(_rsi_v, float) and _rsi_v < 35)
                        else "#ef5350" if (isinstance(_rsi_v, float) and _rsi_v > 65)
                        else "#fbbf24"
                    )
                    _d = _row.get("1G %")
                    _d_color = "#26a69a" if (isinstance(_d, float) and _d > 0) else "#ef5350"
                    _d_str = f"{_d:+.2f}%" if isinstance(_d, float) else "—"
                    _price = _row.get("Fiyat (TL)")
                    _price_str = f"{_price:.2f} TL" if isinstance(_price, (int, float)) else "—"
                    _rsi_str = f"{_rsi_v:.1f}" if isinstance(_rsi_v, float) else "—"
                    _sig = str(_row.get("Sinyaller", ""))[:45]
                    _sect = str(_row.get("Sektör", ""))[:16]
                    _opp_cols[_i].markdown(f"""<div style="background:#1a1d2e;border:1px solid #2a2d42;border-radius:12px;padding:14px 10px;text-align:center">
  <div style="font-size:18px;font-weight:700;color:#e2e8f0;margin-bottom:2px">{_row['Ticker']}</div>
  <div style="font-size:10px;color:#6b7280;margin-bottom:6px">{_sect}</div>
  <div style="font-size:20px;font-weight:600;color:#e2e8f0;margin-bottom:2px">{_price_str}</div>
  <div style="font-size:13px;color:{_d_color};margin-bottom:4px">{_d_str}</div>
  <div style="font-size:12px;color:{_rsi_color};margin-bottom:6px">RSI {_rsi_str}</div>
  <div style="font-size:10px;color:#94a3b8;line-height:1.4">{_sig}</div>
</div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

            # ── Sonuç Tablosu ────────────────────────────────────────────────
            st.markdown(f"**{len(df_filtered_scr)} hisse** gösteriliyor")
            st.dataframe(
                df_filtered_scr,
                hide_index=True,
                width="stretch",
                column_config={
                    "Fiyat (TL)": st.column_config.NumberColumn(format="%.2f TL"),
                    "1G %":       st.column_config.NumberColumn(format="%.2f%%"),
                    "1A %":       st.column_config.NumberColumn(format="%.2f%%"),
                    "RSI":        st.column_config.ProgressColumn(
                        "RSI", min_value=0, max_value=100, format="%.1f"
                    ),
                    "F/K":        st.column_config.NumberColumn(format="%.2f"),
                    "PD/DD":      st.column_config.NumberColumn(format="%.2f"),
                },
            )

            # ── Hızlı Aksiyon ────────────────────────────────────────────────
            if not df_filtered_scr.empty:
                st.markdown("#### ⚡ Hızlı Aksiyon")
                qa1, qa2, qa3 = st.columns([3, 1, 1])
                with qa1:
                    _sel = st.selectbox(
                        "Hisse seç:",
                        df_filtered_scr["Ticker"].tolist(),
                        key="scr_sel",
                    )
                with qa2:
                    if st.button("📈 Analiz Et", key="scr_analyze", use_container_width=True):
                        st.session_state.update({"last_ticker": _sel, "last_period": "1y"})
                        st.session_state.pop("report", None)
                        st.rerun()
                with qa3:
                    if st.button("➕ Portföye Ekle", key="scr_add", use_container_width=True):
                        if _sel not in st.session_state["portfolio_tickers"]:
                            db_add_ticker(_sel)
                            st.session_state["portfolio_tickers"].append(_sel)
                            st.success(f"✅ {_sel} portföye eklendi!")
                        else:
                            st.info(f"{_sel} zaten portföyde.")
    else:
        # ── Henüz tarama yapılmamış ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            f"**🔍 Tara** butonuna basarak taramayı başlatın. "
            f"İlk tarama ~15–25 saniye sürer ({len(BIST_UNIVERSE)} hisse paralel çekilir). "
            "Sonuçlar oturum boyunca önbellekte kalır."
        )
        st.markdown(f"**Tarama Evreni — {len(BIST_UNIVERSE)} Hisse:**")
        _univ_cols = st.columns(7)
        for _ui, _ut in enumerate(BIST_UNIVERSE):
            _univ_cols[_ui % 7].markdown(f"`{_ut}`")


# ══ Tab 4: Portföy ════════════════════════════════════════════════════════════
if tab4 is not None:
    with tab4:
        port_tickers = st.session_state["portfolio_tickers"]

        # Portföy verilerini yükle
        if "portfolio_data" not in st.session_state:
            with st.spinner(f"📊 {len(port_tickers)} tickers loading..."):
                st.session_state["portfolio_data"] = load_watchlist(port_tickers, current_period)
        wl = st.session_state["portfolio_data"]

        # ── Watchlist Karşılaştırma Tablosu ──
        st.markdown("#### 📋 Watchlist Karşılaştırma")
        comp_df = build_comparison_table(wl)
        st.dataframe(comp_df, width="stretch", hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pcol1, pcol2 = st.columns(2)

        # ── Korelasyon Matrisi ──
        with pcol1:
            st.markdown("#### 🔥 Korelasyon Matrisi")
            corr_matrix = compute_correlation_matrix(wl)
            if corr_matrix is not None:
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale=[
                        [0, "#ef5350"], [0.5, "#1a1d2e"], [1, "#26a69a"]
                    ],
                    aspect="auto",
                    zmin=-1, zmax=1,
                )
                fig_corr.update_layout(
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0d0f1a",
                    font=dict(color="#6b7280", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    coloraxis_colorbar=dict(title="Korelasyon"),
                )
                st.plotly_chart(fig_corr, width="stretch", config={"displayModeBar": False})
            else:
                st.info("Correlation matrix needs at least 2 valid tickers.")

        # ── Portföy Simülatörü ──
        with pcol2:
            st.markdown("#### ⚖️ Portfolio simulator")
            valid = [t for t in port_tickers if t in wl and "error" not in wl[t]]
            if len(valid) >= 2:
                st.caption("Adjust weights (total 100%)")
                weights = {}
                default_w = round(100 / len(valid), 1)
                for t in valid:
                    weights[t] = st.slider(
                        f"{t}", 0.0, 100.0, default_w, 1.0, key=f"w_{t}",
                    )

                result = simulate_portfolio(wl, weights)

                # Metrikler
                sm1, sm2, sm3, sm4, sm5 = st.columns(5)
                sm1.metric("Diversifikasyon", f"{result.get('diversifikasyon_skoru', '-')}/100")
                sm2.metric("Volatilite", f"%{result.get('portfoy_volatilite', '-')}")
                sm3.metric("Sharpe", f"{result.get('sharpe_orani', '-')}")
                sm4.metric("Sortino", f"{result.get('sortino_orani', '-')}")
                _mdd = result.get("max_drawdown_pct")
                sm5.metric("Max Drawdown", f"%{_mdd}" if _mdd is not None else "—")

                # Sektör dağılımı pie chart
                sectors = result.get("sektor_dagilimi", {})
                if sectors:
                    fig_pie = px.pie(
                        names=list(sectors.keys()),
                        values=list(sectors.values()),
                        hole=0.45,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_pie.update_layout(
                        height=240,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#6b7280", size=10),
                        margin=dict(l=0, r=0, t=10, b=0),
                        showlegend=True,
                        legend=dict(font=dict(size=10)),
                    )
                    st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})
            else:
                st.info("Simülasyon için en az 2 geçerli hisse gereklidir.")

        # ── Toplu AI Analizi ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🤖 Toplu AI Portföy Analizi")
        if st.button("🧠 Portföyü Analiz Et", type="primary", key="bulk_ai_btn"):
            from config import is_llm_configured
            from llm_analyzer import generate_json_text
            if not is_llm_configured():
                st.warning(
                    "Portföy AI özeti için LLM yapılandırması gerekli "
                    "(Ollama veya Gemini — `.env` dosyasına bakın)."
                )
            else:
                with st.spinner("Tüm portföy analiz ediliyor..."):
                    try:
                        valid = [t for t in port_tickers if t in wl and "error" not in wl[t]]
                        default_w = round(100 / len(valid), 1) if valid else 0
                        w_dict = {t: default_w for t in valid}
                        port_result = simulate_portfolio(wl, w_dict)
                        prompt = build_portfolio_prompt(wl, port_result)

                        import json as _json
                        bulk_report = _json.loads(generate_json_text(prompt))
                        st.session_state["bulk_report"] = bulk_report
                    except Exception as e:
                        st.error(f"Bulk analysis error: {e}")

        if "bulk_report" in st.session_state:
            br = st.session_state["bulk_report"]
            ba1, ba2 = st.columns(2)
            for col, title, key in [
                (ba1, "Portföy Özeti", "portfoy_ozeti"),
                (ba2, "Güçlü Yönler", "guclu_yonler"),
            ]:
                col.markdown(f"""<div class="analysis-card">
                  <div class="card-title">{title}</div>
                  <div class="card-content">{br.get(key, '-')}</div>
                </div>""", unsafe_allow_html=True)
            ba3, ba4 = st.columns(2)
            for col, title, key in [
                (ba3, "Zayıf Yönler & Riskler", "zayif_yonler"),
                (ba4, "Diversifikasyon Yorumu", "diversifikasyon_yorumu"),
            ]:
                col.markdown(f"""<div class="analysis-card">
                  <div class="card-title">{title}</div>
                  <div class="card-content">{br.get(key, '-')}</div>
                </div>""", unsafe_allow_html=True)

            p_risk = br.get("portfoy_risk_skoru", 50)
            if isinstance(p_risk, (int, float)):
                p_color = "#26a69a" if p_risk <= 30 else ("#fbbf24" if p_risk <= 60 else "#ef5350")
                st.markdown(f"""<div style="text-align:center;margin:16px 0">
                  <span style="font-size:36px;font-weight:700;color:{p_color}">{int(p_risk)}</span>
                  <span style="font-size:14px;color:#6b7280">/100 Portföy Risk Skoru</span>
                </div>""", unsafe_allow_html=True)


# ══ PDF Export ════════════════════════════════════════════════════════════════
st.divider()
pdf_col1, pdf_col2 = st.columns([3, 1])

with pdf_col1:
    st.markdown(
        '<span style="font-size:15px;font-weight:600;color:#e2e8f0">'
        '📄 PDF Rapor Export</span><br>'
        '<span style="font-size:12px;color:#6b7280">'
        'Tüm analiz, grafik ve AI yorumlarını kurumsal PDF olarak indirin.'
        '</span>',
        unsafe_allow_html=True,
    )

def _normalize_pdf_bytes(data) -> bytes | None:
    """Streamlit download_button yalnızca bytes/bytearray kabul eder."""
    if data is None:
        return None
    if isinstance(data, bytes):
        return data if len(data) > 0 else None
    if isinstance(data, bytearray):
        return bytes(data) if len(data) > 0 else None
    if isinstance(data, str):
        return data.encode("latin-1") if data else None
    return None


with pdf_col2:
    if st.button("📥 Generate & download PDF", type="primary", width="stretch", key="pdf_btn"):
        with st.spinner("PDF oluşturuluyor..."):
            try:
                chart_fig = build_main_chart(df_chart)
                ai_report = st.session_state.get("report") if st.session_state.get("report_ticker") == current_ticker else None
                pdf_bytes = generate_report(
                    stock_data=stock_data,
                    technicals=technicals,
                    news=news,
                    report=ai_report,
                    chart_fig=chart_fig,
                )
                normalized = _normalize_pdf_bytes(pdf_bytes)
                if not normalized or not normalized.startswith(b"%PDF"):
                    raise ValueError("PDF dosyası geçersiz veya boş üretildi.")
                st.session_state["pdf_data"] = normalized
                st.session_state["pdf_filename"] = f"{current_ticker}_FinSight_Rapor.pdf"
                st.success("PDF ready — download below.")
            except Exception as e:
                st.session_state.pop("pdf_data", None)
                st.session_state.pop("pdf_filename", None)
                st.error(f"PDF generation failed: {e}")

    _pdf_ready = (
        st.session_state.get("pdf_filename", "").startswith(current_ticker)
        and _normalize_pdf_bytes(st.session_state.get("pdf_data")) is not None
    )
    if _pdf_ready:
        st.download_button(
            label="💾 PDF İndir",
            data=st.session_state["pdf_data"],
            file_name=st.session_state["pdf_filename"],
            mime="application/pdf",
            width="stretch",
            key=f"pdf_download_{current_ticker}",
        )
