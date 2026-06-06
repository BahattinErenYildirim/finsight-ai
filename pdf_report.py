"""
PDF report generator — BIST analysis report (fpdf2 + kaleido for charts).
"""
import logging
import os
import tempfile
from datetime import datetime

from fpdf import FPDF
import plotly.graph_objects as go

logger = logging.getLogger("finsight.pdf")

# ── Renk paletleri (RGB) ─────────────────────────────────────────────────────
_C_DARK = {
    "primary":    (99, 102, 241),
    "success":    (38, 166, 154),
    "warning":    (251, 191, 36),
    "danger":     (239, 83, 80),
    "dark":       (15, 17, 26),
    "card_bg":    (26, 29, 46),
    "text":       (226, 232, 240),
    "text_muted": (107, 114, 128),
    "white":      (255, 255, 255),
    "border":     (42, 45, 66),
}

_C_LIGHT = {
    "primary":    (79, 70, 229),
    "success":    (5, 150, 105),
    "warning":    (217, 119, 6),
    "danger":     (220, 38, 38),
    "dark":       (248, 250, 252),
    "card_bg":    (241, 245, 249),
    "text":       (30, 41, 59),
    "text_muted": (100, 116, 139),
    "white":      (255, 255, 255),
    "border":     (203, 213, 225),
}

_C = _C_DARK


class FinSightPDF(FPDF):
    """FinSight AI corporate report template."""

    def __init__(self, ticker: str, company: str, theme: str = "dark"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.ticker = ticker
        self.company = company
        self.theme = theme
        self._toc_entries: list[tuple[str, int]] = []
        self.set_auto_page_break(auto=True, margin=20)

        global _C
        _C = _C_LIGHT if theme == "light" else _C_DARK

        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        if os.path.isdir(font_dir):
            self._load_custom_fonts(font_dir)
        else:
            logger.warning(
                "PDF font klasörü bulunamadı (%s). Helvetica fallback kullanılacak; "
                "Türkçe karakterler sınırlı olabilir.",
                font_dir,
            )

    def _load_custom_fonts(self, font_dir: str):
        """fonts/ klasöründen TTF fontları yükler (fpdf2 yeni API)."""
        regular = os.path.join(font_dir, "DejaVuSans.ttf")
        bold    = os.path.join(font_dir, "DejaVuSans-Bold.ttf")
        try:
            if os.path.isfile(regular):
                self.add_font("DejaVu", style="", fname=regular)
            else:
                logger.warning("DejaVuSans.ttf bulunamadı: %s", regular)
            if os.path.isfile(bold):
                self.add_font("DejaVu", style="B", fname=bold)
            else:
                logger.warning("DejaVuSans-Bold.ttf bulunamadı: %s", bold)
        except Exception as e:
            logger.warning("PDF özel font yüklenemedi, Helvetica kullanılacak: %s", e)

    def add_toc_entry(self, title: str):
        """İçindekiler için bölüm kaydı."""
        self._toc_entries.append((title, self.page_no()))

    def render_table_of_contents(self):
        """İçindekiler sayfası."""
        if not self._toc_entries:
            return
        self.add_page()
        self.section_title("İçindekiler")
        self._use_font(size=10)
        self.set_text_color(*_C["text"])
        for title, page in self._toc_entries:
            self.cell(0, 7, f"  {title}", ln=False)
            self.cell(0, 7, str(page), ln=True, align="R")
        self.ln(6)

    def _use_font(self, bold: bool = False, size: int = 10):
        """Mevcut fontu kullan, yoksa helvetica fallback."""
        if "dejavu" in self.fonts:
            self.set_font("DejaVu", "B" if bold else "", size=size)
        else:
            self.set_font("Helvetica", "B" if bold else "", size=size)

    # ── Header / Footer ──────────────────────────────────────────────────────
    def header(self):
        # Üst gradient bar
        self.set_fill_color(*_C["primary"])
        self.rect(0, 0, 210, 4, "F")

        self.set_y(8)
        self._use_font(bold=True, size=11)
        self.set_text_color(*_C["primary"])
        self.cell(0, 6, "FinSight AI", ln=False, align="L")

        self._use_font(size=8)
        self.set_text_color(*_C["text_muted"])
        self.cell(0, 6, datetime.now().strftime("%d.%m.%Y  %H:%M"), ln=True, align="R")

        # Ayırıcı çizgi
        self.set_draw_color(*_C["border"])
        self.line(10, 16, 200, 16)
        self.set_y(20)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(*_C["border"])
        self.line(10, self.get_y(), 200, self.get_y())
        self._use_font(size=7)
        self.set_text_color(*_C["text_muted"])
        self.cell(0, 8, f"FinSight AI  |  {self.ticker}  |  Sayfa {self.page_no()}/{{nb}}", align="C")

    # ── Bölüm başlığı ────────────────────────────────────────────────────────
    def section_title(self, title: str, color: tuple = _C["primary"]):
        self.ln(4)
        self.set_fill_color(*color)
        self.rect(10, self.get_y(), 3, 7, "F")
        self._use_font(bold=True, size=13)
        self.set_text_color(*_C["white"])
        self.set_x(16)
        self.cell(0, 7, title, ln=True)
        self.ln(3)

    # ── Bilgi kutusu ──────────────────────────────────────────────────────────
    def info_card(self, label: str, value: str, w: float = 45):
        x0 = self.get_x()
        y0 = self.get_y()
        self.set_fill_color(*_C["card_bg"])
        self.set_draw_color(*_C["border"])
        self.rect(x0, y0, w, 18, "DF")

        self._use_font(size=7)
        self.set_text_color(*_C["text_muted"])
        self.set_xy(x0 + 2, y0 + 2)
        self.cell(w - 4, 4, label)

        self._use_font(bold=True, size=12)
        self.set_text_color(*_C["white"])
        self.set_xy(x0 + 2, y0 + 8)
        self.cell(w - 4, 8, str(value)[:20])

        self.set_xy(x0 + w + 2, y0)

    # ── Analiz kartı ──────────────────────────────────────────────────────────
    def analysis_box(self, title: str, content: str, w: float = 90):
        x0 = self.get_x()
        y0 = self.get_y()

        self.set_fill_color(*_C["card_bg"])
        self.set_draw_color(*_C["border"])

        # Başlık
        self._use_font(bold=True, size=8)
        self.set_text_color(*_C["text_muted"])

        # İçerik yüksekliğini hesapla
        self._use_font(size=9)
        content_clean = self._safe_text(content)
        try:
            line_count = max(1, len(self._safe_multi_cell(w - 8, 5, content_clean, dry_run=True, output="LINES")))
        except Exception:
            line_count = 1
        box_h = 10 + (line_count * 5) + 4

        # Sayfa taşması kontrolü
        if y0 + box_h > 275:
            self.add_page()
            y0 = self.get_y()

        self.rect(x0, y0, w, box_h, "DF")

        self._use_font(bold=True, size=8)
        self.set_text_color(*_C["text_muted"])
        self.set_xy(x0 + 4, y0 + 3)
        self.cell(w - 8, 4, title.upper())

        self._use_font(size=9)
        self.set_text_color(*_C["text"])
        self.set_xy(x0 + 4, y0 + 10)
        try:
            self._safe_multi_cell(w - 8, 5, content_clean)
        except Exception:
            self.cell(w - 8, 5, "Metin sigmadi")

        return box_h

    # ── Risk göstergesi ───────────────────────────────────────────────────────
    def risk_indicator(self, score: int, reason: str):
        if score <= 30:
            color, label = _C["success"], "Dusuk Risk"
        elif score <= 60:
            color, label = _C["warning"], "Orta Risk"
        else:
            color, label = _C["danger"], "Yuksek Risk"

        y0 = self.get_y()

        # Sayfa taşması kontrolü
        if y0 + 30 > 275:
            self.add_page()
            y0 = self.get_y()

        # Arka plan kutu
        self.set_fill_color(*_C["card_bg"])
        self.set_draw_color(*color)
        self.rect(10, y0, 190, 28, "DF")

        # Skor
        self._use_font(bold=True, size=28)
        self.set_text_color(*color)
        self.set_xy(16, y0 + 2)
        self.cell(30, 12, str(score))

        self._use_font(bold=True, size=10)
        self.set_xy(16, y0 + 15)
        self.cell(30, 6, f"/ 100  {label}")

        # Gerekçe
        self._use_font(size=9)
        self.set_text_color(*_C["text"])
        self.set_xy(60, y0 + 4)
        try:
            self._safe_multi_cell(135, 5, self._safe_text(reason))
        except Exception:
            self.cell(135, 5, "Metin sigmadi")

        self.set_y(y0 + 32)

    # ── Haber listesi ─────────────────────────────────────────────────────────
    def news_table(self, news_list: list):
        if not news_list:
            self._use_font(size=9)
            self.set_text_color(*_C["text_muted"])
            self.cell(0, 6, "Haber bulunamadi.", ln=True)
            return

        for item in news_list:
            y0 = self.get_y()
            if y0 + 10 > 275:
                self.add_page()

            sentiment = item.get("sentiment", "Notr")
            color = (
                _C["success"] if sentiment == "Pozitif"
                else _C["danger"] if sentiment == "Negatif"
                else _C["text_muted"]
            )

            self.set_fill_color(*color)
            self.rect(10, self.get_y() + 1, 2, 5, "F")

            self._use_font(bold=True, size=8)
            self.set_text_color(*color)
            self.set_x(14)
            self.cell(22, 7, f"[{sentiment}]", ln=False)

            self._use_font(size=8)
            self.set_text_color(*_C["text"])
            title_text = self._safe_text(item.get("title", ""))[:90]
            self.cell(0, 7, title_text, ln=True)

            meta = " · ".join(
                p for p in (
                    item.get("published_display", ""),
                    item.get("publisher", ""),
                )
                if p
            )
            if meta:
                self._use_font(size=7)
                self.set_text_color(*_C["text_muted"])
                self.cell(0, 5, self._safe_text(meta)[:70], ln=True)

    def _safe_multi_cell(self, w, h, text, **kwargs):
        try:
            return self._safe_multi_cell(w, h, text, **kwargs)
        except Exception:
            self.cell(w if w > 0 else 0, h, "Metin sığmadı", ln=True)
            return ["Metin sığmadı"] if kwargs.get("output") == "LINES" else None

    def _safe_text(self, text: str) -> str:
        """Türkçe karakterleri latin1-safe hale getirir (font yoksa)."""
        if "dejavu" in self.fonts:
            return text
        # Fallback: Türkçe → ASCII yaklaşık
        replacements = {
            "ı": "i", "İ": "I", "ğ": "g", "Ğ": "G",
            "ü": "u", "Ü": "U", "ş": "s", "Ş": "S",
            "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
            "â": "a", "î": "i", "û": "u",
            "—": "-", "–": "-", "→": "->", "≈": "~",
            "\u200b": "", "\u00a0": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        try:
            text.encode("latin-1")
        except UnicodeEncodeError:
            text = text.encode("latin-1", errors="replace").decode("latin-1")
        return text


# ══════════════════════════════════════════════════════════════════════════════
# Ana üretici fonksiyon
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    stock_data: dict,
    technicals: dict,
    news: list,
    report: dict | None,
    chart_fig: go.Figure | None = None,
    theme: str = "dark",
) -> bytes:
    """
    Tam PDF raporu üretir ve bytes olarak döndürür.

    Args:
        stock_data: data_fetcher çıktısı
        technicals: technical_analysis çıktısı
        news: news_sentiment çıktısı (liste)
        report: llm_analyzer çıktısı (dict veya None)
        chart_fig: Plotly candlestick Figure (embed edilecek)

    Returns:
        PDF dosyası bytes
    """
    ticker = stock_data.get("hisse_kodu", "BIST")
    company = stock_data.get("sirket_adi", ticker)

    pdf = FinSightPDF(ticker, company, theme=theme)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── 1) Kapak başlık ──────────────────────────────────────────────────────
    pdf.ln(6)
    pdf._use_font(bold=True, size=22)
    pdf.set_text_color(*_C["white"])
    pdf.cell(0, 10, pdf._safe_text(company), ln=True)

    pdf._use_font(size=12)
    pdf.set_text_color(*_C["primary"])
    pdf.cell(0, 7, f"{ticker}  |  Financial Analysis Report", ln=True)

    pdf._use_font(size=9)
    pdf.set_text_color(*_C["text_muted"])
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.add_toc_entry("Temel Göstergeler")
    # ── 2) Temel göstergeler ─────────────────────────────────────────────────
    pdf.section_title("Temel Göstergeler")
    pdf.set_x(10)
    pdf.info_card("Last price", f"{stock_data.get('son_fiyat', '-')} TRY")
    pdf.info_card("P/E", str(stock_data.get("fk_orani", "-")))
    pdf.info_card("P/B", str(stock_data.get("pddd_orani", "-")))
    pdf.info_card("Debt/EBITDA", str(stock_data.get("borc_favok", "-")))
    pdf.ln(22)

    pdf.add_toc_entry("Teknik Göstergeler")
    # ── 3) Teknik göstergeler ────────────────────────────────────────────────
    pdf.section_title("Teknik Göstergeler")
    pdf.set_x(10)
    pdf.info_card("RSI (14)", str(technicals.get("rsi_degeri", "-")))
    pdf.info_card("SMA 50", str(technicals.get("sma50_son", "-")))
    pdf.info_card("SMA 200", str(technicals.get("sma200_son", "-")))
    pdf.ln(22)

    # Sinyal detayları
    y_start = pdf.get_y()
    pdf.set_xy(10, y_start)
    h1 = pdf.analysis_box("MACD", technicals.get("macd_durumu", "-"))
    pdf.set_xy(104, y_start)
    h2 = pdf.analysis_box("Bollinger", technicals.get("bollinger_durumu", "-"))
    pdf.set_y(y_start + max(h1, h2) + 4)

    y_start = pdf.get_y()
    pdf.set_xy(10, y_start)
    h1 = pdf.analysis_box("SMA", technicals.get("sma_durumu", "-"))
    pdf.set_xy(104, y_start)
    h2 = pdf.analysis_box("RSI signal", technicals.get("rsi_sinyal", "-"))
    pdf.set_y(y_start + max(h1, h2) + 4)

    if technicals.get("fibonacci", {}).get("seviyeler"):
        pdf.ln(2)
        fib = technicals["fibonacci"]
        pdf._use_font(size=9)
        pdf.set_text_color(*_C["text_muted"])
        levels_txt = " | ".join(f"{k}: {v}" for k, v in fib["seviyeler"].items())
        pdf._safe_multi_cell(0, 5, pdf._safe_text(f"Fibonacci: {levels_txt}"))
        pdf._safe_multi_cell(0, 5, pdf._safe_text(f"Bölge: {fib.get('bolge', '-')}"))
        pdf.ln(2)

    # ── 4) Grafik embed ──────────────────────────────────────────────────────
    if chart_fig is not None:
        pdf.add_toc_entry("Fiyat Grafiği")
        pdf.section_title("Fiyat Grafiği")
        try:
            chart_bytes = _export_chart_png(chart_fig)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(chart_bytes)
                tmp_path = tmp.name

            # Sayfa taşması kontrolü — grafik yeni sayfada
            if pdf.get_y() + 90 > 275:
                pdf.add_page()

            pdf.image(tmp_path, x=10, w=190)
            os.unlink(tmp_path)
        except Exception as e:
            pdf._use_font(size=9)
            pdf.set_text_color(*_C["text_muted"])
            pdf.cell(0, 6, f"Chart could not be embedded: {e}", ln=True)

    # ── 5) AI Analiz Raporu ──────────────────────────────────────────────────
    if report:
        pdf.add_toc_entry("AI Analiz Raporu")
        pdf.add_page()
        pdf.section_title("AI Analiz Raporu", color=_C["primary"])
        ozet = report.get("analiz_ozeti", {})
        risk = report.get("risk_analizi", {})

        # Risk göstergesi
        try:
            risk_score = int(float(risk.get("risk_skoru", 50)))
        except (TypeError, ValueError):
            risk_score = 50
        risk_score = max(0, min(100, risk_score))
        risk_reason = risk.get("risk_gerekcesi", "-")
        pdf.risk_indicator(risk_score, risk_reason)
        pdf.ln(2)

        # Analiz kartları — 2x2 grid
        cards = [
            ("Overview", ozet.get("genel_gorus", "-")),
            ("Fundamentals", ozet.get("temel_analiz_yorumu", "-")),
            ("Technical", ozet.get("teknik_analiz_yorumu", "-")),
            ("News sentiment", ozet.get("haber_sentiment_yorumu", "-")),
        ]
        for i in range(0, len(cards), 2):
            y_start = pdf.get_y()
            pdf.set_xy(10, y_start)
            h1 = pdf.analysis_box(cards[i][0], cards[i][1])
            if i + 1 < len(cards):
                pdf.set_xy(104, y_start)
                h2 = pdf.analysis_box(cards[i + 1][0], cards[i + 1][1])
            else:
                h2 = 0
            pdf.set_y(y_start + max(h1, h2) + 4)

        # Bollinger yorumu
        if ozet.get("bollinger_yorumu"):
            y_start = pdf.get_y()
            pdf.set_xy(10, y_start)
            pdf.analysis_box("Bollinger", ozet["bollinger_yorumu"], w=184)
    else:
        pdf.section_title("AI analysis")
        pdf._use_font(size=10)
        pdf.set_text_color(*_C["text_muted"])
        pdf.cell(0, 8, "AI analysis not available (configure LLM in .env).", ln=True)

    pdf.add_toc_entry("Haberler")
    # ── 6) News ──────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Haberler ve Duygu Analizi")
    pdf.news_table(news)

    # İçindekiler (kapak sonrası ek sayfa)
    if len(pdf._toc_entries) > 1:
        pdf.render_table_of_contents()

    # ── Footer disclaimer ────────────────────────────────────────────────────
    pdf.ln(10)
    pdf.set_draw_color(*_C["border"])
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf._use_font(size=7)
    pdf.set_text_color(*_C["text_muted"])
    pdf._safe_multi_cell(0, 4, pdf._safe_text(
        "DISCLAIMER: This report is not investment advice. "
        "For informational purposes only. "
        "Investment decisions should rely on your own research and professional advice. "
        "FinSight AI is not liable for losses arising from use of this report."
    ))

    return _pdf_output_bytes(pdf)


def _pdf_output_bytes(pdf: "FinSightPDF") -> bytes:
    """fpdf2 çıktısını Streamlit download_button için güvenli bytes yapar."""
    out = pdf.output(dest="S")
    if out is None:
        raise RuntimeError("PDF generation failed (empty output).")
    if isinstance(out, bytes):
        return out
    if isinstance(out, bytearray):
        return bytes(out)
    if isinstance(out, str):
        return out.encode("latin-1")
    raise RuntimeError(f"PDF beklenmeyen tip: {type(out)!r}")


def _export_chart_png(fig: go.Figure) -> bytes:
    """Plotly Figure'ı PNG bytes olarak export eder."""
    # PDF için beyaz arka planlı tema
    fig_copy = go.Figure(fig)
    fig_copy.update_layout(
        paper_bgcolor="#0d0f1a",
        plot_bgcolor="#0d0f1a",
        width=1200,
        height=500,
        margin=dict(l=50, r=30, t=30, b=30),
    )
    return fig_copy.to_image(format="png", scale=2, engine="kaleido")
