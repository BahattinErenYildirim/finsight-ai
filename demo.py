"""
FinSight AI — Demo Senaryosu
═══════════════════════════════════════════════════════════════
3 dakikalık jüri demo'su için hazırlanmış interaktif script.
Hisseler: THYAO (havacılık), ASELS (savunma), SISE (cam/kimya)

Kullanım:
    python demo.py
"""
import time
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()

# ── Demo hisseleri ────────────────────────────────────────────────────────────
DEMO_TICKERS = ["THYAO", "ASELS", "SISE"]

DEMO_SCRIPT = """
╔══════════════════════════════════════════════════════════════╗
║           🚀  FinSight AI — Demo Senaryosu  🚀              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1️⃣  Tekil Hisse Analizi  (THYAO)           ~60 sn          ║
║     • Temel veri çekimi                                      ║
║     • Teknik göstergeler (RSI, MACD, Bollinger)              ║
║     • Haber sentiment analizi                                ║
║     • Gemini AI rapor üretimi                                ║
║                                                              ║
║  2️⃣  Çoklu Karşılaştırma  (3 hisse)         ~30 sn          ║
║     • Watchlist tablosu                                      ║
║     • Korelasyon matrisi                                     ║
║     • Portföy simülasyonu                                    ║
║                                                              ║
║  3️⃣  Streamlit Dashboard                     ~30 sn          ║
║     • Canlı grafik demo                                      ║
║     • PDF rapor export                                       ║
║     • Portföy sekmesi                                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def step_banner(step_num: int, title: str, desc: str):
    """Demo adım banner'ı gösterir."""
    console.print()
    console.print(Panel(
        f"[bold white]Adım {step_num}:[/] {title}\n"
        f"[dim]{desc}[/]",
        style="bold cyan",
        box=box.DOUBLE,
        expand=False,
    ))
    console.print()


def demo_step1_single_analysis():
    """Adım 1: THYAO tekil analizi."""
    step_banner(1, "Tekil Hisse Analizi — THYAO",
                "Temel veri + teknik gösterge + haber sentiment + AI raporu")

    from data_fetcher import get_stock_info, get_price_history
    from technical_analysis import compute_indicators
    from news_sentiment import get_news_with_sentiment, format_news_for_prompt

    ticker = "THYAO"

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        console=console,
    ) as progress:
        # Temel veri
        task = progress.add_task(f"📊 {ticker} temel verileri çekiliyor...", total=None)
        stock_data = get_stock_info(ticker)
        progress.update(task, description=f"✅ {stock_data['sirket_adi']} — {stock_data['son_fiyat']} TL")
        time.sleep(0.5)

        # Fiyat geçmişi + teknik
        task2 = progress.add_task("📈 Teknik göstergeler hesaplanıyor...", total=None)
        df = get_price_history(ticker, period="1y")
        technicals = compute_indicators(df)
        progress.update(task2, description=f"✅ RSI: {technicals['rsi_degeri']} | MACD: {technicals['macd_durumu'][:30]}")
        time.sleep(0.5)

        # Haberler
        task3 = progress.add_task("📰 Haberler taranıyor...", total=None)
        news = get_news_with_sentiment(ticker, max_items=5)
        progress.update(task3, description=f"✅ {len(news)} haber bulundu")
        time.sleep(0.5)

    # Teknik özet tablosu
    tech_table = Table(title=f"📊 {ticker} Teknik Göstergeler", box=box.ROUNDED, border_style="cyan")
    tech_table.add_column("Gösterge", style="bold")
    tech_table.add_column("Değer")
    tech_table.add_column("Sinyal")
    tech_table.add_row("RSI (14)", str(technicals["rsi_degeri"]), technicals["rsi_sinyal"])
    tech_table.add_row("MACD", "—", technicals["macd_durumu"])
    tech_table.add_row("SMA 50/200", "—", technicals["sma_durumu"])
    tech_table.add_row("Bollinger", "—", technicals["bollinger_durumu"])
    console.print(tech_table)

    # AI analizi
    from config import is_llm_configured
    if is_llm_configured():
        console.print("\n[bold cyan]🤖 Gemini AI analizi başlatılıyor...[/]")
        from llm_analyzer import analyze_stock
        news_text = format_news_for_prompt(news)
        report = analyze_stock(stock_data, technicals, news_text)

        ozet = report.get("analiz_ozeti", {})
        risk = report.get("risk_analizi", {})
        skor = risk.get("risk_skoru", "?")

        if isinstance(skor, (int, float)):
            renk = "green" if skor <= 30 else ("yellow" if skor <= 60 else "red")
        else:
            renk = "white"

        console.print(Panel(
            f"[bold {renk}]Risk Skoru: {skor}/100[/]\n\n"
            f"[bold]Genel Görüş:[/] {ozet.get('genel_gorus', '-')}\n\n"
            f"[bold]Risk Gerekçesi:[/] {risk.get('risk_gerekcesi', '-')}",
            title="🤖 AI Finansal Analiz",
            border_style=renk,
        ))

        # JSON kaydet
        with open(f"{ticker}_demo_rapor.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        console.print(f"[dim]💾 Rapor kaydedildi: {ticker}_demo_rapor.json[/]")
    else:
        console.print("[yellow]⚠️ LLM yapılandırılmamış — AI analizi atlandı (.env / Ollama).[/]")

    return stock_data, technicals, news


def demo_step2_portfolio():
    """Adım 2: Çoklu hisse karşılaştırma."""
    step_banner(2, "Çoklu Hisse Karşılaştırma",
                f"Hisseler: {', '.join(DEMO_TICKERS)} → Watchlist + Korelasyon + Portföy")

    from portfolio import load_watchlist, build_comparison_table, compute_correlation_matrix, simulate_portfolio

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"📊 {len(DEMO_TICKERS)} hisse verisi çekiliyor...", total=None)
        watchlist = load_watchlist(DEMO_TICKERS, period="1y")
        success = sum(1 for v in watchlist.values() if "error" not in v)
        progress.update(task, description=f"✅ {success}/{len(DEMO_TICKERS)} hisse yüklendi")

    # Karşılaştırma tablosu
    comp_df = build_comparison_table(watchlist)
    table = Table(title="📋 Watchlist Karşılaştırma", box=box.ROUNDED, border_style="cyan")
    for col in comp_df.columns:
        table.add_column(col, style="bold" if col == "Hisse" else "")
    for _, row in comp_df.iterrows():
        table.add_row(*[str(v) for v in row.values])
    console.print(table)

    # Korelasyon
    corr = compute_correlation_matrix(watchlist)
    if corr is not None:
        console.print("\n[bold cyan]🔥 Korelasyon Matrisi[/]")
        corr_table = Table(box=box.SIMPLE, border_style="dim")
        corr_table.add_column("")
        for c in corr.columns:
            corr_table.add_column(c, justify="center")
        for idx, row in corr.iterrows():
            vals = []
            for v in row.values:
                color = "green" if v > 0.5 else ("red" if v < -0.3 else "white")
                vals.append(f"[{color}]{v:.3f}[/{color}]")
            corr_table.add_row(str(idx), *vals)
        console.print(corr_table)

    # Portföy simülasyonu (eşit ağırlık)
    valid = [t for t in DEMO_TICKERS if t in watchlist and "error" not in watchlist[t]]
    if len(valid) >= 2:
        eq_w = {t: round(100 / len(valid), 1) for t in valid}
        result = simulate_portfolio(watchlist, eq_w)

        console.print(Panel(
            f"[bold]Hisse Sayısı:[/] {result['hisse_sayisi']}\n"
            f"[bold]Ağırlıklar:[/] {result['agirliklar']}\n"
            f"[bold]Diversifikasyon:[/] {result['diversifikasyon_skoru']}/100\n"
            f"[bold]Yıllık Getiri:[/] %{result.get('portfoy_yillik_getiri', '-')}\n"
            f"[bold]Volatilite:[/] %{result.get('portfoy_volatilite', '-')}\n"
            f"[bold]Sharpe Oranı:[/] {result.get('sharpe_orani', '-')}\n"
            f"[bold]Sektörler:[/] {result.get('sektor_dagilimi', {})}",
            title="⚖️ Portföy Simülasyonu (Eşit Ağırlık)",
            border_style="magenta",
        ))


def demo_step3_dashboard_info():
    """Adım 3: Dashboard başlatma talimatı."""
    step_banner(3, "Streamlit Dashboard",
                "İnteraktif web dashboard'u başlatın")

    console.print(Panel(
        "[bold white]Dashboard'u başlatmak için:[/]\n\n"
        "  [bold cyan]streamlit run app.py[/]\n\n"
        "[bold]Özellikler:[/]\n"
        "  📈 İnteraktif candlestick grafik + RSI + MACD\n"
        "  🤖 AI destekli analiz raporu\n"
        "  📰 Haber sentiment gösterimi\n"
        "  📊 Portföy takibi (2+ hisse ekleyin)\n"
        "  📄 Tek tık PDF rapor export\n"
        "  ⚡ 5 dk cache + rate limit koruması",
        title="🌐 Web Dashboard",
        border_style="green",
    ))


def main():
    """Demo senaryosunu çalıştırır."""
    console.print(DEMO_SCRIPT, style="bold")

    console.print("[bold]Demo başlatılsın mı? (E/h):[/] ", end="")
    answer = input().strip().lower()
    if answer and answer != "e":
        console.print("[dim]Demo iptal edildi.[/]")
        return

    try:
        # Adım 1: Tekil analiz
        demo_step1_single_analysis()
        console.print("\n[dim]— Enter'a basın: Adım 2'ye geçilecek —[/]")
        input()

        # Adım 2: Portföy karşılaştırma
        demo_step2_portfolio()
        console.print("\n[dim]— Enter'a basın: Adım 3'e geçilecek —[/]")
        input()

        # Adım 3: Dashboard
        demo_step3_dashboard_info()

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo sonlandırıldı.[/]")
        return

    # Final
    console.print()
    console.print(Panel(
        "[bold green]✅ Demo tamamlandı![/]\n\n"
        f"[bold]Analiz edilen hisseler:[/] {', '.join(DEMO_TICKERS)}\n"
        "[bold]Kullanılan teknolojiler:[/]\n"
        "  • yfinance + ta (veri & teknik analiz)\n"
        "  • Google Gemini 2.0 Flash (AI rapor)\n"
        "  • Streamlit + Plotly (dashboard)\n"
        "  • fpdf2 + kaleido (PDF export)\n"
        "  • Rich (CLI arayüz)",
        title="🎉 FinSight AI Demo",
        border_style="bold green",
        expand=False,
    ))


if __name__ == "__main__":
    main()
