"""
BIST AI Finansal Analist — Ana giriş noktası (CLI).
Hisse kodu alır, verileri çeker, analiz eder, raporu gösterir.
"""
import config  # noqa: F401 — Windows UTF-8 + SSL (Türkçe klasör yolu)

import json
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data_fetcher import get_stock_info, get_price_history
from technical_analysis import compute_indicators
from news_sentiment import get_news_with_sentiment, format_news_for_prompt
from llm_analyzer import analyze_stock

console = Console()


def run_analysis(ticker: str) -> dict:
    """Tam analiz pipeline'ını çalıştırır."""

    # 1) Temel veriler
    console.print(f"\n[bold cyan]📊 {ticker} verileri çekiliyor...[/]")
    stock_data = get_stock_info(ticker)
    console.print(f"   ✓ Şirket: [bold]{stock_data['sirket_adi']}[/]")
    console.print(f"   ✓ Son Fiyat: [bold green]{stock_data['son_fiyat']} TL[/]")

    # 2) Fiyat geçmişi + Teknik göstergeler
    console.print("[bold cyan]📈 Teknik göstergeler hesaplanıyor...[/]")
    df = get_price_history(ticker, period="1y")
    technicals = compute_indicators(df)
    console.print(f"   ✓ RSI: [bold]{technicals['rsi_degeri']}[/] → {technicals['rsi_sinyal']}")
    console.print(f"   ✓ MACD: [bold]{technicals['macd_durumu']}[/]")
    console.print(f"   ✓ Bollinger: [bold]{technicals['bollinger_durumu']}[/]")

    # 3) Haberler
    console.print("[bold cyan]📰 Son haberler taranıyor...[/]")
    news = get_news_with_sentiment(ticker)
    news_text = format_news_for_prompt(news)
    console.print(f"   ✓ {len(news)} haber bulundu")

    # 4) LLM analizi
    console.print("[bold cyan]🤖 Yapay zeka analizi yapılıyor...[/]\n")
    report = analyze_stock(stock_data, technicals, news_text)

    return report


def display_report(ticker: str, report: dict):
    """Raporu zengin formatta terminale yazdırır."""
    ozet = report.get("analiz_ozeti", {})
    risk = report.get("risk_analizi", {})

    # Başlık
    console.print(Panel(
        f"[bold white]{ticker}[/] — AI Finansal Analiz Raporu",
        style="bold magenta",
        expand=False,
    ))

    # Analiz özeti tablosu
    table = Table(title="📋 Analiz Özeti", show_lines=True, border_style="cyan")
    table.add_column("Alan", style="bold", width=24)
    table.add_column("Yorum", ratio=1)

    table.add_row("Genel Görüş", ozet.get("genel_gorus", "-"))
    table.add_row("Temel Analiz", ozet.get("temel_analiz_yorumu", "-"))
    table.add_row("Teknik Analiz", ozet.get("teknik_analiz_yorumu", "-"))
    table.add_row("Haber Sentiment", ozet.get("haber_sentiment_yorumu", "-"))
    table.add_row("Bollinger Bantları", ozet.get("bollinger_yorumu", "-"))

    console.print(table)

    # Risk paneli
    skor = risk.get("risk_skoru", "?")
    if isinstance(skor, (int, float)):
        if skor <= 30:
            renk = "green"
            etiket = "Düşük Risk"
        elif skor <= 60:
            renk = "yellow"
            etiket = "Orta Risk"
        else:
            renk = "red"
            etiket = "Yüksek Risk"
    else:
        renk = "white"
        etiket = "Belirlenemedi"

    console.print(Panel(
        f"[bold {renk}]Risk Skoru: {skor}/100 — {etiket}[/]\n\n"
        f"{risk.get('risk_gerekcesi', '-')}",
        title="⚠️ Risk Analizi",
        border_style=renk,
    ))

    # JSON çıktı (dosyaya kaydet)
    filename = f"{ticker}_rapor.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]💾 Tam rapor kaydedildi: {filename}[/]\n")


def main():
    """CLI giriş noktası."""
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        console.print(Panel(
            "[bold]BIST AI Finansal Analist[/]\n"
            "Hisse kodunu girerek analiz başlatabilirsiniz.\n"
            "Örnek: [cyan]python main.py THYAO[/]",
            style="blue",
        ))
        ticker = console.input("[bold cyan]Hisse kodu girin: [/]").strip().upper()

    if not ticker:
        console.print("[red]Hisse kodu boş olamaz![/]")
        sys.exit(1)

    try:
        report = run_analysis(ticker)
        display_report(ticker, report)
    except ValueError as e:
        console.print(f"[red]Hata: {e}[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Beklenmeyen hata: {e}[/]")
        raise


if __name__ == "__main__":
    main()
