"""pdf_report — PDF üretim testleri."""
import pytest

from pdf_report import generate_report, FinSightPDF


@pytest.fixture
def minimal_inputs(sample_stock_data, sample_report):
    technicals = {
        "rsi_degeri": 55,
        "rsi_sinyal": "Nötr Bölge",
        "macd_durumu": "Nötr",
        "sma_durumu": "Nötr",
        "sma50_son": 100,
        "sma200_son": 95,
        "bollinger_durumu": "Nötr",
        "fibonacci": {"seviyeler": {"50.0%": 100}, "bolge": "Test"},
        "obv_son": 1e6,
        "obv_trend": "Yükselen",
        "vwap_son": 99.5,
        "vwap_durumu": "VWAP civarı",
    }
    return sample_stock_data, technicals, [], sample_report


class TestGenerateReport:
    def test_dark_theme_bytes(self, minimal_inputs):
        sd, ta, news, report = minimal_inputs
        pdf_bytes = generate_report(sd, ta, news, report, chart_fig=None, theme="dark")
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"

    def test_light_theme_bytes(self, minimal_inputs):
        sd, ta, news, report = minimal_inputs
        pdf_bytes = generate_report(sd, ta, news, report, theme="light")
        assert pdf_bytes[:4] == b"%PDF"


class TestFinSightPDF:
    def test_toc_entries(self):
        pdf = FinSightPDF("THYAO", "Test", theme="dark")
        pdf.add_page()
        pdf.add_toc_entry("Bölüm 1")
        assert len(pdf._toc_entries) == 1
