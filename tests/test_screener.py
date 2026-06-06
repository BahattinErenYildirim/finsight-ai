"""screener — filtre ve paralel tarama testleri."""
import pandas as pd
import pytest
from unittest.mock import patch

from screener import apply_filters, run_screener


@pytest.fixture
def sample_screener_df():
    return pd.DataFrame([
        {"Ticker": "A", "RSI": 30.0, "F/K": 8.0, "PD/DD": 1.2, "Hacim/Ort": 1.8, "1G %": 1.0, "Sektör": "Bank"},
        {"Ticker": "B", "RSI": 50.0, "F/K": 15.0, "PD/DD": 3.0, "Hacim/Ort": 0.9, "1G %": -1.0, "Sektör": "Tech"},
        {"Ticker": "C", "RSI": 70.0, "F/K": 5.0, "PD/DD": 0.8, "Hacim/Ort": 2.1, "1G %": 0.5, "Sektör": "Bank"},
    ])


class TestApplyFilters:
    def test_rsi_oversold(self, sample_screener_df):
        out = apply_filters(sample_screener_df, rsi_filter="Aşırı Satım (RSI<35)")
        assert list(out["Ticker"]) == ["A"]

    def test_pddd_max(self, sample_screener_df):
        out = apply_filters(sample_screener_df, pddd_max=1.5)
        assert set(out["Ticker"]) == {"A", "C"}

    def test_vol_ratio_min(self, sample_screener_df):
        out = apply_filters(sample_screener_df, vol_ratio_min=1.5)
        assert set(out["Ticker"]) == {"A", "C"}


class TestRunScreener:
    @patch("screener._fetch_screener_data")
    def test_skips_failed_futures(self, mock_fetch):
        mock_fetch.side_effect = [None, {"Ticker": "OK", "RSI": 40.0}]
        with patch("screener.BIST_UNIVERSE", ["X", "Y"]):
            df = run_screener(["X", "Y"], max_workers=2)
        assert len(df) == 1
        assert df.iloc[0]["Ticker"] == "OK"
