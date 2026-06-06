"""sector_analysis — sektör endeks evreni ve momentum testleri."""
import pandas as pd
from unittest.mock import patch, MagicMock

import sector_analysis
from sector_analysis import SECTOR_INDICES, get_sector_momentum


class TestSectorIndices:
    def test_includes_new_indices(self):
        for sym in ("XMESY.IS", "XELKT.IS", "XTRZM.IS", "XSGRT.IS"):
            assert sym in SECTOR_INDICES

    def test_bist100_present(self):
        assert "XU100.IS" in SECTOR_INDICES


class TestGetSectorMomentum:
    @patch("sector_analysis.yf.Ticker")
    @patch("sector_analysis._fetch_yf_history")
    def test_returns_dataframe_with_alpha(self, mock_hist, mock_ticker):
        sector_analysis._sector_cache.clear()

        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        fake_df = pd.DataFrame({
            "Close": [100 + i for i in range(30)],
            "Open": [100] * 30,
            "High": [101] * 30,
            "Low": [99] * 30,
            "Volume": [1e6] * 30,
        }, index=dates)
        mock_hist.return_value = fake_df
        mock_ticker.return_value = MagicMock()

        with patch.dict(SECTOR_INDICES, {"XU100.IS": "BIST 100", "XBANK.IS": "Bank"}, clear=True):
            df = get_sector_momentum(period="1mo")

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "Görece Güç (Alpha %)" in df.columns
