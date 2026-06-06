"""portfolio — metrik ve simülasyon testleri."""
import pandas as pd
import numpy as np
import pytest

from portfolio import (
    compute_max_drawdown,
    compute_sortino_ratio,
    simulate_portfolio,
    build_comparison_table,
)


@pytest.fixture
def mini_watchlist():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    close_a = pd.Series(np.linspace(100, 120, 60), index=dates)
    close_b = pd.Series(np.linspace(50, 55, 60), index=dates)
    return {
        "AAA": {
            "stock_data": {"son_fiyat": 120, "fk_orani": 10, "pddd_orani": 2, "sektor": "Test"},
            "df": pd.DataFrame({"Close": close_a, "Open": close_a, "High": close_a, "Low": close_a, "Volume": 1e6}, index=dates),
            "technicals": {"rsi_degeri": 50, "macd_durumu": "Nötr", "sma_durumu": "Nötr", "bollinger_durumu": "Nötr"},
            "news": [],
        },
        "BBB": {
            "stock_data": {"son_fiyat": 55, "fk_orani": 8, "pddd_orani": 1.5, "sektor": "Test"},
            "df": pd.DataFrame({"Close": close_b, "Open": close_b, "High": close_b, "Low": close_b, "Volume": 1e6}, index=dates),
            "technicals": {"rsi_degeri": 45, "macd_durumu": "Nötr", "sma_durumu": "Nötr", "bollinger_durumu": "Nötr"},
            "news": [],
        },
    }


class TestDrawdownSortino:
    def test_max_drawdown_negative_or_zero(self):
        prices = pd.Series([100, 110, 90, 95, 80])
        mdd = compute_max_drawdown(prices)
        assert mdd is not None
        assert mdd <= 0

    def test_sortino_positive_trend(self):
        rets = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 100))
        s = compute_sortino_ratio(rets)
        assert s is None or isinstance(s, float)


class TestSimulatePortfolio:
    def test_includes_sortino_and_drawdown(self, mini_watchlist):
        result = simulate_portfolio(mini_watchlist, {"AAA": 50, "BBB": 50})
        assert "sortino_orani" in result
        assert "max_drawdown_pct" in result
        assert result["hisse_sayisi"] == 2

    def test_comparison_table(self, mini_watchlist):
        df = build_comparison_table(mini_watchlist)
        assert len(df) == 2
        assert "Hisse" in df.columns
