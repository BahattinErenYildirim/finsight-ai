"""
Teknik Analiz Testleri — compute_indicators, support/resistance, backtest.
"""
import pandas as pd

from technical_analysis import (
    compute_indicators,
    compute_support_resistance,
    compute_signal_accuracy,
)


class TestComputeIndicators:
    def test_basic_returns_expected_keys(self, sample_price_df):
        result = compute_indicators(sample_price_df)
        required_keys = [
            "rsi_degeri", "rsi_sinyal",
            "macd_durumu", "sma_durumu",
            "bollinger_durumu",
            "destek_1", "direnc_1", "pivot",
            "backtest",
        ]
        for key in required_keys:
            assert key in result, f"Eksik anahtar: {key}"

    def test_rsi_range(self, sample_price_df):
        result = compute_indicators(sample_price_df)
        rsi = result["rsi_degeri"]
        if isinstance(rsi, (int, float)):
            assert 0 <= rsi <= 100, f"RSI aralık dışı: {rsi}"

    def test_rsi_signal_text(self, sample_price_df):
        result = compute_indicators(sample_price_df)
        valid_signals = [
            "Aşırı Alım Bölgesi (Dikkat!)",
            "Aşırı Satım Bölgesi (Fırsat?)",
            "Alım Baskısı Güçlü",
            "Satım Baskısı Var",
            "Nötr Bölge",
            "Yetersiz Veri",
        ]
        assert result["rsi_sinyal"] in valid_signals

    def test_insufficient_data_graceful(self):
        """Çok az veriyle hata vermemeli."""
        tiny_df = pd.DataFrame(
            {"Open": [100], "High": [105], "Low": [98], "Close": [102], "Volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = compute_indicators(tiny_df)
        assert isinstance(result, dict)

    def test_sma_values_make_sense(self, sample_price_df):
        result = compute_indicators(sample_price_df)
        sma50  = result.get("sma50_son")
        sma200 = result.get("sma200_son")
        close  = sample_price_df["Close"].iloc[-1]
        if sma50 and sma200:
            # SMA değerleri fiyatın makul katları içinde olmalı
            assert 0.1 * close <= sma50  <= 10 * close
            assert 0.1 * close <= sma200 <= 10 * close


class TestSupportResistance:
    def test_returns_all_levels(self, sample_price_df):
        result = compute_support_resistance(sample_price_df)
        for key in ["pivot", "destek_1", "destek_2", "direnc_1", "direnc_2"]:
            assert key in result

    def test_resistance_above_support(self, sample_price_df):
        result = compute_support_resistance(sample_price_df)
        # Sayısal değerlerse direnc > destek olmalı
        d1 = result["destek_1"]
        r1 = result["direnc_1"]
        if isinstance(d1, float) and isinstance(r1, float):
            assert r1 > d1, f"Direnç ({r1}) destek ({d1})'ten küçük olamaz"

    def test_pivot_between_support_and_resistance(self, sample_price_df):
        result = compute_support_resistance(sample_price_df)
        pivot = result["pivot"]
        d1    = result["destek_1"]
        r1    = result["direnc_1"]
        if all(isinstance(v, float) for v in [pivot, d1, r1]):
            assert d1 <= pivot <= r1


class TestSignalAccuracy:
    def test_insufficient_data(self):
        short_df = pd.DataFrame(
            {"Close": [100] * 30},
            index=pd.date_range("2024-01-01", periods=30),
        )
        result = compute_signal_accuracy(short_df)
        assert result["yeterli_veri"] is False

    def test_sufficient_data_returns_results(self, sample_price_df):
        result = compute_signal_accuracy(sample_price_df)
        assert result["yeterli_veri"] is True
        assert "toplam_sinyal" in result
        assert result["toplam_sinyal"] >= 0

    def test_accuracy_percentage_range(self, sample_price_df):
        result = compute_signal_accuracy(sample_price_df)
        for key in ["rsi_asiri_satim_basari", "rsi_asiri_alim_basari"]:
            val = result.get(key)
            if val is not None:
                assert 0 <= val <= 100, f"{key} aralık dışı: {val}"
