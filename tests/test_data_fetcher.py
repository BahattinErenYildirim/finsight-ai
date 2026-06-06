"""data_fetcher — cache ve yardımcı fonksiyon testleri."""
import threading
import time
from unittest.mock import patch


import data_fetcher as df_mod


class TestCacheThreadSafety:
    def test_clear_cache_under_lock(self):
        df_mod._info_cache["TEST"] = (time.time(), {"hisse_kodu": "TEST"})
        df_mod.clear_cache()
        assert "TEST" not in df_mod._info_cache

    def test_concurrent_cache_writes(self):
        df_mod.clear_cache()

        def writer(key: str):
            with df_mod._cache_lock:
                df_mod._info_cache[key] = (time.time(), {"hisse_kodu": key})

        threads = [threading.Thread(target=writer, args=(f"T{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(df_mod._info_cache) == 10
        df_mod.clear_cache()


class TestSafeRound:
    def test_none_returns_yetersiz(self):
        assert df_mod._safe_round(None) == "Yetersiz Veri"

    def test_rounds_float(self):
        assert df_mod._safe_round(3.14159) == 3.14


class TestGetStockInfo:
    @patch.object(df_mod, "_fetch_yf_info")
    def test_cache_hit_skips_fetch(self, mock_info):
      df_mod.clear_cache()
      mock_info.return_value = {
          "longName": "Test A.S.",
          "currentPrice": 100.0,
          "sector": "Industrials",
      }
      with patch.object(df_mod.yf, "Ticker"):
          first = df_mod.get_stock_info("THYAO")
          second = df_mod.get_stock_info("THYAO")
      assert first["hisse_kodu"] == "THYAO"
      assert second["hisse_kodu"] == "THYAO"
      assert mock_info.call_count == 1
      df_mod.clear_cache()
