"""db — watchlist kalıcılık testleri."""
import pytest

import db as db_mod


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "test.db")
    db_mod.init_db()
    yield
    if db_mod.DB_PATH.exists():
        db_mod.DB_PATH.unlink()


class TestWatchlist:
    def test_add_and_get(self):
        db_mod.add_ticker("THYAO")
        db_mod.add_ticker("ASELS")
        wl = db_mod.get_watchlist()
        assert wl == ["THYAO", "ASELS"]

    def test_remove_ticker(self):
        db_mod.add_ticker("THYAO")
        db_mod.add_ticker("ASELS")
        db_mod.remove_ticker("THYAO")
        assert db_mod.get_watchlist() == ["ASELS"]

    def test_clear_watchlist(self):
        db_mod.add_ticker("THYAO")
        db_mod.clear_watchlist()
        assert db_mod.get_watchlist() == []

    def test_duplicate_ignored(self):
        db_mod.add_ticker("THYAO")
        db_mod.add_ticker("THYAO")
        assert db_mod.get_watchlist() == ["THYAO"]
