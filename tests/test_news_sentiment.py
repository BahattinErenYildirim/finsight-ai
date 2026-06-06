"""Haber modülü testleri."""
import json
from unittest.mock import patch

from news_sentiment import (
    _fallback_keyword_sentiment,
    _normalize_sentiment_label,
    _parse_news_item,
    _parse_sentiment_json,
    format_news_for_prompt,
    summarize_news,
)


class TestParseNewsItem:
    def test_legacy_format(self):
        raw = {
            "title": "THYAO kârını artırdı",
            "link": "https://finance.yahoo.com/news/1",
            "summary": "Kısa özet",
        }
        item = _parse_news_item(raw)
        assert item["title"] == "THYAO kârını artırdı"
        assert item["link"].startswith("https://")
        assert item["summary"] == "Kısa özet"

    def test_yfinance_2024_content_format(self):
        raw = {
            "id": "abc",
            "content": {
                "title": "Turkish Airlines expands fleet",
                "summary": "<p>Expansion plans</p>",
                "pubDate": "2026-05-18T10:00:00Z",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://finance.yahoo.com/news/2"},
            },
        }
        item = _parse_news_item(raw)
        assert item is not None
        assert "fleet" in item["title"]
        assert item["publisher"] == "Reuters"
        assert item["link"].startswith("https://")
        assert "Expansion" in item["summary"]

    def test_empty_title_returns_none(self):
        assert _parse_news_item({"content": {"title": ""}}) is None


class TestSentiment:
    def test_keyword_positive(self):
        assert _fallback_keyword_sentiment("Record profit growth surge") == "Pozitif"

    def test_keyword_negative(self):
        assert _fallback_keyword_sentiment("Loss decline lawsuit bankruptcy") == "Negatif"

    def test_normalize_labels(self):
        assert _normalize_sentiment_label("positive") == "Pozitif"
        assert _normalize_sentiment_label("NEGATIF") == "Negatif"
        assert _normalize_sentiment_label("neutral") == "Nötr"

    def test_parse_sentiment_json(self):
        raw = json.dumps(["Pozitif", "Nötr", "Negatif"])
        out = _parse_sentiment_json(raw, 3)
        assert out == ["Pozitif", "Nötr", "Negatif"]


class TestSummarize:
    def test_summarize_empty(self):
        s = summarize_news([])
        assert s["count"] == 0

    def test_summarize_mixed(self):
        news = [
            {"sentiment": "Pozitif"},
            {"sentiment": "Pozitif"},
            {"sentiment": "Negatif"},
        ]
        s = summarize_news(news)
        assert s["count"] == 3
        assert s["pozitif"] == 2
        assert s["skor"] > 0


class TestFormatPrompt:
    def test_format_includes_summary_header(self):
        news = [
            {
                "title": "Test haber",
                "sentiment": "Pozitif",
                "summary": "Özet metni",
                "published_display": "2 saat önce",
                "publisher": "Reuters",
            }
        ]
        text = format_news_for_prompt(news)
        assert "HABER ÖZETİ" in text
        assert "Test haber" in text
        assert "Pozitif" in text


class TestGetNewsWithSentiment:
    @patch("news_sentiment._fetch_raw_news")
    @patch("news_sentiment._batch_analyze_sentiment_with_llm")
    def test_pipeline(self, mock_llm, mock_fetch):
        mock_fetch.return_value = [
            {"title": "Profit beat expectations", "link": "https://example.com/a"},
        ]
        mock_llm.return_value = ["Pozitif"]

        from news_sentiment import get_news_with_sentiment

        result = get_news_with_sentiment("THYAO", max_items=5)
        assert len(result) == 1
        assert result[0]["sentiment"] == "Pozitif"
        assert result[0]["link"].startswith("https://")
