"""
News & sentiment — fetches recent headlines via yfinance.
LLM sentiment (Gemini/Ollama) or TR+EN keyword fallback.

Supports yfinance 2024+ (content.title, canonicalUrl) and legacy shapes.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone

import yfinance as yf

from config import BIST_SUFFIX, GEMINI_API_KEY, GEMINI_MODEL, is_llm_configured, is_ollama_provider

logger = logging.getLogger("finsight.news")

SENTIMENT_LABELS = ("Pozitif", "Negatif", "Nötr")

# TR / EN keywords when LLM is unavailable
_POSITIVE_TR = [
    "kâr", "kar", "büyüme", "artış", "rekor", "yüksel", "anlaşma", "ihracat",
    "temettü", "yatırım", "inovasyon", "genişleme", "olumlu", "güçlü",
    "talep", "pozitif", "başarı", "kapasite", "sözleşme", "ortaklık",
    "iyileşme", "toparlan",
]
_NEGATIVE_TR = [
    "zarar", "düşüş", "kayıp", "risk", "borç", "maliyet", "enflasyon",
    "daralma", "negatif", "ceza", "soruşturma", "dava", "iflas", "küçülme",
    "satış baskı", "belirsizlik", "gerileme", "tehdit", "kötüleş",
    "devalüasyon", "faiz artış", "olumsuz",
]
_POSITIVE_EN = [
    "profit", "growth", "record", "rise", "gain", "agreement", "export",
    "dividend", "investment", "innovation", "expansion", "positive", "strong",
    "demand", "success", "capacity", "contract", "partnership", "recovery",
    "upgrade", "outperform", "beat", "exceed", "surge", "rally", "revenue",
    "earnings beat", "raised guidance", "buyback", "acquisition",
]
_NEGATIVE_EN = [
    "loss", "decline", "fall", "risk", "debt", "cost", "inflation",
    "contraction", "negative", "penalty", "investigation", "lawsuit",
    "bankruptcy", "selling pressure", "uncertainty", "deterioration",
    "threat", "worsen", "downgrade", "underperform", "miss", "below",
    "slump", "drop", "warning", "layoff", "cut", "default", "fine",
]
_ALL_POSITIVE = _POSITIVE_TR + _POSITIVE_EN
_ALL_NEGATIVE = _NEGATIVE_TR + _NEGATIVE_EN

_LLM_SENTIMENT_SYSTEM = (
    "Sen BIST ve küresel piyasalar için haber başlığı duyarlılık analistisin. "
    "Her başlık için yalnızca Pozitif, Negatif veya Nötr etiketi seç. "
    "Yanıtın SADECE JSON dizisi olsun; örnek: [\"Pozitif\", \"Nötr\", \"Negatif\"]"
)


def get_news_with_sentiment(ticker: str, max_items: int = 10) -> list[dict]:
    """Fetch recent headlines and assign sentiment labels."""
    symbol = f"{ticker.upper()}{BIST_SUFFIX}"
    raw_items = _fetch_raw_news(symbol, max_items)
    parsed: list[dict] = []

    for raw in raw_items:
        item = _parse_news_item(raw)
        if item and not _is_duplicate(item, parsed):
            parsed.append(item)

    parsed.sort(key=_published_sort_key, reverse=True)
    parsed = parsed[:max_items]

    if not parsed:
        return []

    titles = [p["title"] for p in parsed]
    sentiments = _batch_analyze_sentiment_with_llm(titles)

    for item, sentiment in zip(parsed, sentiments):
        item["sentiment"] = sentiment
        item["sentiment_score"] = _sentiment_to_score(sentiment)

    return parsed


def summarize_news(news_list: list[dict]) -> dict:
    """Aggregate stats for the news tab and summary cards."""
    if not news_list:
        return {
            "count": 0,
            "pozitif": 0,
            "negatif": 0,
            "notr": 0,
            "skor": 0,
            "etiket": "No data",
            "yuzde_pozitif": 0.0,
        }

    pos = sum(1 for n in news_list if n.get("sentiment") == "Pozitif")
    neg = sum(1 for n in news_list if n.get("sentiment") == "Negatif")
    neu = len(news_list) - pos - neg
    total = len(news_list)
    # score: -100 (very negative) .. +100 (very positive)
    skor = round(((pos - neg) / total) * 100) if total else 0

    if skor >= 35:
        etiket = "Positive mood"
    elif skor <= -35:
        etiket = "Negative mood"
    else:
        etiket = "Mixed / neutral mood"

    return {
        "count": total,
        "pozitif": pos,
        "negatif": neg,
        "notr": neu,
        "skor": skor,
        "etiket": etiket,
        "yuzde_pozitif": round(100 * pos / total, 1) if total else 0.0,
    }


def format_news_for_prompt(news_list: list) -> str:
    """LLM raporu için zenginleştirilmiş haber metni."""
    if not news_list:
        return "Son dönemde bu hisseyle ilgili haber bulunamadı."

    summary = summarize_news(news_list)
    header = (
        f"[HABER ÖZETİ] {summary['count']} haber — "
        f"Pozitif: {summary['pozitif']}, Negatif: {summary['negatif']}, "
        f"Nötr: {summary['notr']}. Gündem skoru: {summary['skor']}/100 ({summary['etiket']}).\n"
    )
    lines = []
    for i, item in enumerate(news_list, 1):
        meta = item.get("published_display") or ""
        pub = f" | {item['publisher']}" if item.get("publisher") else ""
        date_part = f" ({meta}{pub})" if meta or pub else ""
        body = item.get("summary", "").strip()
        extra = f"\n   Özet: {body[:200]}..." if len(body) > 200 else (f"\n   Özet: {body}" if body else "")
        lines.append(
            f"{i}. [{item['sentiment']}] {item['title']}{date_part}{extra}"
        )
    return header + "\n".join(lines)


# ── Veri çekme & normalizasyon ────────────────────────────────────────────────

def _fetch_raw_news(symbol: str, max_items: int) -> list:
    want = min(max(max_items * 3, 15), 40)
    for attempt in range(3):
        try:
            stock = yf.Ticker(symbol)
            raw: list = []
            if hasattr(stock, "get_news"):
                try:
                    raw = stock.get_news(count=want) or []
                except TypeError:
                    raw = stock.get_news(count=want) or []
            if not raw:
                raw = stock.news or []
            if raw:
                return raw
        except Exception as e:
            logger.debug("Haber çekme denemesi %d: %s", attempt + 1, e)
        time.sleep(0.5 * (attempt + 1))
    return []


def _parse_news_item(item: dict) -> dict | None:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}

    title = (item.get("title") or content.get("title") or "").strip()
    if not title:
        return None

    link = _extract_link(item, content)
    summary = (
        content.get("summary")
        or content.get("description")
        or item.get("summary")
        or ""
    )
    if isinstance(summary, str):
        summary = re.sub(r"<[^>]+>", "", summary).strip()[:600]
    else:
        summary = ""

    publisher = ""
    prov = content.get("provider")
    if isinstance(prov, dict):
        publisher = (prov.get("displayName") or prov.get("name") or "").strip()
    publisher = publisher or str(item.get("publisher", "")).strip()

    pub_raw = (
        content.get("pubDate")
        or content.get("displayTime")
        or item.get("providerPublishTime")
        or item.get("published_at")
    )

    return {
        "title": title,
        "summary": summary,
        "link": link,
        "publisher": publisher,
        "published_at": pub_raw,
        "published_display": _format_published(pub_raw),
    }


def _extract_link(item: dict, content: dict) -> str:
    link = (item.get("link") or "").strip()
    if link.startswith("http"):
        return link
    for key in ("canonicalUrl", "clickThroughUrl", "previewUrl"):
        url_obj = content.get(key)
        if isinstance(url_obj, dict):
            u = (url_obj.get("url") or "").strip()
            if u.startswith("http"):
                return u
        elif isinstance(url_obj, str) and url_obj.startswith("http"):
            return url_obj
    return ""


def _format_published(pub) -> str:
    dt = _parse_datetime(pub)
    if not dt:
        return ""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt.astimezone(timezone.utc)
    if delta.total_seconds() < 0:
        return dt.strftime("%d.%m.%Y %H:%M")
    if delta.days == 0:
        hours = delta.seconds // 3600
        if hours < 1:
            mins = max(1, delta.seconds // 60)
            return f"{mins}m ago"
        return f"{hours}h ago"
    if delta.days == 1:
        return "Yesterday"
    if delta.days < 7:
        return f"{delta.days}d ago"
    return dt.strftime("%d.%m.%Y")


def _parse_datetime(pub) -> datetime | None:
    if pub is None:
        return None
    try:
        if isinstance(pub, (int, float)):
            ts = float(pub)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        s = str(pub).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (TypeError, ValueError, OSError):
        return None


def _published_sort_key(item: dict) -> float:
    dt = _parse_datetime(item.get("published_at"))
    return dt.timestamp() if dt else 0.0


def _is_duplicate(item: dict, existing: list[dict]) -> bool:
    key = re.sub(r"\s+", " ", item["title"].lower())[:80]
    for ex in existing:
        if re.sub(r"\s+", " ", ex["title"].lower())[:80] == key:
            return True
    return False


# ── Sentiment ─────────────────────────────────────────────────────────────────

def _batch_analyze_sentiment_with_llm(titles: list[str]) -> list[str]:
    if not is_llm_configured():
        return [_fallback_keyword_sentiment(t) for t in titles]

    prompt = (
        "Aşağıdaki haber başlıklarının BIST yatırımcısı açısından duyarlılığını analiz et. "
        "Her başlık için SADECE 'Pozitif', 'Negatif' veya 'Nötr' kullan. "
        f"Tam {len(titles)} elemanlı JSON dizisi döndür.\n\nBaşlıklar:\n"
    )
    for i, t in enumerate(titles, 1):
        prompt += f"{i}. {t}\n"

    try:
        if is_ollama_provider():
            from llm_analyzer import generate_json_text

            raw = generate_json_text(prompt, system_prompt=_LLM_SENTIMENT_SYSTEM)
        else:
            raw = _gemini_sentiment_json(prompt)

        result_list = _parse_sentiment_json(raw, len(titles))
        if result_list:
            return result_list
        logger.warning("LLM haber sentiment uzunluk/format hatası, fallback.")
    except Exception as e:
        logger.warning("LLM haber sentiment hatası: %s", e)

    return [_fallback_keyword_sentiment(t) for t in titles]


def _gemini_sentiment_json(prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    return response.text.strip()


def _parse_sentiment_json(raw: str, expected_len: int) -> list[str] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(data, list) or len(data) != expected_len:
        return None

    return [_normalize_sentiment_label(str(x)) for x in data]


def _normalize_sentiment_label(label: str) -> str:
    t = label.strip().lower()
    if any(k in t for k in ("pozitif", "positive", "bull", "olumlu", "iyi")):
        return "Pozitif"
    if any(k in t for k in ("negatif", "negative", "bear", "olumsuz", "kötü")):
        return "Negatif"
    return "Nötr"


def _fallback_keyword_sentiment(text: str) -> str:
    text_lower = text.lower()
    pos_count = sum(1 for kw in _ALL_POSITIVE if kw in text_lower)
    neg_count = sum(1 for kw in _ALL_NEGATIVE if kw in text_lower)
    if pos_count > neg_count:
        return "Pozitif"
    if neg_count > pos_count:
        return "Negatif"
    return "Nötr"


def _sentiment_to_score(sentiment: str) -> int:
    return {"Pozitif": 1, "Negatif": -1}.get(sentiment, 0)
