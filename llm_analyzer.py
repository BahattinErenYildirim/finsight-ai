"""
LLM analyzer — structured financial reports via Gemini or Ollama.

Gemini: exponential backoff on 429 rate limits.
Graceful handling of JSON parse and API errors.
"""
import json
import os
import re
import time
import logging
import requests
from google import genai
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    is_llm_configured,
    is_ollama_provider,
)

logger = logging.getLogger("finsight.llm")

# Gemini rate-limit retries
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
BASE_DELAY = 2.0
BACKOFF_FACTOR = 2.0

# Extra JSON constraints for Ollama (reasoning models)
OLLAMA_JSON_SUFFIX = (
    "\n\n[OLLAMA] Yanıtın TEK parça geçerli JSON olsun. "
    "Düşünme adımları, liste, markdown veya \\boxed kullanma."
)

SYSTEM_PROMPT = (
    "Sen, Borsa İstanbul (BIST) piyasalarında uzmanlaşmış, SPK mevzuatlarına "
    "ve uluslararası finansal analiz standartlarına (CFA) hakim, kıdemli bir "
    "Yapay Zeka Finansal Analistisin.\n\n"
    "Görevin: Sana sağlanan ham finansal verileri, teknik göstergeleri ve son "
    "haber duyarlılıklarını (sentiment) sentezleyerek, bireysel yatırımcıların "
    "anlayabileceği, rasyonel, objektif ve veri odaklı bir Türkçe yatırım özeti "
    "ve risk analizi raporu üretmektir.\n\n"
    "Kritik Kurallar:\n"
    '1. Kesinlikle doğrudan "AL", "SAT" veya "TUT" gibi yatırım tavsiyesi verme. '
    'Bunun yerine "Güçlü Görünüm", "Yüksek Risk", "Nötr-Temkinli" gibi analitik ifadeler kullan.\n'
    "2. Sana verilen ham veriler dışında KENDİNDEN RAKAM VEYA VERİ UYDURMA. "
    '"Yetersiz Veri" olan alanlar için o notu koru.\n'
    "3. Çıktıyı KESİNLİKLE VE SADECE sana verilen JSON formatında döndür.\n"
    "4. Her yorum en az 2-3 cümle olsun, somut veriye referans versin.\n"
    "5. Destek ve direnç seviyelerini analizde mutlaka değerlendir.\n\n"
    "ÖNEMLİ: Aşağıdaki örnek, beklenen kalite ve derinlik standardını gösterir:\n\n"
    "ÖRNEK ÇIKTI (kalite referansı):\n"
    '{"analiz_ozeti":{"genel_gorus":"Nötr-Temkinli — THYAO teknik göstergelerde '
    "karışık sinyaller veriyor. RSI 52 ile nötr bölgede seyrederken, MACD histogram "
    "daralması kısa vadeli momentum kaybına işaret ediyor. Ancak F/K oranı 8.2 ile "
    "sektör ortalaması 12.5'in belirgin altında, bu da görece ucuz bir değerlemeye "
    'işaret ediyor. Destek seviyesi 285 TL korunduğu sürece orta vadeli görünüm olumlu.",'
    '"temel_analiz_yorumu":"F/K oranı 8.2 ile havacılık sektörü ortalamasının (12.5) '
    "önemli ölçüde altında, değerleme açısından cazip bir seviyeye işaret ediyor. "
    "PD/DD 3.1 makul düzeyde. Net kâr büyümesi %18.5 ile güçlü seyrediyor, ancak "
    'yüksek borç/FAVÖK oranı (2.8) finansal kaldıraç riskini artırıyor.",'
    '"teknik_analiz_yorumu":"RSI 52 ile nötr bölgede, aşırı alım/satım baskısı yok. '
    "MACD pozitif bölgede ancak histogram daralıyor — momentum zayıflıyor. "
    "SMA50 (295) > SMA200 (278) Golden Cross formasyonu devam ediyor, bu uzun vadeli "
    'yükseliş trendinin bozulmadığını teyit ediyor.",'
    '"haber_sentiment_yorumu":"5 haberin 3\'ü pozitif, 1 negatif, 1 nötr tonlu. '
    "İhracat anlaşması ve yolcu artışı haberleri olumlu operasyonel sinyaller veriyor. "
    'Yakıt maliyeti artışı riski ise kâr marjı üzerinde baskı oluşturabilir.",'
    '"bollinger_yorumu":"Fiyat (298 TL) Bollinger orta bandının (290 TL) üstünde, '
    "üst banda (312 TL) yaklaşıyor. Bant genişliği %7.6 ile normal seviyede — "
    'aşırı volatilite yok. Direnç seviyesi 310 TL test edilebilir."}'
    ',"risk_analizi":{"risk_skoru":42,"risk_gerekcesi":"Orta seviye risk — güçlü '
    "temel göstergeler (düşük F/K, %18.5 kâr büyümesi) riski sınırlıyor, ancak "
    "yüksek borç oranı ve MACD momentum kaybı kısa vadeli dikkat gerektiriyor. "
    'Destek seviyesi 285 TL altına sarkması durumunda risk profili yükselir."}}'
)

USER_PROMPT_TEMPLATE = """Aşağıdaki finansal verileri analiz et ve JSON formatında rapor üret.

[HİSSE BİLGİSİ]
- Hisse Kodu: {hisse_kodu}
- Şirket Adı: {sirket_adi}
- Son Fiyat: {son_fiyat} TL
- Sektör: {sektor}

[FİNANSAL GÖSTERGELER]
- F/K Oranı: {fk_orani}
- Sektör F/K Ortalaması: {sektor_fk_ort}
- PD/DD Oranı: {pddd_orani}
- Net Kâr Büyümesi (Yıllık): {net_kar_buyumesi_formatted}
- Borç / FAVÖK Oranı: {borc_favok}

[TEKNİK GÖSTERGELER]
- RSI (14): {rsi_degeri} — {rsi_sinyal}
- MACD Durumu: {macd_durumu}
- Hareketli Ortalamalar: {sma_durumu}
- Bollinger Bantları: {bollinger_durumu}
- Destek Seviyesi: {destek_1} TL
- Direnç Seviyesi: {direnc_1} TL
- Pivot Noktası: {pivot} TL

[SON HABERLER & SENTIMENT]
{haberler_listesi}

JSON Çıktı Formatı (başka hiçbir şey ekleme):
{{"analiz_ozeti":{{"genel_gorus":"...","temel_analiz_yorumu":"...","teknik_analiz_yorumu":"...","haber_sentiment_yorumu":"...","bollinger_yorumu":"..."}},"risk_analizi":{{"risk_skoru":75,"risk_gerekcesi":"..."}}}}

risk_skoru: 0-100 arası integer. 0=Düşük Risk, 100=Çok Riskli.
Her yorum en az 2-3 cümle olsun ve verilen destek/direnç seviyelerine referans versin."""

_PROMPT_DEFAULTS = {
    "rsi_degeri": "Yetersiz Veri",
    "rsi_sinyal": "Yetersiz Veri",
    "macd_durumu": "Yetersiz Veri",
    "sma_durumu": "Yetersiz Veri",
    "bollinger_durumu": "Yetersiz Veri",
    "destek_1": "Yetersiz Veri",
    "direnc_1": "Yetersiz Veri",
    "pivot": "Yetersiz Veri",
}


def _retry_wait_seconds(exc: Exception, attempt: int) -> float:
    text = str(exc)
    m = re.search(r"retryDelay['\"]\s*:\s*['\"](\d+)s", text, re.IGNORECASE)
    backoff = BASE_DELAY * (BACKOFF_FACTOR ** (attempt - 1))
    if m:
        return max(float(m.group(1)), backoff)
    return backoff


def _build_user_prompt(stock_data: dict, technicals: dict, news_text: str) -> str:
    nk = stock_data.get("net_kar_buyumesi", "Yetersiz Veri")
    net_kar_formatted = f"%{nk}" if isinstance(nk, (int, float)) else str(nk)
    prompt_data = {
        **_PROMPT_DEFAULTS,
        **stock_data,
        **technicals,
        "haberler_listesi": news_text,
        "net_kar_buyumesi_formatted": net_kar_formatted,
    }
    return USER_PROMPT_TEMPLATE.format(**prompt_data)


def _clean_llm_text(text: str) -> str:
    """Ollama reasoning modellerinin düşünme / boxed çıktılarını temizler."""
    text = text.strip()
    text = re.sub(
        r"<think>.*?(?:</think>|$)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", text)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _extract_json_object(text: str) -> str:
    """Metin içindeki ilk JSON nesnesini çıkarır."""
    text = _clean_llm_text(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def generate_json_text(user_prompt: str, system_prompt: str | None = None) -> str:
    """
    Yapılandırılmış JSON metni üretir (Gemini veya Ollama).
    Portföy özeti gibi ek çağrılar için de kullanılır.
    """
    system = system_prompt or SYSTEM_PROMPT
    if is_ollama_provider():
        return _generate_ollama(user_prompt, system)
    return _generate_gemini(user_prompt, system)


def _generate_ollama(user_prompt: str, system_prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt + OLLAMA_JSON_SUFFIX},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3, "num_predict": 4096},
    }
    logger.info("Ollama çağrısı — model=%s", OLLAMA_MODEL)
    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
    except requests.ConnectionError as e:
        raise RuntimeError(
            f"Ollama'ya bağlanılamadı ({OLLAMA_BASE_URL}). "
            "Ollama çalışıyor mu? Terminalde: ollama serve"
        ) from e
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP hatası: {e}") from e

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama boş yanıt döndü.")
    return _extract_json_object(content)


def _generate_gemini(user_prompt: str, system_prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY ayarlanmamış. .env dosyasını kontrol edin veya LLM_PROVIDER=ollama kullanın."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Gemini çağrısı (deneme %d/%d)", attempt, MAX_RETRIES)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            return response.text

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            if "429" in error_str or "resource_exhausted" in error_str or "503" in error_str:
                wait = _retry_wait_seconds(e, attempt)
                logger.warning(
                    "Rate limit! %s — %.1f sn beklenecek (deneme %d/%d)",
                    e, wait, attempt, MAX_RETRIES,
                )
                time.sleep(wait)
                continue

            if (
                "api_key_invalid" in error_str
                or "api key not found" in error_str
                or ("invalid_argument" in error_str and "api key" in error_str)
            ):
                raise RuntimeError(
                    "Gemini API anahtarı geçersiz. https://aistudio.google.com/apikey"
                ) from e

            if "403" in error_str or "401" in error_str or "permission_denied" in error_str:
                raise RuntimeError(f"Gemini API yetki hatası: {e}") from e

            if attempt < MAX_RETRIES:
                wait = BASE_DELAY * (BACKOFF_FACTOR ** (attempt - 1))
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Gemini API {MAX_RETRIES} denemede başarısız: {last_error}")


def analyze_stock(stock_data: dict, technicals: dict, news_text: str) -> dict:
    """
    Tüm verileri birleştirip LLM'e gönderir, JSON rapor döndürür.
    Sağlayıcı: config.LLM_PROVIDER (gemini | ollama).
    """
    if not is_llm_configured():
        if is_ollama_provider():
            raise RuntimeError(
                "Ollama yapılandırılmamış. .env içinde OLLAMA_MODEL kontrol edin."
            )
        raise RuntimeError(
            "GEMINI_API_KEY ayarlanmamış. .env dosyasını kontrol edin veya LLM_PROVIDER=ollama kullanın."
        )

    user_prompt = _build_user_prompt(stock_data, technicals, news_text)
    logger.info(
        "Hisse analizi — %s (%s)",
        stock_data.get("hisse_kodu", "?"),
        LLM_PROVIDER,
    )
    raw = generate_json_text(user_prompt)
    return _parse_response(raw)


def _parse_response(raw_text: str) -> dict:
    text = _extract_json_object(raw_text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_clean_llm_text(raw_text))
    except json.JSONDecodeError as e:
        logger.error("JSON parse hatası: %s — İlk 200 karakter: %s", e, text[:200])
        return {
            "analiz_ozeti": {
                "genel_gorus": "AI analizi tamamlanamadı — yanıt formatı beklenen JSON'a uymadı.",
                "temel_analiz_yorumu": "Yetersiz Veri",
                "teknik_analiz_yorumu": "Yetersiz Veri",
                "haber_sentiment_yorumu": "Yetersiz Veri",
                "bollinger_yorumu": "Yetersiz Veri",
            },
            "risk_analizi": {
                "risk_skoru": 50,
                "risk_gerekcesi": f"Analiz tamamlanamadı: {e}",
            },
        }
