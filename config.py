"""
Configuration — loads API keys and settings from environment variables.
"""
import os
import sys


def _configure_windows_utf8() -> None:
    """Windows konsolunda Türkçe karakter (ı, ü, ş) kaynaklı charmap hatalarını önler."""
    os.environ.setdefault("PYTHONUTF8", "1")
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _configure_ssl_cert_bundle() -> None:
    """
    venv, Türkçe karakterli klasördeyse (örn. Atılım, Masaüstü) certifi yolu
    requests/yfinance SSL adımında charmap hatası verir. CA dosyasını ASCII
    bir yola kopyalayıp ortam değişkenlerine yönlendirir.
    """
    if os.environ.get("REQUESTS_CA_BUNDLE") and os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
        import shutil

        source = certifi.where()
        try:
            source.encode("ascii")
            os.environ.setdefault("SSL_CERT_FILE", source)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", source)
            return
        except UnicodeEncodeError:
            pass

        cache_dir = os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
            "FinSightAI",
        )
        os.makedirs(cache_dir, exist_ok=True)
        dest = os.path.join(cache_dir, "cacert.pem")
        if not os.path.isfile(dest) or os.path.getmtime(dest) < os.path.getmtime(source):
            shutil.copy2(source, dest)
        os.environ["SSL_CERT_FILE"] = dest
        os.environ["REQUESTS_CA_BUNDLE"] = dest
    except Exception:
        pass


_configure_windows_utf8()
_configure_ssl_cert_bundle()

from pathlib import Path  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

# Proje kökündeki .env — override=True: terminaldeki eski GEMINI_API_KEY=test vb. ezilir
_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_FILE, override=True)

_GEMINI_RAW: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# LLM sağlayıcı: gemini (bulut) | ollama (yerel)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "RefinedNeuro/RN_TR_R2").strip()
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# yfinance BIST hisseleri ".IS" soneki kullanır (Örn: THYAO.IS)
BIST_SUFFIX: str = ".IS"

_PLACEHOLDER_KEYS = frozenset({
    "",
    "buraya_api_keyinizi_yazin",
    "test",
    "your_api_key_here",
})


def is_gemini_key_configured(key: str | None = None) -> bool:
    """Geçerli Gemini API anahtarı var mı? (AI sekmesi için)"""
    k = (key if key is not None else _GEMINI_RAW).strip()
    if k.lower() in _PLACEHOLDER_KEYS:
        return False
    # Google AI Studio anahtarları genelde AIza ile başlar
    return k.startswith("AIza") and len(k) >= 30


GEMINI_API_KEY: str = _GEMINI_RAW if is_gemini_key_configured(_GEMINI_RAW) else ""

_env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
if _env_provider in ("gemini", "ollama"):
    LLM_PROVIDER: str = _env_provider
elif is_gemini_key_configured():
    LLM_PROVIDER = "gemini"
else:
    LLM_PROVIDER = "ollama"


def is_ollama_provider() -> bool:
    return LLM_PROVIDER == "ollama"


def is_llm_configured() -> bool:
    """AI raporu için Gemini anahtarı veya Ollama sağlayıcısı hazır mı?"""
    if is_ollama_provider():
        return bool(OLLAMA_MODEL)
    return bool(GEMINI_API_KEY)


def llm_provider_label() -> str:
    if is_ollama_provider():
        return f"Ollama ({OLLAMA_MODEL})"
    return f"Gemini ({GEMINI_MODEL})"


if LLM_PROVIDER == "gemini":
    if _GEMINI_RAW and not GEMINI_API_KEY:
        import warnings
        warnings.warn(
            "[UYARI] GEMINI_API_KEY geçersiz veya placeholder! "
            "https://aistudio.google.com/apikey — veya LLM_PROVIDER=ollama kullanın.",
            stacklevel=2,
        )
    elif not _GEMINI_RAW:
        import warnings
        warnings.warn(
            "[UYARI] GEMINI_API_KEY ayarlanmamış! "
            "LLM_PROVIDER=ollama ile yerel model kullanabilirsiniz.",
            stacklevel=2,
        )
