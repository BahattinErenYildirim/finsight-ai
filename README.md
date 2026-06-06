<p align="center">
  <h1 align="center">📈 FinSight AI</h1>
  <p align="center">
    <strong>AI-Powered Financial Analysis Platform for BIST Equities</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/Gemini-2.0_Flash-orange?logo=google&logoColor=white" alt="Gemini"/>
    <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white" alt="Streamlit"/>
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  </p>
</p>

---

## Overview

**FinSight AI** is a professional financial analysis platform for Borsa Istanbul (BIST) equities. It pulls near-real-time market data, computes technical indicators, analyzes news sentiment, and generates structured AI reports via Google Gemini or local Ollama.

### Key features

| Feature | Description |
|---------|-------------|
| Technical analysis | RSI, MACD, SMA 50/200, Bollinger Bands (20, 2σ) |
| AI reports | Structured JSON output (Gemini 2.0 Flash or Ollama) |
| News sentiment | LLM-based or TR+EN keyword fallback |
| Dashboard | Streamlit + Plotly (candlestick, RSI, MACD) |
| Portfolio | Multi-ticker comparison, correlation matrix, simulator |
| PDF export | Report with embedded charts and risk gauge |
| Resilience | 5 min TTL cache, exponential backoff on rate limits |
| CLI | Rich terminal interface |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FinSight AI                            │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Data     │ Technical│ News     │ LLM      │ Presentation    │
├──────────┼──────────┼──────────┼──────────┼─────────────────┤
│ yfinance │ RSI,MACD │ Sentiment│ Gemini / │ Streamlit       │
│ cache    │ SMA, BB  │ TR + EN  │ Ollama   │ Plotly, PDF     │
└──────────┴──────────┴──────────┴──────────┴─────────────────┘
```

### Project layout

```
├── app.py                  # Streamlit dashboard
├── main.py                 # CLI entry
├── demo.py                 # 3-minute demo script
├── config.py               # Environment configuration
├── data_fetcher.py         # yfinance + cache + retry
├── technical_analysis.py   # Indicators
├── news_sentiment.py       # News fetch + sentiment
├── llm_analyzer.py         # LLM integration
├── portfolio.py            # Watchlist & simulator
├── sector_analysis.py      # Sector rotation
├── pdf_report.py           # PDF generator
├── backend/                # FastAPI REST API
└── tests/                  # pytest suite
```

---

## Setup

### Requirements

- Python 3.11+
- Google Gemini API key **or** [Ollama](https://ollama.com) with a Turkish-capable model

### Install

```bash
git clone <repo-url>
cd Hackhatlon

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
```

Edit `.env`:

```env
# Gemini (default)
GEMINI_API_KEY=your_key_here
LLM_PROVIDER=gemini

# Or local Ollama
LLM_PROVIDER=ollama
OLLAMA_MODEL=RefinedNeuro/RN_TR_R2
OLLAMA_BASE_URL=http://localhost:11434
```

Get a Gemini key: [Google AI Studio](https://aistudio.google.com/apikey)

---

## Usage

### Web dashboard (recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501`, enter a ticker (e.g. `THYAO`), click **Analyze**.

### CLI

```bash
python main.py THYAO
python main.py          # interactive mode
```

### Demo script

```bash
python demo.py
```

### REST API

```bash
uvicorn backend.api.main:app --reload --port 8000
```

- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Tests

```bash
pytest -v tests/
```

CI runs on push/PR via `.github/workflows/ci.yml`.

---

## AI report schema

```json
{
  "analiz_ozeti": {
    "genel_gorus": "...",
    "temel_analiz_yorumu": "...",
    "teknik_analiz_yorumu": "...",
    "haber_sentiment_yorumu": "...",
    "bollinger_yorumu": "..."
  },
  "risk_analizi": {
    "risk_skoru": 42,
    "risk_gerekcesi": "..."
  }
}
```

> **Disclaimer:** For informational purposes only. Not investment advice.

---

## Tech stack

| Layer | Tools |
|-------|--------|
| Data | yfinance, pandas, numpy |
| Technical | ta |
| LLM | Google Gemini, Ollama |
| Web | Streamlit, Plotly |
| API | FastAPI, SQLAlchemy, Redis |
| PDF | fpdf2, kaleido |
| Tests | pytest, httpx |

---

## License

MIT
