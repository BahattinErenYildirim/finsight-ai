# FinSight AI — Jury Demo Script (~3 minutes)

> **Goal:** Live flow: data → technical analysis → AI report → portfolio comparison.

---

## Prep (5 minutes before presentation)

```powershell
cd Hackhatlon
.\venv\Scripts\activate

copy .env.example .env   # if missing; set GEMINI_API_KEY or LLM_PROVIDER=ollama

streamlit run app.py
```

Browser: **http://localhost:8501**

Optional API backup:

```powershell
$env:SECRET_KEY="demo-secret-key-at-least-32-chars-long"
uvicorn backend.api.main:app --port 8000
# Swagger: http://localhost:8000/docs
```

---

## Scenario A — Streamlit (recommended, ~3 min)

### Step 1 — Single ticker (60s)

| Action | Talking point |
|--------|----------------|
| Sidebar → `THYAO` → **Analyze** | "We pull live BIST data via yfinance." |
| **Technical Analysis** tab | "RSI, MACD, Bollinger, support/resistance are computed automatically." |
| Point at charts | "Candlestick + SMA 50/200 — signals like Golden Cross are visualized." |

### Step 2 — AI report (60s)

| Action | Talking point |
|--------|----------------|
| **AI Analysis** tab → generate report | "Gemini or Ollama synthesizes all inputs into structured JSON." |
| Risk score | "0–100 risk score — no direct BUY/SELL; regulatory-friendly language." |
| Summary cards | "Fundamental, technical, and news sentiment in one view." |

### Step 3 — Portfolio + PDF (60s)

| Action | Talking point |
|--------|----------------|
| **Portfolio** → add `THYAO`, `ASELS`, `SISE` | "Multi-ticker correlation and comparison table." |
| Correlation heatmap | "Highlights diversification opportunities." |
| **Download PDF** | "Shareable institutional-style report." |

**Closing line:**  
*"FinSight AI unifies data, technicals, sentiment, and LLM in one platform — web dashboard and REST API."*

---

## Scenario B — CLI (backup)

```bash
python demo.py
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Gemini 429 quota | Wait ~1 min or switch to `LLM_PROVIDER=ollama` |
| No news for ticker | Normal for some BIST symbols on Yahoo; try THYAO |
| Ollama slow first run | First report may take 1–3 minutes |

---

## Checklist

- [ ] `.env` configured
- [ ] `streamlit run app.py` running
- [ ] Test ticker `THYAO` loads charts
- [ ] AI report button works once
