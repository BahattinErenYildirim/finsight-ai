# FinSight AI — 1-minute promo video

## Technical specs (typical hackathon requirements)

| Property | Recommended |
|----------|-------------|
| Duration | **45–60 seconds** (max 60s) |
| Resolution | **1920×1080** (min 1280×720) |
| Format | **MP4** (H.264) |
| Size | Usually **≤ 50–100 MB** |
| Audio | Voiceover or subtitles (silent may be allowed) |

---

## 60-second script (Turkish voiceover — on-screen UI can stay English)

| Sec | On screen | Voiceover (TR) |
|-----|-----------|----------------|
| 0–8 | Logo **FinSight AI** | "FinSight AI is an AI-powered financial analysis platform for Borsa Istanbul equities." |
| 8–18 | Streamlit, type `THYAO`, click **Analyze** | "We enter a ticker; the system pulls live market data." |
| 18–30 | **Technical Analysis** tab | "RSI, MACD, Bollinger, and support/resistance are computed automatically." |
| 30–42 | **AI Analysis** — risk score | "Gemini or Ollama produces a Turkish narrative report and risk score — not investment advice." |
| 42–52 | **Portfolio** or PDF download | "Compare multiple tickers and export a PDF report." |
| 52–60 | Closing — team / Atılım | "FinSight AI — Atılım University Hackathon 2026. Thank you." |

---

## Recording (Windows)

```powershell
cd Hackhatlon
.\venv\Scripts\activate
streamlit run app.py
```

- Full-screen browser (F11) or window capture.
- Use Xbox Game Bar (`Win+G`) or OBS Studio.
- Export MP4, upload with your application.
