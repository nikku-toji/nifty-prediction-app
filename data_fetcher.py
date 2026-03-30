"""
data_fetcher.py
================
FastAPI service that fetches data from:
  - Angel One SmartAPI (Nifty historical + live data)
  - Yahoo Finance (US markets, Crude Oil, VIX)
  - NewsAPI (Geopolitical sentiment)

Runs inside the 'data-fetcher' Kubernetes POD on port 8001.

Install: pip install -r requirements.txt
Run locally: uvicorn data_fetcher:app --reload --port 8001
"""

import os
import pyotp
import logging
import httpx
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from SmartApi import SmartConnect

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Nifty Data Fetcher", version="1.0.0")

# ── Angel One Auth ─────────────────────────────────────────────────────────────

def get_angel_session() -> SmartConnect:
    """Authenticate with Angel One SmartAPI and return session object."""
    api_key     = os.environ["ANGEL_ONE_API_KEY"]
    client_id   = os.environ["ANGEL_ONE_CLIENT_ID"]
    password    = os.environ["ANGEL_ONE_PASSWORD"]
    totp_secret = os.environ["ANGEL_ONE_TOTP_SECRET"]

    totp = pyotp.TOTP(totp_secret).now()
    obj  = SmartConnect(api_key=api_key)
    data = obj.generateSession(client_id, password, totp)

    if data["status"] is False:
        raise RuntimeError(f"Angel One auth failed: {data['message']}")

    log.info("Angel One session created for client %s", client_id)
    return obj


# ── Nifty Historical Data ──────────────────────────────────────────────────────

@app.get("/fetch/nifty")
def fetch_nifty(days: int = Query(365, description="Number of past days to fetch")):
    """
    Returns Nifty 50 OHLCV candles from Angel One API.
    Token 99926000 = NIFTY index on NSE.
    """
    try:
        smart = get_angel_session()
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
        to_date   = datetime.now().strftime("%Y-%m-%d %H:%M")

        params = {
            "exchange":    "NSE",
            "symboltoken": "99926000",   # Nifty 50 token
            "interval":    "ONE_DAY",
            "fromdate":    from_date,
            "todate":      to_date,
        }

        resp = smart.getCandleData(params)
        if resp["status"] is False:
            raise HTTPException(status_code=502, detail=resp["message"])

        candles = resp["data"]
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        log.info("Fetched %d Nifty candles", len(df))
        return df.to_dict(orient="records")

    except Exception as e:
        log.exception("Error fetching Nifty data")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch/nifty/live")
def fetch_nifty_live():
    """Returns the latest Nifty 50 LTP (Last Traded Price)."""
    try:
        smart = get_angel_session()
        resp = smart.ltpData("NSE", "Nifty 50", "99926000")
        return resp["data"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch/nifty/options-chain")
def fetch_options_chain(expiry: str = Query(..., description="e.g. 27MAR2025")):
    """
    Returns Nifty options chain (Call + Put OI, IV, PCR) for a given expiry.
    Uses NSE public API as Angel One does not expose full options chain.
    """
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com",
    }
    try:
        with httpx.Client(headers=headers, timeout=15) as client:
            # NSE requires a cookies handshake
            client.get("https://www.nseindia.com", timeout=10)
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()

        records = data["records"]["data"]
        rows = []
        for r in records:
            if r.get("expiryDate", "").replace(" ", "").upper() != expiry.upper():
                continue
            row = {"strike": r["strikePrice"]}
            if "CE" in r:
                row.update({
                    "ce_oi": r["CE"].get("openInterest", 0),
                    "ce_chg_oi": r["CE"].get("changeinOpenInterest", 0),
                    "ce_iv": r["CE"].get("impliedVolatility", 0),
                    "ce_ltp": r["CE"].get("lastPrice", 0),
                })
            if "PE" in r:
                row.update({
                    "pe_oi": r["PE"].get("openInterest", 0),
                    "pe_chg_oi": r["PE"].get("changeinOpenInterest", 0),
                    "pe_iv": r["PE"].get("impliedVolatility", 0),
                    "pe_ltp": r["PE"].get("lastPrice", 0),
                })
            rows.append(row)

        df = pd.DataFrame(rows)

        # Put-Call Ratio
        total_ce_oi = df["ce_oi"].sum() if "ce_oi" in df.columns else 1
        total_pe_oi = df["pe_oi"].sum() if "pe_oi" in df.columns else 0
        pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi else 0

        return {"pcr": pcr, "expiry": expiry, "chain": df.to_dict(orient="records")}

    except Exception as e:
        log.exception("Options chain error")
        raise HTTPException(status_code=500, detail=str(e))


# ── US Markets + Crude + VIX via Yahoo Finance ────────────────────────────────

@app.get("/fetch/us-markets")
def fetch_us_markets(days: int = 90):
    """
    Returns S&P 500 and Nasdaq historical data.
    Used to compute US-India market correlation.
    """
    tickers = {"sp500": "^GSPC", "nasdaq": "^IXIC"}
    result  = {}
    for name, symbol in tickers.items():
        df = yf.download(symbol, period=f"{days}d", auto_adjust=True, progress=False)
        df.index = df.index.astype(str)
        result[name] = df[["Open", "High", "Low", "Close", "Volume"]].round(2).to_dict()
    return result


@app.get("/fetch/crude-oil")
def fetch_crude_oil(days: int = 90):
    """Returns Brent Crude Oil (BZ=F) historical prices."""
    df = yf.download("BZ=F", period=f"{days}d", auto_adjust=True, progress=False)
    df.index = df.index.astype(str)
    return df[["Open", "High", "Low", "Close"]].round(2).to_dict()


@app.get("/fetch/vix")
def fetch_vix(days: int = 90):
    """Returns India VIX and CBOE VIX historical data."""
    result = {}
    for name, symbol in [("india_vix", "^INDIAVIX"), ("cboe_vix", "^VIX")]:
        df = yf.download(symbol, period=f"{days}d", auto_adjust=True, progress=False)
        df.index = df.index.astype(str)
        result[name] = df[["Close"]].round(2).to_dict()
    return result


# ── Geopolitical Sentiment ────────────────────────────────────────────────────

@app.get("/sentiment/geopolitical")
def fetch_sentiment(query: str = "india economy stock market geopolitical", days: int = 7):
    """
    Fetches recent news headlines and computes average sentiment score.
    Positive score → bullish signal, Negative → bearish.
    Uses TextBlob for lightweight sentiment (VADER can be substituted).
    """
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        return {"error": "NEWS_API_KEY not configured", "sentiment": 0, "articles": []}

    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}&from={from_date}&language=en&sortBy=relevancy"
            f"&apiKey={api_key}&pageSize=30"
        )
        with httpx.Client(timeout=15) as client:
            resp = client.get(url)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])

        from textblob import TextBlob
        sentiments = []
        results    = []
        for a in articles:
            title = a.get("title", "") or ""
            desc  = a.get("description", "") or ""
            text  = f"{title}. {desc}"
            score = TextBlob(text).sentiment.polarity  # -1 to +1
            sentiments.append(score)
            results.append({"title": title[:120], "score": round(score, 3)})

        avg_sentiment = round(sum(sentiments) / len(sentiments), 4) if sentiments else 0
        label = "BULLISH" if avg_sentiment > 0.05 else ("BEARISH" if avg_sentiment < -0.05 else "NEUTRAL")

        return {
            "average_sentiment": avg_sentiment,
            "label": label,
            "articles_analyzed": len(sentiments),
            "articles": results[:10],
        }

    except Exception as e:
        log.exception("Sentiment error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Health Probes ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "data-fetcher"}

@app.get("/ready")
def ready():
    # Could add actual connectivity check here
    return {"status": "ready"}
