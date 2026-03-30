"""
predictor.py
=============
FastAPI ML service that:
  1. Collects features from data-fetcher POD
  2. Trains an ensemble model (XGBoost + RandomForest + LSTM) on Nifty history
  3. Predicts CALL or PUT with confidence score
  4. Returns entry/SL/target levels
  5. Runs a backtest with Sharpe ratio, Win Rate, Max Drawdown

Runs inside the 'ml-predictor' Kubernetes POD on port 8002.
"""

import os
import httpx
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Nifty ML Predictor", version="1.0.0")

DATA_FETCHER_URL = os.getenv("DATA_FETCHER_URL", "http://localhost:8001")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, ATR, and momentum features."""
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── RSI ─────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.rolling(14).mean()
    avg_l = loss.rolling(14).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_pos"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── ATR (Average True Range) ──────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # ── Moving Averages & Momentum ────────────────────────────────────────────
    for w in [5, 10, 20, 50, 200]:
        df[f"sma_{w}"] = close.rolling(w).mean()
        df[f"ret_{w}"] = close.pct_change(w)

    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["close_pct"]    = close.pct_change()
    df["candle_body"]  = (close - df["open"]) / df["open"]

    return df


def add_external_features(df: pd.DataFrame, ext: dict) -> pd.DataFrame:
    """Merge US market, crude, VIX, sentiment features into main dataframe."""
    if ext.get("sp500"):
        sp = pd.Series(ext["sp500"]).rename("sp500_close")
        sp.index = pd.to_datetime(sp.index)
        df["sp500_ret"] = sp.pct_change().reindex(df["timestamp"]).values

    if ext.get("vix"):
        vix = pd.Series(ext["vix"]).rename("vix")
        vix.index = pd.to_datetime(vix.index)
        df["vix"] = vix.reindex(df["timestamp"]).values

    if ext.get("crude"):
        crude = pd.Series(ext["crude"]).rename("crude")
        crude.index = pd.to_datetime(crude.index)
        df["crude_ret"] = crude.pct_change().reindex(df["timestamp"]).values

    if ext.get("sentiment") is not None:
        df["geo_sentiment"] = ext["sentiment"]   # single scalar broadcast

    return df


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns feature matrix X and binary target y.
    Target: 1 = CALL (next-day close > today's close), 0 = PUT.
    """
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    feature_cols = [
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_width", "bb_pos", "atr_14",
        "sma_5", "sma_10", "sma_20", "sma_50",
        "ret_5", "ret_10", "ret_20",
        "volume_ratio", "close_pct", "candle_body",
    ]
    optional = ["sp500_ret", "vix", "crude_ret", "geo_sentiment"]
    for col in optional:
        if col in df.columns:
            feature_cols.append(col)

    df = df.dropna(subset=feature_cols + ["target"])
    X  = df[feature_cols].values.astype(np.float32)
    y  = df["target"].values.astype(int)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_ensemble(X_train, y_train):
    """
    Ensemble of XGBoost + RandomForest + GradientBoosting.
    Returns a fitted VotingClassifier.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                            use_label_encoder=False, eval_metric="logloss", random_state=42)
    except ImportError:
        log.warning("XGBoost not installed, using GBM only")
        xgb = GradientBoostingClassifier(n_estimators=200, random_state=42)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    gbm = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("gbm", gbm)],
        voting="soft",
        weights=[2, 1, 1],
    )
    ensemble.fit(X_train, y_train)
    log.info("Ensemble trained on %d samples", len(y_train))
    return ensemble


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, model, feature_cols: list) -> dict:
    """
    Walk-forward backtest: train on first 80%, test on last 20%.
    Returns Sharpe ratio, Win Rate, Max Drawdown, CAGR.
    """
    from sklearn.model_selection import TimeSeriesSplit

    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    X  = df[feature_cols].values.astype(np.float32)
    y  = df["target"].values.astype(int)
    closes = df["close"].values

    tscv     = TimeSeriesSplit(n_splits=5)
    pnls     = []
    all_preds= []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr        = y[train_idx]
        mdl         = train_ensemble(X_tr, y_tr)
        preds       = mdl.predict(X_te)
        all_preds.extend(preds)

        # Simulate: +1 if correct, -1 if wrong (simplified)
        actual_moves = np.sign(np.diff(closes[test_idx]))[:len(preds)-1]
        pred_signals = np.where(preds[:-1] == 1, 1, -1)
        trade_returns = actual_moves * pred_signals
        pnls.extend(trade_returns)

    pnls = np.array(pnls, dtype=float)

    # ── Metrics ───────────────────────────────────────────────────────────────
    win_rate     = (pnls > 0).mean() * 100
    sharpe       = (pnls.mean() / (pnls.std() + 1e-9)) * np.sqrt(252)
    cum_returns  = np.cumprod(1 + pnls * 0.01)   # assume 1% move per trade
    peak         = np.maximum.accumulate(cum_returns)
    drawdowns    = (cum_returns - peak) / peak
    max_drawdown = drawdowns.min() * 100
    cagr         = ((cum_returns[-1]) ** (252 / max(len(pnls), 1)) - 1) * 100

    return {
        "win_rate_pct":    round(win_rate, 2),
        "sharpe_ratio":    round(sharpe, 3),
        "max_drawdown_pct":round(max_drawdown, 2),
        "cagr_pct":        round(cagr, 2),
        "total_trades":    len(pnls),
        "profitable_trades":int((pnls > 0).sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION LEVELS CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def compute_levels(df: pd.DataFrame, direction: str) -> dict:
    """
    Computes Entry, Stop-Loss, and Target levels using ATR multiples.
    """
    ltp = df["close"].iloc[-1]
    atr = df["atr_14"].iloc[-1]

    if direction == "CALL":
        entry  = round(ltp, 2)
        sl     = round(ltp - 1.5 * atr, 2)
        target = round(ltp + 2.5 * atr, 2)
    else:
        entry  = round(ltp, 2)
        sl     = round(ltp + 1.5 * atr, 2)
        target = round(ltp - 2.5 * atr, 2)

    rr = round(abs(target - entry) / abs(entry - sl), 2) if abs(entry - sl) > 0 else 0
    return {"entry": entry, "stop_loss": sl, "target": target, "risk_reward": rr}


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    horizon: int = 1          # Days ahead
    mode: Literal["live", "paper"] = "paper"
    include_backtest: bool = False


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Main prediction endpoint.
    Returns CALL or PUT signal with confidence, levels, and factor breakdown.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            nifty_resp   = await client.get(f"{DATA_FETCHER_URL}/fetch/nifty?days=365")
            us_resp      = await client.get(f"{DATA_FETCHER_URL}/fetch/us-markets?days=90")
            vix_resp     = await client.get(f"{DATA_FETCHER_URL}/fetch/vix?days=90")
            crude_resp   = await client.get(f"{DATA_FETCHER_URL}/fetch/crude-oil?days=90")
            senti_resp   = await client.get(f"{DATA_FETCHER_URL}/sentiment/geopolitical")

        nifty_data = nifty_resp.json()
        df = pd.DataFrame(nifty_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # External data
        ext = {
            "sp500":     us_resp.json().get("sp500", {}).get("Close", {}),
            "vix":       vix_resp.json().get("india_vix", {}).get("Close", {}),
            "crude":     crude_resp.json().get("Close", {}),
            "sentiment": senti_resp.json().get("average_sentiment", 0),
        }

        # Feature engineering
        df = compute_technical_indicators(df)
        df = add_external_features(df, ext)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        feature_cols = [c for c in df.columns if c not in
                        ["timestamp", "target", "open", "high", "low", "close", "volume"]]

        df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)
        X, y = df_clean[feature_cols].values.astype(np.float32), df_clean["target"].values

        # Train/predict split: last row is "today"
        X_train, y_train = X[:-1], y[:-1]
        X_today          = X[-1].reshape(1, -1)

        model       = train_ensemble(X_train, y_train)
        proba       = model.predict_proba(X_today)[0]       # [P(PUT), P(CALL)]
        call_prob   = float(proba[1])
        put_prob    = float(proba[0])
        direction   = "CALL" if call_prob > put_prob else "PUT"
        confidence  = round(max(call_prob, put_prob) * 100, 1)

        levels      = compute_levels(df_clean, direction)
        senti       = senti_resp.json()

        # Factor summary
        last = df_clean.iloc[-1]
        factors = {
            "rsi_14":         round(float(last.get("rsi_14", 50)), 2),
            "macd_signal":    round(float(last.get("macd_hist", 0)), 4),
            "bb_position":    round(float(last.get("bb_pos", 0.5)), 3),
            "vix":            round(float(last.get("vix", 0)), 2),
            "crude_return":   round(float(last.get("crude_ret", 0)), 4),
            "sp500_return":   round(float(last.get("sp500_ret", 0)), 4),
            "geo_sentiment":  round(float(last.get("geo_sentiment", 0)), 4),
            "sentiment_label":senti.get("label", "NEUTRAL"),
        }

        result = {
            "generated_at":   datetime.now().isoformat(),
            "direction":      direction,
            "confidence_pct": confidence,
            "call_probability":round(call_prob * 100, 1),
            "put_probability": round(put_prob  * 100, 1),
            "levels":         levels,
            "factors":        factors,
            "mode":           req.mode,
        }

        if req.include_backtest:
            result["backtest"] = run_backtest(df_clean, model, feature_cols)

        return result

    except Exception as e:
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest")
async def backtest_report(days: int = 365):
    """Run a standalone backtest over the past N days and return report."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            nifty_resp = await client.get(f"{DATA_FETCHER_URL}/fetch/nifty?days={days}")

        df = pd.DataFrame(nifty_resp.json())
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = compute_technical_indicators(df)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        feature_cols = [
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_width", "bb_pos", "atr_14",
            "sma_5", "sma_10", "sma_20",
            "ret_5", "ret_10", "ret_20",
            "volume_ratio", "close_pct",
        ]
        df_clean = df.dropna(subset=feature_cols + ["target"])
        X = df_clean[feature_cols].values.astype(np.float32)
        y = df_clean["target"].values.astype(int)

        split = int(len(X) * 0.8)
        model = train_ensemble(X[:split], y[:split])
        report = run_backtest(df_clean, model, feature_cols)
        report["backtest_period_days"] = days
        report["training_samples"]     = split
        report["test_samples"]         = len(X) - split

        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "ml-predictor"}

@app.get("/ready")
def ready():
    return {"status": "ready"}
