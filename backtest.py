"""
backtest.py
============
Standalone backtest runner — can be executed directly from CLI or called by the predictor POD.

Usage:
    python backtest.py --days 365 --model ensemble
    python backtest.py --days 252 --model xgboost --report-html report.html

Output:
    - Console summary table
    - Optional HTML report with equity curve + drawdown chart
"""

import argparse
import os
import sys
import httpx
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_FETCHER_URL = os.getenv("DATA_FETCHER_URL", "http://localhost:8001")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(days: int) -> pd.DataFrame:
    log.info("Fetching Nifty data (%d days)...", days)
    resp = httpx.get(f"{DATA_FETCHER_URL}/fetch/nifty?days={days}", timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Loaded %d candles", len(df))
    return df


# ── Indicators (same as predictor.py) ─────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"]  = ema12 - ema26
    df["macd_s"]= df["macd"].ewm(span=9).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_pos"] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-9)

    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    for w in [5, 10, 20]:
        df[f"ret_{w}"] = close.pct_change(w)
        df[f"sma_{w}"] = close.rolling(w).mean()

    df["vol_ratio"]  = df["volume"] / df["volume"].rolling(20).mean()
    df["close_pct"]  = close.pct_change()
    df["target"]     = (close.shift(-1) > close).astype(int)
    return df


# ── Walk-Forward Backtest ──────────────────────────────────────────────────────

def walk_forward_backtest(df: pd.DataFrame, model_type: str) -> dict:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score

    FEATURES = ["rsi", "macd", "macd_s", "bb_pos", "atr",
                "ret_5", "ret_10", "ret_20", "sma_5", "sma_10", "sma_20",
                "vol_ratio", "close_pct"]

    df = df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    X = df[FEATURES].values.astype(np.float32)
    y = df["target"].values.astype(int)
    closes = df["close"].values

    tscv = TimeSeriesSplit(n_splits=5)
    equity  = [1.0]
    trades  = []
    accs    = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        if model_type == "random_forest":
            mdl = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        elif model_type == "gradient_boosting":
            mdl = GradientBoostingClassifier(n_estimators=150, random_state=42)
        else:  # ensemble (default)
            rf  = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
            gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
            mdl = VotingClassifier([("rf", rf), ("gbm", gbm)], voting="soft")

        mdl.fit(X_tr, y_tr)
        preds = mdl.predict(X_te)
        accs.append(accuracy_score(y_te, preds))

        for i in range(len(te_idx) - 1):
            c_today  = closes[te_idx[i]]
            c_next   = closes[te_idx[i+1]]
            pred_dir = preds[i]
            actual_move = (c_next - c_today) / c_today

            pnl = actual_move if pred_dir == 1 else -actual_move
            equity.append(equity[-1] * (1 + pnl))
            trades.append({"fold": fold+1, "date": str(df["timestamp"].iloc[te_idx[i]])[:10],
                           "pred": "CALL" if pred_dir==1 else "PUT",
                           "pnl_pct": round(pnl*100, 3)})

    equity = np.array(equity)
    pnls   = np.diff(equity) / equity[:-1]
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak

    return {
        "model":           model_type,
        "folds":           5,
        "total_trades":    len(trades),
        "accuracy_pct":    round(np.mean(accs)*100, 2),
        "win_rate_pct":    round((np.array([t["pnl_pct"] for t in trades]) > 0).mean()*100, 2),
        "sharpe_ratio":    round((pnls.mean()/(pnls.std()+1e-9))*np.sqrt(252), 3),
        "max_drawdown_pct":round(dd.min()*100, 2),
        "final_equity":    round(equity[-1], 4),
        "cagr_pct":        round(((equity[-1])**(252/max(len(pnls),1))-1)*100, 2),
        "trades":          trades[-20:],   # last 20 trades for display
    }


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def print_report(r: dict):
    sep = "─" * 50
    print(f"\n{'═'*50}")
    print(f"  NIFTY BACKTEST REPORT — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'═'*50}")
    print(f"  Model          : {r['model']}")
    print(f"  Total Trades   : {r['total_trades']}")
    print(sep)
    print(f"  Accuracy       : {r['accuracy_pct']}%")
    print(f"  Win Rate       : {r['win_rate_pct']}%")
    print(f"  Sharpe Ratio   : {r['sharpe_ratio']}")
    print(f"  Max Drawdown   : {r['max_drawdown_pct']}%")
    print(f"  CAGR           : {r['cagr_pct']}%")
    print(f"  Final Equity   : {r['final_equity']}x")
    print(f"{'═'*50}")
    print("\n  Recent Trades (last 20):")
    print(f"  {'Date':<12} {'Dir':<6} {'P&L %':>8}")
    print(f"  {'─'*30}")
    for t in r["trades"]:
        icon = "✅" if t["pnl_pct"] > 0 else "❌"
        print(f"  {t['date']:<12} {t['pred']:<6} {t['pnl_pct']:>7.3f}% {icon}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nifty Backtest Runner")
    parser.add_argument("--days",  type=int, default=365, help="Lookback days")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["ensemble", "random_forest", "gradient_boosting"])
    args = parser.parse_args()

    try:
        df = load_data(args.days)
        df = add_indicators(df)
        report = walk_forward_backtest(df, args.model)
        print_report(report)
    except httpx.ConnectError:
        log.error("Cannot connect to data-fetcher at %s", DATA_FETCHER_URL)
        log.error("Make sure the data-fetcher POD is running and port-forwarded.")
        sys.exit(1)
