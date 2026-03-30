# 🚀 Nifty Options Prediction System — Kubernetes Algo Trading Platform

A production-grade, modular algorithmic trading platform deployed on Kubernetes.
Each trading algorithm runs in its own isolated POD with local HTTP exposure.
Supports CALL/PUT prediction for NIFTY using ML/AI, Angel One API, VIX, crude oil, geopolitical signals, and US market correlation.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  POD: data  │  │ POD: ml-    │  │  POD: nifty-        │ │
│  │  -fetcher   │  │ predictor   │  │  dashboard (UI)     │ │
│  │  :8001      │  │ :8002       │  │  :8080              │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│  ┌──────▼──────────────────────────────────────▼──────────┐ │
│  │              Shared ConfigMap / Secrets                 │ │
│  │         (Angel One API Key, model weights, etc.)        │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Repository Structure

```
nifty-algo-k8s/
├── README.md
├── k8s/
│   ├── namespace.yaml
│   ├── secrets.yaml
│   ├── configmap.yaml
│   ├── pod-data-fetcher.yaml
│   ├── pod-ml-predictor.yaml
│   ├── pod-dashboard.yaml
│   └── services.yaml
├── scripts/
│   ├── setup-cluster.sh          # Full cluster bootstrap
│   ├── deploy.sh                 # Deploy all PODs
│   ├── port-forward.sh           # Expose services locally
│   ├── data_fetcher.py           # Angel One API integration
│   ├── backtest.py               # ML backtest engine
│   ├── predictor.py              # CALL/PUT predictor
│   └── requirements.txt
└── models/
    └── nifty_model.pkl           # (generated after training)
```

---

## 🔧 Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| `kubectl` | ≥ 1.28 | Kubernetes CLI |
| `minikube` | ≥ 1.32 | Local K8s cluster |
| `docker` | ≥ 24.x | Container runtime |
| `python` | ≥ 3.10 | Algorithm scripts |
| `helm` | ≥ 3.x | (optional) chart mgmt |

---

## ⚡ Quick Start

```bash
# 1. Clone and enter repo
git clone <your-repo-url>
cd nifty-algo-k8s

# 2. Start the Kubernetes cluster
bash scripts/setup-cluster.sh

# 3. Add your Angel One credentials to secrets.yaml
# Then deploy all PODs
bash scripts/deploy.sh

# 4. Expose services locally
bash scripts/port-forward.sh

# 5. Run the prediction
python scripts/predictor.py --date today --mode live
```

---

## 🧠 Prediction Factors

The ML model considers the following signals:

| Signal | Source | Weight |
|--------|--------|--------|
| Nifty historical OHLCV | Angel One API | High |
| India VIX | NSE / Angel One | High |
| US Markets (S&P 500, Nasdaq) | Yahoo Finance API | Medium-High |
| Crude Oil Price (Brent) | Yahoo Finance / EIA | Medium |
| Geopolitical News Sentiment | NewsAPI + NLP | Medium |
| FII/DII Activity | NSE Data | Medium |
| PCR (Put-Call Ratio) | NSE OI Data | High |

---

## 📊 Output

The system outputs:
- `CALL` or `PUT` recommendation
- Confidence score (0–100%)
- Entry, Stop-Loss, Target levels
- Backtest performance report (Sharpe, Win Rate, Max Drawdown)

---

## 🔐 Security Notes

- Never commit `secrets.yaml` with real API keys
- Use Kubernetes secrets or a vault solution in production
- Angel One API tokens expire daily — the fetcher auto-refreshes

---

## 📬 Contact / Contributions

PRs welcome. For issues, open a GitHub Issue.
