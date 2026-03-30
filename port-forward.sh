#!/usr/bin/env bash
# =============================================================================
# port-forward.sh
# Exposes all Kubernetes services to localhost for local development/testing
# Usage: bash scripts/port-forward.sh
# Stop: Ctrl+C (kills all background forwards)
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }

NS="algo-trading"

# Track PIDs to kill on exit
PIDS=()
cleanup() {
    echo ""
    echo -e "${YELLOW}[STOP]${NC}  Stopping all port forwards..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    echo "Done."
}
trap cleanup INT TERM EXIT

# ── Data Fetcher: localhost:8001 ──────────────────────────────────────────────
info "Forwarding Data Fetcher  → localhost:8001"
kubectl port-forward svc/data-fetcher-service 8001:8001 -n "$NS" &>/tmp/pf-data.log &
PIDS+=($!)
sleep 1
success "http://localhost:8001  (Data Fetcher API)"

# ── ML Predictor: localhost:8002 ─────────────────────────────────────────────
info "Forwarding ML Predictor  → localhost:8002"
kubectl port-forward svc/ml-predictor-service 8002:8002 -n "$NS" &>/tmp/pf-ml.log &
PIDS+=($!)
sleep 1
success "http://localhost:8002  (ML Predictor API)"

# ── Dashboard: localhost:8080 ─────────────────────────────────────────────────
info "Forwarding Dashboard     → localhost:8080"
kubectl port-forward svc/dashboard-service 8080:8080 -n "$NS" &>/tmp/pf-dash.log &
PIDS+=($!)
sleep 1
success "http://localhost:8080  (Nifty Dashboard UI)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Services Available Locally 🌐                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  📊 Dashboard     → http://localhost:8080"
echo "  📡 Data Fetcher  → http://localhost:8001/docs"
echo "  🤖 ML Predictor  → http://localhost:8002/docs"
echo ""
echo "  Endpoints:"
echo "  GET  http://localhost:8001/fetch/nifty?days=365"
echo "  GET  http://localhost:8001/fetch/us-markets"
echo "  GET  http://localhost:8001/fetch/crude-oil"
echo "  GET  http://localhost:8001/sentiment/geopolitical"
echo "  POST http://localhost:8002/predict   {\"horizon\":1,\"mode\":\"live\"}"
echo "  GET  http://localhost:8002/backtest  ?days=365"
echo ""
echo "  Press Ctrl+C to stop all port forwards."
echo ""

# Keep running until killed
wait
