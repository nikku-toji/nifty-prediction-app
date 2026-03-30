#!/usr/bin/env bash
# =============================================================================
# deploy.sh
# Deploys all algo-trading PODs to the local Kubernetes cluster
# Usage: bash scripts/deploy.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

NS="algo-trading"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "Checking cluster connectivity..."
kubectl cluster-info --request-timeout=5s >/dev/null 2>&1 || error "Cannot reach cluster. Run setup-cluster.sh first."
success "Cluster reachable"

kubectl get namespace "$NS" >/dev/null 2>&1 || error "Namespace '$NS' missing. Run setup-cluster.sh first."
success "Namespace exists"

kubectl get secret algo-trading-secrets -n "$NS" >/dev/null 2>&1 \
  || error "Secrets not found. Apply k8s/secrets.yaml first."
success "Secrets found"

# ── Deploy ────────────────────────────────────────────────────────────────────
info "Deploying Data Fetcher POD..."
kubectl apply -f k8s/pod-data-fetcher.yaml
success "Data Fetcher deployed"

info "Deploying ML Predictor POD..."
kubectl apply -f k8s/pod-ml-predictor.yaml
success "ML Predictor deployed"

# ── Wait for rollout ──────────────────────────────────────────────────────────
info "Waiting for all deployments to be ready (timeout: 3min)..."
kubectl rollout status deployment/data-fetcher   -n "$NS" --timeout=180s
kubectl rollout status deployment/ml-predictor   -n "$NS" --timeout=180s
kubectl rollout status deployment/nifty-dashboard -n "$NS" --timeout=180s
success "All deployments ready"

# ── Status ────────────────────────────────────────────────────────────────────
echo ""
info "Current pod status:"
kubectl get pods -n "$NS" -o wide

echo ""
info "Services:"
kubectl get services -n "$NS"

echo ""
echo -e "${GREEN}✅ All PODs deployed successfully!${NC}"
echo ""
echo "  Next: bash scripts/port-forward.sh"
echo ""
