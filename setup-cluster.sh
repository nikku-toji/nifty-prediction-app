#!/usr/bin/env bash
# =============================================================================
# setup-cluster.sh
# Bootstraps a local Minikube Kubernetes cluster for the Nifty Algo platform
# Usage: bash scripts/setup-cluster.sh
# =============================================================================

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Check prerequisites ───────────────────────────────────────────────────────
info "Checking prerequisites..."

check_tool() {
    command -v "$1" >/dev/null 2>&1 || error "$1 is not installed. Please install it first."
    success "$1 found: $(command -v $1)"
}

check_tool minikube
check_tool kubectl
check_tool docker

# ── Start Minikube ────────────────────────────────────────────────────────────
info "Starting Minikube cluster (4 CPUs, 8GB RAM, 30GB disk)..."
minikube start \
  --driver=docker \
  --cpus=4 \
  --memory=8192 \
  --disk-size=30g \
  --kubernetes-version=v1.28.0 \
  --addons=metrics-server \
  --addons=dashboard

success "Minikube cluster started"

# ── Verify cluster ────────────────────────────────────────────────────────────
info "Verifying cluster health..."
kubectl cluster-info
kubectl get nodes -o wide

# ── Enable addons ─────────────────────────────────────────────────────────────
info "Enabling useful addons..."
minikube addons enable ingress
minikube addons enable storage-provisioner
success "Addons enabled"

# ── Create namespace ──────────────────────────────────────────────────────────
info "Creating algo-trading namespace..."
kubectl apply -f k8s/namespace.yaml
success "Namespace 'algo-trading' created"

# ── Configure secrets ─────────────────────────────────────────────────────────
info "Setting up secrets..."
echo ""
warn "⚠️  You must now fill in your API credentials."
warn "Edit k8s/secrets.yaml and run: kubectl apply -f k8s/secrets.yaml"
echo ""
echo "  To encode your credentials:"
echo "  echo -n 'YOUR_API_KEY' | base64"
echo ""

# ── Apply ConfigMap ───────────────────────────────────────────────────────────
info "Applying ConfigMap..."
kubectl apply -f k8s/configmap.yaml
success "ConfigMap applied"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Kubernetes Cluster Ready! ✅                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Cluster IP  : $(minikube ip)"
echo "  Dashboard   : run 'minikube dashboard' to open"
echo ""
echo "  Next steps:"
echo "  1. Fill credentials: edit k8s/secrets.yaml"
echo "  2. kubectl apply -f k8s/secrets.yaml"
echo "  3. bash scripts/deploy.sh"
echo ""
