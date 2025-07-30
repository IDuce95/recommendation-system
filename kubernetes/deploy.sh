#!/bin/bash

# ML Recommendation System Helm Deployment Script
# This script helps deploy the ML recomme        "--timeout" "10m"
    )
    
    # Check if release exists
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        log_info "Upgrading existing release..."
        helm upgrade "${helm_args[@]}"
    else
        log_info "Installing new release..."
        helm install "${helm_args[@]}"
    fising Helm charts

set -e

# Configuration
CHART_DIR="./kubernetes/helm-chart"
NAMESPACE="ml-recommendation-system"
RELEASE_NAME="ml-system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed. Please install Helm 3.8.0+"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl"
        exit 1
    fi
    
    # Check Helm version
    HELM_VERSION=$(helm version --short | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+')
    log_info "Helm version: $HELM_VERSION"
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

setup_helm_repos() {
    log_info "Setting up Helm repositories..."
    
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    log_success "Helm repositories configured"
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE created"
    fi
}

validate_chart() {
    log_info "Validating Helm chart..."
    
    cd "$CHART_DIR"
    helm lint .
    helm template "$RELEASE_NAME" . --validate > /dev/null
    cd - > /dev/null
    
    log_success "Chart validation passed"
}

deploy_chart() {
    log_info "Deploying ML Recommendation System..."
    
    local helm_args=(
        "$RELEASE_NAME"
        "$CHART_DIR"
        "--namespace" "$NAMESPACE"
        "--create-namespace"
        "--wait"
        "--timeout" "600s"
    )
    
    if [[ -n "$values_file" ]]; then
        helm_args+=("--values" "$CHART_DIR/$values_file")
        log_info "Using values file: $values_file"
    fi
    
    # Check if release exists
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        log_info "Upgrading existing release..."
        helm upgrade "${helm_args[@]}"
    else
        log_info "Installing new release..."
        helm install "${helm_args[@]}"
    fi
    
    log_success "Deployment completed successfully"
}

check_deployment() {
    log_info "Checking deployment status..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance="$RELEASE_NAME" -n "$NAMESPACE" --timeout=300s
    
    # Show deployment status
    echo
    log_info "Pod Status:"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME"
    
    echo
    log_info "Service Status:"
    kubectl get services -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME"
    
    echo
    log_info "HPA Status:"
    kubectl get hpa -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" 2>/dev/null || log_warning "No HPA found"
    
    log_success "Deployment is healthy"
}

show_access_info() {
    log_info "Access Information:"
    
    # Get Streamlit UI service
    local streamlit_svc=$(kubectl get svc -n "$NAMESPACE" -l app.kubernetes.io/component=streamlit-ui -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$streamlit_svc" ]]; then
        echo
        log_info "To access Streamlit UI:"
        echo "kubectl port-forward -n $NAMESPACE svc/$streamlit_svc 8501:8501"
        echo "Then open: http://localhost:8501"
    fi
    
    # Get Recommendation API service
    local api_svc=$(kubectl get svc -n "$NAMESPACE" -l app.kubernetes.io/component=recommendation-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$api_svc" ]]; then
        echo
        log_info "To access Recommendation API:"
        echo "kubectl port-forward -n $NAMESPACE svc/$api_svc 8000:8000"
        echo "Then open: http://localhost:8000/docs"
    fi
    
    echo
    log_info "To view logs:"
    echo "kubectl logs -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME -f"
    
    echo
    log_info "To get Helm release info:"
    echo "helm status $RELEASE_NAME -n $NAMESPACE"
}

cleanup() {
    log_warning "Cleaning up deployment..."
    
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        helm uninstall "$RELEASE_NAME" -n "$NAMESPACE"
        log_success "Helm release uninstalled"
    fi
    
    read -p "Do you want to delete the namespace $NAMESPACE? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace "$NAMESPACE"
        log_success "Namespace deleted"
    fi
}

show_help() {
    echo "ML Recommendation System Helm Deployment Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  deploy          Deploy the system"
    echo "  upgrade         Upgrade existing deployment"
    echo "  status          Check deployment status"
    echo "  cleanup         Remove deployment and optionally namespace"
    echo "  access          Show access information"
    echo "  help            Show this help message"
    echo
    echo "Examples:"
    echo "  $0 deploy                     # Deploy system"
    echo "  $0 upgrade                    # Upgrade deployment"
    echo "  $0 status                     # Check current status"
    echo "  $0 cleanup                    # Remove deployment"
}

case "${1:-help}" in
    "deploy")
        check_dependencies
        setup_helm_repos
        create_namespace
        validate_chart
        deploy_chart
        check_deployment
        show_access_info
        ;;
    "upgrade")
        check_dependencies
        validate_chart
        deploy_chart
        check_deployment
        ;;
    "status")
        check_deployment
        show_access_info
        ;;
    "cleanup")
        cleanup
        ;;
    "access")
        show_access_info
        ;;
    "help"|*)
        show_help
        ;;
esac
