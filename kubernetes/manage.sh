#!/bin/bash

set -e

CLUSTER_NAME="recommendation-system"
NAMESPACE="recommendation-system"

show_help() {
    echo "Kubernetes Management Script for Recommendation System"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup           - Create cluster and install nginx ingress"
    echo "  deploy          - Deploy all services to Kubernetes"
    echo "  delete          - Delete all deployments"
    echo "  status          - Show status of all pods and services"
    echo "  logs [service]  - Show logs for specific service"
    echo "  port-forward    - Setup port forwarding for local access"
    echo "  build-images    - Build Docker images for Kubernetes"
    echo "  load-images     - Load Docker images into kind cluster"
    echo "  clean           - Delete cluster and cleanup"
    echo "  help            - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 deploy"
    echo "  $0 logs fastapi"
    echo "  $0 port-forward"
}

create_cluster() {
    echo "Creating kind cluster: $CLUSTER_NAME"
    if kind get clusters | grep -q "$CLUSTER_NAME"; then
        echo "Cluster $CLUSTER_NAME already exists"
    else
        kind create cluster --name "$CLUSTER_NAME"
    fi
    
    echo "Installing nginx ingress controller..."
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
    
    echo "Waiting for ingress controller to be ready..."
    kubectl wait --namespace ingress-nginx \
        --for=condition=ready pod \
        --selector=app.kubernetes.io/component=controller \
        --timeout=90s
}

build_images() {
    echo "Building Docker images..."
    
    cd /home/palianm/Desktop/recommendation-system
    
    echo "Building FastAPI image..."
    docker build -t recommendation-api:latest -f docker/Dockerfile.api .
    
    echo "Building Streamlit image..."
    docker build -t recommendation-streamlit:latest -f docker/Dockerfile.streamlit .
    
    echo "Images built successfully!"
}

load_images() {
    echo "Loading images into kind cluster..."
    
    kind load docker-image recommendation-api:latest --name "$CLUSTER_NAME"
    kind load docker-image recommendation-streamlit:latest --name "$CLUSTER_NAME"
    
    echo "Images loaded successfully!"
}

deploy_all() {
    echo "Deploying all services to Kubernetes..."
    
    kubectl apply -f kubernetes/namespace.yaml
    kubectl apply -f kubernetes/configmaps.yaml
    kubectl apply -f kubernetes/persistent-volumes.yaml
    kubectl apply -f kubernetes/postgres.yaml
    
    echo "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout=120s
    
    kubectl apply -f kubernetes/mlflow.yaml
    kubectl apply -f kubernetes/fastapi.yaml
    kubectl apply -f kubernetes/streamlit.yaml
    kubectl apply -f kubernetes/monitoring.yaml
    kubectl apply -f kubernetes/ingress.yaml
    
    echo "Deployment completed!"
    echo "Use '$0 status' to check deployment status"
}

delete_all() {
    echo "Deleting all deployments..."
    
    kubectl delete -f kubernetes/ingress.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/monitoring.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/streamlit.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/fastapi.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/mlflow.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/postgres.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/persistent-volumes.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/configmaps.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/namespace.yaml --ignore-not-found=true
    
    echo "All deployments deleted!"
}

show_status() {
    echo "=== Cluster Info ==="
    kubectl cluster-info --context "kind-$CLUSTER_NAME"
    
    echo ""
    echo "=== Nodes ==="
    kubectl get nodes
    
    echo ""
    echo "=== Namespaces ==="
    kubectl get namespaces
    
    echo ""
    echo "=== Pods in $NAMESPACE ==="
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo ""
    echo "=== Services in $NAMESPACE ==="
    kubectl get services -n "$NAMESPACE"
    
    echo ""
    echo "=== Ingress in $NAMESPACE ==="
    kubectl get ingress -n "$NAMESPACE"
    
    echo ""
    echo "=== HPA in $NAMESPACE ==="
    kubectl get hpa -n "$NAMESPACE"
}

show_logs() {
    local service=$1
    if [ -z "$service" ]; then
        echo "Available services:"
        kubectl get pods -n "$NAMESPACE" --no-headers | awk '{print $1}' | cut -d'-' -f1 | sort -u
        return 1
    fi
    
    echo "Showing logs for $service..."
    kubectl logs -l app="$service" -n "$NAMESPACE" --tail=50 -f
}

port_forward() {
    echo "Setting up port forwarding..."
    echo "You can access services at:"
    echo "  FastAPI:   http://localhost:5000"
    echo "  Streamlit: http://localhost:8501"
    echo "  MLflow:    http://localhost:5555"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana:   http://localhost:3000"
    echo ""
    echo "Press Ctrl+C to stop port forwarding"
    
    kubectl port-forward -n "$NAMESPACE" service/fastapi-service 5000:5000 &
    kubectl port-forward -n "$NAMESPACE" service/streamlit-service 8501:8501 &
    kubectl port-forward -n "$NAMESPACE" service/mlflow-service 5555:5555 &
    kubectl port-forward -n "$NAMESPACE" service/prometheus-service 9090:9090 &
    kubectl port-forward -n "$NAMESPACE" service/grafana-service 3000:3000 &
    
    wait
}

clean_cluster() {
    echo "Deleting kind cluster: $CLUSTER_NAME"
    kind delete cluster --name "$CLUSTER_NAME"
    echo "Cluster deleted!"
}

case "$1" in
    setup)
        create_cluster
        ;;
    build-images)
        build_images
        ;;
    load-images)
        load_images
        ;;
    deploy)
        deploy_all
        ;;
    delete)
        delete_all
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    port-forward)
        port_forward
        ;;
    clean)
        clean_cluster
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac
