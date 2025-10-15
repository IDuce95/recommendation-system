# Kubernetes Deployment Guide for ML Recommendation System

## Prerequisites

1. **Kubernetes Cluster**:
   - Local: Minikube, Kind, or Docker Desktop with Kubernetes
   - Cloud: GKE, EKS, AKS, or any managed Kubernetes service

2. **Required Tools**:
   ```bash
   # Install kubectl
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   chmod +x kubectl
   sudo mv kubectl /usr/local/bin/
   
   # Install Helm
   curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
   ```

## Quick Start (Local Development)

### 1. Start Minikube
```bash
# Start Minikube with sufficient resources
minikube start --cpus=4 --memory=8192 --driver=docker

# Enable necessary addons
minikube addons enable ingress
minikube addons enable metrics-server
minikube addons enable dashboard
```

### 2. Build Docker Images
```bash
# Point Docker to Minikube's Docker daemon
eval $(minikube docker-env)

# Build images
docker compose build recommendation_api streamlit_ui mlflow
```

### 3. Deploy with Helm
```bash
cd kubernetes/helm-chart

# Install the application
helm install ml-recommendation . \
  --namespace ml-system \
  --create-namespace \
  --values values.yaml

# Check deployment status
kubectl get pods -n ml-system
kubectl get svc -n ml-system
```

### 4. Access Services

```bash
# Port-forward to access services
kubectl port-forward -n ml-system svc/ml-recommendation-api 8000:8000 &
kubectl port-forward -n ml-system svc/ml-recommendation-streamlit 8501:8501 &
kubectl port-forward -n ml-system svc/ml-recommendation-mlflow 5555:5000 &
kubectl port-forward -n ml-system svc/ml-recommendation-grafana 3000:3000 &
```

## Production Deployment

### 1. Prepare Secrets
```bash
# Create namespace
kubectl create namespace ml-system

# Create database secrets
kubectl create secret generic postgres-secret \
  --from-literal=password=your-secure-password \
  -n ml-system

# Create MLflow secrets
kubectl create secret generic mlflow-secret \
  --from-literal=backend-store-uri=postgresql://user:pass@postgres:5432/mlflow \
  -n ml-system
```

### 2. Configure values.yaml
```yaml
# Edit kubernetes/helm-chart/values.yaml
global:
  environment: production
  
api:
  replicaCount: 3
  image:
    repository: your-registry/ml-recommendation-api
    tag: latest
  resources:
    requests:
      memory: "2Gi"
      cpu: "1000m"
    limits:
      memory: "4Gi"
      cpu: "2000m"

postgresql:
  enabled: true
  auth:
    existingSecret: postgres-secret
  persistence:
    enabled: true
    size: 50Gi
```

### 3. Deploy to Production
```bash
# Deploy with production values
helm upgrade --install ml-recommendation ./kubernetes/helm-chart \
  --namespace ml-system \
  --values ./kubernetes/helm-chart/values.yaml \
  --values ./kubernetes/helm-chart/values-production.yaml \
  --wait

# Monitor rollout
kubectl rollout status deployment/ml-recommendation-api -n ml-system
```

### 4. Configure Ingress
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-recommendation-ingress
  namespace: ml-system
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  tls:
  - hosts:
    - api.ml-recommendation.yourdomain.com
    - ui.ml-recommendation.yourdomain.com
    secretName: ml-recommendation-tls
  rules:
  - host: api.ml-recommendation.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-recommendation-api
            port:
              number: 8000
  - host: ui.ml-recommendation.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-recommendation-streamlit
            port:
              number: 8501
```

## Monitoring & Observability

### Deploy Prometheus Stack
```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin123
```

### Configure Service Monitors
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-recommendation-api
  namespace: ml-system
spec:
  selector:
    matchLabels:
      app: ml-recommendation-api
  endpoints:
  - port: metrics
    interval: 30s
```

## Scaling & Performance

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: ml-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-recommendation-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Apply HPA
```bash
kubectl apply -f hpa.yaml
```

## Backup & Recovery

### Database Backup
```bash
# Create backup
kubectl exec -n ml-system postgres-pod -- pg_dump -U ml_user ml_system > backup.sql

# Restore from backup
kubectl exec -i -n ml-system postgres-pod -- psql -U ml_user ml_system < backup.sql
```

### Persistent Volume Snapshots
```bash
# Create volume snapshot
kubectl apply -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: postgres-snapshot
  namespace: ml-system
spec:
  volumeSnapshotClassName: csi-hostpath-snapclass
  source:
    persistentVolumeClaimName: postgres-pvc
EOF
```

## Troubleshooting

### Common Issues

1. **Pods not starting**:
```bash
kubectl describe pod <pod-name> -n ml-system
kubectl logs <pod-name> -n ml-system --previous
```

2. **Service not accessible**:
```bash
kubectl get endpoints -n ml-system
kubectl get svc -n ml-system
```

3. **Resource issues**:
```bash
kubectl top nodes
kubectl top pods -n ml-system
```

### Useful Commands
```bash
# Get all resources
kubectl get all -n ml-system

# Debug pod
kubectl exec -it <pod-name> -n ml-system -- /bin/bash

# View logs
kubectl logs -f deployment/ml-recommendation-api -n ml-system

# Delete and reinstall
helm uninstall ml-recommendation -n ml-system
kubectl delete namespace ml-system
```

## CI/CD Integration

### GitHub Actions Deployment
```yaml
# .github/workflows/deploy.yml
name: Deploy to Kubernetes
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Configure kubectl
      uses: azure/setup-kubectl@v1
    - name: Deploy
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        helm upgrade --install ml-recommendation ./kubernetes/helm-chart \
          --namespace ml-system \
          --wait
```

## Security Best Practices

1. **Network Policies**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
  namespace: ml-system
spec:
  podSelector:
    matchLabels:
      app: ml-recommendation-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: ml-recommendation-streamlit
    ports:
    - protocol: TCP
      port: 8000
```

2. **Pod Security Policies**:
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## Support

For issues or questions:
- Check logs: `kubectl logs -f <pod-name> -n ml-system`
- Review events: `kubectl get events -n ml-system`
- Dashboard: `minikube dashboard` or `kubectl proxy`
