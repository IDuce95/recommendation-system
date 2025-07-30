# ML Recommendation System Helm Chart

This Helm chart deploys a complete ML recommendation system with Feature Store, Real-time Recommendation API, A/B Testing Framework, and monitoring stack on Kubernetes.

## Features

- **Feature Store**: Redis-based feature storage with TTL management
- **Real-time Recommendation API**: FastAPI-based service with Kafka streaming
- **A/B Testing Framework**: Statistical analysis with multi-armed bandit optimization
- **Streamlit UI**: Interactive web interface for system management
- **ML Orchestrator**: Unified system coordinator and health monitor
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Auto-scaling**: HPA support for all services
- **Persistence**: PVC for ML models storage

## Prerequisites

- Kubernetes 1.19+
- Helm 3.8.0+
- PV provisioner support for persistent volumes (if persistence is enabled)

## Installing the Chart

To install the chart with the release name `ml-recommendation-system`:

```bash
# Add Bitnami repository for dependencies
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install the chart
helm install ml-recommendation-system ./kubernetes/helm-chart
```

The command deploys the ML recommendation system on the Kubernetes cluster with default configuration. The [Parameters](#parameters) section lists the parameters that can be configured during installation.

> **Tip**: List all releases using `helm list`

## Uninstalling the Chart

To uninstall/delete the `ml-recommendation-system` deployment:

```bash
helm delete ml-recommendation-system
```

## Parameters

### Global parameters

| Name                      | Description                                     | Value |
| ------------------------- | ----------------------------------------------- | ----- |
| `global.imageRegistry`    | Global Docker image registry                    | `""`  |
| `global.imagePullSecrets` | Global Docker registry secret names as an array| `[]`  |

### Common parameters

| Name               | Description                                        | Value                        |
| ------------------ | -------------------------------------------------- | ---------------------------- |
| `nameOverride`     | String to partially override ml-recommendation-system.fullname | `""`                         |
| `fullnameOverride` | String to fully override ml-recommendation-system.fullname    | `""`                         |

### Application configuration

| Name                    | Description                           | Value           |
| ----------------------- | ------------------------------------- | --------------- |
| `app.environment`       | Application environment               | `production`    |
| `app.debug`             | Enable debug mode                     | `false`         |
| `app.corsOrigins`       | CORS allowed origins                  | `["*"]`         |

### Recommendation API parameters

| Name                                          | Description                                    | Value           |
| --------------------------------------------- | ---------------------------------------------- | --------------- |
| `recommendationApi.enabled`                   | Enable recommendation API                     | `true`          |
| `recommendationApi.replicaCount`              | Number of replicas                            | `2`             |
| `recommendationApi.image.registry`            | Image registry                                | `docker.io`     |
| `recommendationApi.image.repository`          | Image repository                              | `your-org/recommendation-api` |
| `recommendationApi.image.tag`                 | Image tag                                     | `latest`        |
| `recommendationApi.image.pullPolicy`          | Image pull policy                             | `IfNotPresent`  |
| `recommendationApi.workers`                   | Number of worker processes                    | `4`             |
| `recommendationApi.batchSize`                 | Batch size for processing                     | `32`            |
| `recommendationApi.service.type`              | Service type                                  | `ClusterIP`     |
| `recommendationApi.service.port`              | Service port                                  | `8000`          |
| `recommendationApi.resources.limits.cpu`      | CPU resource limits                           | `1000m`         |
| `recommendationApi.resources.limits.memory`   | Memory resource limits                        | `2Gi`           |
| `recommendationApi.resources.requests.cpu`    | CPU resource requests                         | `500m`          |
| `recommendationApi.resources.requests.memory` | Memory resource requests                      | `1Gi`           |

### A/B Testing Service parameters

| Name                                         | Description                                    | Value           |
| -------------------------------------------- | ---------------------------------------------- | --------------- |
| `abTestingService.enabled`                   | Enable A/B testing service                    | `true`          |
| `abTestingService.replicaCount`              | Number of replicas                            | `1`             |
| `abTestingService.image.registry`            | Image registry                                | `docker.io`     |
| `abTestingService.image.repository`          | Image repository                              | `your-org/ab-testing-service` |
| `abTestingService.image.tag`                 | Image tag                                     | `latest`        |
| `abTestingService.confidenceLevel`           | Statistical confidence level                  | `0.95`          |
| `abTestingService.minSampleSize`             | Minimum sample size for experiments           | `100`           |
| `abTestingService.service.type`              | Service type                                  | `ClusterIP`     |
| `abTestingService.service.port`              | Service port                                  | `8001`          |

### Streamlit UI parameters

| Name                                    | Description                                    | Value           |
| --------------------------------------- | ---------------------------------------------- | --------------- |
| `streamlitUi.enabled`                   | Enable Streamlit UI                           | `true`          |
| `streamlitUi.replicaCount`              | Number of replicas                            | `1`             |
| `streamlitUi.image.registry`            | Image registry                                | `docker.io`     |
| `streamlitUi.image.repository`          | Image repository                              | `your-org/streamlit-ui` |
| `streamlitUi.image.tag`                 | Image tag                                     | `latest`        |
| `streamlitUi.service.type`              | Service type                                  | `ClusterIP`     |
| `streamlitUi.service.port`              | Service port                                  | `8501`          |

### ML Orchestrator parameters

| Name                                    | Description                                    | Value           |
| --------------------------------------- | ---------------------------------------------- | --------------- |
| `mlOrchestrator.enabled`                | Enable ML orchestrator                        | `true`          |
| `mlOrchestrator.image.registry`         | Image registry                                | `docker.io`     |
| `mlOrchestrator.image.repository`       | Image repository                              | `your-org/ml-orchestrator` |
| `mlOrchestrator.image.tag`              | Image tag                                     | `latest`        |

### Ingress parameters

| Name                  | Description                            | Value   |
| --------------------- | -------------------------------------- | ------- |
| `ingress.enabled`     | Enable ingress record generation       | `false` |
| `ingress.className`   | IngressClass that will be used         | `""`    |
| `ingress.annotations` | Additional annotations for the Ingress | `{}`    |
| `ingress.hosts`       | An array with hosts and paths          | `[]`    |
| `ingress.tls`         | TLS configuration for hosts            | `[]`    |

### Persistence parameters

| Name                                | Description                     | Value           |
| ----------------------------------- | ------------------------------- | --------------- |
| `models.persistence.enabled`       | Enable persistence for models   | `true`          |
| `models.persistence.storageClass`   | Storage class of backing PVC    | `""`            |
| `models.persistence.accessModes`    | Persistent Volume access modes  | `["ReadWriteOnce"]` |
| `models.persistence.size`           | Persistent Volume size          | `10Gi`          |

### Dependencies

| Name                   | Description                     | Value   |
| ---------------------- | ------------------------------- | ------- |
| `redis.enabled`        | Enable Redis subchart           | `true`  |
| `kafka.enabled`        | Enable Kafka subchart           | `true`  |
| `postgresql.enabled`   | Enable PostgreSQL subchart      | `true`  |
| `prometheus.enabled`   | Enable Prometheus subchart      | `true`  |
| `grafana.enabled`      | Enable Grafana subchart         | `true`  |

### Autoscaling parameters

| Name                                                          | Description                                               | Value   |
| ------------------------------------------------------------- | --------------------------------------------------------- | ------- |
| `recommendationApi.autoscaling.enabled`                      | Enable Horizontal Pod Autoscaler                         | `true`  |
| `recommendationApi.autoscaling.minReplicas`                  | Minimum number of replicas                               | `1`     |
| `recommendationApi.autoscaling.maxReplicas`                  | Maximum number of replicas                               | `10`    |
| `recommendationApi.autoscaling.targetCPUUtilizationPercentage` | Target CPU utilization percentage                       | `70`    |
| `recommendationApi.autoscaling.targetMemoryUtilizationPercentage` | Target Memory utilization percentage                 | `80`    |

## Configuration and installation details

### Resource Requirements

The default resource allocation for each component:

- **Recommendation API**: 500m CPU, 1Gi RAM (request) / 1000m CPU, 2Gi RAM (limit)
- **A/B Testing Service**: 250m CPU, 512Mi RAM (request) / 500m CPU, 1Gi RAM (limit)
- **Streamlit UI**: 100m CPU, 256Mi RAM (request) / 200m CPU, 512Mi RAM (limit)
- **ML Orchestrator**: 100m CPU, 128Mi RAM (request) / 200m CPU, 256Mi RAM (limit)

### External Dependencies

The chart includes the following external services as dependencies:

- **Redis** (Feature Store): Bitnami Redis chart
- **Kafka** (Message Streaming): Bitnami Kafka chart  
- **PostgreSQL** (Metadata Storage): Bitnami PostgreSQL chart
- **Prometheus** (Metrics): Prometheus Community chart
- **Grafana** (Dashboards): Grafana Community chart

### Networking

By default, all services are exposed as ClusterIP. For external access:

1. **Ingress**: Configure `ingress.enabled=true` and set appropriate hosts
2. **LoadBalancer**: Set service type to LoadBalancer
3. **NodePort**: Set service type to NodePort
4. **Port Forward**: Use kubectl port-forward for development

### Storage

Models are stored in a persistent volume when `models.persistence.enabled=true`. The default storage size is 10Gi. Ensure your cluster has a default storage class configured.

### Monitoring

When monitoring is enabled, the following endpoints are available:

- Prometheus: `/metrics` endpoint on port 9090
- Grafana: Dashboard interface on port 3000
- Health checks: `/health` and `/ready` endpoints on all services

### Security

- Pod Security Context: Non-root user (1001:1001)
- Read-only root filesystem for application containers
- Security Context constraints compatible
- RBAC enabled by default

## Troubleshooting

### Common Issues

1. **Pods in Pending state**: Check resource quotas and node capacity
2. **Services not accessible**: Verify service configuration and network policies
3. **Persistent Volume issues**: Ensure storage class exists and has available capacity
4. **Dependency failures**: Check if external charts are properly installed

### Debug Commands

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/instance=ml-recommendation-system

# View logs
kubectl logs -l app.kubernetes.io/component=recommendation-api -f

# Describe problematic pods
kubectl describe pod <pod-name>

# Check services
kubectl get svc -l app.kubernetes.io/instance=ml-recommendation-system

# Test connectivity
kubectl exec -it <pod-name> -- curl http://service-name:port/health
```

### Resource Monitoring

```bash
# View resource usage
kubectl top pods -l app.kubernetes.io/instance=ml-recommendation-system

# Check HPA status
kubectl get hpa

# Monitor events
kubectl get events --sort-by=.metadata.creationTimestamp
```

## Upgrading

To upgrade the chart:

```bash
helm upgrade ml-recommendation-system ./kubernetes/helm-chart
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `helm lint` and `helm template`
5. Submit a pull request

## License

This chart is licensed under the Apache 2.0 License.
