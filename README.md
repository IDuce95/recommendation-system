# ML Recommendation System

Kompletny system rekomendacji z uczeniem maszynowym i infrastrukturą produkcyjną.

## Zaimplementowane narzędzia

### Core ML Components
- **Feature Store** - magazyn cech z cache'owaniem i wersjonowaniem
- **Recommendation Engine** - silnik rekomendacji z modelem ensemble
- **Text/Image Processing** - przetwarzanie tekstu i obrazów
- **Model Manager** - zarządzanie modelami ML z LangGraph

### API & Services
- **FastAPI Server** - REST API z dokumentacją OpenAPI
- **A/B Testing Service** - testowanie wersji algorytmów
- **Real-time Recommendations** - rekomendacje w czasie rzeczywistym
- **Batch Processing** - przetwarzanie wsadowe

### UI & Monitoring
- **Streamlit Dashboard** - interfejs użytkownika
- **Grafana Dashboards** - monitorowanie systemu
- **Prometheus Metrics** - zbieranie metryk
- **Health Checks** - sprawdzanie stanu systemu

### Data Pipeline
- **Apache Airflow** - orkiestracja ETL
- **Data Generator** - generowanie danych testowych
- **PostgreSQL** - baza danych
- **Redis** - cache i sesje
- **Apache Kafka** - streaming danych

### Infrastructure
- **Docker Compose** - lokalne środowisko
- **Kubernetes Helm** - deployment produkcyjny
- **Load Balancing** - równoważenie obciążenia
- **Auto-scaling** - automatyczne skalowanie

### Development Tools
- **MLflow Tracking** - śledzenie eksperymentów
- **Configuration Management** - zarządzanie konfiguracją
- **Environment Setup** - konfiguracja środowisk
- **Build Automation** - automatyzacja budowy

## Uruchomienie

```bash
# Lokalne środowisko
make docker-up

# Kubernetes
make k8s-deploy

# Streamlit UI
make streamlit-run
```