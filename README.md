# Product Recommendation System

System rekomendacji produktów wykorzystujący AI z embeddings tekstowych i wizualnych.

## Uruchomienie

```bash
cp .env.example .env
./docker/manage.sh up
```

## Dostęp
- UI: http://localhost:8501
- API: http://localhost:5000
- MLflow: http://localhost:5555
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Stack
FastAPI • Streamlit • PostgreSQL • MLflow • Prometheus • Grafana • LangChain