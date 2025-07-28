# Product Recommendation System

An AI-powered product recommendation system that combines textual and visual embeddings to provide intelligent product suggestions. This project serves as a comprehensive learning platform for modern ML/AI and DevOps technologies.

## Project Overview

The system leverages multiple AI technologies to create sophisticated product recommendations:

- **Text Analysis**: Uses LLM embeddings to understand product descriptions and categories
- **Visual Recognition**: Processes product images to extract visual features
- **Smart Agent**: LangChain/LangGraph agent that integrates different recommendation components
- **Real-time API**: FastAPI backend with comprehensive monitoring and health checks
- **Interactive UI**: Streamlit frontend for easy product exploration and recommendations

## Key Features

- **Multi-modal Recommendations**: Combines text and image data for better accuracy
- **Experiment Tracking**: MLflow integration for model versioning and experiment management
- **Production Monitoring**: Prometheus metrics collection with Grafana dashboards
- **Containerized Deployment**: Full Docker setup with orchestration
- **Scalable Architecture**: Designed for easy scaling and production deployment

## Quick Start

```bash
# Setup environment
cp .env.example .env

# Launch all services
./docker/manage.sh up
```

## Access Points
- **Streamlit UI**: http://localhost:8501 - Main user interface
- **FastAPI API**: http://localhost:5000 - REST API endpoints
- **MLflow UI**: http://localhost:5555 - Experiment tracking
- **Prometheus**: http://localhost:9090 - Metrics collection
- **Grafana**: http://localhost:3000 - Monitoring dashboards

## Technology Stack
FastAPI • Streamlit • PostgreSQL • MLflow • Prometheus • Grafana • LangChain • PyTorch