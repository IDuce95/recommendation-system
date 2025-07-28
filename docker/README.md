# Docker Setup Documentation

## 🐳 Docker Configuration for Product Recommendation System

This directory contains all Docker-related files for the Product Recommendation System, optimized for development, testing, and production environments.

## 📁 Directory Structure

```
docker/
├── Dockerfile.api              # FastAPI backend image
├── Dockerfile.streamlit        # Streamlit frontend image  
├── Dockerfile.postgres         # PostgreSQL database image
├── postgresql.conf             # PostgreSQL configuration
├── streamlit_config.toml       # Streamlit configuration
├── daemon.json                 # Docker daemon configuration
├── setup-docker-home.sh        # Script to configure Docker for /home partition
├── manage.sh                   # Docker management script
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Configure Docker for /home partition (One-time setup)

```bash
# Run this once to configure Docker to store data on /home partition
sudo ./docker/setup-docker-home.sh
```

This script:
- Creates `/home/docker-data` directory for Docker storage
- Configures Docker daemon to use the new location
- Restarts Docker service with new configuration

### 2. Start the Application

```bash
# Using the management script (recommended)
./docker/manage.sh up

# Or using docker-compose directly
docker compose up -d
```

### 3. Access the Application

- **Streamlit UI**: http://localhost:8501
- **FastAPI API**: http://localhost:5000
- **MLflow UI**: http://localhost:5555
- **Database**: localhost:5432
- **API Documentation**: http://localhost:5000/docs
- **Health Check**: http://localhost:5000/health

## 🛠️ Management Script Usage

The `manage.sh` script provides convenient commands for managing the Docker environment:

```bash
# Build images
./docker/manage.sh build [env]

# Start services
./docker/manage.sh up [env]

# Stop services  
./docker/manage.sh down [env]

# Restart services
./docker/manage.sh restart [env]

# View logs
./docker/manage.sh logs [env] [service]

# Check status
./docker/manage.sh status [env]

# Clean up everything
./docker/manage.sh clean

# Show help
./docker/manage.sh help
```

## 🔧 Docker Compose File

### docker-compose.yml (Simple)
- Single compose file for all use cases
- Uses environment variables from `.env` file
- Configured for development with hot reload
- All services: postgres, api, streamlit, mlflow

### Environment File
- `.env` - Simple configuration with essential variables only
- Variables control ports, database settings, and development features

## 🏗️ Docker Images

### FastAPI Backend (Dockerfile.api)
- Based on `python:3.11-slim`
- Multi-stage build for optimization
- Non-root user for security
- Health check endpoint at `/health`
- Optimized for production with multiple workers

### Streamlit Frontend (Dockerfile.streamlit)
- Based on `python:3.11-slim`
- Custom Streamlit configuration
- Non-root user for security
- Health check via Streamlit's built-in endpoint

### PostgreSQL Database (Dockerfile.postgres)
- Based on `postgres:15-alpine`
- Custom PostgreSQL configuration
- Initialization scripts included
- Optimized for development workloads

## 📊 Volume Management

All persistent data is stored on the `/home` partition:

```bash
/home/docker-volumes/
├── postgres_data_dev/          # Development database data
├── postgres_data_prod/         # Production database data
└── ...
```

Project logs are stored locally:

```bash
./logs/
├── api/                        # FastAPI logs
├── streamlit/                  # Streamlit logs
└── postgres/                   # PostgreSQL logs
```

## 🔒 Security Features

- Non-root users in all containers
- Read-only volume mounts in production
- Network isolation with custom bridge networks
- Health checks for all services
- Resource limits to prevent resource exhaustion

## 🚨 Troubleshooting

### Common Issues

1. **Permission denied errors**
   ```bash
   sudo chown -R $USER:$USER /home/docker-volumes/
   ```

2. **Port already in use**
   ```bash
   ./docker/manage.sh down
   # Or check what's using the port
   sudo netstat -tulpn | grep :5000
   ```

3. **Database connection issues**
   ```bash
   # Check if database is healthy
   ./docker/manage.sh status
   
   # View database logs
   ./docker/manage.sh logs dev postgres
   ```

4. **Docker daemon not using /home partition**
   ```bash
   # Verify Docker data directory
   docker info | grep "Docker Root Dir"
   
   # Should show: /home/docker-data
   ```

### Health Checks

All services include health checks:

- **PostgreSQL**: `pg_isready` command
- **FastAPI**: HTTP GET to `/health` endpoint
- **Streamlit**: HTTP GET to `/_stcore/health` endpoint

View health status:
```bash
./docker/manage.sh status [env]
```

## 🔄 Development Workflow

1. **Setup** (one-time):
   ```bash
   ./docker/manage.sh setup-docker
   ./docker/manage.sh build dev
   ```

2. **Daily development**:
   ```bash
   ./docker/manage.sh up dev
   # Make changes to code (auto-reload enabled)
   ./docker/manage.sh logs dev api  # View logs
   ```

3. **Testing changes**:
   ```bash
   ./docker/manage.sh restart dev
   ```

4. **Cleanup**:
   ```bash
   ./docker/manage.sh down dev
   ```

## 📈 Production Deployment

1. **Build production images**:
   ```bash
   ./docker/manage.sh build prod
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with production values
   ```

3. **Deploy**:
   ```bash
   ./docker/manage.sh up prod
   ```

4. **Monitor**:
   ```bash
   ./docker/manage.sh status prod
   ./docker/manage.sh logs prod
   ```

## 🔗 Next Steps

After completing Phase 1.1, the following phases will add:

- **Phase 1.2**: MLflow integration for model tracking
- **Phase 2.1**: Prometheus monitoring
- **Phase 2.2**: Grafana dashboards
- **Phase 3.1**: Kubernetes deployment
- **Phase 4.1**: Apache Airflow data pipelines

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. View service logs: `./docker/manage.sh logs [env] [service]`
3. Check service status: `./docker/manage.sh status [env]`
