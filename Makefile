.PHONY: help setup build up down clean test lint format

help:
	@echo "ML Recommendation System Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  setup              - Install dependencies and setup environment"
	@echo "  install-deps       - Install Python dependencies"
	@echo "  setup-data         - Initialize database and load sample data"
	@echo ""
	@echo "Docker:"
	@echo "  build              - Build Docker images"
	@echo "  up                 - Start all services"
	@echo "  down               - Stop all services"
	@echo "  restart            - Restart all services"
	@echo "  logs               - Show service logs"
	@echo ""
	@echo "Development:"
	@echo "  test               - Run tests"
	@echo "  lint               - Run code linting"
	@echo "  format             - Format code"
	@echo "  clean              - Clean up containers and cache"
	@echo ""
	@echo "Kubernetes:"
	@echo "  helm-deploy        - Deploy using Helm"
	@echo "  helm-status        - Check Helm deployment status"
	@echo "  helm-cleanup       - Remove Helm deployment"

DOCKER_COMPOSE_FILE = docker-compose.yml

setup: install-deps setup-data
	@echo "✅ Setup complete"

install-deps:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

setup-data:
	python data/data_generator.py
	python data/load_data_to_db.py

build:
	docker compose -f $(DOCKER_COMPOSE_FILE) build

up:
	docker compose -f $(DOCKER_COMPOSE_FILE) up -d
	@echo "Services available:"
	@echo "  - API: http://localhost:8000"
	@echo "  - UI: http://localhost:8501"
	@echo "  - Orchestrator: http://localhost:8003"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000"

down:
	docker compose -f $(DOCKER_COMPOSE_FILE) down

restart: down up

logs:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f

test:
	python -m pytest tests/ -v

lint:
	flake8 app/ ai/ config/ data/ --max-line-length=100
	pylint app/ ai/ config/ data/ --fail-under=8.0

format:
	black app/ ai/ config/ data/
	isort app/ ai/ config/ data/

clean:
	docker compose -f $(DOCKER_COMPOSE_FILE) down -v --remove-orphans
	docker system prune -f
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

helm-deploy:
	./kubernetes/deploy.sh deploy production

helm-status:
	./kubernetes/deploy.sh status

helm-cleanup:
	./kubernetes/deploy.sh cleanup

quick-start: setup build up
	@echo "🚀 ML Recommendation System ready!"