.PHONY: help install run-api run-streamlit run-docker test clean setup lint format security ci-local generate_data load_data_to_db create_table preprocess_data api streamlit

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  run-api     - Run FastAPI server"
	@echo "  run-streamlit - Run Streamlit app"
	@echo "  run-docker  - Run with Docker Compose"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code with black and isort"
	@echo "  security    - Run security checks"
	@echo "  ci-local    - Run full CI pipeline locally"
	@echo "  clean       - Clean up generated files"
	@echo "  setup       - Initial project setup"

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov flake8 black isort safety bandit mypy radon

generate_data:
	- python data/data_generator.py

load_data_to_db:
	- python data/load_data_to_db.py

create_table:
	- PGPASSWORD=password psql -U postgres -d recommendation_system -f data/create_table.sql

preprocess_data:
	- python app/data_processing/data_preprocessor.py

api:
	- python app/api/fastapi_server.py

streamlit:
	- streamlit run app/streamlit_app/streamlit_ui.py

run-api:
	export PYTHONPATH=$$(pwd):$$PYTHONPATH && python app/api/fastapi_server.py

run-streamlit:
	export PYTHONPATH=$$(pwd):$$PYTHONPATH && streamlit run app/streamlit_app/streamlit_ui.py

run-docker:
	docker-compose up

test:
	export PYTHONPATH=$$(pwd):$$PYTHONPATH && python -m pytest tests/ -v

test-cov:
	export PYTHONPATH=$$(pwd):$$PYTHONPATH && python -m pytest tests/ -v --cov=app --cov=ai --cov=config --cov=data --cov-report=xml --cov-report=term-missing

lint:
	flake8 app/ ai/ config/ data/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 app/ ai/ config/ data/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy app/ ai/ config/ data/ --ignore-missing-imports || true

format:
	black app/ ai/ config/ data/ tests/
	isort app/ ai/ config/ data/ tests/

format-check:
	black --check app/ ai/ config/ data/ tests/
	isort --check-only app/ ai/ config/ data/ tests/

security:
	safety check
	bandit -r app/ ai/ config/ data/ -f json || true

complexity:
	radon cc app/ ai/ config/ data/ --min B
	radon mi app/ ai/ config/ data/ --min B

ci-local: format-check lint security test-cov complexity
	@echo "✅ All CI checks passed locally!"

build-images:
	docker build -t recommendation-api:latest -f docker/Dockerfile.api .
	docker build -t recommendation-streamlit:latest -f docker/Dockerfile.streamlit .

k8s-deploy:
	./kubernetes/manage.sh deploy

k8s-status:
	./kubernetes/manage.sh status

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf .mypy_cache

setup:
	@echo "Setting up project..."
	cp .env.example .env
	@echo "Please edit .env file with your configuration"
	@echo "Then run 'make install' to install dependencies"