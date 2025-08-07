# ML Prediction API Makefile

.PHONY: help install test lint format clean build run-local deploy-local deploy-ec2 docker-build docker-run docker-stop docker-logs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

test: ## Run all tests
	pytest tests/ -v

test-api: ## Run API tests only
	pytest tests/test_api.py -v

test-advanced: ## Run advanced feature tests
	pytest tests/test_advanced_features.py -v

test-logging: ## Run logging tests
	pytest tests/test_logging.py -v

lint: ## Run linting checks
	flake8 src/ tests/ --config=config/.flake8

format: ## Format code with black and isort
	black src/ tests/ --config=config/pyproject.toml
	isort src/ tests/ --config=config/pyproject.toml

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

build: ## Build Docker image
	docker build -t ml-prediction-api .

run-local: ## Run API locally
	python run_api.py

deploy-local: ## Deploy locally using Docker
	./scripts/deploy_local.sh

deploy-ec2: ## Deploy to EC2
	./scripts/deploy_ec2.sh

docker-build: ## Build Docker image
	docker-compose build

docker-run: ## Run with Docker Compose
	docker-compose up -d

docker-stop: ## Stop Docker containers
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

monitoring: ## Start monitoring stack
	docker-compose -f monitoring/docker-compose-monitoring.yml up --build

monitoring-stop: ## Stop monitoring stack
	docker-compose -f monitoring/docker-compose-monitoring.yml down

monitoring-logs: ## View monitoring logs
	docker-compose -f monitoring/docker-compose-monitoring.yml logs -f

train-models: ## Train ML models
	python src/train_housing.py
	python src/train_iris.py

monitor-dashboard: ## Start monitoring dashboard
	python monitoring/monitor.py

ci: ## Run CI checks (lint, format, test)
	$(MAKE) lint
	$(MAKE) format
	$(MAKE) test

dev-setup: ## Set up development environment
	$(MAKE) install
	$(MAKE) train-models
	$(MAKE) format
	$(MAKE) lint

all: ## Run everything: setup, test, build, run
	$(MAKE) dev-setup
	$(MAKE) test
	$(MAKE) build
	$(MAKE) monitoring 