# ML Prediction API - Project Structure

This document outlines the comprehensive structure of the ML Prediction API project with MLOps best practices.

## Overview

A production-ready machine learning prediction API with comprehensive MLOps features including:
- FastAPI-based prediction service for Housing and Iris models
- Data Version Control (DVC) for reproducible ML pipelines
- Prometheus + Grafana monitoring stack
- Docker containerization with multi-stage builds
- GitHub Actions CI/CD pipeline
- MLflow experiment tracking and model registry
- Comprehensive testing suite
- Automated logging and error tracking

## Directory Layout

```
mlops_assignment/
├── README.md                           # Main project overview and setup guide
├── Makefile                           # Development and deployment commands
├── run_api.py                         # Local API runner script
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Multi-stage Docker build configuration
├── docker-compose.yml                # Basic Docker Compose setup
├── .dockerignore                      # Docker ignore patterns
├── PROJECT_STRUCTURE.md               # This file - comprehensive project structure
│
├── src/                              # Core application source code
│   ├── api.py                        # FastAPI application with enhanced model loading
│   ├── logger.py                     # Structured logging with multiple handlers
│   ├── schemas.py                    # Pydantic models and API contracts
│   ├── prometheus_metrics.py         # Comprehensive Prometheus metrics collection
│   ├── model_retraining.py          # Automated model retraining workflows
│   ├── model_evaluation.py          # Model evaluation and comparison (DVC)
│   ├── data_pipeline.py             # Data preprocessing pipeline
│   ├── train_housing.py             # Housing price prediction model training
│   ├── train_iris.py                # Iris classification model training
│   ├── data_load_housing.py         # Housing dataset loading and preprocessing
│   └── data_load_iris.py            # Iris dataset loading and preprocessing
│
├── tests/                            # Comprehensive test suite
│   ├── __init__.py                   # Test package initialization
│   ├── test_api.py                   # API endpoint integration tests
│   ├── test_advanced_features.py    # Advanced ML features testing
│   └── test_logging.py              # Logging functionality unit tests
│
├── scripts/                          # Deployment and utility scripts
│   └── deploy_local.sh               # Enhanced local deployment with health checks
│
├── monitoring/                       # Complete monitoring stack
│   ├── monitor.py                    # Real-time monitoring dashboard
│   ├── prometheus.yml                # Prometheus scraping configuration
│   ├── grafana-dashboard.json        # Legacy dashboard (deprecated)
│   ├── docker-compose-monitoring.yml # Monitoring services orchestration
│   │
│   └── grafana/                      # Grafana configuration structure
│       ├── dashboards/               # Dashboard definitions
│       │   └── ml-dashboard.json     # ML metrics dashboard with Prometheus datasource
│       └── provisioning/             # Auto-provisioning configuration
│           ├── datasources/          # Datasource configurations
│           │   └── prometheus.yml    # Prometheus datasource (prometheus:9090)
│           └── dashboards/           # Dashboard provisioning
│               └── dashboard.yml     # Dashboard provider configuration
│
├── config/                           # Configuration management
│   ├── pyproject.toml               # Black, isort, and tool configurations
│   └── pytest.ini                   # Pytest configuration and coverage settings
│
├── .github/                          # GitHub Actions CI/CD workflows
│   └── workflows/
│       └── ci-cd.yml                # Complete CI/CD pipeline with actual deployment
│
├── logs/                            # Application logging
│   ├── api.log                      # Structured API application logs
│   └── predictions.db               # SQLite prediction audit trail
│
├── data/                            # Dataset storage
│   ├── housing_raw.csv              # Raw housing market data
│   ├── housing_processed.csv        # Cleaned and preprocessed housing data
│   ├── iris_raw.csv                 # Raw iris classification dataset
│   └── iris_processed.csv           # Preprocessed iris features
│
├── mlruns/                          # MLflow experiment tracking
│   └── [experiment_id]/             # Experiment run artifacts
│       ├── artifacts/               # Model artifacts and metadata
│       ├── metrics/                 # Training and validation metrics
│       └── params/                  # Hyperparameters and configuration
│
├── .dvc/                            # DVC version control system
│   ├── config                       # DVC configuration
│   └── cache/                       # DVC cached artifacts
│
├── dvc.yaml                         # DVC pipeline definition (6-stage ML workflow)
├── dvc.lock                         # DVC pipeline lock file
├── params.yaml                      # DVC pipeline parameters
│
└── models/                          # Model registry and artifacts
    ├── housing/                     # Housing prediction models
    │   ├── model.pkl                # Trained model artifacts
    │   └── metrics.json             # Model performance metrics
    └── iris/                        # Iris classification models
        ├── model.pkl                # Trained model artifacts
        └── metrics.json             # Model performance metrics
```

## Key Features & Components

### Machine Learning Pipeline
- **Models**: Housing price prediction (regression) + Iris classification
- **Training**: Automated training scripts with MLflow tracking
- **Evaluation**: Comprehensive model evaluation and comparison
- **Registry**: MLflow model registry with versioning

### Data Version Control (DVC)
- **Pipeline Stages**:
  1. `data_preparation` - Raw data loading and validation
  2. `feature_engineering` - Feature preprocessing and transformation
  3. `train_housing` - Housing model training with hyperparameters
  4. `train_iris` - Iris model training and optimization
  5. `model_evaluation` - Cross-model performance comparison
  6. `deployment_validation` - Production readiness checks
- **Reproducibility**: Parameterized pipelines with `params.yaml`
- **Artifacts**: Versioned models, metrics, and datasets

### API Service
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Endpoints**: Health checks, model info, predictions, metrics
- **Model Loading**: Multi-strategy loading (MLflow, local files, training fallback)
- **Error Handling**: Comprehensive error responses and logging
- **Validation**: Pydantic schema validation for all inputs/outputs

### Monitoring & Observability
- **Metrics**: Prometheus metrics for requests, predictions, errors, latency
- **Dashboards**: Grafana dashboards with ML-specific visualizations
- **Logging**: Structured logging with multiple output formats
- **Health Checks**: Container and service health monitoring
- **Alerting**: Performance and error rate monitoring

### Containerization
- **Multi-stage Build**: Optimized Docker images with model training
- **Orchestration**: Docker Compose for local development
- **Networking**: Proper service discovery (prometheus:9090, ml-api:8000)
- **Volumes**: Persistent storage for models, logs, and metrics

### CI/CD Pipeline
- **Stages**: Lint → Test → Build → Deploy → Verify
- **Quality Gates**: Black, isort, flake8, pytest with coverage
- **Deployment**: Automated local deployment with health verification
- **Monitoring**: Optional monitoring stack deployment
- **Artifacts**: Docker image building and registry push

### Testing Strategy
- **Unit Tests**: Core functionality and edge cases
- **Integration Tests**: API endpoints and workflows
- **Coverage**: Automated coverage reporting with Codecov
- **Fixtures**: Reusable test data and mock objects

## Usage Examples

### Quick Start
```bash
# Clone and setup
git clone <repository>
cd mlops_assignment
pip install -r requirements.txt

# Train models
cd src && python data_pipeline.py && python train_housing.py && python train_iris.py

# Run API
python run_api.py

# Access API docs
open http://localhost:8000/docs
```

### DVC Workflow
```bash
# Reproduce entire pipeline
dvc repro

# Run specific stage
dvc repro train_housing

# Show pipeline status
dvc dag

# Check metrics
dvc metrics show
```

### Docker Deployment
```bash
# API only
./scripts/deploy_local.sh

# Full stack (API + monitoring)
./scripts/deploy_comprehensive.sh --full

# Check status
./scripts/deploy_comprehensive.sh --status
```

### Monitoring Access
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Prometheus Metrics**: http://localhost:9090
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **API Metrics**: http://localhost:8000/metrics/prometheus

### Testing Commands
```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific test categories
pytest tests/test_api.py -v
pytest tests/test_advanced_features.py -v

# Generate coverage report
pytest --cov=src --cov-report=html
```

### Development Workflow
```bash
# Format code
python -m black src/
python -m isort src/

# Lint code  
python -m flake8 src/

# Run quality checks
python -m black --check src/
python -m isort --check-only src/
python -m flake8 src/
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow tracking server URI |
| `API_PORT` | `8000` | FastAPI service port |
| `LOG_LEVEL` | `INFO` | Application logging level |
| `DEPLOY_MONITORING` | `false` | Deploy monitoring stack in CI/CD |
| `PROMETHEUS_PORT` | `9090` | Prometheus metrics port |
| `GRAFANA_PORT` | `3000` | Grafana dashboard port |

## Troubleshooting

### Common Issues
1. **Model Loading Failures**: Check MLflow registry and model artifacts
2. **Prometheus Connection**: Verify service names in docker-compose
3. **Grafana Datasource**: Ensure Prometheus URL is `http://prometheus:9090`
4. **Port Conflicts**: Check for existing services on ports 8000, 9090, 3000

### Debug Commands
```bash
# Check container logs
docker logs ml-api-local
docker logs monitoring_prometheus_1

# Verify model availability
curl http://localhost:8000/models/info

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```
