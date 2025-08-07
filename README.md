# ML Prediction API with Monitoring

A comprehensive Machine Learning prediction API built with FastAPI, featuring monitoring, logging, model retraining, and data drift detection.

## Project Structure

```
ml-prediction-api/
├── src/                    # Core application code
│   ├── api.py             # FastAPI application
│   ├── logger.py          # Logging functionality
│   ├── schemas.py         # Pydantic models
│   ├── prometheus_metrics.py  # Prometheus metrics
│   ├── model_retraining.py    # Model retraining logic
│   ├── train_*.py         # Model training scripts
│   └── data_load_*.py     # Data loading scripts
├── tests/                 # Test files
│   ├── test_api.py        # API tests
│   ├── test_advanced_features.py  # Advanced feature tests
│   └── test_logging.py    # Logging tests
├── scripts/               # Deployment scripts
│   ├── deploy_local.sh    # Local deployment
│   └── deploy_ec2.sh      # EC2 deployment
├── monitoring/            # Monitoring stack
│   ├── monitor.py         # Real-time monitoring
│   ├── prometheus.yml     # Prometheus config
│   ├── grafana-dashboard.json  # Grafana dashboard
│   └── docker-compose-monitoring.yml  # Monitoring stack
├── config/                # Configuration files
│   ├── .flake8           # Linting config
│   ├── pytest.ini        # Test config
│   └── pyproject.toml    # Code formatting config
├── .github/              # GitHub Actions workflows
├── logs/                 # Application logs
├── data/                 # Data files
└── mlruns/              # MLflow runs
```

### 1. Train Models
```bash
python src/train_housing.py
python src/train_iris.py
```

### 2. Start the Complete Stack
```bash
docker-compose -f monitoring/docker-compose-monitoring.yml up --build
```

### 3. Access Services
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Documentation

- **[How to Run the Pipeline](docs/RUN_PIPELINE.md)** - Complete setup and usage guide
- **[API Documentation](docs/README_API.md)** - Detailed API reference
- **[CI/CD Guide](docs/README_CI_CD.md)** - Deployment and automation

## Testing

```bash
# Run all tests
pytest tests/

# Test specific components
python tests/test_api.py
python tests/test_advanced_features.py
python tests/test_logging.py
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python run_api.py

# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
```

## Monitoring

```bash
# Start monitoring dashboard
python monitoring/monitor.py

# View logs
tail -f logs/api.log
```

## Deployment

### Local
```bash
./scripts/deploy_local.sh
```

### EC2
```bash
./scripts/deploy_ec2.sh
```

## Features

- **FastAPI Prediction API** with two models (Housing & Iris)
- **Comprehensive Input Validation** using Pydantic
- **SQLite Logging** of all prediction requests
- **Prometheus Metrics** for monitoring
- **Grafana Dashboard** with pre-configured panels
- **Model Retraining** with background jobs
- **Data Drift Detection** using statistical tests
- **CI/CD Pipeline** with GitHub Actions
- **Docker Containerization** for easy deployment
