# Project Structure

This document outlines the structure of the ML Prediction API project.

## Directory Layout

```
ml-prediction-api/
├── README.md                      # Main project overview
├── Makefile                       # Development and deployment commands
├── run_api.py                     # Local API runner
├── requirements.txt               # dependencies
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Basic Docker Compose setup
├── .dockerignore                  # Docker ignore patterns
│
├── src/                           # Core application code
│   ├── api.py                     # FastAPI application
│   ├── logger.py                  # Logging functionality
│   ├── schemas.py                 # Pydantic models
│   ├── prometheus_metrics.py      # Prometheus metrics
│   ├── model_retraining.py        # Model retraining logic
│   ├── train_housing.py           # Housing model training
│   ├── train_iris.py              # Iris model training
│   ├── data_load_housing.py       # Housing data loader
│   └── data_load_iris.py          # Iris data loader
│
├── tests/                         # Test files
│   ├── __init__.py                # Test package init
│   ├── test_api.py                # API endpoint tests
│   ├── test_advanced_features.py  # Advanced feature tests
│   └── test_logging.py            # Logging functionality tests
│
├── scripts/                       # Deployment scripts
│   ├── deploy_local.sh            # Local deployment script
│   └── deploy_ec2.sh              # EC2 deployment script
│
├── monitoring/                    # Monitoring stack
│   ├── monitor.py                 # Real-time monitoring dashboard
│   ├── prometheus.yml             # Prometheus configuration
│   ├── grafana-dashboard.json     # Grafana dashboard definition
│   └── docker-compose-monitoring.yml # Monitoring stack setup
│
├── config/                       # Configuration files
│   ├── .flake8                   # Linting configuration
│   ├── pytest.ini                # Test configuration
│   └── pyproject.toml            # Code formatting configuration
│
├── docs/                         # Documentation
│   ├── README_API.md             # API documentation
│   ├── README_CI_CD.md           # CI/CD documentation
│   └── RUN_PIPELINE.md           # How to run the pipeline
│
├── .github/                      # GitHub Actions workflows
│   └── workflows/
│       ├── ci-cd.yml             # Main CI/CD pipeline
│       └── pr-check.yml          # Pull request checks
│
├── logs/                         # Application logs
│   ├── api.log                   # API application logs
│   └── predictions.db            # SQLite prediction logs
│
├── data/                         # Data files
│   ├── housing_raw.csv           # Raw housing data
│   ├── housing_processed.csv     # Processed housing data
│   ├── iris_raw.csv              # Raw iris data
│   └── iris_processed.csv        # Processed iris data
│
└── mlruns/                       # MLflow experiment tracking
    └── [experiment_id]/          # Experiment runs
	│
	└── models/                   # Model artifacts
		└── [model files]         # Saved model files
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test
python tests/test_api.py
```

### Running Monitoring
```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose-monitoring.yml up --build

# Run monitoring dashboard
python monitoring/monitor.py
```

### Development Commands
```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test
```
