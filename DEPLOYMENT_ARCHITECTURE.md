# MLOps Assignment - Deployment Architecture Summary

## Project Overview
A production-ready ML prediction API with automated CI/CD, monitoring, and model management capabilities for California Housing and Iris classification models.

## Core Components

### 1. ML Prediction API (FastAPI)
- **Location**: `src/api.py`, `run_api.py`
- **Port**: 8000
- **Features**: 
  - Housing price predictions (`/predict/housing`)
  - Iris species classification (`/predict/iris`)
  - Model retraining endpoints (`/retrain`)
  - Health checks (`/health`)
  - Prometheus metrics (`/metrics/prometheus`)

### 2. Model Management (MLflow)
- **Purpose**: Model versioning, experiment tracking, and model registry
- **Storage**: Local `mlruns/` directory
- **Models**: CaliforniaHousingBestModel, IrisBestModel

### 3. Data Version Control (DVC)
- **Purpose**: Reproducible ML pipelines and data versioning
- **Pipeline**: 6-stage workflow (data prep, feature engineering, training, evaluation, validation)
- **Configuration**: `dvc.yaml`, `params.yaml`

### 4. Monitoring Stack (Prometheus + Grafana)
- **Prometheus**: Metrics collection on port 9090
- **Grafana**: Dashboard visualization on port 3000 (admin/admin)
- **Configuration**: `monitoring/docker-compose-monitoring.yml`

### 5. CI/CD Pipeline (GitHub Actions)
- **Trigger**: Push to main/develop branches
- **Stages**: Lint → Test → Build → Deploy → Verify
- **Quality Gates**: Black, isort, flake8, pytest with coverage

## Architecture Diagram

```
Client Request
     |
     v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   MLflow        │    │   Prometheus    │
│   (Port 8000)   │───▶│   Registry      │    │   (Port 9090)   │
│   - Predictions │    │   - Models      │    │   - Metrics     │
│   - Health      │    │   - Experiments │    │   - Monitoring  │
│   - Metrics     │    │   - Artifacts   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Logs/SQLite   │    │   Local Storage │    │   Grafana       │
│   - Predictions │    │   - mlruns/     │    │   (Port 3000)   │
│   - Errors      │    │   - models/     │    │   - Dashboards  │
│   - Metrics     │    │   - data/       │    │   - Alerts      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Container Architecture

### Docker Services
1. **ml-api**: Main API container
   - Built from `Dockerfile`
   - Volumes: `mlruns/`, `logs/`
   - Environment: MLflow tracking URI

2. **prometheus**: Metrics collection
   - Image: `prom/prometheus:latest`
   - Config: `monitoring/prometheus.yml`
   - Scrapes: `ml-api:8000/metrics/prometheus`

3. **grafana**: Dashboard visualization
   - Image: `grafana/grafana:latest`
   - Config: Auto-provisioned datasources and dashboards
   - Datasource: `http://prometheus:9090`

## Deployment Options

### Option 1: API Only
```bash
./scripts/deploy_local.sh
```
- Deploys FastAPI service
- Includes model training during build
- Health verification

### Option 2: Full Stack
```bash
docker-compose -f monitoring/docker-compose-monitoring.yml up -d
```
- Deploys API + Prometheus + Grafana
- Complete monitoring solution
- Automated service discovery

### Option 3: CI/CD Deployment
- Automated via GitHub Actions
- Triggered on push to main
- Includes testing and verification

## Key URLs and Endpoints

### API Endpoints
- **Base URL**: `http://localhost:8000`
- **Documentation**: `/docs` (Swagger UI)
- **Health Check**: `/health`
- **Model Info**: `/models/info`
- **Housing Prediction**: `POST /predict/housing`
- **Iris Prediction**: `POST /predict/iris`
- **Retraining**: `POST /retrain`
- **Metrics**: `/metrics/prometheus`

### Monitoring URLs
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin/admin)

## Data Flow

### Prediction Flow
1. Client sends POST request to `/predict/{model}`
2. API validates input using Pydantic schemas
3. Model performs inference
4. Prediction logged to SQLite database
5. Metrics updated in Prometheus
6. Response returned to client

### Monitoring Flow
1. Prometheus scrapes metrics from API every 10 seconds
2. Metrics stored in Prometheus time-series database
3. Grafana queries Prometheus for dashboard data
4. Real-time visualization of API performance

### Training Flow
1. DVC pipeline triggered via `dvc repro`
2. Data preprocessing and feature engineering
3. Model training with MLflow tracking
4. Model evaluation and comparison
5. Best model registered in MLflow registry
6. Deployment validation

## Configuration Management

### Environment Variables
- `MLFLOW_TRACKING_URI`: MLflow server location
- `API_PORT`: FastAPI service port (default: 8000)
- `LOG_LEVEL`: Application logging level
- `DEPLOY_MONITORING`: Enable monitoring in CI/CD

### Configuration Files
- `requirements.txt`: Python dependencies
- `config/pyproject.toml`: Code formatting settings
- `config/pytest.ini`: Test configuration
- `monitoring/prometheus.yml`: Metrics scraping config
- `dvc.yaml`: ML pipeline definition
- `params.yaml`: Pipeline parameters

## Security Considerations

### API Security
- Input validation with Pydantic
- Error handling without data exposure
- Health checks for service monitoring

### Container Security
- Non-root user in containers
- Minimal base images
- Environment variable management

## Troubleshooting Guide

### Common Issues
1. **Model Loading Failures**
   - Check MLflow registry: `mlflow ui`
   - Verify model artifacts in `mlruns/`
   - Run model training: `python src/train_housing.py`

2. **Prometheus Connection Issues**
   - Verify service names in docker-compose
   - Check API metrics endpoint: `curl localhost:8000/metrics/prometheus`
   - Restart monitoring stack

3. **Grafana Dashboard Issues**
   - Verify Prometheus datasource: `http://prometheus:9090`
   - Check Grafana logs: `docker logs <grafana_container>`
   - Re-provision dashboards

### Debug Commands
```bash
# Check API health
curl http://localhost:8000/health

# View container logs
docker logs ml-api-local

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Test model loading
python -c "from src.api import load_models; print(load_models())"
```

## Performance Optimization

### API Performance
- Model loading optimization with caching
- Async request handling
- Connection pooling for database operations

### Monitoring Performance
- Efficient metrics collection
- Optimized Prometheus queries
- Dashboard refresh intervals

## Maintenance Tasks

### Regular Maintenance
1. Monitor disk space for logs and model artifacts
2. Update dependencies in `requirements.txt`
3. Review and rotate API logs
4. Backup MLflow registry and experiments

### Model Maintenance
1. Monitor model performance metrics
2. Trigger retraining when performance degrades
3. Update model versions in production
4. Validate new models before deployment
