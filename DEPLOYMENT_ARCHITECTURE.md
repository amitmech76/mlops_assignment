# MLOps Assignment - Deployment Architecture Summary

## 🏗️ **Project Overview**
A production-ready ML prediction API with automated CI/CD, monitoring, and model management capabilities for California Housing and Iris classification models.

## 🔧 **Core Components**

### **1. ML Prediction API (FastAPI)**
- **Location**: `src/api.py`, `run_api.py`
- **Port**: 8000
- **Features**: 
  - Housing price predictions (`/predict/housing`)
  - Iris species classification (`/predict/iris`)
  - Model retraining endpoints (`/retrain`)
  - Health checks (`/health`)
  - Prometheus metrics (`/metrics/prometheus`)

### **2. Model Management (MLflow)**
- **Purpose**: Model versioning, experiment tracking, and model registry
- **Storage**: Local `mlruns/` directory
- **Models**: CaliforniaHousingBestModel, IrisBestModel
- **Features**: Automatic model registration and lifecycle management

### **3. Monitoring Stack**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML API        │    │   Prometheus    │    │    Grafana      │
│   Port: 8000    │───▶│   Port: 9090    │───▶│   Port: 3000    │
│                 │    │                 │    │                 │
│ /metrics/       │    │ Metrics         │    │ Dashboards &    │
│ prometheus      │    │ Collection      │    │ Visualization   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 **Deployment**

```bash
# Build and deploy entire stack
docker-compose -f monitoring/docker-compose-monitoring.yml up -d
```

## 🔄 **CI/CD Pipeline (GitHub Actions)**

```
┌──────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Code Push  │──▶│ Lint & Test │──▶│Build Docker │──▶│   Deploy    │
│   (main)     │   │             │   │   Image     │   │  to EC2     │
└──────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
                         │
                         ▼
                   ┌─────────────┐
                   │ Train Models│
                   │ Run Tests   │
                   │ (API Live)  │
                   └─────────────┘
```

**Pipeline Steps:**
1. **Linting**: flake8, black, isort code formatting
2. **Model Training**: Train housing & iris models for testing
3. **Testing**: 12 comprehensive tests with live API
4. **Docker Build**: Multi-stage build with optimization
5. **Deployment**: Automated deployment to local

## 📊 **Monitoring & Observability**

### **Metrics Collected**
- **API Metrics**: Request count, duration, errors by endpoint
- **Model Metrics**: Prediction count, accuracy, confidence scores
- **System Metrics**: Model load status, active requests
- **Business Metrics**: Custom KPIs and retraining events

### **Dashboards Available**
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **API Docs**: http://localhost:8000/docs

## 🗂️ **Data Management**

### **Data Pipeline**
```
Raw Data (CSV) ──▶ Processing Scripts ──▶ Trained Models ──▶ MLflow Registry
     │                    │                     │                │
data/housing_raw.csv  train_housing.py    housing_model    CaliforniaHousing
data/iris_raw.csv     train_iris.py       iris_model       BestModel
```

### **Storage Structure**
- **Data**: `data/` (raw and processed datasets)
- **Models**: `mlruns/` (MLflow artifacts and metadata)
- **Logs**: `logs/` (API logs and prediction database)
- **Config**: `config/pyproject.toml` (tool configurations)

## 🔐 **Security & Configuration**

### **Environment Variables**
- `MLFLOW_TRACKING_URI`: MLflow server location
- `DOCKERHUB_USERNAME`: Docker registry credentials
- `DOCKERHUB_TOKEN`: Docker registry access token

### **Health Checks**
- **API Health**: `/health` endpoint with model status
- **Docker Health**: Built-in container health checks
- **Monitoring**: Prometheus alerting (configurable)

## 🚦 **Key Features**

### **Production Ready**
✅ **Automated Testing**: 12 comprehensive test cases  
✅ **Code Quality**: Enforced linting and formatting  
✅ **Containerization**: Docker with multi-stage builds  
✅ **Monitoring**: Full observability stack  
✅ **Model Versioning**: MLflow integration  
✅ **Auto-scaling**: Container orchestration ready  

### **ML Operations**
✅ **Model Retraining**: Manual and automated triggers  
✅ **Data Drift Detection**: Built-in drift monitoring  
✅ **Performance Tracking**: Comprehensive metrics  
✅ **A/B Testing**: Model version comparison support  
✅ **Rollback**: Model version management  

### **Developer Experience**
✅ **Interactive API**: Swagger UI documentation  
✅ **Local Development**: Easy setup and debugging  
✅ **CI/CD Integration**: Automated deployment pipeline  
✅ **Monitoring Dashboards**: Real-time insights  

## 🌐 **Access Endpoints**

| Service | Local URL | Production URL | Purpose |
|---------|-----------|----------------|---------|
| ML API | http://localhost:8000 | http://{ec2-ip}:8000 | Predictions & Management |
| API Docs | http://localhost:8000/docs | http://{ec2-ip}:8000/docs | Interactive Documentation |
| Prometheus | http://localhost:9090 | http://{ec2-ip}:9090 | Metrics & Alerting |
| Grafana | http://localhost:3000 | http://{ec2-ip}:3000 | Dashboards & Visualization |

