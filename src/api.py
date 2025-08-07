from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import List, Union, Dict, Any, Optional
import os
from datetime import datetime
import uuid
import time
from logger import log_prediction_request, prediction_logger
from schemas import (
    HousingPredictionRequest,
    IrisPredictionRequest,
    PredictionResponse,
    RetrainingRequest,
    RetrainingResponse,
    DataDriftRequest,
    DataDriftResponse,
    HealthCheckResponse,
    ErrorResponse,
)
from prometheus_metrics import (
    metrics_collector,
    get_prometheus_metrics,
    get_metrics_content_type,
)
from model_retraining import model_retrainer


# Global variables to store loaded models
housing_model = None
iris_model = None

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    metrics_collector.set_model_load_status("housing", housing_model is not None)
    metrics_collector.set_model_load_status("iris", iris_model is not None)
    yield


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="ML Model Prediction API",
    description="API for making predictions using trained ML models",
    version="1.0.0",
    lifespan=lifespan,
)


def load_models():
    """Load the trained models from MLflow"""
    global housing_model, iris_model

    try:
        # Load California Housing model
        housing_model = mlflow.pyfunc.load_model(
            "models:/CaliforniaHousingBestModel/latest"
        )
        print("California Housing model loaded successfully")
    except Exception as e:
        print(f"Error loading California Housing model: {e}")
        housing_model = None

    try:
        # Load Iris model
        iris_model = mlflow.pyfunc.load_model("models:/IrisBestModel/latest")
        print("Iris model loaded successfully")
    except Exception as e:
        print(f"Error loading Iris model: {e}")
        iris_model = None


# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Start request timer
    metrics_collector.start_request_timer(request.url.path, request_id)

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as e:
        status = 500
        response = Response(content=str(e), status_code=500)

    # Record request metrics
    duration = time.time() - start_time
    metrics_collector.record_request(
        endpoint=request.url.path,
        method=request.method,
        status=str(status),
        duration=duration,
    )

    # End request timer
    metrics_collector.end_request_timer(request_id)

    return response


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ML Model Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict/housing": "California Housing price prediction",
            "/predict/iris": "Iris flower classification",
            "/health": "Health check endpoint",
            "/health/advanced": "Advanced health check",
            "/models/info": "Model information",
            "/metrics": "Metrics and statistics",
            "/metrics/prometheus": "Prometheus metrics",
            "/logs/recent": "Recent prediction logs",
            "/logs/stats": "Prediction statistics",
            "/retrain": "Trigger model retraining",
            "/retrain/status/{job_id}": "Get retraining job status",
            "/retrain/jobs": "Get all retraining jobs",
            "/drift/detect": "Detect data drift",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "housing_model": housing_model is not None,
            "iris_model": iris_model is not None,
        },
    }


@app.post("/predict/housing", response_model=PredictionResponse)
async def predict_housing(request: HousingPredictionRequest):
    """Predict California Housing prices"""
    if housing_model is None:
        metrics_collector.record_error("/predict/housing", "model_not_loaded")
        raise HTTPException(status_code=500, detail="Housing model not loaded")

    start_time = time.time()

    # Log the prediction request
    with log_prediction_request("housing", request.dict()) as response_data:
        try:
            # Convert request to DataFrame
            input_data = pd.DataFrame([request.dict()])

            # Make prediction
            prediction = housing_model.predict(input_data)[0]

            response = PredictionResponse(
                prediction=float(prediction),
                confidence=None,
                model_info={
                    "model_type": "California Housing Regression",
                    "features": list(request.dict().keys()),
                },
            )

            # Update response data for logging
            response_data.update(response.dict())

            # Record prediction metrics
            duration = time.time() - start_time
            metrics_collector.record_prediction("housing", "success", duration)

            return response

        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            duration = time.time() - start_time
            metrics_collector.record_prediction("housing", "error", duration)
            metrics_collector.record_error("/predict/housing", "prediction_error")
            raise HTTPException(status_code=500, detail=error_msg)


@app.post("/predict/iris", response_model=PredictionResponse)
async def predict_iris(request: IrisPredictionRequest):
    """Predict Iris flower species"""
    if iris_model is None:
        metrics_collector.record_error("/predict/iris", "model_not_loaded")
        raise HTTPException(status_code=500, detail="Iris model not loaded")

    start_time = time.time()

    # Log the prediction request
    with log_prediction_request("iris", request.dict()) as response_data:
        try:
            # Convert request to DataFrame
            input_data = pd.DataFrame([request.dict()])

            # Make prediction
            prediction = iris_model.predict(input_data)[0]
            confidence = None  # Not available for pyfunc models

            # Map prediction to species name (if model returns int)
            species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
            # If prediction is int, map to species; else, return as is
            try:
                species = species_map.get(int(prediction), f"class_{prediction}")
            except Exception:
                species = str(prediction)

            response = PredictionResponse(
                prediction=species,
                confidence=confidence,
                model_info={
                    "model_type": "Iris Classification",
                    "features": list(request.dict().keys()),
                    "classes": list(species_map.values()),
                },
            )

            # Update response data for logging
            response_data.update(response.dict())

            # Record prediction metrics
            duration = time.time() - start_time
            metrics_collector.record_prediction("iris", "success", duration, confidence)

            return response

        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            duration = time.time() - start_time
            metrics_collector.record_prediction("iris", "error", duration)
            metrics_collector.record_error("/predict/iris", "prediction_error")
            raise HTTPException(status_code=500, detail=error_msg)


@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    return {
        "housing_model": {
            "loaded": housing_model is not None,
            "type": "California Housing Regression" if housing_model else None,
        },
        "iris_model": {
            "loaded": iris_model is not None,
            "type": "Iris Classification" if iris_model else None,
        },
    }


@app.get("/metrics")
async def get_metrics():
    """Get metrics and statistics"""
    try:
        # Get overall stats
        overall_stats = prediction_logger.get_prediction_stats()

        # Get model-specific stats
        housing_stats = prediction_logger.get_prediction_stats("housing")
        iris_stats = prediction_logger.get_prediction_stats("iris")

        # Get recent predictions
        recent_predictions = prediction_logger.get_recent_predictions(limit=5)

        # Get metrics data
        metrics_data = prediction_logger.get_metrics()

        return {
            "overall_stats": overall_stats,
            "model_stats": {"housing": housing_stats, "iris": iris_stats},
            "recent_predictions": recent_predictions,
            "metrics": metrics_data,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")


@app.get("/logs/recent")
async def get_recent_logs(limit: int = 10):
    """Get recent prediction logs"""
    try:
        recent_predictions = prediction_logger.get_recent_predictions(limit=limit)
        return {
            "recent_predictions": recent_predictions,
            "count": len(recent_predictions),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting logs: {str(e)}")


@app.get("/logs/stats")
async def get_log_stats(hours: int = 24, model_name: Optional[str] = None):
    """Get prediction statistics for a time period"""
    try:
        stats = prediction_logger.get_prediction_stats(
            model_name=model_name, hours=hours
        )
        return {
            "stats": stats,
            "time_window_hours": hours,
            "model_name": model_name or "all",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Get Prometheus metrics"""
    return Response(
        content=get_prometheus_metrics(), media_type=get_metrics_content_type()
    )


@app.post("/retrain", response_model=RetrainingResponse)
async def trigger_retraining(request: RetrainingRequest):
    """Trigger model retraining"""
    try:
        result = model_retrainer.trigger_retraining(
            model_name=request.model_name,
            trigger_type=request.trigger_type,
            data_path=request.data_path,
            parameters=request.parameters,
        )
        return RetrainingResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error triggering retraining: {str(e)}"
        )


@app.get("/retrain/status/{job_id}")
async def get_retraining_status(job_id: str):
    """Get retraining job status"""
    try:
        status = model_retrainer.get_retraining_status(job_id)
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        return status
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting retraining status: {str(e)}"
        )


@app.get("/retrain/jobs")
async def get_all_retraining_jobs():
    """Get all retraining jobs"""
    try:
        return {"jobs": model_retrainer.get_all_jobs()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting retraining jobs: {str(e)}"
        )


@app.post("/drift/detect", response_model=DataDriftResponse)
async def detect_data_drift(request: DataDriftRequest):
    """Detect data drift"""
    try:
        # Load reference and current data
        reference_data = pd.read_csv(request.reference_data_path)
        current_data = pd.read_csv(request.current_data_path)

        # Check for drift
        drift_result = model_retrainer.check_data_drift(
            model_name=request.model_name,
            reference_data=reference_data,
            current_data=current_data,
        )

        return DataDriftResponse(
            model_name=request.model_name,
            drift_detected=drift_result["drift_detected"],
            drift_score=drift_result["drift_score"],
            features_with_drift=drift_result["features_with_drift"],
            recommendation=drift_result["recommendation"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error detecting data drift: {str(e)}"
        )


@app.get("/health/advanced", response_model=HealthCheckResponse)
async def advanced_health_check():
    """Advanced health check with detailed status"""
    try:
        # Check database connection
        db_status = True
        try:
            prediction_logger.get_prediction_stats()
        except:
            db_status = False

        # Check Prometheus connection
        prometheus_status = True
        try:
            get_prometheus_metrics()
        except:
            prometheus_status = False

        return HealthCheckResponse(
            status=(
                "healthy"
                if all([housing_model is not None, iris_model is not None, db_status])
                else "unhealthy"
            ),
            models_loaded={
                "housing_model": housing_model is not None,
                "iris_model": iris_model is not None,
            },
            database_connection=db_status,
            prometheus_connection=prometheus_status,
            last_updated=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in health check: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
