import threading
import time
from typing import Any, Dict, Optional

from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               Summary, generate_latest)
from prometheus_client.core import CollectorRegistry

# Create a custom registry for our metrics
registry = CollectorRegistry()

# API Metrics
REQUEST_COUNT = Counter(
    "ml_api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status"],
    registry=registry,
)

REQUEST_DURATION = Histogram(
    "ml_api_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "method"],
    registry=registry,
)

# Model-specific Metrics
PREDICTION_COUNT = Counter(
    "ml_model_predictions_total",
    "Total number of predictions made",
    ["model_name", "status"],
    registry=registry,
)

PREDICTION_DURATION = Histogram(
    "ml_model_prediction_duration_seconds",
    "Prediction duration in seconds",
    ["model_name"],
    registry=registry,
)

MODEL_ACCURACY = Gauge(
    "ml_model_accuracy", "Model accuracy score", ["model_name"], registry=registry
)

MODEL_CONFIDENCE = Histogram(
    "ml_model_confidence",
    "Model prediction confidence scores",
    ["model_name"],
    registry=registry,
)

# System Metrics
MODEL_LOAD_STATUS = Gauge(
    "ml_model_loaded",
    "Model loading status (1=loaded, 0=not loaded)",
    ["model_name"],
    registry=registry,
)

ACTIVE_REQUESTS = Gauge(
    "ml_api_active_requests",
    "Number of active requests",
    ["endpoint"],
    registry=registry,
)

# Error Metrics
ERROR_COUNT = Counter(
    "ml_api_errors_total",
    "Total number of errors",
    ["endpoint", "error_type"],
    registry=registry,
)

# Retraining Metrics
RETRAINING_COUNT = Counter(
    "ml_model_retraining_total",
    "Total number of model retraining events",
    ["model_name", "trigger_type"],
    registry=registry,
)

RETRAINING_DURATION = Histogram(
    "ml_model_retraining_duration_seconds",
    "Model retraining duration in seconds",
    ["model_name"],
    registry=registry,
)

# Data Drift Metrics
DATA_DRIFT_SCORE = Gauge(
    "ml_data_drift_score",
    "Data drift detection score",
    ["model_name"],
    registry=registry,
)

DRIFT_DETECTED = Gauge(
    "ml_data_drift_detected",
    "Data drift detection status (1=detected, 0=not detected)",
    ["model_name"],
    registry=registry,
)

# Performance Metrics
MODEL_PERFORMANCE = Gauge(
    "ml_model_performance_metric",
    "Model performance metrics",
    ["model_name", "metric_name"],
    registry=registry,
)

# Custom metrics for business logic
BUSINESS_METRICS = Counter(
    "ml_business_metrics_total",
    "Business-specific metrics",
    ["metric_name", "model_name"],
    registry=registry,
)


class MetricsCollector:
    """Collector for ML model metrics"""

    def __init__(self):
        self.request_times = {}
        self.active_requests = {}

    def record_request(self, endpoint: str, method: str, status: str, duration: float):
        """Record API request metrics"""
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
        REQUEST_DURATION.labels(endpoint=endpoint, method=method).observe(duration)

    def record_prediction(
        self,
        model_name: str,
        status: str,
        duration: float,
        confidence: Optional[float] = None,
    ):
        """Record model prediction metrics"""
        PREDICTION_COUNT.labels(model_name=model_name, status=status).inc()
        PREDICTION_DURATION.labels(model_name=model_name).observe(duration)

        if confidence is not None:
            MODEL_CONFIDENCE.labels(model_name=model_name).observe(confidence)

    def record_model_accuracy(self, model_name: str, accuracy: float):
        """Record model accuracy"""
        MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)

    def record_model_performance(self, model_name: str, metric_name: str, value: float):
        """Record model performance metrics"""
        MODEL_PERFORMANCE.labels(model_name=model_name, metric_name=metric_name).set(
            value
        )

    def set_model_load_status(self, model_name: str, loaded: bool):
        """Set model loading status"""
        status_value = 1 if loaded else 0
        MODEL_LOAD_STATUS.labels(model_name=model_name).set(status_value)

    def record_error(self, endpoint: str, error_type: str):
        """Record error metrics"""
        ERROR_COUNT.labels(endpoint=endpoint, error_type=error_type).inc()

    def record_retraining(self, model_name: str, trigger_type: str, duration: float):
        """Record model retraining metrics"""
        RETRAINING_COUNT.labels(model_name=model_name, trigger_type=trigger_type).inc()
        RETRAINING_DURATION.labels(model_name=model_name).observe(duration)

    def record_data_drift(self, model_name: str, drift_score: float, detected: bool):
        """Record data drift metrics"""
        DATA_DRIFT_SCORE.labels(model_name=model_name).set(drift_score)
        drift_status = 1 if detected else 0
        DRIFT_DETECTED.labels(model_name=model_name).set(drift_status)

    def record_business_metric(
        self, metric_name: str, model_name: str, value: float = 1
    ):
        """Record business-specific metrics"""
        BUSINESS_METRICS.labels(metric_name=metric_name, model_name=model_name).inc(
            value
        )

    def start_request_timer(self, endpoint: str, request_id: str):
        """Start timing a request"""
        self.request_times[request_id] = time.time()
        self.active_requests[request_id] = endpoint
        ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()

    def end_request_timer(self, request_id: str):
        """End timing a request"""
        if request_id in self.request_times:
            duration = time.time() - self.request_times[request_id]
            endpoint = self.active_requests.get(request_id, "unknown")
            ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()
            del self.request_times[request_id]
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            return duration
        return 0.0


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_prometheus_metrics():
    """Get Prometheus metrics"""
    return generate_latest(registry)


def get_metrics_content_type():
    """Get content type for metrics response"""
    return CONTENT_TYPE_LATEST


# Context manager for timing operations
class MetricsTimer:
    """Context manager for timing operations and recording metrics"""

    def __init__(self, operation_type: str, model_name: str = None):
        self.operation_type = operation_type
        self.model_name = model_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if self.operation_type == "prediction":
            status = "success" if exc_type is None else "error"
            metrics_collector.record_prediction(
                model_name=self.model_name, status=status, duration=duration
            )
        elif self.operation_type == "retraining":
            metrics_collector.record_retraining(
                model_name=self.model_name, trigger_type="manual", duration=duration
            )


# Decorator for automatic metrics collection
def collect_metrics(operation_type: str, model_name: str = None):
    """Decorator to automatically collect metrics for functions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with MetricsTimer(operation_type, model_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
