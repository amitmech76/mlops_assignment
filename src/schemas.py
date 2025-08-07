from pydantic import BaseModel, Field, validator
from typing import Optional, Union, Dict, Any
import numpy as np

# Base prediction response schema
class PredictionResponse(BaseModel):
    prediction: Union[float, str]
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence score")
    model_info: Dict[str, Any]

# California Housing prediction schemas
class HousingPredictionRequest(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income in block group")
    HouseAge: float = Field(..., ge=0, le=100, description="Median house age in block group")
    AveRooms: float = Field(..., gt=0, le=20, description="Average number of rooms per household")
    AveBedrms: float = Field(..., gt=0, le=10, description="Average number of bedrooms per household")
    Population: float = Field(..., gt=0, le=10000, description="Block group population")
    AveOccup: float = Field(..., gt=0, le=50, description="Average number of household members")
    Latitude: float = Field(..., ge=32.0, le=42.0, description="Block group latitude")
    Longitude: float = Field(..., ge=-125.0, le=-114.0, description="Block group longitude")

    @validator('MedInc')
    def validate_medinc(cls, v):
        if v > 50:
            raise ValueError('Median income seems too high (>50)')
        return v

    @validator('AveRooms')
    def validate_ave_rooms(cls, v):
        if v < 1:
            raise ValueError('Average rooms must be at least 1')
        return v

    @validator('AveBedrms')
    def validate_ave_bedrms(cls, v):
        if v < 0.5:
            raise ValueError('Average bedrooms must be at least 0.5')
        return v

    @validator('Latitude', 'Longitude')
    def validate_coordinates(cls, v, values):
        # Check if coordinates are within California bounds
        if 'Latitude' in values and 'Longitude' in values:
            lat, lon = values['Latitude'], values['Longitude']
            if not (32 <= lat <= 42 and -125 <= lon <= -114):
                raise ValueError('Coordinates must be within California bounds')
        return v

    class Config:
        schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }

# Iris prediction schemas
class IrisPredictionRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, le=10, description="Petal width in cm")

    @validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    def validate_measurements(cls, v):
        if v <= 0:
            raise ValueError('All measurements must be positive')
        if v > 10:
            raise ValueError('All measurements must be reasonable (< 10cm)')
        return v

    @validator('sepal_length')
    def validate_sepal_length(cls, v, values):
        if 'sepal_width' in values:
            if v < values['sepal_width']:
                raise ValueError('Sepal length should typically be greater than sepal width')
        return v

    @validator('petal_length')
    def validate_petal_length(cls, v, values):
        if 'petal_width' in values:
            if v < values['petal_width']:
                raise ValueError('Petal length should typically be greater than petal width')
        return v

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

# Model retraining schemas
class RetrainingRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to retrain")
    trigger_type: str = Field(..., description="Type of retraining trigger")
    data_path: Optional[str] = Field(None, description="Path to new training data")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters for retraining")

    @validator('model_name')
    def validate_model_name(cls, v):
        valid_models = ['housing', 'iris', 'CaliforniaHousingBestModel', 'IrisBestModel']
        if v not in valid_models:
            raise ValueError(f'Model name must be one of: {valid_models}')
        return v

    @validator('trigger_type')
    def validate_trigger_type(cls, v):
        valid_triggers = ['manual', 'scheduled', 'performance_drop', 'data_drift']
        if v not in valid_triggers:
            raise ValueError(f'Trigger type must be one of: {valid_triggers}')
        return v

class RetrainingResponse(BaseModel):
    status: str = Field(..., description="Retraining status")
    model_name: str = Field(..., description="Name of the model being retrained")
    job_id: Optional[str] = Field(None, description="Retraining job ID")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")
    message: str = Field(..., description="Status message")

# Metrics and monitoring schemas
class MetricsRequest(BaseModel):
    metric_name: str = Field(..., description="Name of the metric to collect")
    value: float = Field(..., description="Metric value")
    labels: Optional[Dict[str, str]] = Field(None, description="Metric labels")

class ModelPerformanceMetrics(BaseModel):
    model_name: str
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    precision: Optional[float] = Field(None, ge=0, le=1)
    recall: Optional[float] = Field(None, ge=0, le=1)
    f1_score: Optional[float] = Field(None, ge=0, le=1)
    rmse: Optional[float] = Field(None, ge=0)
    mae: Optional[float] = Field(None, ge=0)
    timestamp: str

# Data drift detection schemas
class DataDriftRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to check for drift")
    reference_data_path: str = Field(..., description="Path to reference data")
    current_data_path: str = Field(..., description="Path to current data")
    threshold: float = Field(0.05, ge=0, le=1, description="Drift detection threshold")

class DataDriftResponse(BaseModel):
    model_name: str
    drift_detected: bool
    drift_score: float
    features_with_drift: Optional[list] = None
    recommendation: str

# Health check schemas
class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    database_connection: bool = Field(..., description="Database connection status")
    prometheus_connection: bool = Field(..., description="Prometheus connection status")
    last_updated: str = Field(..., description="Last health check timestamp")

# API error response schema
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking") 