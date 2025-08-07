import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import threading
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import logging
from prometheus_metrics import metrics_collector

logger = logging.getLogger(__name__)

class ModelRetrainer:
    """Handles model retraining with various triggers"""
    
    def __init__(self):
        self.retraining_jobs = {}
        self.performance_history = {}
        self.drift_thresholds = {
            'housing': 0.1,  # 10% performance drop
            'iris': 0.05     # 5% accuracy drop
        }
        self.last_retraining = {}
    
    def trigger_retraining(self, model_name: str, trigger_type: str, 
                          data_path: Optional[str] = None, 
                          parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Trigger model retraining"""
        job_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start retraining in background thread
        thread = threading.Thread(
            target=self._retrain_model,
            args=(job_id, model_name, trigger_type, data_path, parameters)
        )
        thread.daemon = True
        thread.start()
        
        # Record retraining metrics
        metrics_collector.record_retraining(model_name, trigger_type, 0)
        
        return {
            "status": "started",
            "job_id": job_id,
            "model_name": model_name,
            "trigger_type": trigger_type,
            "estimated_duration": self._get_estimated_duration(model_name),
            "message": f"Retraining job {job_id} started for {model_name}"
        }
    
    def _retrain_model(self, job_id: str, model_name: str, trigger_type: str,
                       data_path: Optional[str], parameters: Optional[Dict[str, Any]]):
        """Retrain model in background"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting retraining for {model_name} (Job: {job_id})")
            
            # Update job status
            self.retraining_jobs[job_id] = {
                "status": "running",
                "start_time": start_time,
                "model_name": model_name,
                "trigger_type": trigger_type
            }
            
            # Load and prepare data
            if model_name == 'housing':
                success = self._retrain_housing_model(job_id, data_path, parameters)
            elif model_name == 'iris':
                success = self._retrain_iris_model(job_id, data_path, parameters)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Update job status
            duration = time.time() - start_time
            self.retraining_jobs[job_id].update({
                "status": "completed" if success else "failed",
                "duration": duration,
                "end_time": time.time()
            })
            
            # Update last retraining time
            self.last_retraining[model_name] = datetime.now()
            
            # Record metrics
            metrics_collector.record_retraining(model_name, trigger_type, duration)
            
            logger.info(f"Retraining completed for {model_name} (Job: {job_id})")
            
        except Exception as e:
            logger.error(f"Retraining failed for {model_name} (Job: {job_id}): {e}")
            self.retraining_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "duration": time.time() - start_time,
                "end_time": time.time()
            })
    
    def _retrain_housing_model(self, job_id: str, data_path: Optional[str], 
                              parameters: Optional[Dict[str, Any]]) -> bool:
        """Retrain California Housing model"""
        try:
            # Load data
            if data_path and os.path.exists(data_path):
                df = pd.read_csv(data_path)
            else:
                from sklearn.datasets import fetch_california_housing
                data = fetch_california_housing(as_frame=True)
                df = data.frame
            
            # Prepare features and target
            X = df.drop('MedHouseVal', axis=1)
            y = df['MedHouseVal']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train models
            models = {
                'LinearRegression': LinearRegression(),
                'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = float('-inf')
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            # Log model with MLflow
            mlflow.set_experiment('CaliforniaHousing-Regression')
            with mlflow.start_run(run_name=f'Retraining-{job_id}'):
                mlflow.log_param('retraining_job_id', job_id)
                mlflow.log_param('trigger_type', self.retraining_jobs[job_id]['trigger_type'])
                mlflow.log_metric('r2_score', best_score)
                mlflow.log_metric('rmse', np.sqrt(mean_squared_error(y_test, best_model.predict(X_test))))
                
                # Log the model
                mlflow.sklearn.log_model(best_model, artifact_path='model')
                
                # Register new version
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                mlflow.register_model(model_uri, "CaliforniaHousingBestModel")
            
            # Update performance history
            self.performance_history[model_name] = {
                'r2_score': best_score,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error retraining housing model: {e}")
            return False
    
    def _retrain_iris_model(self, job_id: str, data_path: Optional[str], 
                           parameters: Optional[Dict[str, Any]]) -> bool:
        """Retrain Iris model"""
        try:
            # Load data
            if data_path and os.path.exists(data_path):
                df = pd.read_csv(data_path)
            else:
                from sklearn.datasets import load_iris
                data = load_iris(as_frame=True)
                df = data.frame
            
            # Prepare features and target
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train models
            models = {
                'LogisticRegression': LogisticRegression(max_iter=200),
                'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            # Log model with MLflow
            mlflow.set_experiment('Iris-Classification')
            with mlflow.start_run(run_name=f'Retraining-{job_id}'):
                mlflow.log_param('retraining_job_id', job_id)
                mlflow.log_param('trigger_type', self.retraining_jobs[job_id]['trigger_type'])
                mlflow.log_metric('accuracy', best_score)
                
                # Log the model
                mlflow.sklearn.log_model(best_model, artifact_path='model')
                
                # Register new version
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                mlflow.register_model(model_uri, "IrisBestModel")
            
            # Update performance history
            self.performance_history['iris'] = {
                'accuracy': best_score,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error retraining iris model: {e}")
            return False
    
    def check_performance_drop(self, model_name: str, current_performance: float) -> bool:
        """Check if performance has dropped significantly"""
        if model_name not in self.performance_history:
            return False
        
        historical_performance = self.performance_history[model_name]
        if 'r2_score' in historical_performance:
            historical_score = historical_performance['r2_score']
            threshold = self.drift_thresholds.get(model_name, 0.1)
            return (historical_score - current_performance) > threshold
        elif 'accuracy' in historical_performance:
            historical_score = historical_performance['accuracy']
            threshold = self.drift_thresholds.get(model_name, 0.05)
            return (historical_score - current_performance) > threshold
        
        return False
    
    def check_data_drift(self, model_name: str, reference_data: pd.DataFrame, 
                        current_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift between reference and current data"""
        try:
            # Simple drift detection using statistical tests
            drift_scores = {}
            features = reference_data.columns
            
            for feature in features:
                if feature in current_data.columns:
                    # Kolmogorov-Smirnov test for distribution difference
                    from scipy.stats import ks_2samp
                    stat, p_value = ks_2samp(
                        reference_data[feature].dropna(),
                        current_data[feature].dropna()
                    )
                    drift_scores[feature] = p_value
            
            # Calculate overall drift score
            avg_drift_score = np.mean(list(drift_scores.values()))
            drift_detected = avg_drift_score < 0.05  # 5% significance level
            
            # Record drift metrics
            metrics_collector.record_data_drift(model_name, avg_drift_score, drift_detected)
            
            return {
                "drift_detected": drift_detected,
                "drift_score": avg_drift_score,
                "features_with_drift": [f for f, p in drift_scores.items() if p < 0.05],
                "recommendation": "Retrain model" if drift_detected else "No action needed"
            }
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "features_with_drift": [],
                "recommendation": "Error in drift detection"
            }
    
    def get_retraining_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a retraining job"""
        if job_id not in self.retraining_jobs:
            return {"error": "Job not found"}
        
        job = self.retraining_jobs[job_id]
        return {
            "job_id": job_id,
            "status": job["status"],
            "model_name": job["model_name"],
            "trigger_type": job["trigger_type"],
            "start_time": job["start_time"],
            "duration": job.get("duration", 0),
            "error": job.get("error", None)
        }
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all retraining jobs"""
        return [
            {
                "job_id": job_id,
                **job_info
            }
            for job_id, job_info in self.retraining_jobs.items()
        ]
    
    def _get_estimated_duration(self, model_name: str) -> int:
        """Get estimated retraining duration in minutes"""
        estimates = {
            'housing': 5,   # 5 minutes
            'iris': 3       # 3 minutes
        }
        return estimates.get(model_name, 10)


# Global retrainer instance
model_retrainer = ModelRetrainer() 