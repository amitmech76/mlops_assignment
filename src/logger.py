import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Logger for prediction requests and responses"""

    def __init__(self, db_path: str = "logs/predictions.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with required tables"""
        Path(self.db_path).parent.mkdir(exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create predictions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    request_data TEXT NOT NULL,
                    response_data TEXT NOT NULL,
                    prediction_value TEXT NOT NULL,
                    confidence REAL,
                    processing_time_ms REAL,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
            """
            )

            # Create metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    model_name TEXT
                )
            """
            )

            conn.commit()

    def log_prediction(
        self,
        model_name: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        processing_time_ms: float,
        status: str = "success",
        error_message: Optional[str] = None,
    ):
        """Log a prediction request and response"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO predictions 
                    (timestamp, model_name, request_data, response_data, 
                     prediction_value, confidence, processing_time_ms, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        datetime.now().isoformat(),
                        model_name,
                        json.dumps(request_data),
                        json.dumps(response_data),
                        str(response_data.get("prediction", "")),
                        response_data.get("confidence"),
                        processing_time_ms,
                        status,
                        error_message,
                    ),
                )

                conn.commit()

                # Log to file as well
                logger.info(
                    f"Prediction logged - Model: {model_name}, Status: {status}, Time: {processing_time_ms:.2f}ms"
                )

        except Exception as e:
            logger.error(f"Error logging prediction: {e}")

    def log_metric(
        self, metric_name: str, metric_value: float, model_name: Optional[str] = None
    ):
        """Log a metric value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO metrics (timestamp, metric_name, metric_value, model_name)
                    VALUES (?, ?, ?, ?)
                """,
                    (datetime.now().isoformat(), metric_name, metric_value, model_name),
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Error logging metric: {e}")

    def get_prediction_stats(
        self, model_name: Optional[str] = None, hours: int = 24
    ) -> Dict[str, Any]:
        """Get prediction statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Calculate time threshold
                time_threshold = (
                    datetime.now().replace(hour=datetime.now().hour - hours).isoformat()
                )

                where_clause = "WHERE timestamp > ?"
                params = [time_threshold]

                if model_name:
                    where_clause += " AND model_name = ?"
                    params.append(model_name)

                # Total predictions
                cursor.execute(
                    f"""
                    SELECT COUNT(*) FROM predictions {where_clause}
                """,
                    params,
                )
                total_predictions = cursor.fetchone()[0]

                # Successful predictions
                cursor.execute(
                    f"""
                    SELECT COUNT(*) FROM predictions {where_clause} AND status = 'success'
                """,
                    params,
                )
                successful_predictions = cursor.fetchone()[0]

                # Average processing time
                cursor.execute(
                    f"""
                    SELECT AVG(processing_time_ms) FROM predictions {where_clause} AND status = 'success'
                """,
                    params,
                )
                avg_processing_time = cursor.fetchone()[0] or 0

                # Error rate
                error_rate = 0
                if total_predictions > 0:
                    error_rate = (
                        (total_predictions - successful_predictions) / total_predictions
                    ) * 100

                return {
                    "total_predictions": total_predictions,
                    "successful_predictions": successful_predictions,
                    "error_rate_percent": round(error_rate, 2),
                    "avg_processing_time_ms": round(avg_processing_time, 2),
                    "time_window_hours": hours,
                }

        except Exception as e:
            logger.error(f"Error getting prediction stats: {e}")
            return {}

    def get_recent_predictions(self, limit: int = 10) -> list:
        """Get recent predictions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT timestamp, model_name, prediction_value, confidence, 
                           processing_time_ms, status
                    FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (limit,),
                )

                return [
                    {
                        "timestamp": row[0],
                        "model_name": row[1],
                        "prediction": row[2],
                        "confidence": row[3],
                        "processing_time_ms": row[4],
                        "status": row[5],
                    }
                    for row in cursor.fetchall()
                ]

        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return []

    def get_metrics(
        self, metric_name: Optional[str] = None, hours: int = 24
    ) -> Dict[str, Any]:
        """Get metrics data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                time_threshold = (
                    datetime.now().replace(hour=datetime.now().hour - hours).isoformat()
                )

                where_clause = "WHERE timestamp > ?"
                params = [time_threshold]

                if metric_name:
                    where_clause += " AND metric_name = ?"
                    params.append(metric_name)

                # Get all metrics
                cursor.execute(
                    f"""
                    SELECT metric_name, AVG(metric_value), COUNT(*)
                    FROM metrics {where_clause}
                    GROUP BY metric_name
                """,
                    params,
                )

                metrics = {}
                for row in cursor.fetchall():
                    metrics[row[0]] = {"average": round(row[1], 2), "count": row[2]}

                return metrics

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}


# Global logger instance
prediction_logger = PredictionLogger()


@contextmanager
def log_prediction_request(model_name: str, request_data: Dict[str, Any]):
    """Context manager for logging prediction requests"""
    start_time = time.time()
    status = "success"
    error_message = None
    response_data = {}

    try:
        yield response_data
    except Exception as e:
        status = "error"
        error_message = str(e)
        raise
    finally:
        processing_time_ms = (time.time() - start_time) * 1000

        # Log the prediction
        prediction_logger.log_prediction(
            model_name=model_name,
            request_data=request_data,
            response_data=response_data,
            processing_time_ms=processing_time_ms,
            status=status,
            error_message=error_message,
        )

        # Log metrics
        prediction_logger.log_metric(
            "processing_time_ms", processing_time_ms, model_name
        )
        prediction_logger.log_metric("request_count", 1, model_name)

        if status == "error":
            prediction_logger.log_metric("error_count", 1, model_name)
