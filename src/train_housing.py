"""
California Housing Model Training Script

This script trains multiple regression models on the California Housing dataset
and logs them with MLflow for experiment tracking and model versioning.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_california_housing_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the California Housing dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)

    Raises:
        AttributeError: If DataFrame cannot be extracted from the dataset
    """
    logger.info("Loading California Housing dataset...")

    data = fetch_california_housing(as_frame=True)
    df = getattr(data, "frame", None)

    if df is None:
        raise AttributeError(
            "Could not extract DataFrame from fetch_california_housing result."
        )

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def prepare_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data by splitting into train and test sets.

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data: {test_size*100}% for testing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Dictionary containing evaluation metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {"rmse": rmse, "r2": r2, "mae": mae}


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Train a Linear Regression model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    logger.info("Training Linear Regression model...")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    logger.info(
        f"Linear Regression - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}"
    )

    return model, metrics


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[DecisionTreeRegressor, Dict[str, float]]:
    """
    Train a Decision Tree Regressor model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    logger.info("Training Decision Tree Regressor model...")

    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    logger.info(f"Decision Tree - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

    return model, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train a Random Forest Regressor model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    logger.info("Training Random Forest Regressor model...")

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    logger.info(f"Random Forest - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

    return model, metrics


def log_model_with_mlflow(
    model: Any,
    model_name: str,
    metrics: Dict[str, float],
    X_test: pd.DataFrame,
    run_name: str,
) -> str:
    """
    Log model and metrics with MLflow.

    Args:
        model: Trained model
        model_name: Name of the model type
        metrics: Evaluation metrics
        X_test: Test features for input example
        run_name: Name for the MLflow run

    Returns:
        Run ID of the logged experiment
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("random_state", 42)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model with input example
        input_example = X_test.iloc[:2]
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        logger.info(f"Logged {model_name} model with run ID: {run.info.run_id}")
        return run.info.run_id


def register_best_model(
    best_run_id: str, best_rmse: float, model_name: str = "CaliforniaHousingBestModel"
) -> None:
    """
    Register the best performing model in MLflow Model Registry.

    Args:
        best_run_id: Run ID of the best model
        best_rmse: RMSE score of the best model
        model_name: Name for the registered model
    """
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, model_name)
        logger.info(
            f"Best model registered as '{model_name}' from run {best_run_id} with RMSE {best_rmse:.4f}"
        )
    else:
        logger.warning("No model was registered.")


def train_all_models(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[str, float]:
    """
    Train multiple models and return the best one.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets

    Returns:
        Tuple of (best_run_id, best_rmse)
    """
    models_to_train = [
        ("LinearRegression", train_linear_regression),
        ("DecisionTreeRegressor", train_decision_tree),
        ("RandomForestRegressor", train_random_forest),
    ]

    best_rmse = float("inf")
    best_run_id = None

    for model_name, train_func in models_to_train:
        try:
            model, metrics = train_func(X_train, y_train, X_test, y_test)
            run_id = log_model_with_mlflow(
                model, model_name, metrics, X_test, model_name
            )

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_run_id = run_id

        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")

    return best_run_id, best_rmse


def main():
    """
    Main function to orchestrate the training pipeline.
    """
    try:
        # Set up MLflow experiment
        mlflow.set_experiment("CaliforniaHousing-Regression")
        logger.info("Starting California Housing model training...")

        # Load and prepare data
        X, y = load_california_housing_data()
        X_train, X_test, y_train, y_test = prepare_data(X, y)

        # Train all models and find the best one
        best_run_id, best_rmse = train_all_models(X_train, y_train, X_test, y_test)

        # Register the best model
        register_best_model(best_run_id, best_rmse)

        logger.info("Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
