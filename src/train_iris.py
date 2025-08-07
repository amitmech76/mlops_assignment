"""
Iris Classification Model Training Script

This script trains multiple classification models on the Iris dataset
and logs them with MLflow for experiment tracking and model versioning.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_iris_data(data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the processed Iris dataset.

    Args:
        data_path: Path to the processed data file. If None, uses default path.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)

    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "iris_processed.csv"
        )

    logger.info(f"Loading Iris dataset from {data_path}...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    # Exclude non-numeric/string columns from features
    feature_cols = [
        col
        for col in df.columns
        if col not in ["target", "species"] and pd.api.types.is_numeric_dtype(df[col])
    ]
    X = df[feature_cols]
    y = df["target"]

    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target classes: {y.unique()}")
    logger.info(f"Feature columns used for training: {feature_cols}")

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
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Convert all integer columns to float64 to avoid MLflow schema warnings
    int_cols_train = X_train.select_dtypes(include="int").columns
    int_cols_test = X_test.select_dtypes(include="int").columns
    if len(int_cols_train) > 0:
        X_train[int_cols_train] = X_train[int_cols_train].astype("float64")
    if len(int_cols_test) > 0:
        X_test[int_cols_test] = X_test[int_cols_test].astype("float64")

    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def evaluate_classification_model(
    y_true: pd.Series, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate classification model performance using multiple metrics.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Dictionary containing evaluation metrics
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted"))
    recall = float(recall_score(y_true, y_pred, average="weighted"))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter: int = 200,
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Train a Logistic Regression model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        max_iter: Maximum iterations for convergence

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    logger.info("Training Logistic Regression model...")

    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_classification_model(y_test, y_pred)

    logger.info(
        f"Logistic Regression - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}"
    )

    return model, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train a Random Forest Classifier model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        n_estimators: Number of trees in the forest

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    logger.info("Training Random Forest Classifier model...")

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_classification_model(y_test, y_pred)

    logger.info(
        f"Random Forest - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}"
    )

    return model, metrics


def train_support_vector_machine(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[SVC, Dict[str, float]]:
    """
    Train a Support Vector Machine model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    logger.info("Training Support Vector Machine model...")

    model = SVC(random_state=42, probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_classification_model(y_test, y_pred)

    logger.info(
        f"SVM - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}"
    )

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
        mlflow.sklearn.log_model(model, name="model", input_example=input_example)

        logger.info(f"Logged {model_name} model with run ID: {run.info.run_id}")
        return run.info.run_id


from typing import Optional


def register_best_model(
    best_run_id: Optional[str], best_accuracy: float, model_name: str = "IrisBestModel"
) -> None:
    """
    Register the best performing model in MLflow Model Registry.

    Args:
        best_run_id: Run ID of the best model
        best_accuracy: Accuracy score of the best model
        model_name: Name for the registered model
    """
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, model_name)
        logger.info(
            f"Best model registered as '{model_name}' from run {best_run_id} with accuracy {best_accuracy:.4f}"
        )
    else:
        logger.warning("No model was registered.")


from typing import Optional


def train_all_models(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Optional[str], float]:
    """
    Train multiple classification models and return the best one.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets

    Returns:
        Tuple of (best_run_id, best_accuracy)
    """
    models_to_train = [
        ("LogisticRegression", train_logistic_regression),
        ("RandomForestClassifier", train_random_forest),
        ("SupportVectorMachine", train_support_vector_machine),
    ]

    best_accuracy = 0.0
    best_run_id = None

    for model_name, train_func in models_to_train:
        try:
            model, metrics = train_func(X_train, y_train, X_test, y_test)
            run_id = log_model_with_mlflow(
                model, model_name, metrics, X_test, model_name
            )

            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_run_id = run_id

        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")

    return best_run_id, best_accuracy


def print_classification_report(
    y_test: pd.Series, y_pred: np.ndarray, model_name: str
) -> None:
    """
    Print detailed classification report.

    Args:
        y_test: True target values
        y_pred: Predicted target values
        model_name: Name of the model for reporting
    """
    logger.info(f"\nClassification Report for {model_name}:")
    report = classification_report(y_test, y_pred)
    if isinstance(report, dict):
        logger.info(str(report))
    else:
        logger.info("\n" + report)


def main():
    """
    Main function to orchestrate the training pipeline.
    """
    try:
        # Set up MLflow experiment
        mlflow.set_experiment("Iris-Classification")
        logger.info("Starting Iris classification model training...")

        # Load and prepare data
        X, y = load_iris_data()
        X_train, X_test, y_train, y_test = prepare_data(X, y)

        # Train all models and find the best one
        best_run_id, best_accuracy = train_all_models(X_train, y_train, X_test, y_test)

        # Register the best model
        register_best_model(best_run_id, best_accuracy)

        logger.info("Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
