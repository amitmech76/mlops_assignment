#!/usr/bin/env python3
"""
Model Evaluation Script for DVC Pipeline
Evaluates and compares trained models, generates metrics and plots
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare ML models for DVC pipeline"""

    def __init__(
        self,
        mlruns_path: str = "mlruns",
        metrics_path: str = "metrics",
        plots_path: str = "plots",
    ):
        self.mlruns_path = Path(mlruns_path)
        self.metrics_path = Path(metrics_path)
        self.plots_path = Path(plots_path)

        # Create directories if they don't exist
        self.metrics_path.mkdir(exist_ok=True)
        self.plots_path.mkdir(exist_ok=True)

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f"file://{self.mlruns_path.absolute()}")

    def get_latest_models(self) -> Dict[str, Any]:
        """Get latest model information from MLflow"""
        models_info = {}

        try:
            # Get California Housing model
            housing_model = mlflow.pyfunc.load_model(
                "models:/CaliforniaHousingBestModel/latest"
            )
            models_info["housing"] = {
                "model": housing_model,
                "type": "regression",
                "dataset": "california_housing",
            }
            logger.info("Loaded California Housing model")
        except Exception as e:
            logger.warning(f"Could not load housing model: {e}")
            models_info["housing"] = None

        try:
            # Get Iris model
            iris_model = mlflow.pyfunc.load_model("models:/IrisBestModel/latest")
            models_info["iris"] = {
                "model": iris_model,
                "type": "classification",
                "dataset": "iris",
            }
            logger.info("Loaded Iris model")
        except Exception as e:
            logger.warning(f"Could not load iris model: {e}")
            models_info["iris"] = None

        return models_info

    def load_test_data(self) -> Dict[str, pd.DataFrame]:
        """Load test datasets"""
        test_data = {}

        try:
            # Load housing data
            housing_data = pd.read_csv("data/housing_processed.csv")
            test_data["housing"] = housing_data
            logger.info(f"Loaded housing data: {housing_data.shape}")
        except Exception as e:
            logger.error(f"Could not load housing data: {e}")

        try:
            # Load iris data
            iris_data = pd.read_csv("data/iris_processed.csv")
            test_data["iris"] = iris_data
            logger.info(f"Loaded iris data: {iris_data.shape}")
        except Exception as e:
            logger.error(f"Could not load iris data: {e}")

        return test_data

    def evaluate_regression_model(
        self, model, data: pd.DataFrame, model_name: str
    ) -> Dict[str, float]:
        """Evaluate regression model"""
        # Prepare data
        target_col = (
            "MedHouseVal" if "MedHouseVal" in data.columns else data.columns[-1]
        )
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Make predictions
        y_pred = model.predict(X)

        # Calculate metrics
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2_score": float(r2_score(y, y_pred)),
            "mape": float(np.mean(np.abs((y - y_pred) / y)) * 100),
            "samples_evaluated": len(y),
            "evaluation_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"{model_name} Regression Metrics: RMSE={metrics['rmse']:.4f}, R²={metrics['r2_score']:.4f}"
        )
        return metrics

    def evaluate_classification_model(
        self, model, data: pd.DataFrame, model_name: str
    ) -> Dict[str, Any]:
        """Evaluate classification model"""
        # Prepare data
        target_col = "target" if "target" in data.columns else data.columns[-1]
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Make predictions
        y_pred = model.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="weighted"
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "samples_evaluated": len(y),
            "evaluation_timestamp": datetime.now().isoformat(),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
        }

        logger.info(
            f"{model_name} Classification Metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}"
        )
        return metrics

    def create_performance_plots(self, all_metrics: Dict[str, Dict]):
        """Create performance visualization plots"""
        plt.style.use("seaborn-v0_8")

        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Performance Evaluation", fontsize=16, fontweight="bold")

        # Plot 1: Housing Model Metrics
        if "housing" in all_metrics:
            housing_metrics = all_metrics["housing"]
            metrics_names = ["RMSE", "MAE", "R² Score", "MAPE"]
            metrics_values = [
                housing_metrics.get("rmse", 0),
                housing_metrics.get("mae", 0),
                housing_metrics.get("r2_score", 0),
                housing_metrics.get("mape", 0),
            ]

            bars1 = ax1.bar(
                metrics_names,
                metrics_values,
                color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"],
            )
            ax1.set_title("California Housing Model Performance", fontweight="bold")
            ax1.set_ylabel("Metric Value")
            ax1.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars1, metrics_values):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        # Plot 2: Iris Model Metrics
        if "iris" in all_metrics:
            iris_metrics = all_metrics["iris"]
            metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
            metrics_values = [
                iris_metrics.get("accuracy", 0),
                iris_metrics.get("precision", 0),
                iris_metrics.get("recall", 0),
                iris_metrics.get("f1_score", 0),
            ]

            bars2 = ax2.bar(
                metrics_names,
                metrics_values,
                color=["#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"],
            )
            ax2.set_title("Iris Classification Model Performance", fontweight="bold")
            ax2.set_ylabel("Metric Value")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars2, metrics_values):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        # Plot 3: Confusion Matrix for Iris (if available)
        if "iris" in all_metrics and "confusion_matrix" in all_metrics["iris"]:
            cm = np.array(all_metrics["iris"]["confusion_matrix"])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax3,
                xticklabels=["Setosa", "Versicolor", "Virginica"],
                yticklabels=["Setosa", "Versicolor", "Virginica"],
            )
            ax3.set_title("Iris Model Confusion Matrix", fontweight="bold")
            ax3.set_xlabel("Predicted")
            ax3.set_ylabel("Actual")

        # Plot 4: Model Comparison Summary
        models_data = []
        if "housing" in all_metrics:
            models_data.append(
                {
                    "Model": "Housing",
                    "Type": "Regression",
                    "Primary Metric": all_metrics["housing"].get("r2_score", 0),
                    "Metric Name": "R² Score",
                }
            )
        if "iris" in all_metrics:
            models_data.append(
                {
                    "Model": "Iris",
                    "Type": "Classification",
                    "Primary Metric": all_metrics["iris"].get("accuracy", 0),
                    "Metric Name": "Accuracy",
                }
            )

        if models_data:
            df_comparison = pd.DataFrame(models_data)
            bars4 = ax4.bar(
                df_comparison["Model"],
                df_comparison["Primary Metric"],
                color=["#FF6B6B", "#4ECDC4"],
            )
            ax4.set_title("Model Performance Comparison", fontweight="bold")
            ax4.set_ylabel("Primary Metric Value")
            ax4.set_ylim(0, 1)

            # Add value labels and metric names
            for i, (bar, row) in enumerate(zip(bars4, models_data)):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f'{height:.3f}\n({row["Metric Name"]})',
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()

        # Save plot
        plot_path = self.plots_path / "performance_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance plots saved to {plot_path}")

    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete model evaluation"""
        logger.info("Starting model evaluation...")

        # Load models and data
        models_info = self.get_latest_models()
        test_data = self.load_test_data()

        evaluation_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "models_evaluated": 0,
            "results": {},
        }

        # Evaluate each model
        for model_name, model_info in models_info.items():
            if model_info is None or model_name not in test_data:
                logger.warning(f"Skipping {model_name} - model or data not available")
                continue

            logger.info(f"Evaluating {model_name} model...")

            if model_info["type"] == "regression":
                metrics = self.evaluate_regression_model(
                    model_info["model"], test_data[model_name], model_name
                )
            elif model_info["type"] == "classification":
                metrics = self.evaluate_classification_model(
                    model_info["model"], test_data[model_name], model_name
                )
            else:
                logger.warning(f"Unknown model type for {model_name}")
                continue

            evaluation_results["results"][model_name] = metrics
            evaluation_results["models_evaluated"] += 1

        # Create visualization plots
        if evaluation_results["results"]:
            self.create_performance_plots(evaluation_results["results"])

        # Save evaluation results
        results_path = self.metrics_path / "evaluation_report.json"
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        logger.info(f"Evaluation complete. Results saved to {results_path}")
        logger.info(f"Evaluated {evaluation_results['models_evaluated']} models")

        return evaluation_results


def main():
    """Main function to run model evaluation"""
    evaluator = ModelEvaluator()
    results = evaluator.run_evaluation()

    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {results['evaluation_timestamp']}")
    print(f"Models Evaluated: {results['models_evaluated']}")

    for model_name, metrics in results["results"].items():
        print(f"\n📊 {model_name.upper()} MODEL:")
        if "accuracy" in metrics:
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1 Score: {metrics['f1_score']:.4f}")
        if "rmse" in metrics:
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   R² Score: {metrics['r2_score']:.4f}")
        print(f"   Samples: {metrics['samples_evaluated']}")

    print(f"\n📈 Plots saved to: plots/")
    print(f"📋 Metrics saved to: metrics/evaluation_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
