"""
Unified Data Pipeline for ML Models

This script provides a comprehensive data pipeline for both Iris and California Housing datasets,
including downloading, validation, preprocessing, feature engineering, and data splitting.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Unified data pipeline for handling multiple datasets.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data pipeline.

        Args:
            data_dir: Directory to save processed data. If None, uses default path.
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Initialize scalers
        self.scalers = {}
        self.label_encoders = {}

    def download_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Download a dataset from sklearn.

        Args:
            dataset_name: Name of the dataset ('iris' or 'housing')

        Returns:
            pd.DataFrame: Raw dataset

        Raises:
            ValueError: If dataset name is not supported
        """
        logger.info(f"Downloading {dataset_name} dataset...")

        try:
            if dataset_name.lower() == "iris":
                data = load_iris(as_frame=True)
                df = data.frame
            elif dataset_name.lower() == "housing":
                data = fetch_california_housing(as_frame=True)
                df = data.frame
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            logger.info(
                f"{dataset_name.title()} dataset downloaded: {df.shape[0]} samples, {df.shape[1]} features"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to download {dataset_name} dataset: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """
        Validate dataset for data quality issues.

        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for specific validation rules

        Returns:
            bool: True if data is valid, False otherwise
        """
        logger.info(f"Validating {dataset_name} dataset...")

        # Common validations
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(
                f"Found missing values: {missing_values[missing_values > 0].to_dict()}"
            )
            return False

        infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum()
        if infinite_values.sum() > 0:
            logger.warning(
                f"Found infinite values: {infinite_values[infinite_values > 0].to_dict()}"
            )
            return False

        # Dataset-specific validations
        if dataset_name.lower() == "iris":
            expected_columns = [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
                "target",
            ]
            if not all(col in df.columns for col in expected_columns):
                logger.error(f"Missing expected columns. Found: {df.columns.tolist()}")
                return False

            target_values = df["target"].unique()
            expected_targets = [0, 1, 2]
            if not all(val in expected_targets for val in target_values):
                logger.error(f"Unexpected target values: {target_values}")
                return False

        elif dataset_name.lower() == "housing":
            expected_columns = [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
                "MedHouseVal",
            ]
            if not all(col in df.columns for col in expected_columns):
                logger.error(f"Missing expected columns. Found: {df.columns.tolist()}")
                return False

            # Check for reasonable value ranges
            if (df["MedHouseVal"] > 500000).any():
                logger.warning("Found unusually high house values (>500k)")

            if (df["MedInc"] > 50).any():
                logger.warning("Found unusually high median income (>50)")

        logger.info(f"{dataset_name.title()} dataset validation passed")
        return True

    def preprocess_iris_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Iris dataset with feature engineering.

        Args:
            df: Raw Iris DataFrame

        Returns:
            pd.DataFrame: Preprocessed Iris DataFrame
        """
        logger.info("Preprocessing Iris dataset...")

        processed_df = df.copy()

        # Rename columns for consistency
        column_mapping = {
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "target",
        }
        processed_df = processed_df.rename(columns=column_mapping)

        # Create target labels
        target_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
        processed_df["species"] = processed_df["target"].map(target_mapping)

        # Feature engineering
        processed_df["sepal_area"] = (
            processed_df["sepal_length"] * processed_df["sepal_width"]
        )
        processed_df["petal_area"] = (
            processed_df["petal_length"] * processed_df["petal_width"]
        )
        processed_df["sepal_petal_ratio"] = (
            processed_df["sepal_area"] / processed_df["petal_area"]
        )

        # Handle infinite values
        processed_df["sepal_petal_ratio"] = processed_df["sepal_petal_ratio"].replace(
            [np.inf, -np.inf], np.nan
        )
        processed_df["sepal_petal_ratio"] = processed_df["sepal_petal_ratio"].fillna(
            processed_df["sepal_petal_ratio"].mean()
        )

        # Statistical features
        for col in ["sepal_length", "petal_length"]:
            processed_df[f"{col}_zscore"] = (
                processed_df[col] - processed_df[col].mean()
            ) / processed_df[col].std()

        logger.info(
            f"Iris preprocessing completed. Features: {list(processed_df.columns)}"
        )
        return processed_df

    def preprocess_housing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess California Housing dataset with feature engineering.

        Args:
            df: Raw California Housing DataFrame

        Returns:
            pd.DataFrame: Preprocessed California Housing DataFrame
        """
        logger.info("Preprocessing California Housing dataset...")

        processed_df = df.copy()

        # Handle outliers using IQR method
        numeric_columns = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "MedHouseVal",
        ]

        for column in numeric_columns:
            Q1 = processed_df[column].quantile(0.25)
            Q3 = processed_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed_df[column] = processed_df[column].clip(
                lower=lower_bound, upper=upper_bound
            )

        # Feature engineering
        processed_df["total_rooms"] = (
            processed_df["AveRooms"] * processed_df["Population"]
        )
        processed_df["total_bedrooms"] = (
            processed_df["AveBedrms"] * processed_df["Population"]
        )
        processed_df["room_to_bedroom_ratio"] = (
            processed_df["AveRooms"] / processed_df["AveBedrms"]
        )
        processed_df["income_per_person"] = (
            processed_df["MedInc"] / processed_df["AveOccup"]
        )
        processed_df["house_age_squared"] = processed_df["HouseAge"] ** 2

        # Handle infinite values
        for col in ["room_to_bedroom_ratio", "income_per_person"]:
            processed_df[col] = processed_df[col].replace([np.inf, -np.inf], np.nan)
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())

        # Geographic features
        processed_df["distance_from_coast"] = np.sqrt(
            processed_df["Latitude"] ** 2 + processed_df["Longitude"] ** 2
        )
        processed_df["lat_long_interaction"] = (
            processed_df["Latitude"] * processed_df["Longitude"]
        )

        # Statistical features
        for column in ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]:
            processed_df[f"{column}_zscore"] = (
                processed_df[column] - processed_df[column].mean()
            ) / processed_df[column].std()

        # Price categories
        processed_df["price_category"] = pd.cut(
            processed_df["MedHouseVal"],
            bins=[0, 100000, 200000, 300000, float("inf")],
            labels=["Low", "Medium", "High", "Very High"],
        )

        logger.info(
            f"Housing preprocessing completed. Features: {list(processed_df.columns)}"
        )
        return processed_df

    def preprocess_data(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Preprocess dataset based on dataset name.

        Args:
            df: Raw DataFrame
            dataset_name: Name of the dataset

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        if dataset_name.lower() == "iris":
            return self.preprocess_iris_data(df)
        elif dataset_name.lower() == "housing":
            return self.preprocess_housing_data(df)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def save_data(
        self, df_raw: pd.DataFrame, df_processed: pd.DataFrame, dataset_name: str
    ) -> Tuple[str, str]:
        """
        Save raw and processed data to CSV files.

        Args:
            df_raw: Raw DataFrame
            df_processed: Processed DataFrame
            dataset_name: Name of the dataset

        Returns:
            Tuple[str, str]: Paths to raw and processed data files
        """
        # Save raw data
        raw_path = self.data_dir / f"{dataset_name}_raw.csv"
        df_raw.to_csv(raw_path, index=False)
        logger.info(f"Raw data saved to: {raw_path}")

        # Save processed data
        processed_path = self.data_dir / f"{dataset_name}_processed.csv"
        df_processed.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to: {processed_path}")

        return str(raw_path), str(processed_path)

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/test split for dataset.

        Args:
            df: Processed DataFrame
            dataset_name: Name of the dataset
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Creating train/test split for {dataset_name}...")

        if dataset_name.lower() == "iris":
            # Select features (exclude target and species columns)
            feature_columns = [
                col for col in df.columns if col not in ["target", "species"]
            ]
            X = df[feature_columns]
            y = df["target"]

            # Stratified split for classification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

        elif dataset_name.lower() == "housing":
            # Select features (exclude target and categorical columns)
            exclude_columns = ["MedHouseVal", "price_category"]
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            X = df[feature_columns]
            y = df["MedHouseVal"]

            # Regular split for regression
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        logger.info(
            f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features"
        )
        logger.info(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

        return X_train, X_test, y_train, y_test

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        dataset_name: str,
        scaler_type: str = "standard",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features for better model performance.

        Args:
            X_train: Training features
            X_test: Test features
            dataset_name: Name of the dataset
            scaler_type: Type of scaler ('standard' or 'robust')

        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        logger.info(
            f"Scaling features for {dataset_name} using {scaler_type} scaler..."
        )

        if scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        # Fit scaler on training data only
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )

        # Transform test data
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # Store scaler for later use
        self.scalers[dataset_name] = scaler

        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled

    def get_data_summary(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive dataset summary.

        Args:
            df: DataFrame
            dataset_name: Name of the dataset

        Returns:
            Dict containing dataset summary statistics
        """
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict(),
        }

        if dataset_name.lower() == "iris":
            summary["target_distribution"] = df["target"].value_counts().to_dict()
            summary["species_distribution"] = df["species"].value_counts().to_dict()

        elif dataset_name.lower() == "housing":
            summary["target_summary"] = {
                "mean": df["MedHouseVal"].mean(),
                "median": df["MedHouseVal"].median(),
                "std": df["MedHouseVal"].std(),
                "min": df["MedHouseVal"].min(),
                "max": df["MedHouseVal"].max(),
            }
            summary["price_category_distribution"] = (
                df["price_category"].value_counts().to_dict()
            )

        return summary

    def run_pipeline(
        self,
        dataset_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        scaler_type: str = "standard",
    ) -> Dict[str, Any]:
        """
        Run complete data pipeline for a dataset.

        Args:
            dataset_name: Name of the dataset ('iris' or 'housing')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            scaler_type: Type of scaler to use

        Returns:
            Dict containing all pipeline outputs
        """
        try:
            logger.info(f"Starting {dataset_name} data pipeline...")

            # Download data
            df_raw = self.download_dataset(dataset_name)

            # Validate data
            if not self.validate_data(df_raw, dataset_name):
                raise ValueError(f"{dataset_name} data validation failed")

            # Preprocess data
            df_processed = self.preprocess_data(df_raw, dataset_name)

            # Save data
            raw_path, processed_path = self.save_data(
                df_raw, df_processed, dataset_name
            )

            # Generate summary
            summary = self.get_data_summary(df_processed, dataset_name)
            logger.info(f"Dataset summary: {summary['shape']}")

            # Create train/test split
            X_train, X_test, y_train, y_test = self.create_train_test_split(
                df_processed, dataset_name, test_size, random_state
            )

            # Scale features
            X_train_scaled, X_test_scaled = self.scale_features(
                X_train, X_test, dataset_name, scaler_type
            )

            logger.info(f"{dataset_name.title()} data pipeline completed successfully!")

            return {
                "raw_data": df_raw,
                "processed_data": df_processed,
                "train_test_split": (X_train_scaled, X_test_scaled, y_train, y_test),
                "scaler": self.scalers[dataset_name],
                "summary": summary,
                "raw_path": raw_path,
                "processed_path": processed_path,
            }

        except Exception as e:
            logger.error(f"{dataset_name} data pipeline failed: {str(e)}")
            raise


def main():
    """
    Main function to run data pipelines for both datasets.
    """
    try:
        # Initialize pipeline
        pipeline = DataPipeline()

        # Run pipeline for both datasets
        results = {}

        for dataset_name in ["iris", "housing"]:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {dataset_name.upper()} dataset")
            logger.info(f"{'='*50}")

            results[dataset_name] = pipeline.run_pipeline(dataset_name)

        logger.info("\n" + "=" * 50)
        logger.info("ALL DATA PIPELINES COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)

        return results

    except Exception as e:
        logger.error(f"Data pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
