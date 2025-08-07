"""
California Housing Dataset Loading and Preprocessing Script

This script downloads, loads, validates, and preprocesses the California Housing dataset
for machine learning model training.
"""

import logging
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_housing_data() -> pd.DataFrame:
    """
    Download the California Housing dataset from sklearn.

    Returns:
        pd.DataFrame: Raw California Housing dataset

    Raises:
        Exception: If dataset cannot be loaded
    """
    logger.info("Downloading California Housing dataset...")

    try:
        data = fetch_california_housing(as_frame=True)
        df = data.frame

        logger.info(
            f"California Housing dataset downloaded successfully: {df.shape[0]} samples, {df.shape[1]} features"
        )
        return df

    except Exception as e:
        logger.error(f"Failed to download California Housing dataset: {str(e)}")
        raise


def validate_housing_data(df: pd.DataFrame) -> bool:
    """
    Validate the California Housing dataset for data quality issues.

    Args:
        df: DataFrame to validate

    Returns:
        bool: True if data is valid, False otherwise
    """
    logger.info("Validating California Housing dataset...")

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(
            f"Found missing values: {missing_values[missing_values > 0].to_dict()}"
        )
        return False

    # Check for infinite values
    infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if infinite_values.sum() > 0:
        logger.warning(
            f"Found infinite values: {infinite_values[infinite_values > 0].to_dict()}"
        )
        return False

    # Check data types
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

    # Check for negative values in features that should be positive
    positive_features = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "MedHouseVal",
    ]
    for feature in positive_features:
        if (df[feature] < 0).any():
            logger.warning(f"Found negative values in {feature}")

    # Check for reasonable value ranges
    if (df["MedHouseVal"] > 500000).any():
        logger.warning("Found unusually high house values (>500k)")

    if (df["MedInc"] > 50).any():
        logger.warning("Found unusually high median income (>50)")

    logger.info("California Housing dataset validation passed")
    return True


def preprocess_housing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the California Housing dataset for machine learning.

    Args:
        df: Raw California Housing DataFrame

    Returns:
        pd.DataFrame: Preprocessed California Housing DataFrame
    """
    logger.info("Preprocessing California Housing dataset...")

    # Create a copy to avoid modifying original data
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

        # Cap outliers instead of removing them
        processed_df[column] = processed_df[column].clip(
            lower=lower_bound, upper=upper_bound
        )

    # Add feature engineering
    processed_df["total_rooms"] = processed_df["AveRooms"] * processed_df["Population"]
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

    # Handle any infinite values from division
    processed_df["room_to_bedroom_ratio"] = processed_df[
        "room_to_bedroom_ratio"
    ].replace([np.inf, -np.inf], np.nan)
    processed_df["room_to_bedroom_ratio"] = processed_df[
        "room_to_bedroom_ratio"
    ].fillna(processed_df["room_to_bedroom_ratio"].mean())

    processed_df["income_per_person"] = processed_df["income_per_person"].replace(
        [np.inf, -np.inf], np.nan
    )
    processed_df["income_per_person"] = processed_df["income_per_person"].fillna(
        processed_df["income_per_person"].mean()
    )

    # Add geographic features
    processed_df["distance_from_coast"] = np.sqrt(
        processed_df["Latitude"] ** 2 + processed_df["Longitude"] ** 2
    )
    processed_df["lat_long_interaction"] = (
        processed_df["Latitude"] * processed_df["Longitude"]
    )

    # Add statistical features
    for column in ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]:
        processed_df[f"{column}_zscore"] = (
            processed_df[column] - processed_df[column].mean()
        ) / processed_df[column].std()

    # Create price categories for potential classification tasks
    processed_df["price_category"] = pd.cut(
        processed_df["MedHouseVal"],
        bins=[0, 100000, 200000, 300000, float("inf")],
        labels=["Low", "Medium", "High", "Very High"],
    )

    logger.info(
        f"Preprocessing completed. New features added: {list(processed_df.columns)}"
    )
    return processed_df


def save_housing_data(
    df_raw: pd.DataFrame, df_processed: pd.DataFrame, data_dir: Optional[str] = None
) -> Tuple[str, str]:
    """
    Save raw and processed California Housing data to CSV files.

    Args:
        df_raw: Raw California Housing DataFrame
        df_processed: Processed California Housing DataFrame
        data_dir: Directory to save data. If None, uses default path.

    Returns:
        Tuple[str, str]: Paths to raw and processed data files
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Save raw data
    raw_path = os.path.join(data_dir, "housing_raw.csv")
    df_raw.to_csv(raw_path, index=False)
    logger.info(f"Raw data saved to: {raw_path}")

    # Save processed data
    processed_path = os.path.join(data_dir, "housing_processed.csv")
    df_processed.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to: {processed_path}")

    return raw_path, processed_path


def create_housing_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train/test split for California Housing dataset.

    Args:
        df: Processed California Housing DataFrame
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Creating train/test split with {test_size*100}% for testing...")

    # Select features (exclude target and categorical columns)
    exclude_columns = ["MedHouseVal", "price_category"]
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    X = df[feature_columns]
    y = df["MedHouseVal"]

    # Create split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    return X_train, X_test, y_train, y_test


def get_housing_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the California Housing dataset.

    Args:
        df: California Housing DataFrame

    Returns:
        Dict containing dataset summary statistics
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "target_summary": {
            "mean": df["MedHouseVal"].mean(),
            "median": df["MedHouseVal"].median(),
            "std": df["MedHouseVal"].std(),
            "min": df["MedHouseVal"].min(),
            "max": df["MedHouseVal"].max(),
        },
        "price_category_distribution": (
            df["price_category"].value_counts().to_dict()
            if "price_category" in df.columns
            else {}
        ),
        "numeric_summary": df.describe().to_dict(),
    }

    return summary


def scale_housing_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type: str = "standard"
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Scale housing features for better model performance.

    Args:
        X_train: Training features
        X_test: Test features
        scaler_type: Type of scaler ('standard' or 'robust')

    Returns:
        Tuple of (scaled_X_train, scaled_X_test, fitted_scaler)
    """
    logger.info(f"Scaling features using {scaler_type} scaler...")

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

    logger.info("Feature scaling completed")
    return X_train_scaled, X_test_scaled, scaler


def main():
    """
    Main function to orchestrate the California Housing data loading and preprocessing pipeline.
    """
    try:
        logger.info(
            "Starting California Housing data loading and preprocessing pipeline..."
        )

        # Download data
        df_raw = download_housing_data()

        # Validate data
        if not validate_housing_data(df_raw):
            raise ValueError("Data validation failed")

        # Preprocess data
        df_processed = preprocess_housing_data(df_raw)

        # Save data
        raw_path, processed_path = save_housing_data(df_raw, df_processed)

        # Generate summary
        summary = get_housing_data_summary(df_processed)
        logger.info("Dataset summary:")
        logger.info(f"  Shape: {summary['shape']}")
        logger.info(f"  Features: {len(summary['columns'])}")
        logger.info(f"  Target mean: ${summary['target_summary']['mean']:.2f}")
        logger.info(
            f"  Target range: ${summary['target_summary']['min']:.2f} - ${summary['target_summary']['max']:.2f}"
        )

        # Create train/test split
        X_train, X_test, y_train, y_test = create_housing_train_test_split(df_processed)

        # Scale features
        X_train_scaled, X_test_scaled, scaler = scale_housing_features(X_train, X_test)

        logger.info(
            "California Housing data loading and preprocessing completed successfully!"
        )

        return {
            "raw_data": df_raw,
            "processed_data": df_processed,
            "train_test_split": (X_train_scaled, X_test_scaled, y_train, y_test),
            "scaler": scaler,
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"California Housing data pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
