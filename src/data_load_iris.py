"""
Iris Dataset Loading and Preprocessing Script

This script downloads, loads, validates, and preprocesses the Iris dataset
for machine learning model training.
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_iris_data() -> pd.DataFrame:
    """
    Download the Iris dataset from sklearn.
    
    Returns:
        pd.DataFrame: Raw Iris dataset
    
    Raises:
        Exception: If dataset cannot be loaded
    """
    logger.info("Downloading Iris dataset...")
    
    try:
        data = load_iris(as_frame=True)
        df = data.frame
        
        logger.info(f"Iris dataset downloaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
        return df
        
    except Exception as e:
        logger.error(f"Failed to download Iris dataset: {str(e)}")
        raise


def validate_iris_data(df: pd.DataFrame) -> bool:
    """
    Validate the Iris dataset for data quality issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        bool: True if data is valid, False otherwise
    """
    logger.info("Validating Iris dataset...")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Found missing values: {missing_values[missing_values > 0].to_dict()}")
        return False
    
    # Check for infinite values
    infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if infinite_values.sum() > 0:
        logger.warning(f"Found infinite values: {infinite_values[infinite_values > 0].to_dict()}")
        return False
    
    # Check data types
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
    if not all(col in df.columns for col in expected_columns):
        logger.error(f"Missing expected columns. Found: {df.columns.tolist()}")
        return False
    
    # Check target values
    target_values = df['target'].unique()
    expected_targets = [0, 1, 2]  # Iris species encoded as 0, 1, 2
    if not all(val in expected_targets for val in target_values):
        logger.error(f"Unexpected target values: {target_values}")
        return False
    
    logger.info("Iris dataset validation passed")
    return True


def preprocess_iris_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Iris dataset for machine learning.
    
    Args:
        df: Raw Iris DataFrame
    
    Returns:
        pd.DataFrame: Preprocessed Iris DataFrame
    """
    logger.info("Preprocessing Iris dataset...")
    
    # Create a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Rename columns for consistency
    column_mapping = {
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width',
        'target': 'target'
    }
    processed_df = processed_df.rename(columns=column_mapping)
    
    # Create target labels for better interpretability
    target_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    processed_df['species'] = processed_df['target'].map(target_mapping)
    
    # Add feature engineering
    processed_df['sepal_area'] = processed_df['sepal_length'] * processed_df['sepal_width']
    processed_df['petal_area'] = processed_df['petal_length'] * processed_df['petal_width']
    processed_df['sepal_petal_ratio'] = processed_df['sepal_area'] / processed_df['petal_area']
    
    # Handle any infinite values from division
    processed_df['sepal_petal_ratio'] = processed_df['sepal_petal_ratio'].replace([np.inf, -np.inf], np.nan)
    processed_df['sepal_petal_ratio'] = processed_df['sepal_petal_ratio'].fillna(processed_df['sepal_petal_ratio'].mean())
    
    # Add statistical features
    processed_df['sepal_length_zscore'] = (processed_df['sepal_length'] - processed_df['sepal_length'].mean()) / processed_df['sepal_length'].std()
    processed_df['petal_length_zscore'] = (processed_df['petal_length'] - processed_df['petal_length'].mean()) / processed_df['petal_length'].std()
    
    logger.info(f"Preprocessing completed. New features added: {list(processed_df.columns)}")
    return processed_df


def save_iris_data(df_raw: pd.DataFrame, df_processed: pd.DataFrame, 
                  data_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Save raw and processed Iris data to CSV files.
    
    Args:
        df_raw: Raw Iris DataFrame
        df_processed: Processed Iris DataFrame
        data_dir: Directory to save data. If None, uses default path.
    
    Returns:
        Tuple[str, str]: Paths to raw and processed data files
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save raw data
    raw_path = os.path.join(data_dir, 'iris_raw.csv')
    df_raw.to_csv(raw_path, index=False)
    logger.info(f"Raw data saved to: {raw_path}")
    
    # Save processed data
    processed_path = os.path.join(data_dir, 'iris_processed.csv')
    df_processed.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to: {processed_path}")
    
    return raw_path, processed_path


def create_iris_train_test_split(df: pd.DataFrame, test_size: float = 0.2, 
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train/test split for Iris dataset with stratification.
    
    Args:
        df: Processed Iris DataFrame
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Creating train/test split with {test_size*100}% for testing...")
    
    # Select features (exclude target and species columns)
    feature_columns = [col for col in df.columns if col not in ['target', 'species']]
    X = df[feature_columns]
    y = df['target']
    
    # Create stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, X_test, y_train, y_test


def get_iris_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the Iris dataset.
    
    Args:
        df: Iris DataFrame
    
    Returns:
        Dict containing dataset summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'target_distribution': df['target'].value_counts().to_dict(),
        'species_distribution': df['species'].value_counts().to_dict(),
        'numeric_summary': df.describe().to_dict()
    }
    
    return summary


def main():
    """
    Main function to orchestrate the Iris data loading and preprocessing pipeline.
    """
    try:
        logger.info("Starting Iris data loading and preprocessing pipeline...")
        
        # Download data
        df_raw = download_iris_data()
        
        # Validate data
        if not validate_iris_data(df_raw):
            raise ValueError("Data validation failed")
        
        # Preprocess data
        df_processed = preprocess_iris_data(df_raw)
        
        # Save data
        raw_path, processed_path = save_iris_data(df_raw, df_processed)
        
        # Generate summary
        summary = get_iris_data_summary(df_processed)
        logger.info("Dataset summary:")
        logger.info(f"  Shape: {summary['shape']}")
        logger.info(f"  Features: {len(summary['columns'])}")
        logger.info(f"  Target distribution: {summary['target_distribution']}")
        
        # Create train/test split
        X_train, X_test, y_train, y_test = create_iris_train_test_split(df_processed)
        
        logger.info("Iris data loading and preprocessing completed successfully!")
        
        return {
            'raw_data': df_raw,
            'processed_data': df_processed,
            'train_test_split': (X_train, X_test, y_train, y_test),
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Iris data pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
