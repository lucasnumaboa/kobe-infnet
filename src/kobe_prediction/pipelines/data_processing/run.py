"""
Data Processing Pipeline for Kobe Bryant Shot Prediction Project.
This module handles data loading, cleaning, and preprocessing.
"""

import os
import pandas as pd
import mlflow
import logging
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the absolute path to the project root directory."""
    # Start from the current file and go up until we find the project root
    current_path = Path(__file__).resolve()
    # Go up multiple levels: pipelines/data_processing/run.py -> pipelines -> kobe_prediction -> src -> root
    project_root = current_path.parent.parent.parent.parent.parent
    return project_root

def log_step(message):
    """Log a step to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as f:
        f.write(f"[{timestamp}] DATA PROCESSING: {message}\n")
    logger.info(message)

def load_data(filepath):
    """Load data from parquet file."""
    log_step(f"Loading data from {filepath}")
    return pd.read_parquet(filepath)

def filter_data(df):
    """
    Filter data to only include specified columns and remove rows with missing values.
    """
    log_step("Filtering data to include only specified columns")
    
    # Verificar colunas disponíveis no dataset
    log_step(f"Columns available in dataset: {df.columns.tolist()}")
    
    # Columns to keep based on requirements
    columns_to_keep = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    
    # Filter columns
    df_filtered = df[columns_to_keep]
    
    # Log initial shape
    log_step(f"Initial filtered data shape: {df_filtered.shape}")
    
    # Remove rows with missing values
    df_filtered = df_filtered.dropna()
    
    # Log shape after removing missing values
    log_step(f"Data shape after removing missing values: {df_filtered.shape}")
    
    return df_filtered

def stratified_split_data(df, target_col='shot_made_flag', test_size=0.2, random_state=42):
    """Split the data into training and testing sets using stratified sampling."""
    log_step(f"Splitting data with test_size={test_size} using stratified sampling")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Use stratified sampling to ensure class distribution is preserved
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Reconstruct dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    log_step(f"Training set shape: {train_df.shape}, Testing set shape: {test_df.shape}")
    
    return train_df, test_df

def main():
    """Main function to run the data processing pipeline."""
    log_step("Starting data processing pipeline")
    
    # Get project root directory
    project_root = get_project_root()
    log_step(f"Project root directory: {project_root}")
    
    # Define data paths with absolute paths
    dev_data_path = os.path.join(project_root, "data", "raw", "dataset_kobe_dev.parquet")
    prod_data_path = os.path.join(project_root, "data", "raw", "dataset_kobe_prod.parquet")
    
    # Create absolute paths for output
    processed_dir = os.path.join(project_root, "data", "processed")
    filtered_data_path = os.path.join(processed_dir, "data_filtered.parquet")
    train_output_path = os.path.join(processed_dir, "base_train.parquet")
    test_output_path = os.path.join(processed_dir, "base_test.parquet")
    
    # Fixed test size parameter
    test_size = 0.2
    
    # Process development data
    try:
        # Step 1: Load development data
        log_step("Processing development dataset")
        log_step(f"Reading data from: {dev_data_path}")
        dev_data = load_data(dev_data_path)
        log_step(f"Original development data shape: {dev_data.shape}")
        
        # Step 2: Filter columns and remove missing values
        dev_data_filtered = filter_data(dev_data)
        
        # Log dataset dimensions after filtering
        filtered_rows, filtered_cols = dev_data_filtered.shape
        log_step(f"Filtered development data dimension: {filtered_rows} rows, {filtered_cols} columns")
        
        # Step 3: Save filtered data
        os.makedirs(processed_dir, exist_ok=True)
        log_step(f"Saving filtered data to: {filtered_data_path}")
        dev_data_filtered.to_parquet(filtered_data_path)
        
        # Step 4: Split into train and test sets using stratified sampling
        train_dev, test_dev = stratified_split_data(dev_data_filtered, test_size=test_size)
        
        # Calculate sizes for MLflow logging
        train_size = len(train_dev)
        test_size_actual = len(test_dev)
        
        # Step 5: Save processed data
        log_step(f"Saving training data to: {train_output_path}")
        train_dev.to_parquet(train_output_path)
        log_step(f"Saving test data to: {test_output_path}")
        test_dev.to_parquet(test_output_path)
        log_step("Development data processed and saved")
        
        # Load production data (just to check, not processing yet)
        log_step("Loading production dataset")
        log_step(f"Reading data from: {prod_data_path}")
        prod_data = load_data(prod_data_path)
        log_step(f"Production data shape: {prod_data.shape}")
        
        # Log metrics and parameters in MLflow
        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("train_size", train_size)
        mlflow.log_metric("test_size", test_size_actual)
        mlflow.log_metric("filtered_dataset_size", filtered_rows)
        
    except Exception as e:
        log_step(f"Error in data processing: {str(e)}")
        raise
    
    log_step("Data processing pipeline completed successfully")
    return True

if __name__ == "__main__":
    # Start MLflow run for tracking with the specified name
    with mlflow.start_run(run_name="PreparacaoDados"):
        main()