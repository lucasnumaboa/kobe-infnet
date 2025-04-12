"""
Regression Model Training Pipeline for Kobe Bryant Shot Prediction.
This module trains various regression models to predict the probability of a shot being made.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from pathlib import Path

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
    # Go up multiple levels: pipelines/model_training/run_regression.py -> model_training -> pipelines -> kobe_prediction -> src -> root
    project_root = current_path.parent.parent.parent.parent.parent
    return project_root

def log_step(message):
    """Log a step to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as f:
        f.write(f"[{timestamp}] REGRESSION TRAINING: {message}\n")
    logger.info(message)

def load_training_data():
    """Load the preprocessed training data."""
    log_step("Loading training data")
    
    project_root = get_project_root()
    train_data_path = os.path.join(project_root, "data", "processed", "base_train.parquet")
    
    log_step(f"Loading training data from: {train_data_path}")
    return pd.read_parquet(train_data_path)

def prepare_features(df, target_col='shot_made_flag'):
    """Prepare features and target variable for model training."""
    log_step("Preparing features and target variable")
    
    # Drop any columns that shouldn't be used for prediction
    columns_to_drop = ['shot_id', 'team_id', 'team_name', 'matchup', 'game_date', 'game_id']
    features_df = df.copy()
    
    # Only drop columns that actually exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in features_df.columns]
    if columns_to_drop:
        log_step(f"Dropping non-feature columns: {columns_to_drop}")
        features_df = features_df.drop(columns=columns_to_drop)
    
    # Split features and target
    if target_col in features_df.columns:
        X = features_df.drop(columns=[target_col])
        y = features_df[target_col]
    else:
        raise ValueError(f"Target column '{target_col}' not found in the dataset")
    
    log_step(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return evaluation metrics."""
    log_step(f"Evaluating {model.__class__.__name__} model")
    
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    log_step(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    log_step("Training Linear Regression model")
    
    # Define the model
    lr_model = LinearRegression()
    
    # Train the model
    lr_model.fit(X_train, y_train)
    
    log_step("Linear Regression model trained successfully")
    return lr_model

def train_elastic_net(X_train, y_train):
    """Train an ElasticNet model with hyperparameter tuning."""
    log_step("Training ElasticNet model with hyperparameter tuning")
    
    # Define the parameter grid
    param_grid = {
        'alpha': [0.1, 0.5, 1.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    }
    
    # Define the model
    elastic_net = ElasticNet(max_iter=10000)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        elastic_net, param_grid, cv=5, 
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    
    # Train with grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    log_step(f"ElasticNet best parameters: {grid_search.best_params_}")
    log_step("ElasticNet model trained successfully")
    
    return best_model

def train_random_forest(X_train, y_train):
    """Train a Random Forest Regressor model."""
    log_step("Training Random Forest model")
    
    # Define the model
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None, 
        min_samples_split=2,
        random_state=42
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    log_step("Random Forest model trained successfully")
    return rf_model

def train_gradient_boosting(X_train, y_train):
    """Train a Gradient Boosting Regressor model."""
    log_step("Training Gradient Boosting model")
    
    # Define the model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # Train the model
    gb_model.fit(X_train, y_train)
    
    log_step("Gradient Boosting model trained successfully")
    return gb_model

def save_model(model, model_name, metrics):
    """Save the trained model and register it with MLflow."""
    log_step(f"Saving {model_name} model")
    
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models", "regression")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model locally using joblib
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    log_step(f"Model saved to {model_path}")
    
    # Cria um novo run do MLflow para cada modelo
    with mlflow.start_run(run_name=f"regression_{model_name}", nested=True):
        # Log model and metrics with MLflow
        mlflow.log_params({
            "model_type": model_name,
            "model_class": model.__class__.__name__,
        })
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        mlflow.sklearn.log_model(
            model, 
            f"regression_models/{model_name}",
            registered_model_name=f"kobe_shot_regression_{model_name}"
        )
    
    log_step(f"Model {model_name} registered with MLflow")

def main():
    """Main function to run the regression model training pipeline."""
    log_step("Starting regression model training pipeline")
    
    try:
        # Get project root
        project_root = get_project_root()
        log_step(f"Project root directory: {project_root}")
        
        # Load data
        train_df = load_training_data()
        
        # Load test data for final evaluation
        log_step("Loading test data for evaluation")
        test_data_path = os.path.join(project_root, "data", "processed", "base_test.parquet")
        log_step(f"Loading test data from: {test_data_path}")
        test_df = pd.read_parquet(test_data_path)
        
        # Prepare features
        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)
        
        # Train different regression models
        models = {}
        
        # Linear Regression
        lr_model = train_linear_regression(X_train, y_train)
        lr_metrics = evaluate_model(lr_model, X_test, y_test)
        models['linear_regression'] = (lr_model, lr_metrics)
        
        # ElasticNet
        elastic_model = train_elastic_net(X_train, y_train)
        elastic_metrics = evaluate_model(elastic_model, X_test, y_test)
        models['elastic_net'] = (elastic_model, elastic_metrics)
        
        # Random Forest
        rf_model = train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(rf_model, X_test, y_test)
        models['random_forest'] = (rf_model, rf_metrics)
        
        # Gradient Boosting
        gb_model = train_gradient_boosting(X_train, y_train)
        gb_metrics = evaluate_model(gb_model, X_test, y_test)
        models['gradient_boosting'] = (gb_model, gb_metrics)
        
        # Save all models (cada um em seu próprio run MLflow)
        for model_name, (model, metrics) in models.items():
            save_model(model, model_name, metrics)
        
        # Find best model based on R² score
        best_model_name = max(
            models.keys(),
            key=lambda name: models[name][1]['r2']
        )
        best_model, best_metrics = models[best_model_name]
        
        log_step(f"Best regression model: {best_model_name} with R² = {best_metrics['r2']:.4f}")
        
        # Save best model information in the parent run
        mlflow.log_params({
            "best_model": best_model_name,
            "best_r2_score": best_metrics['r2']
        })
        
        log_step("Regression model training pipeline completed successfully")
        return True
        
    except Exception as e:
        log_step(f"Error in regression model training: {str(e)}")
        raise

if __name__ == "__main__":
    # Start MLflow run for tracking
    with mlflow.start_run(run_name="regression_model_training"):
        main()