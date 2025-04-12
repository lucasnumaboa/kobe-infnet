"""
Classification Model Training Pipeline for Kobe Bryant Shot Prediction.
This module trains classification models directly using scikit-learn to predict whether a shot will be made or missed.
"""

import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime
from pathlib import Path
from sklearn.metrics import log_loss, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

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
    # Go up multiple levels: pipelines/model_training/run_classification.py -> model_training -> pipelines -> kobe_prediction -> src -> root
    project_root = current_path.parent.parent.parent.parent.parent
    return project_root

def log_step(message):
    """Log a step to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as f:
        f.write(f"[{timestamp}] CLASSIFICATION TRAINING: {message}\n")
    logger.info(message)

def load_data():
    """Load the preprocessed training and testing data."""
    log_step("Loading training and testing data")
    project_root = get_project_root()
    train_data_path = os.path.join(project_root, "data", "processed", "base_train.parquet")
    test_data_path = os.path.join(project_root, "data", "processed", "base_test.parquet")
    
    log_step(f"Loading training data from: {train_data_path}")
    train_df = pd.read_parquet(train_data_path)
    
    log_step(f"Loading testing data from: {test_data_path}")
    test_df = pd.read_parquet(test_data_path)
    
    return train_df, test_df

def prepare_features(df, target_col='shot_made_flag'):
    """Prepare features and target variable for model training."""
    log_step("Preparing features and target variable")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    log_step(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train a Logistic Regression model using scikit-learn."""
    log_step("Training Logistic Regression model")
    
    # Create and train the model with optimal hyperparameters
    model = LogisticRegression(
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    log_step("Logistic Regression model trained successfully")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    log_step(f"Logistic Regression Log Loss: {logloss:.4f}")
    log_step(f"Logistic Regression F1 Score: {f1:.4f}")
    log_step("\nClassification Report:\n" + classification_report(y_test, y_pred))
    
    # Save results
    results = {
        'model': model,
        'log_loss': logloss,
        'f1_score': f1
    }
    
    return results

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train a Decision Tree model using scikit-learn."""
    log_step("Training Decision Tree model")
    
    # Create and train the model with optimal hyperparameters
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    log_step("Decision Tree model trained successfully")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    log_step(f"Decision Tree Log Loss: {logloss:.4f}")
    log_step(f"Decision Tree F1 Score: {f1:.4f}")
    log_step("\nClassification Report:\n" + classification_report(y_test, y_pred))
    
    # Save results
    results = {
        'model': model,
        'log_loss': logloss,
        'f1_score': f1
    }
    
    return results

def save_model_with_mlflow(model, model_name, metrics):
    """Save the model and register it with MLflow."""
    log_step(f"Saving {model_name} model")
    
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models", "classification")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model locally using joblib
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    
    log_step(f"Model saved to {model_path}")
    
    # Criar um novo run do MLflow para cada modelo
    with mlflow.start_run(run_name=f"classification_{model_name}", nested=True):
        # Log model type
        mlflow.log_param("model_type", model_name)
        
        # Log metrics with MLflow
        for metric_name, metric_value in metrics.items():
            if metric_name != 'model':
                mlflow.log_metric(metric_name, metric_value)
        
        # Log model in MLflow usando sklearn
        mlflow.sklearn.log_model(
            model, 
            artifact_path=f"models/{model_name}",
            registered_model_name=f"kobe_shot_prediction_{model_name}"
        )
    
    log_step(f"Model {model_name} registered with MLflow")

def main():
    """Main function to run the classification model training pipeline."""
    log_step("Starting classification model training pipeline")
    
    try:
        # Load data
        train_data, test_data = load_data()
        log_step(f"Training data shape: {train_data.shape}, Testing data shape: {test_data.shape}")
        
        # Prepare features and target variables
        X_train, y_train = prepare_features(train_data)
        X_test, y_test = prepare_features(test_data)
        
        # Train logistic regression model
        lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Save logistic regression model
        save_model_with_mlflow(lr_results['model'], "logistic_regression", {
            'log_loss': lr_results['log_loss'],
            'f1_score': lr_results['f1_score']
        })
        
        # Train decision tree model
        dt_results = train_decision_tree(X_train, y_train, X_test, y_test)
        
        # Save decision tree model
        save_model_with_mlflow(dt_results['model'], "decision_tree", {
            'log_loss': dt_results['log_loss'], 
            'f1_score': dt_results['f1_score']
        })
        
        # Compare models and select the best one based on log loss (lower is better)
        if (lr_results['log_loss'] < dt_results['log_loss']):
            log_step("Logistic Regression model selected as the best model (lower log loss)")
            best_model_name = "logistic_regression"
            best_metrics = {'log_loss': lr_results['log_loss'], 'f1_score': lr_results['f1_score']}
        else:
            log_step("Decision Tree model selected as the best model (lower log loss)")
            best_model_name = "decision_tree"
            best_metrics = {'log_loss': dt_results['log_loss'], 'f1_score': dt_results['f1_score']}
        
        # Log the selected model name to parent run
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metrics(best_metrics)
        
        log_step("Classification model training pipeline completed successfully")
        return True
        
    except Exception as e:
        log_step(f"Error in classification model training: {str(e)}")
        raise

if __name__ == "__main__":
    # Start MLflow run for tracking with the specified name
    with mlflow.start_run(run_name="Treinamento"):
        main()