"""
Unified Model Training Pipeline for Kobe Bryant Shot Prediction.
This module combines regression and classification model training in a single pipeline.
"""

import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime
from pathlib import Path
import argparse
import joblib

# Import classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, f1_score, classification_report

# Import regression models
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the absolute path to the project root directory."""
    # Start from the current file location
    current_path = Path(__file__).resolve()
    # Go up one level to reach the project root
    project_root = current_path.parent
    return project_root

def log_step(message, model_type="TRAINING"):
    """Log a step to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as f:
        f.write(f"[{timestamp}] {model_type}: {message}\n")
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

# ---- CLASSIFICATION MODEL FUNCTIONS ----

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train a Logistic Regression model using scikit-learn."""
    log_step("Training Logistic Regression model", "CLASSIFICATION")
    
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
    log_step("Logistic Regression model trained successfully", "CLASSIFICATION")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    log_step(f"Logistic Regression Log Loss: {logloss:.4f}", "CLASSIFICATION")
    log_step(f"Logistic Regression F1 Score: {f1:.4f}", "CLASSIFICATION")
    log_step("\nClassification Report:\n" + classification_report(y_test, y_pred), "CLASSIFICATION")
    
    # Save results
    results = {
        'model': model,
        'log_loss': logloss,
        'f1_score': f1
    }
    
    return results

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train a Decision Tree model using scikit-learn."""
    log_step("Training Decision Tree model", "CLASSIFICATION")
    
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
    log_step("Decision Tree model trained successfully", "CLASSIFICATION")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    log_step(f"Decision Tree Log Loss: {logloss:.4f}", "CLASSIFICATION")
    log_step(f"Decision Tree F1 Score: {f1:.4f}", "CLASSIFICATION")
    log_step("\nClassification Report:\n" + classification_report(y_test, y_pred), "CLASSIFICATION")
    
    # Save results
    results = {
        'model': model,
        'log_loss': logloss,
        'f1_score': f1
    }
    
    return results

def train_random_forest_classifier(X_train, y_train, X_test, y_test):
    """Train a Random Forest Classifier."""
    log_step("Training Random Forest Classifier", "CLASSIFICATION")
    
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    log_step("Random Forest Classifier trained successfully", "CLASSIFICATION")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    log_step(f"Random Forest Classifier Log Loss: {logloss:.4f}", "CLASSIFICATION")
    log_step(f"Random Forest Classifier F1 Score: {f1:.4f}", "CLASSIFICATION")
    log_step("\nClassification Report:\n" + classification_report(y_test, y_pred), "CLASSIFICATION")
    
    # Save results
    results = {
        'model': model,
        'log_loss': logloss,
        'f1_score': f1
    }
    
    return results

def save_classification_model(model, model_name, metrics):
    """Save the classification model and register it with MLflow."""
    log_step(f"Saving {model_name} model", "CLASSIFICATION")
    
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models", "classification")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model locally using joblib
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    
    log_step(f"Model saved to {model_path}", "CLASSIFICATION")
    
    # Create a new MLflow run for each model
    with mlflow.start_run(run_name=f"classification_{model_name}", nested=True):
        # Log model type
        mlflow.log_param("model_type", model_name)
        
        # Log metrics with MLflow
        for metric_name, metric_value in metrics.items():
            if metric_name != 'model':
                mlflow.log_metric(metric_name, metric_value)
        
        # Log model in MLflow using sklearn
        mlflow.sklearn.log_model(
            model, 
            artifact_path=f"models/{model_name}",
            registered_model_name=f"kobe_shot_prediction_{model_name}"
        )
    
    log_step(f"Model {model_name} registered with MLflow", "CLASSIFICATION")

# ---- REGRESSION MODEL FUNCTIONS ----

def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    log_step("Training Linear Regression model", "REGRESSION")
    
    # Define the model
    lr_model = LinearRegression()
    
    # Train the model
    lr_model.fit(X_train, y_train)
    
    log_step("Linear Regression model trained successfully", "REGRESSION")
    return lr_model

def train_elastic_net(X_train, y_train):
    """Train an ElasticNet model with hyperparameter tuning."""
    log_step("Training ElasticNet model with hyperparameter tuning", "REGRESSION")
    
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
    
    log_step(f"ElasticNet best parameters: {grid_search.best_params_}", "REGRESSION")
    log_step("ElasticNet model trained successfully", "REGRESSION")
    
    return best_model

def train_random_forest_regressor(X_train, y_train):
    """Train a Random Forest Regressor model."""
    log_step("Training Random Forest Regressor", "REGRESSION")
    
    # Define the model
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None, 
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    log_step("Random Forest Regressor trained successfully", "REGRESSION")
    return rf_model

def train_gradient_boosting(X_train, y_train):
    """Train a Gradient Boosting Regressor model."""
    log_step("Training Gradient Boosting model", "REGRESSION")
    
    # Define the model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # Train the model
    gb_model.fit(X_train, y_train)
    
    log_step("Gradient Boosting model trained successfully", "REGRESSION")
    return gb_model

def evaluate_regression_model(model, X_test, y_test):
    """Evaluate the regression model and return evaluation metrics."""
    log_step(f"Evaluating {model.__class__.__name__} model", "REGRESSION")
    
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    log_step(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}", "REGRESSION")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def save_regression_model(model, model_name, metrics):
    """Save the regression model and register it with MLflow."""
    log_step(f"Saving {model_name} model", "REGRESSION")
    
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models", "regression")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model locally using joblib
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    log_step(f"Model saved to {model_path}", "REGRESSION")
    
    # Create a new MLflow run for each model
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
    
    log_step(f"Model {model_name} registered with MLflow", "REGRESSION")

# ---- MAIN PIPELINE FUNCTIONS ----

def train_classification_models(X_train, y_train, X_test, y_test, force_retrain=False):
    """Train all classification models and return the best one."""
    log_step("Starting classification model training", "CLASSIFICATION")
    
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models", "classification")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    model_results = {}
    
    # Check if models exist and whether to retrain
    lr_path = os.path.join(models_dir, "logistic_regression.pkl")
    dt_path = os.path.join(models_dir, "decision_tree.pkl")
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    
    # Train or load logistic regression
    if force_retrain or not os.path.exists(lr_path):
        log_step("Training new logistic regression model", "CLASSIFICATION")
        lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
        save_classification_model(lr_results['model'], "logistic_regression", {
            'log_loss': lr_results['log_loss'],
            'f1_score': lr_results['f1_score']
        })
    else:
        log_step("Loading existing logistic regression model", "CLASSIFICATION")
        lr_model = joblib.load(lr_path)
        y_pred = lr_model.predict(X_test)
        y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        
        logloss = log_loss(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        lr_results = {
            'model': lr_model,
            'log_loss': logloss,
            'f1_score': f1
        }
        log_step(f"Loaded model metrics - Log Loss: {logloss:.4f}, F1: {f1:.4f}", "CLASSIFICATION")
    
    # Add to results
    model_results["logistic_regression"] = lr_results
    
    # Train or load decision tree
    if force_retrain or not os.path.exists(dt_path):
        log_step("Training new decision tree model", "CLASSIFICATION")
        dt_results = train_decision_tree(X_train, y_train, X_test, y_test)
        save_classification_model(dt_results['model'], "decision_tree", {
            'log_loss': dt_results['log_loss'], 
            'f1_score': dt_results['f1_score']
        })
    else:
        log_step("Loading existing decision tree model", "CLASSIFICATION")
        dt_model = joblib.load(dt_path)
        y_pred = dt_model.predict(X_test)
        y_pred_proba = dt_model.predict_proba(X_test)[:, 1]
        
        logloss = log_loss(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        dt_results = {
            'model': dt_model,
            'log_loss': logloss,
            'f1_score': f1
        }
        log_step(f"Loaded model metrics - Log Loss: {logloss:.4f}, F1: {f1:.4f}", "CLASSIFICATION")
    
    # Add to results
    model_results["decision_tree"] = dt_results
    
    # Train or load random forest classifier
    if force_retrain or not os.path.exists(rf_path):
        log_step("Training new random forest classifier model", "CLASSIFICATION")
        rf_results = train_random_forest_classifier(X_train, y_train, X_test, y_test)
        save_classification_model(rf_results['model'], "random_forest", {
            'log_loss': rf_results['log_loss'], 
            'f1_score': rf_results['f1_score']
        })
    else:
        log_step("Loading existing random forest classifier model", "CLASSIFICATION")
        rf_model = joblib.load(rf_path)
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        logloss = log_loss(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        rf_results = {
            'model': rf_model,
            'log_loss': logloss,
            'f1_score': f1
        }
        log_step(f"Loaded model metrics - Log Loss: {logloss:.4f}, F1: {f1:.4f}", "CLASSIFICATION")
    
    # Add to results
    model_results["random_forest"] = rf_results
    
    # Compare models and select the best one based on log loss (lower is better)
    best_model_name = min(model_results, key=lambda x: model_results[x]['log_loss'])
    best_result = model_results[best_model_name]
    
    log_step(f"Best classification model: {best_model_name} (Log Loss: {best_result['log_loss']:.4f})", "CLASSIFICATION")
    
    return best_model_name, best_result

def train_regression_models(X_train, y_train, X_test, y_test, force_retrain=False):
    """Train all regression models and return the best one."""
    log_step("Starting regression model training", "REGRESSION")
    
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models", "regression")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    models = {}
    
    # Check if models exist and whether to retrain
    lr_path = os.path.join(models_dir, "linear_regression.joblib")
    en_path = os.path.join(models_dir, "elastic_net.joblib")
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    gb_path = os.path.join(models_dir, "gradient_boosting.joblib")
    
    # Linear Regression
    if force_retrain or not os.path.exists(lr_path):
        log_step("Training new linear regression model", "REGRESSION")
        lr_model = train_linear_regression(X_train, y_train)
        lr_metrics = evaluate_regression_model(lr_model, X_test, y_test)
        save_regression_model(lr_model, "linear_regression", lr_metrics)
    else:
        log_step("Loading existing linear regression model", "REGRESSION")
        lr_model = joblib.load(lr_path)
        lr_metrics = evaluate_regression_model(lr_model, X_test, y_test)
        log_step(f"Loaded model metrics - MSE: {lr_metrics['mse']:.4f}, RMSE: {lr_metrics['rmse']:.4f}, MAE: {lr_metrics['mae']:.4f}, R²: {lr_metrics['r2']:.4f}", "REGRESSION")
    
    models["linear_regression"] = (lr_model, lr_metrics)
    
    # ElasticNet
    if force_retrain or not os.path.exists(en_path):
        log_step("Training new elastic net model", "REGRESSION")
        en_model = train_elastic_net(X_train, y_train)
        en_metrics = evaluate_regression_model(en_model, X_test, y_test)
        save_regression_model(en_model, "elastic_net", en_metrics)
    else:
        log_step("Loading existing elastic net model", "REGRESSION")
        en_model = joblib.load(en_path)
        en_metrics = evaluate_regression_model(en_model, X_test, y_test)
        log_step(f"Loaded model metrics - MSE: {en_metrics['mse']:.4f}, RMSE: {en_metrics['rmse']:.4f}, MAE: {en_metrics['mae']:.4f}, R²: {en_metrics['r2']:.4f}", "REGRESSION")
    
    models["elastic_net"] = (en_model, en_metrics)
    
    # Random Forest
    if force_retrain or not os.path.exists(rf_path):
        log_step("Training new random forest model", "REGRESSION")
        rf_model = train_random_forest_regressor(X_train, y_train)
        rf_metrics = evaluate_regression_model(rf_model, X_test, y_test)
        save_regression_model(rf_model, "random_forest", rf_metrics)
    else:
        log_step("Loading existing random forest model", "REGRESSION")
        rf_model = joblib.load(rf_path)
        rf_metrics = evaluate_regression_model(rf_model, X_test, y_test)
        log_step(f"Loaded model metrics - MSE: {rf_metrics['mse']:.4f}, RMSE: {rf_metrics['rmse']:.4f}, MAE: {rf_metrics['mae']:.4f}, R²: {rf_metrics['r2']:.4f}", "REGRESSION")
    
    models["random_forest"] = (rf_model, rf_metrics)
    
    # Gradient Boosting
    if force_retrain or not os.path.exists(gb_path):
        log_step("Training new gradient boosting model", "REGRESSION")
        gb_model = train_gradient_boosting(X_train, y_train)
        gb_metrics = evaluate_regression_model(gb_model, X_test, y_test)
        save_regression_model(gb_model, "gradient_boosting", gb_metrics)
    else:
        log_step("Loading existing gradient boosting model", "REGRESSION")
        gb_model = joblib.load(gb_path)
        gb_metrics = evaluate_regression_model(gb_model, X_test, y_test)
        log_step(f"Loaded model metrics - MSE: {gb_metrics['mse']:.4f}, RMSE: {gb_metrics['rmse']:.4f}, MAE: {gb_metrics['mae']:.4f}, R²: {gb_metrics['r2']:.4f}", "REGRESSION")
    
    models["gradient_boosting"] = (gb_model, gb_metrics)
    
    # Compare models and select the best one based on RMSE (lower is better)
    best_model_name = min(models, key=lambda x: models[x][1]['rmse'])
    best_model, best_metrics = models[best_model_name]
    
    log_step(f"Best regression model: {best_model_name} (RMSE: {best_metrics['rmse']:.4f})", "REGRESSION")
    
    return best_model_name, (best_model, best_metrics)

def update_model_info(class_model_name, reg_model_name, class_metric, reg_metric):
    """Update model info for deployment by storing best model information."""
    log_step("Updating model information for deployment")
    
    project_root = get_project_root()
    deployment_dir = os.path.join(project_root, "models", "deployment")
    
    # Create deployment directory if it doesn't exist
    os.makedirs(deployment_dir, exist_ok=True)
    
    # Create a dictionary with model info
    model_info = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'classification_model': class_model_name,
        'classification_log_loss': class_metric,
        'regression_model': reg_model_name,
        'regression_r2': reg_metric
    }
    
    # Save the model info as JSON
    info_path = os.path.join(deployment_dir, "model_info.json")
    import json
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    log_step(f"Model info saved to {info_path}")
    
    return True

def main(model_type='both', force_retrain=False):
    """Main function to run the unified model training pipeline."""
    log_step(f"Starting unified model training pipeline. Mode: {model_type}, Force retrain: {force_retrain}")
    
    try:
        # Load data
        train_data, test_data = load_data()
        log_step(f"Training data shape: {train_data.shape}, Testing data shape: {test_data.shape}")
        
        # Prepare features and target variables
        X_train, y_train = prepare_features(train_data)
        X_test, y_test = prepare_features(test_data)
        
        if model_type == 'classification' or model_type == 'both':
            # Train classification models
            with mlflow.start_run(run_name="classification_training", nested=True):
                best_class_model, best_class_result = train_classification_models(X_train, y_train, X_test, y_test, force_retrain)
                
                # Log the selected model name to parent run
                mlflow.log_param("best_model", best_class_model)
                mlflow.log_metric("log_loss", best_class_result['log_loss'])
                mlflow.log_metric("f1_score", best_class_result['f1_score'])
                
                log_step("Classification model training completed")
        
        if model_type == 'regression' or model_type == 'both':
            # Train regression models
            with mlflow.start_run(run_name="regression_training", nested=True):
                best_reg_model, (_, best_reg_metrics) = train_regression_models(X_train, y_train, X_test, y_test, force_retrain)
                
                # Log the selected model name to parent run
                mlflow.log_param("best_model", best_reg_model)
                mlflow.log_metrics(best_reg_metrics)
                
                log_step("Regression model training completed")
        
        if model_type == 'both':
            # Update model info for deployment
            update_model_info(
                best_class_model, best_reg_model, 
                best_class_result['log_loss'],
                best_reg_metrics['r2']
            )
        
        log_step("Unified model training pipeline completed successfully")
        return True
        
    except Exception as e:
        log_step(f"Error in unified model training pipeline: {str(e)}")
        raise
