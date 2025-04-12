"""
Model Evaluation Pipeline for Kobe Bryant Shot Prediction.
This module evaluates and compares both regression and classification models.
"""

import pandas as pd
import numpy as np
import mlflow
import joblib
import os
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, roc_curve, auc,
    confusion_matrix, classification_report, roc_auc_score
)

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
    # Go up multiple levels: pipelines/model_evaluation/run.py -> model_evaluation -> pipelines -> kobe_prediction -> src -> root
    project_root = current_path.parent.parent.parent.parent.parent
    return project_root

def log_step(message):
    """Log a step to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as f:
        f.write(f"[{timestamp}] MODEL EVALUATION: {message}\n")
    logger.info(message)

def load_test_data():
    """Load the test data for model evaluation."""
    log_step("Loading test data")
    project_root = get_project_root()
    test_data_path = os.path.join(project_root, "data", "processed", "base_test.parquet")
    log_step(f"Reading test data from: {test_data_path}")
    return pd.read_parquet(test_data_path)

def prepare_features(df, target_col='shot_made_flag'):
    """Prepare features and target variable from the test data."""
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
        
        # For classification, ensure target is binary
        y_binary = (y > 0.5).astype(int)
    else:
        raise ValueError(f"Target column '{target_col}' not found in the dataset")
    
    log_step(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y, y_binary

def load_models():
    """Load the saved regression and classification models."""
    log_step("Loading trained models")
    
    project_root = get_project_root()
    regression_models = {}
    classification_models = {}
    
    # Load regression models
    reg_model_dir = os.path.join(project_root, "models", "regression")
    log_step(f"Looking for regression models in: {reg_model_dir}")
    if os.path.exists(reg_model_dir):
        for model_file in os.listdir(reg_model_dir):
            if model_file.endswith(".joblib") or model_file.endswith(".pkl"):
                model_name = os.path.splitext(model_file)[0]
                model_path = os.path.join(reg_model_dir, model_file)
                log_step(f"Loading regression model: {model_name} from {model_path}")
                regression_models[model_name] = joblib.load(model_path)
    else:
        log_step(f"Regression models directory not found: {reg_model_dir}")
    
    # Load classification models
    class_model_dir = os.path.join(project_root, "models", "classification")
    log_step(f"Looking for classification models in: {class_model_dir}")
    if os.path.exists(class_model_dir):
        for model_file in os.listdir(class_model_dir):
            if model_file.endswith(".joblib") or model_file.endswith(".pkl"):
                model_name = os.path.splitext(model_file)[0]
                model_path = os.path.join(class_model_dir, model_file)
                log_step(f"Loading classification model: {model_name} from {model_path}")
                classification_models[model_name] = joblib.load(model_path)
    else:
        log_step(f"Classification models directory not found: {class_model_dir}")
    
    log_step(f"Loaded {len(regression_models)} regression models and {len(classification_models)} classification models")
    return regression_models, classification_models

def evaluate_regression_models(models, X_test, y_test, threshold=0.5):
    """Evaluate regression models on test data."""
    log_step("Evaluating regression models")
    
    project_root = get_project_root()
    results = {}
    
    # Criar diretório de avaliação se não existir
    evaluation_dir = os.path.join(project_root, "models", "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    for model_name, model in models.items():
        if model_name == "best_regression_model" and any(m == model for m in models.values() if m != model):
            # Skip the best model if it's a duplicate of another model
            continue
            
        log_step(f"Evaluating regression model: {model_name}")
        
        try:
            # Garantir que as colunas estão na ordem correta (mesma ordem do treinamento)
            X_test_aligned = X_test.copy()
            
            # Se o modelo tiver o atributo 'feature_names_in_', ajustar as colunas de acordo
            if hasattr(model, 'feature_names_in_'):
                expected_feature_names = model.feature_names_in_.tolist()
                log_step(f"Feature names esperadas pelo modelo: {expected_feature_names}")
                
                # Verificar se todas as colunas necessárias estão presentes nos dados de teste
                missing_cols = set(expected_feature_names) - set(X_test_aligned.columns)
                if missing_cols:
                    log_step(f"AVISO: Colunas ausentes nos dados de teste: {missing_cols}")
                    # Adicionar colunas faltantes com zeros
                    for col in missing_cols:
                        X_test_aligned[col] = 0
                
                # Reordenar colunas para garantir a mesma ordem do treinamento
                X_test_aligned = X_test_aligned[expected_feature_names]
                log_step(f"Colunas reordenadas para avaliação: {X_test_aligned.columns.tolist()}")
            else:
                # Se não tiver feature_names_in_, usar todas as colunas disponíveis
                log_step("Modelo não tem feature_names_in_, usando todas as colunas disponíveis")
            
            # Get regression predictions
            y_pred_reg = model.predict(X_test_aligned)
            
            # Convert regression predictions to binary for classification metrics
            y_pred_binary = (y_pred_reg > threshold).astype(int)
            y_test_binary = (y_test > threshold).astype(int)
            
            # Calculate regression metrics
            mse = mean_squared_error(y_test, y_pred_reg)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred_reg)
            
            # Calculate classification metrics from regression predictions
            accuracy = accuracy_score(y_test_binary, y_pred_binary)
            precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test_binary, y_pred_reg)
            roc_auc = auc(fpr, tpr)
            
            # Log metrics
            log_step(f"Regression Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            log_step(f"Classification Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
            
            # Log confusion matrix
            cm = confusion_matrix(y_test_binary, y_pred_binary)
            log_step(f"Confusion Matrix:\n{cm}")
            
            # Create and save confusion matrix visualization
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title(f"Confusion Matrix - {model_name} (Regression)")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            cm_path = os.path.join(evaluation_dir, f"{model_name}_reg_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()
            
            # Create and save ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name} (Regression)')
            plt.legend(loc="lower right")
            roc_path = os.path.join(evaluation_dir, f"{model_name}_reg_roc_curve.png")
            plt.tight_layout()
            plt.savefig(roc_path)
            plt.close()
            
            # Log metrics with MLflow - usando nested=True para criar runs aninhados
            with mlflow.start_run(run_name=f"eval_regression_{model_name}", nested=True):
                mlflow.log_params({
                    "model_name": model_name,
                    "threshold": threshold
                })
                mlflow.log_metrics({
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc
                })
                mlflow.log_artifact(cm_path)
                mlflow.log_artifact(roc_path)
            
            # Store results
            results[model_name] = {
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            }
        
        except Exception as e:
            log_step(f"Erro ao avaliar modelo {model_name}: {str(e)}")
            
            # Fornecer detalhes adicionais para diagnóstico
            if "feature names" in str(e).lower():
                if hasattr(model, 'feature_names_in_'):
                    log_step(f"Features esperadas pelo modelo: {model.feature_names_in_.tolist()}")
                log_step(f"Features fornecidas: {X_test.columns.tolist()}")
            
            # Continuar com o próximo modelo
            continue
    
    return results

def evaluate_classification_models(models, X_test, y_test_binary):
    """Evaluate classification models on test data."""
    log_step("Evaluating classification models")
    
    project_root = get_project_root()
    results = {}
    
    # Criar diretório de avaliação se não existir
    evaluation_dir = os.path.join(project_root, "models", "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    for model_name, model in models.items():
        if model_name == "best_classification_model" and any(m == model for m in models.values() if m != model):
            # Skip the best model if it's a duplicate of another model
            continue
            
        log_step(f"Evaluating classification model: {model_name}")
        
        try:
            # Garantir que as colunas estão na ordem correta (mesma ordem do treinamento)
            X_test_aligned = X_test.copy()
            
            # Se o modelo tiver o atributo 'feature_names_in_', ajustar as colunas de acordo
            if hasattr(model, 'feature_names_in_'):
                expected_feature_names = model.feature_names_in_.tolist()
                log_step(f"Feature names esperadas pelo modelo: {expected_feature_names}")
                
                # Verificar se todas as colunas necessárias estão presentes nos dados de teste
                missing_cols = set(expected_feature_names) - set(X_test_aligned.columns)
                if missing_cols:
                    log_step(f"AVISO: Colunas ausentes nos dados de teste: {missing_cols}")
                    # Adicionar colunas faltantes com zeros
                    for col in missing_cols:
                        X_test_aligned[col] = 0
                
                # Reordenar colunas para garantir a mesma ordem do treinamento
                X_test_aligned = X_test_aligned[expected_feature_names]
                log_step(f"Colunas reordenadas para avaliação: {X_test_aligned.columns.tolist()}")
            else:
                # Se não tiver feature_names_in_, usar todas as colunas disponíveis
                log_step("Modelo não tem feature_names_in_, usando todas as colunas disponíveis")
            
            # Get predictions
            y_pred = model.predict(X_test_aligned)
            
            # Get probability estimates if available
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test_aligned)[:, 1]
                has_proba = True
            else:
                y_pred_proba = None
                has_proba = False
            
            # Calculate classification metrics
            accuracy = accuracy_score(y_test_binary, y_pred)
            precision = precision_score(y_test_binary, y_pred, zero_division=0)
            recall = recall_score(y_test_binary, y_pred, zero_division=0)
            f1 = f1_score(y_test_binary, y_pred, zero_division=0)
            
            # Calculate ROC curve and AUC if probability estimates are available
            if has_proba:
                fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
                roc_auc = auc(fpr, tpr)
            else:
                fpr, tpr = None, None
                roc_auc = None
            
            # Log metrics
            log_step(f"Classification Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            if roc_auc is not None:
                log_step(f"AUC: {roc_auc:.4f}")
            
            # Log confusion matrix
            cm = confusion_matrix(y_test_binary, y_pred)
            log_step(f"Confusion Matrix:\n{cm}")
            
            # Generate classification report
            class_report = classification_report(y_test_binary, y_pred)
            log_step(f"Classification Report:\n{class_report}")
            
            # Create and save confusion matrix visualization
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title(f"Confusion Matrix - {model_name} (Classification)")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            cm_path = os.path.join(evaluation_dir, f"{model_name}_class_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()
            
            # Create and save ROC curve if probability estimates are available
            if has_proba:
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name} (Classification)')
                plt.legend(loc="lower right")
                roc_path = os.path.join(evaluation_dir, f"{model_name}_class_roc_curve.png")
                plt.tight_layout()
                plt.savefig(roc_path)
                plt.close()
            else:
                roc_path = None
            
            # Log metrics with MLflow - usando nested=True para criar runs aninhados
            with mlflow.start_run(run_name=f"eval_classification_{model_name}", nested=True):
                mlflow.log_params({
                    "model_name": model_name
                })
                metrics_dict = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }
                if roc_auc is not None:
                    metrics_dict["roc_auc"] = roc_auc
                    
                mlflow.log_metrics(metrics_dict)
                mlflow.log_artifact(cm_path)
                if roc_path:
                    mlflow.log_artifact(roc_path)
            
            # Store results
            results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            if roc_auc is not None:
                results[model_name]["roc_auc"] = roc_auc
        
        except Exception as e:
            log_step(f"Erro ao avaliar modelo {model_name}: {str(e)}")
            
            # Fornecer detalhes adicionais para diagnóstico
            if "feature names" in str(e).lower():
                if hasattr(model, 'feature_names_in_'):
                    log_step(f"Features esperadas pelo modelo: {model.feature_names_in_.tolist()}")
                log_step(f"Features fornecidas: {X_test.columns.tolist()}")
            
            # Continuar com o próximo modelo
            continue
    
    return results

def compare_approaches(reg_results, class_results):
    """Compare regression and classification approaches to determine the best overall model."""
    log_step("Comparing regression and classification approaches")
    
    project_root = get_project_root()
    evaluation_dir = os.path.join(project_root, "models", "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # Create comparison dataframes for visualization
    reg_df = pd.DataFrame.from_dict(reg_results, orient='index')
    class_df = pd.DataFrame.from_dict(class_results, orient='index')
    
    # Add model type column
    reg_df['model_type'] = 'Regression'
    class_df['model_type'] = 'Classification'
    
    # Combine results
    all_results = pd.concat([reg_df, class_df])
    
    # Find best models based on F1 score and ROC AUC
    best_f1_model = all_results['f1_score'].idxmax()
    best_f1_score = all_results.loc[best_f1_model, 'f1_score']
    best_f1_type = all_results.loc[best_f1_model, 'model_type']
    
    log_step(f"Best model by F1 score: {best_f1_model} ({best_f1_type}) with F1 = {best_f1_score:.4f}")
    
    # Check if ROC AUC is available for all models
    if 'roc_auc' in all_results.columns and all_results['roc_auc'].notna().all():
        best_auc_model = all_results['roc_auc'].idxmax()
        best_auc_score = all_results.loc[best_auc_model, 'roc_auc']
        best_auc_type = all_results.loc[best_auc_model, 'model_type']
        log_step(f"Best model by ROC AUC: {best_auc_model} ({best_auc_type}) with AUC = {best_auc_score:.4f}")
    
    # Create comparison visualizations
    
    # F1 Score comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x=all_results.index, y='f1_score', hue='model_type', data=all_results)
    plt.title('F1 Score Comparison: Regression vs Classification')
    plt.xticks(rotation=45)
    plt.tight_layout()
    f1_comp_path = os.path.join(evaluation_dir, "f1_score_comparison.png")
    plt.savefig(f1_comp_path)
    plt.close()
    
    # ROC AUC comparison if available
    if 'roc_auc' in all_results.columns and all_results['roc_auc'].notna().all():
        plt.figure(figsize=(12, 6))
        sns.barplot(x=all_results.index, y='roc_auc', hue='model_type', data=all_results)
        plt.title('ROC AUC Comparison: Regression vs Classification')
        plt.xticks(rotation=45)
        plt.tight_layout()
        auc_comp_path = os.path.join(evaluation_dir, "auc_comparison.png")
        plt.savefig(auc_comp_path)
        plt.close()
        
        # Log results with MLflow - usando nested=True para criar runs aninhados
        with mlflow.start_run(run_name="model_comparison", nested=True):
            mlflow.log_artifact(f1_comp_path)
            mlflow.log_artifact(auc_comp_path)
            mlflow.log_params({
                "best_f1_model": best_f1_model,
                "best_f1_type": best_f1_type,
                "best_auc_model": best_auc_model,
                "best_auc_type": best_auc_type
            })
            mlflow.log_metrics({
                "best_f1_score": best_f1_score,
                "best_auc_score": best_auc_score
            })
    else:
        # Log results with just F1 score comparison - usando nested=True para criar runs aninhados
        with mlflow.start_run(run_name="model_comparison", nested=True):
            mlflow.log_artifact(f1_comp_path)
            mlflow.log_params({
                "best_f1_model": best_f1_model,
                "best_f1_type": best_f1_type
            })
            mlflow.log_metrics({
                "best_f1_score": best_f1_score
            })
    
    # Save comparison results
    results_path = os.path.join(evaluation_dir, "model_comparison_results.csv")
    all_results.to_csv(results_path)
    log_step(f"Comparison results saved to {results_path}")
    
    return all_results

def main():
    """Main function to run the model evaluation pipeline."""
    log_step("Starting model evaluation pipeline")
    
    try:
        # Load test data
        test_df = load_test_data()
        
        # Prepare features and target
        X_test, y_test, y_test_binary = prepare_features(test_df)
        
        # Load models
        regression_models, classification_models = load_models()
        
        # Evaluate models
        if regression_models:
            reg_results = evaluate_regression_models(regression_models, X_test, y_test)
            log_step(f"Evaluated {len(regression_models)} regression models")
        else:
            reg_results = {}
            log_step("No regression models found for evaluation")
            
        if classification_models:
            class_results = evaluate_classification_models(classification_models, X_test, y_test_binary)
            log_step(f"Evaluated {len(classification_models)} classification models")
        else:
            class_results = {}
            log_step("No classification models found for evaluation")
        
        # Compare approaches if both types of models are available
        if reg_results and class_results:
            comparison_results = compare_approaches(reg_results, class_results)
            log_step("Completed comparison of regression and classification approaches")
        
        log_step("Model evaluation pipeline completed successfully")
        return True
        
    except Exception as e:
        log_step(f"Error in model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    # Start MLflow run for tracking
    with mlflow.start_run(run_name="model_evaluation"):
        main()