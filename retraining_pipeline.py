"""
Pipeline de retreinamento para o projeto de previsão de arremessos do Kobe Bryant.
Este módulo implementa o retreinamento incremental do modelo usando os dados de produção.
"""

import os
import pandas as pd
import numpy as np
import mlflow
import yaml
import joblib
from datetime import datetime
import logging
from pathlib import Path
from sklearn.metrics import log_loss, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the absolute path to the project root directory."""
    current_path = Path(__file__).resolve()
    project_root = current_path.parent
    return project_root

def log_step(message):
    """Log a step to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as f:
        f.write(f"[{timestamp}] PIPELINE RETREINAMENTO: {message}\n")
    logger.info(message)

def load_training_data():
    """Load the existing training dataset."""
    log_step("Carregando dados de treinamento existentes")
    project_root = get_project_root()
    train_data_path = os.path.join(project_root, "data", "processed", "base_train.parquet")
    log_step(f"Lendo dados de treinamento de: {train_data_path}")
    
    return pd.read_parquet(train_data_path)

def load_current_model():
    """Load the current best model."""
    log_step("Carregando o modelo atual")
    
    project_root = get_project_root()
    
    # Verificar se temos informações do melhor modelo no arquivo de deployment
    deployment_info_path = os.path.join(project_root, "models", "deployment", "model_info.yaml")
    
    if os.path.exists(deployment_info_path):
        log_step(f"Carregando informações de deployment de: {deployment_info_path}")
        try:
            with open(deployment_info_path, "r") as f:
                model_info = yaml.safe_load(f)
            
            model_name = model_info.get("model_name")
            model_type = model_info.get("model_type")
            
            log_step(f"Informações do modelo encontradas: {model_name} ({model_type})")
            
            if model_type == "Regression":
                model_path = os.path.join(project_root, "models", "regression", f"{model_name}.joblib")
            else:  # Classification
                model_path = os.path.join(project_root, "models", "classification", f"{model_name}.pkl")
                
            if os.path.exists(model_path):
                log_step(f"Carregando modelo de: {model_path}")
                model = joblib.load(model_path)
                return model, model_info
        except Exception as e:
            log_step(f"Erro ao carregar modelo a partir das informações de deployment: {str(e)}")
    
    # Se não encontrou o modelo pelo arquivo de deployment, tenta encontrar diretamente
    log_step("Informações de deployment não encontradas, procurando modelo padrão")
    models_dir = os.path.join(project_root, "models", "classification")
    
    # Procurar pelo melhor modelo salvo
    model_path = os.path.join(models_dir, "logistic_regression.pkl")
    if os.path.exists(model_path):
        log_step(f"Carregando modelo de regressão logística de: {model_path}")
        model = joblib.load(model_path)
        model_info = {"model_name": "logistic_regression", "model_type": "Classification"}
    else:
        model_path = os.path.join(models_dir, "decision_tree.pkl")
        if os.path.exists(model_path):
            log_step(f"Carregando modelo de árvore de decisão de: {model_path}")
            model = joblib.load(model_path)
            model_info = {"model_name": "decision_tree", "model_type": "Classification"}
        else:
            raise FileNotFoundError("Nenhum modelo encontrado para retreinamento")
    
    return model, model_info

def prepare_production_data_for_training(production_data):
    """Prepare production data for training by selecting only relevant features."""
    log_step("Preparando dados de produção para retreinamento")
    
    if 'shot_made_flag' not in production_data.columns:
        log_step("Dados de produção não contêm a coluna target 'shot_made_flag'. Não é possível usar para retreinamento.")
        return None
        
    # Selecionar apenas as colunas relevantes para treinamento
    columns_to_keep = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    
    # Verificar quais colunas estão disponíveis
    available_columns = [col for col in columns_to_keep if col in production_data.columns]
    
    if len(available_columns) < 6:  # Se não tiver pelo menos 5 features + target
        log_step(f"Dados de produção não contêm features suficientes. Encontradas: {available_columns}")
        return None
    
    # Filtrar colunas
    filtered_data = production_data[available_columns].copy()
    
    # Remover linhas com valores faltantes
    filtered_data = filtered_data.dropna()
    log_step(f"Dados preparados para retreinamento. Formato: {filtered_data.shape}")
    
    return filtered_data

def combine_training_data(existing_data, new_data):
    """Combine existing training data with new production data."""
    log_step("Combinando dados de treinamento existentes com novos dados de produção")
    
    # Verificar se as colunas são compatíveis
    existing_columns = set(existing_data.columns)
    new_columns = set(new_data.columns)
    common_columns = existing_columns.intersection(new_columns)
    
    if 'shot_made_flag' not in common_columns:
        log_step("Coluna target 'shot_made_flag' não está presente em ambos os conjuntos de dados")
        return existing_data
    
    # Selecionar apenas colunas comuns
    common_columns_list = list(common_columns)
    existing_filtered = existing_data[common_columns_list].copy()
    new_filtered = new_data[common_columns_list].copy()
    
    # Concatenar datasets
    combined_data = pd.concat([existing_filtered, new_filtered], ignore_index=True)
    
    log_step(f"Dados combinados. Formato original: {existing_filtered.shape}, Novos dados: {new_filtered.shape}, Combinados: {combined_data.shape}")
    
    return combined_data

def train_models(training_data):
    """Train multiple models on the combined training data."""
    log_step("Treinando modelos com dados combinados")
    
    # Separar features e target
    X = training_data.drop('shot_made_flag', axis=1)
    y = training_data['shot_made_flag']
    
    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Iniciar run do MLflow
    with mlflow.start_run(run_name="ModelRetraining"):
        # Treinar modelo de regressão logística
        log_step("Treinando modelo de regressão logística")
        logreg = LogisticRegression(max_iter=1000, random_state=42)
        logreg.fit(X_train, y_train)
        
        # Avaliar modelo de regressão logística
        logreg_preds = logreg.predict(X_test)
        logreg_proba = logreg.predict_proba(X_test)[:, 1]
        
        logreg_accuracy = accuracy_score(y_test, logreg_preds)
        logreg_f1 = f1_score(y_test, logreg_preds)
        logreg_auc = roc_auc_score(y_test, logreg_proba)
        logreg_logloss = log_loss(y_test, logreg_proba)
        
        log_step(f"Regressão Logística - Acurácia: {logreg_accuracy:.4f}, F1: {logreg_f1:.4f}, AUC: {logreg_auc:.4f}, Log Loss: {logreg_logloss:.4f}")
        
        # Registrar métricas no MLflow
        mlflow.log_metrics({
            "logreg_accuracy": logreg_accuracy,
            "logreg_f1": logreg_f1,
            "logreg_auc": logreg_auc,
            "logreg_logloss": logreg_logloss
        })
        
        # Treinar modelo Random Forest
        log_step("Treinando modelo Random Forest")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Avaliar modelo Random Forest
        rf_preds = rf.predict(X_test)
        rf_proba = rf.predict_proba(X_test)[:, 1]
        
        rf_accuracy = accuracy_score(y_test, rf_preds)
        rf_f1 = f1_score(y_test, rf_preds)
        rf_auc = roc_auc_score(y_test, rf_proba)
        rf_logloss = log_loss(y_test, rf_proba)
        
        log_step(f"Random Forest - Acurácia: {rf_accuracy:.4f}, F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}, Log Loss: {rf_logloss:.4f}")
        
        # Registrar métricas no MLflow
        mlflow.log_metrics({
            "rf_accuracy": rf_accuracy,
            "rf_f1": rf_f1,
            "rf_auc": rf_auc,
            "rf_logloss": rf_logloss
        })
        
        # Salvar modelos no MLflow
        mlflow.sklearn.log_model(logreg, "logistic_regression")
        mlflow.sklearn.log_model(rf, "random_forest")
        
        # Determinar o melhor modelo com base no Log Loss (menor é melhor)
        models = {
            "logistic_regression": {"model": logreg, "log_loss": logreg_logloss, "f1": logreg_f1, "auc": logreg_auc},
            "random_forest": {"model": rf, "log_loss": rf_logloss, "f1": rf_f1, "auc": rf_auc}
        }
        
        # Retornar todos os modelos treinados e suas métricas
        return models, X_test, y_test

def save_best_model(models, X_test, y_test, current_model_info):
    """Save the best model based on performance metrics."""
    log_step("Selecionando e salvando o melhor modelo")
    
    # Determinar o melhor modelo com base no Log Loss
    sorted_models = sorted(models.items(), key=lambda x: x[1]["log_loss"])
    best_model_name, best_model_info = sorted_models[0]
    
    log_step(f"Melhor modelo: {best_model_name} com Log Loss: {best_model_info['log_loss']:.4f}")
    
    # Comparar com o modelo atual
    project_root = get_project_root()
    current_model_name = current_model_info.get("model_name")
    current_model_type = current_model_info.get("model_type")
    
    # Verificar se o novo modelo é melhor que o atual
    # Para isso, precisamos avaliar o modelo atual com os mesmos dados de teste
    try:
        if current_model_name in ["logistic_regression", "random_forest"]:
            current_proba = current_model_info["model"].predict_proba(X_test)[:, 1]
            current_logloss = log_loss(y_test, current_proba)
            log_step(f"Modelo atual ({current_model_name}) - Log Loss: {current_logloss:.4f}")
            
            if best_model_info["log_loss"] >= current_logloss:
                log_step(f"O modelo atual ({current_model_name}) tem desempenho igual ou melhor. Mantendo modelo atual.")
                return current_model_name, current_model_type
    except Exception as e:
        log_step(f"Erro ao avaliar modelo atual: {str(e)}. Prosseguindo com o novo modelo.")
    
    # Salvar o melhor modelo
    best_model = best_model_info["model"]
    models_dir = os.path.join(project_root, "models", "classification")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{best_model_name}.pkl")
    joblib.dump(best_model, model_path)
    log_step(f"Novo modelo salvo em: {model_path}")
    
    # Atualizar informações de deployment
    deployment_dir = os.path.join(project_root, "models", "deployment")
    os.makedirs(deployment_dir, exist_ok=True)
    
    deployment_info = {
        "model_name": best_model_name,
        "model_type": "Classification",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "log_loss": float(best_model_info["log_loss"]),
            "f1_score": float(best_model_info["f1"]),
            "auc": float(best_model_info["auc"])
        }
    }
    
    deployment_info_path = os.path.join(deployment_dir, "model_info.yaml")
    with open(deployment_info_path, "w") as f:
        yaml.dump(deployment_info, f)
    
    log_step(f"Informações de deployment atualizadas: {deployment_info}")
    
    # Gerar gráficos de avaliação
    try:
        create_evaluation_plots(models, X_test, y_test)
    except Exception as e:
        log_step(f"Erro ao criar gráficos de avaliação: {str(e)}")
    
    return best_model_name, "Classification"

def create_evaluation_plots(models, X_test, y_test):
    """Create and save evaluation plots for model comparison."""
    log_step("Criando gráficos de avaliação de modelos")
    
    project_root = get_project_root()
    eval_dir = os.path.join(project_root, "models", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # F1 Score Comparison
    f1_scores = {name: info["f1"] for name, info in models.items()}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(f1_scores.keys(), f1_scores.values())
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    f1_plot_path = os.path.join(eval_dir, "f1_score_comparison.png")
    plt.savefig(f1_plot_path)
    plt.close(fig)
    
    # AUC Comparison
    auc_scores = {name: info["auc"] for name, info in models.items()}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(auc_scores.keys(), auc_scores.values())
    ax.set_ylabel("AUC")
    ax.set_title("AUC Comparison")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    auc_plot_path = os.path.join(eval_dir, "auc_comparison.png")
    plt.savefig(auc_plot_path)
    plt.close(fig)
    
    # Create confusion matrices and ROC curves for each model
    for model_name, model_info in models.items():
        model = model_info["model"]
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        cm = pd.DataFrame(
            confusion_matrix(y_test, preds),
            index=["Actual Missed", "Actual Made"],
            columns=["Predicted Missed", "Predicted Made"]
        )
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        cm_path = os.path.join(eval_dir, f"{model_name}_class_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close(fig)
        
        # ROC Curve
        fig, ax = plt.subplots(figsize=(8, 8))
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(eval_dir, f"{model_name}_class_roc_curve.png")
        plt.savefig(roc_path)
        plt.close(fig)
    
    log_step("Gráficos de avaliação salvos com sucesso")

def save_combined_data(combined_data):
    """Save the combined dataset for future training."""
    log_step("Salvando conjunto de dados combinado para uso futuro")
    
    project_root = get_project_root()
    data_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(data_dir, exist_ok=True)
    
    # Salvar dataset combinado
    combined_data_path = os.path.join(data_dir, "base_train.parquet")
    combined_data.to_parquet(combined_data_path)
    log_step(f"Dataset combinado salvo em: {combined_data_path}")
    
    return combined_data_path

def retrain_model(production_data):
    """Main function to retrain the model with new production data."""
    log_step("Iniciando pipeline de retreinamento")
    
    try:
        # Etapa 1: Carregar dados de treinamento existentes
        training_data = load_training_data()
        
        # Etapa 2: Carregar modelo atual
        current_model, current_model_info = load_current_model()
        current_model_info["model"] = current_model  # Adicionar o modelo carregado ao dicionário de informações
        
        # Etapa 3: Preparar dados de produção para treinamento
        prepared_data = prepare_production_data_for_training(production_data)
        
        if prepared_data is None:
            log_step("Não foi possível preparar os dados de produção para retreinamento. Abortando.")
            return current_model
        
        # Etapa 4: Combinar dados existentes com novos dados
        combined_data = combine_training_data(training_data, prepared_data)
        
        # Etapa 5: Treinar modelos com dados combinados
        models, X_test, y_test = train_models(combined_data)
        
        # Etapa 6: Selecionar e salvar o melhor modelo
        best_model_name, best_model_type = save_best_model(models, X_test, y_test, current_model_info)
        
        # Etapa 7: Salvar dados combinados para uso futuro
        save_combined_data(combined_data)
        
        log_step(f"Pipeline de retreinamento concluído com sucesso. Melhor modelo: {best_model_name}")
        
        # Retornar o melhor modelo treinado
        if best_model_name in models:
            return models[best_model_name]["model"]
        else:
            return current_model
    
    except Exception as e:
        log_step(f"Erro no pipeline de retreinamento: {str(e)}")
        # Retornar o modelo atual em caso de erro
        return current_model

# Funções auxiliares que faltaram nas importações
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Execução direta (para testes)
if __name__ == "__main__":
    log_step("Iniciando teste de pipeline de retreinamento")
    
    # Carregar dados de produção simulados
    try:
        project_root = get_project_root()
        prod_data_path = os.path.join(project_root, "data", "raw", "dataset_kobe_prod.parquet")
        production_data = pd.read_parquet(prod_data_path)
        
        # Executar pipeline de retreinamento
        retrain_model(production_data)
    except Exception as e:
        log_step(f"Erro no teste de pipeline: {str(e)}")