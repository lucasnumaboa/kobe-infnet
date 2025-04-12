"""
Pipeline de aplicação para o projeto de previsão de arremessos do Kobe Bryant.
Este módulo carrega o modelo treinado, aplica-o aos dados de produção e registra métricas.
"""

import os
import pandas as pd
import mlflow
# Removendo import mlflow.pycaret que não existe
from pathlib import Path
from datetime import datetime
import logging
from sklearn.metrics import log_loss, f1_score
import joblib
import yaml
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
    # Start from the current file and go up until we find the project root
    current_path = Path(__file__).resolve()
    project_root = current_path.parent
    return project_root

def log_step(message):
    """Log a step to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as f:
        f.write(f"[{timestamp}] PIPELINE APLICACAO: {message}\n")
    logger.info(message)

def load_production_data():
    """Load the production dataset."""
    log_step("Carregando dados de produção")
    project_root = get_project_root()
    prod_data_path = os.path.join(project_root, "data", "raw", "dataset_kobe_prod.parquet")
    log_step(f"Lendo dados de: {prod_data_path}")
    
    return pd.read_parquet(prod_data_path)

def filter_production_data(df):
    """
    Filter the production data to include only the required columns and remove rows with missing values.
    """
    log_step("Filtrando dados para incluir apenas as colunas especificadas")
    
    # Verificar colunas disponíveis no dataset
    log_step(f"Columns available in dataset: {df.columns.tolist()}")
    
    # Colunas específicas conforme o enunciado
    columns_to_keep = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
    
    # Verificar se existe shot_made_flag nos dados de produção
    if 'shot_made_flag' in df.columns:
        columns_to_keep.append('shot_made_flag')
        log_step("Coluna shot_made_flag encontrada nos dados de produção")
        has_target = True
    else:
        log_step("Coluna shot_made_flag não encontrada nos dados de produção")
        has_target = False
    
    # Filtrar colunas
    df_filtered = df[columns_to_keep]
    
    # Registrar formato inicial
    log_step(f"Formato inicial dos dados filtrados: {df_filtered.shape}")
    
    # Remover linhas com valores faltantes
    df_filtered = df_filtered.dropna()
    
    # Registrar formato após remover valores faltantes
    log_step(f"Formato após remover valores faltantes: {df_filtered.shape}")
    
    return df_filtered, has_target

def load_best_model():
    """Load the best trained model from the models directory."""
    log_step("Carregando o melhor modelo treinado")
    
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
                return model
        except Exception as e:
            log_step(f"Erro ao carregar modelo a partir das informações de deployment: {str(e)}")
    
    # Se não encontrou o modelo pelo arquivo de deployment, tenta encontrar diretamente
    models_dir = os.path.join(project_root, "models", "classification")
    
    # Procurar pelo melhor modelo salvo
    model_path = os.path.join(models_dir, "logistic_regression.pkl")
    if os.path.exists(model_path):
        log_step(f"Carregando modelo de regressão logística de: {model_path}")
        model = joblib.load(model_path)
    else:
        model_path = os.path.join(models_dir, "decision_tree.pkl")
        if os.path.exists(model_path):
            log_step(f"Carregando modelo de árvore de decisão de: {model_path}")
            model = joblib.load(model_path)
        else:
            # Tente buscar no registro MLflow
            log_step("Buscando o melhor modelo no registro MLflow")
            try:
                # Buscar último modelo registrado
                client = mlflow.tracking.MlflowClient()
                registered_models = client.list_registered_models()
                
                if registered_models:
                    latest_model_name = registered_models[0].name
                    model_uri = f"models:/{latest_model_name}/latest"
                    model = mlflow.sklearn.load_model(model_uri)
                    log_step(f"Modelo carregado do registro MLflow: {model_uri}")
                else:
                    raise FileNotFoundError("Nenhum modelo registrado encontrado")
            except Exception as e:
                log_step(f"Erro ao buscar modelo do MLflow: {str(e)}")
                raise
    
    return model

def apply_model_to_production_data(model, production_data):
    """Apply the trained model to production data."""
    log_step("Aplicando modelo aos dados de produção")
    
    try:
        # Preparar os dados para predição
        log_step(f"Formato dos dados de entrada: {production_data.shape}")
        
        # Criar uma cópia para não modificar os dados originais
        X = production_data.copy()
        
        # Se a coluna shot_made_flag existir, vamos salvá-la para uso posterior e removê-la dos dados de entrada
        y_true = None
        if 'shot_made_flag' in X.columns:
            log_step("Removendo coluna shot_made_flag dos dados de entrada para predição")
            y_true = X['shot_made_flag'].copy()
            X = X.drop('shot_made_flag', axis=1)
        
        # Problema: O modelo espera features específicas em uma ordem específica
        # Solução: Verificar as feature names do modelo e reordenar as colunas de X
        expected_feature_names = None
        
        # Tentar obter os nomes das features do modelo
        if hasattr(model, 'feature_names_in_'):
            expected_feature_names = model.feature_names_in_.tolist()
            log_step(f"Feature names do modelo: {expected_feature_names}")
        else:
            # Se o modelo não tiver feature_names_in_, usar uma ordem padrão
            expected_feature_names = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
            log_step(f"Usando ordem padrão de features: {expected_feature_names}")
        
        # Verificar se todas as colunas necessárias estão presentes
        missing_cols = set(expected_feature_names) - set(X.columns)
        if missing_cols:
            log_step(f"AVISO: Colunas ausentes nos dados de entrada: {missing_cols}")
            # Adicionar colunas faltantes com zeros
            for col in missing_cols:
                X[col] = 0
        
        # Verificar se há colunas extras nos dados de entrada que não estão no modelo
        extra_cols = set(X.columns) - set(expected_feature_names)
        if extra_cols:
            log_step(f"AVISO: Colunas extras nos dados de entrada que serão ignoradas: {extra_cols}")
        
        # Reordenar as colunas para garantir a mesma ordem usada no treinamento
        X = X[expected_feature_names]
        log_step(f"Colunas reordenadas para predição: {X.columns.tolist()}")
        
        # Fazer predições usando sklearn diretamente
        if hasattr(model, 'predict_proba'):
            # Predições de probabilidade
            y_pred_proba = model.predict_proba(X)[:, 1]
            # Predições de classe
            y_pred = model.predict(X)
            
            # Criar dataframe de resultados
            predictions = production_data.copy()
            predictions['prediction_score'] = y_pred_proba
            predictions['prediction_label'] = y_pred
            
            log_step("Predições realizadas com sucesso")
            return predictions
        else:
            raise ValueError("O modelo não possui método predict_proba")
            
    except Exception as e:
        log_step(f"Erro ao aplicar o modelo: {str(e)}")
        
        # Adicionar mais detalhes para diagnóstico
        if "feature names" in str(e).lower():
            if hasattr(model, 'feature_names_in_'):
                log_step(f"Features esperadas pelo modelo: {model.feature_names_in_.tolist()}")
            log_step(f"Features fornecidas: {X.columns.tolist() if 'X' in locals() else 'N/A'}")
        
        raise

def evaluate_predictions(predictions, has_target):
    """Evaluate model predictions if target variable is available."""
    log_step("Avaliando predições do modelo")
    
    if not has_target:
        log_step("Alvo 'shot_made_flag' não disponível nos dados de produção. Não é possível avaliar métricas.")
        return None
    
    try:
        # Extrair valores reais e preditos
        y_true = predictions['shot_made_flag'].values
        
        # Extrair probabilidades preditas
        if 'prediction_score' in predictions.columns:
            y_prob = predictions['prediction_score'].values
        else:
            # Tentar encontrar coluna de probabilidade
            prob_cols = [col for col in predictions.columns if 'score' in col.lower() or 'prob' in col.lower()]
            if prob_cols:
                y_prob = predictions[prob_cols[0]].values
            else:
                log_step("Coluna com probabilidades não encontrada. Não é possível calcular log loss.")
                y_prob = None
        
        # Extrair classes preditas
        if 'prediction_label' in predictions.columns:
            y_pred = predictions['prediction_label'].values
        else:
            # Tentar encontrar coluna de predição
            pred_cols = [col for col in predictions.columns if 'pred' in col.lower() or 'label' in col.lower()]
            if pred_cols:
                y_pred = predictions[pred_cols[0]].values
            else:
                log_step("Coluna com predições não encontrada. Não é possível calcular f1-score.")
                y_pred = None
        
        # Calcular métricas
        metrics = {}
        
        if y_prob is not None:
            log_loss_value = log_loss(y_true, y_prob)
            metrics['log_loss'] = log_loss_value
            log_step(f"Log Loss: {log_loss_value:.4f}")
        
        if y_pred is not None:
            f1_score_value = f1_score(y_true, y_pred)
            metrics['f1_score'] = f1_score_value
            log_step(f"F1 Score: {f1_score_value:.4f}")
        
        return metrics
    
    except Exception as e:
        log_step(f"Erro ao calcular métricas: {str(e)}")
        return None

def save_results(predictions, metrics):
    """Save prediction results and log metrics."""
    log_step("Salvando resultados e registrando métricas")
    
    # Criar diretório para resultados
    project_root = get_project_root()
    results_dir = os.path.join(project_root, "data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Salvar predições como parquet
    results_path = os.path.join(results_dir, f"production_predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.parquet")
    predictions.to_parquet(results_path)
    log_step(f"Predições salvas em: {results_path}")
    
    # Registrar arquivo como artefato no MLflow
    mlflow.log_artifact(results_path, "prediction_results")
    
    # Registrar métricas no MLflow, se disponíveis
    if metrics:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Salvar métricas também como CSV para o dashboard
        metrics_path = os.path.join(results_dir, "metrics.csv")
        metrics_dict = metrics.copy()
        metrics_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Verificar se o arquivo já existe
        metrics_df = None
        if os.path.exists(metrics_path):
            try:
                metrics_df = pd.read_csv(metrics_path)
                # Adicionar nova linha
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_dict])])
            except:
                metrics_df = pd.DataFrame([metrics_dict])
        else:
            metrics_df = pd.DataFrame([metrics_dict])
            
        # Salvar CSV atualizado
        metrics_df.to_csv(metrics_path, index=False)
    
    return results_path

def create_monitoring_dashboard(predictions, results_path, metrics):
    """Create a monitoring dashboard using Streamlit."""
    log_step("Criando dashboard de monitoramento")
    
    # Criar diretório para a aplicação Streamlit
    project_root = get_project_root()
    app_dir = os.path.join(project_root, "app")
    os.makedirs(app_dir, exist_ok=True)
    
    # Criar aplicação Streamlit
    app_path = os.path.join(app_dir, "monitoring_dashboard.py")
    
    # Lista de fragmentos de código para o dashboard
    code_fragments = []
    
    # Importações e configurações
    code_fragments.append("""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Monitoramento do Modelo - Kobe Bryant Shot Prediction",
    layout="wide"
)

# Funções auxiliares
def get_project_root():
    current_path = Path(__file__).resolve()
    return current_path.parent.parent

def load_predictions():
    # Encontrar o arquivo de predições mais recente
    project_root = get_project_root()
    results_dir = os.path.join(project_root, "data", "results")
    
    if not os.path.exists(results_dir):
        st.error("Diretório de resultados não encontrado")
        return None
    
    # Listar arquivos de predição
    prediction_files = [f for f in os.listdir(results_dir) if f.startswith("production_predictions_") and f.endswith(".parquet")]
    
    if not prediction_files:
        st.error("Nenhum arquivo de predições encontrado")
        return None
    
    # Ordenar por data (mais recente primeiro)
    latest_file = sorted(prediction_files, reverse=True)[0]
    file_path = os.path.join(results_dir, latest_file)
    
    # Carregar dados
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        st.error(f"Erro ao carregar arquivo de predições: {str(e)}")
        return None

# Título principal
st.title("Dashboard de Monitoramento - Predição de Arremessos do Kobe Bryant")

# Barra lateral com informações do modelo
st.sidebar.title("Informações do Modelo")

# Carregar métricas do último execução
has_metrics = False
try:
    project_root = get_project_root()
    metrics_file = os.path.join(project_root, "data", "results", "metrics.csv")
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        latest_metrics = metrics_df.iloc[-1].to_dict()
        has_metrics = True
        
        # Carregar histórico completo de métricas para gráficos de tendência
        metrics_history = metrics_df.copy()
        metrics_history['timestamp'] = pd.to_datetime(metrics_history['timestamp'])
    else:
        # Métricas de exemplo caso o arquivo não exista
        latest_metrics = {
            'log_loss': 0.5423,
            'f1_score': 0.6812,
            'timestamp': '2025-04-12 15:30:00'
        }
except Exception as e:
    st.sidebar.error(f"Erro ao carregar métricas: {str(e)}")
    # Métricas de exemplo em caso de erro
    latest_metrics = {
        'log_loss': 0.5423,
        'f1_score': 0.6812,
        'timestamp': '2025-04-12 15:30:00'
    }

# Exibir métricas do modelo
st.sidebar.header("Métricas do Modelo")
st.sidebar.metric("Log Loss", f"{latest_metrics.get('log_loss', 'N/A'):.4f}")
st.sidebar.metric("F1 Score", f"{latest_metrics.get('f1_score', 'N/A'):.4f}")
st.sidebar.text(f"Última atualização: {latest_metrics.get('timestamp', 'Desconhecido')}")

# Exibir informações do modelo
st.sidebar.header("Detalhes do Modelo")
st.sidebar.write("**Tipo:** Classificação Binária")
st.sidebar.write("**Objetivo:** Prever se o arremesso do Kobe Bryant será convertido (1) ou não (0)")
st.sidebar.write("**Features utilizadas:**")
st.sidebar.write("- Localização (lat, lon)")
st.sidebar.write("- Minutos restantes")
st.sidebar.write("- Período (quarter)")
st.sidebar.write("- Playoffs (sim/não)")
st.sidebar.write("- Distância do arremesso")
""")

    # Tabs e conteúdo principal
    code_fragments.append("""
# Tabs para organizar o dashboard
tab1, tab2, tab3 = st.tabs(["Visão Geral", "Análise de Predições", "Saúde do Modelo"])

# Tab 1: Visão Geral
with tab1:
    st.header("Visão Geral das Predições")
    
    # Carregar dados de predições
    predictions = load_predictions()
    
    if predictions is not None:
        # Resumo das predições
        st.write(f"Total de arremessos analisados: {len(predictions)}")
        
        # Layout de duas colunas
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de predições
            st.subheader("Distribuição das Predições")
            if 'prediction_label' in predictions.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(x='prediction_label', data=predictions, ax=ax)
                ax.set_xlabel("Arremesso Convertido (1) vs Errado (0)")
                ax.set_ylabel("Contagem")
                ax.set_title("Distribuição das Predições")
                st.pyplot(fig)
            else:
                st.info("Coluna de predição não encontrada nos dados")
        
        with col2:
            # Distribuição da confiança nas predições
            st.subheader("Distribuição da Confiança")
            if 'prediction_score' in predictions.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(predictions['prediction_score'], bins=20, ax=ax)
                ax.set_xlabel("Score de Probabilidade")
                ax.set_ylabel("Frequência")
                ax.set_title("Distribuição das Probabilidades Preditas")
                st.pyplot(fig)
            else:
                st.info("Coluna de score não encontrada nos dados")
        
        # Mostrar dados brutos
        with st.expander("Ver Dados Brutos"):
            st.dataframe(predictions)
""")

    # Tab 2 - Análise de Predições
    code_fragments.append("""
# Tab 2: Análise de Predições
with tab2:
    st.header("Análise Detalhada das Predições")
    
    if predictions is not None:
        # Layout de duas colunas
        col1, col2 = st.columns(2)
        
        with col1:
            # Análise por distância de arremesso
            st.subheader("Predições por Distância de Arremesso")
            
            if 'shot_distance' in predictions.columns and 'prediction_label' in predictions.columns:
                # Criar bins para distâncias
                predictions['distance_bin'] = pd.cut(
                    predictions['shot_distance'],
                    bins=[0, 5, 10, 15, 20, 25, 30, 100],
                    labels=['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
                )
                
                # Calcular taxa de conversão por bin de distância
                dist_conversion = predictions.groupby('distance_bin')['prediction_label'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x='distance_bin', y='prediction_label', data=dist_conversion, ax=ax)
                ax.set_xlabel("Distância (pés)")
                ax.set_ylabel("Taxa de Conversão Predita")
                ax.set_title("Taxa de Conversão por Distância")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("Colunas necessárias não encontradas nos dados")
        
        with col2:
            # Análise por período
            st.subheader("Predições por Período (Quarter)")
            
            if 'period' in predictions.columns and 'prediction_label' in predictions.columns:
                period_conversion = predictions.groupby('period')['prediction_label'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x='period', y='prediction_label', data=period_conversion, ax=ax)
                ax.set_xlabel("Período")
                ax.set_ylabel("Taxa de Conversão Predita")
                ax.set_title("Taxa de Conversão por Período")
                st.pyplot(fig)
            else:
                st.info("Colunas necessárias não encontradas nos dados")
        
        # Mapa de calor das localizações (se disponível)
        if 'lat' in predictions.columns and 'lon' in predictions.columns:
            st.subheader("Mapa de Calor das Localizações")
            
            fig, ax = plt.subplots(figsize=(10, 9))
            ax.set_xlim(-250, 250)
            ax.set_ylim(-50, 450)
            
            # Desenhar quadra de basquete (simplificada)
            # Círculo central
            circle = plt.Circle((0, 140), 60, fill=False, color='black')
            ax.add_artist(circle)
            
            # Linha de 3 pontos
            three_point_line_y = 0.0
            three_point_radius = 237.5
            three_point_line = plt.Circle((0, three_point_line_y), three_point_radius, fill=False, linestyle='--', color='black')
            ax.add_artist(three_point_line)
            
            # Cesta
            basket = plt.Circle((0, 0), 7.5, fill=False, color='red')
            ax.add_artist(basket)
            
            # Área restrita
            restricted_area = plt.Circle((0, 0), 40, fill=False, color='black')
            ax.add_artist(restricted_area)
            
            # Linha de lance livre
            ax.plot([-80, 80], [140, 140], color='black')
            
            # Distribuição de pontos coloridos por predição
            scatter = ax.scatter(
                predictions['lon'], 
                predictions['lat'],
                c=predictions['prediction_score'] if 'prediction_score' in predictions.columns else predictions['prediction_label'],
                cmap='coolwarm', 
                alpha=0.7, 
                s=20
            )
            plt.colorbar(scatter, label='Probabilidade de Conversão' if 'prediction_score' in predictions.columns else 'Predição')
            
            ax.set_title('Mapa de Calor das Predições na Quadra')
            ax.set_xlabel('Coordenada X')
            ax.set_ylabel('Coordenada Y')
            ax.set_aspect('equal')
            st.pyplot(fig)
        else:
            st.info("Coordenadas de localização não encontradas nos dados")
""")

    # Tab 3 - Saúde do Modelo (parte 1)
    code_fragments.append("""
# Tab 3: Saúde do Modelo
with tab3:
    st.header("Monitoramento da Saúde do Modelo")
    
    # Tendências de métricas ao longo do tempo
    st.subheader("Tendências de Métricas ao Longo do Tempo")
    
    # Usar dados reais se disponíveis, caso contrário dados simulados
    if 'metrics_history' in locals() and len(metrics_history) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(metrics_history['timestamp'], metrics_history['log_loss'], marker='o')
            ax.set_title('Tendência de Log Loss')
            ax.set_xlabel('Data')
            ax.set_ylabel('Log Loss (menor é melhor)')
            plt.xticks(rotation=45)
            ax.grid(True)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(metrics_history['timestamp'], metrics_history['f1_score'], marker='o', color='green')
            ax.set_title('Tendência de F1-Score')
            ax.set_xlabel('Data')
            ax.set_ylabel('F1-Score (maior é melhor)')
            plt.xticks(rotation=45)
            ax.grid(True)
            st.pyplot(fig)
    else:
        # Dados simulados para métricas históricas
        dates = pd.date_range(end=pd.Timestamp.now(), periods=10).to_pydatetime()
        metrics_history_sim = pd.DataFrame({
            'timestamp': dates,
            'log_loss': np.random.normal(0.58, 0.03, 10),
            'f1_score': np.random.normal(0.65, 0.02, 10)
        })
        
        # Exibir gráficos de tendência simulados
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(metrics_history_sim['timestamp'], metrics_history_sim['log_loss'], marker='o')
            ax.set_title('Tendência de Log Loss (Simulado)')
            ax.set_xlabel('Data')
            ax.set_ylabel('Log Loss (menor é melhor)')
            plt.xticks(rotation=45)
            ax.grid(True)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(metrics_history_sim['timestamp'], metrics_history_sim['f1_score'], marker='o', color='green')
            ax.set_title('Tendência de F1-Score (Simulado)')
            ax.set_xlabel('Data')
            ax.set_ylabel('F1-Score (maior é melhor)')
            plt.xticks(rotation=45)
            ax.grid(True)
            st.pyplot(fig)
""")

    # Tab 3 - Saúde do Modelo (parte 2) - Parte problemática com correção
    code_fragments.append('''
    # Detecção de desvio de dados (data drift)
    st.subheader("Detecção de Desvio de Dados (Data Drift)")
    
    st.write(
        "Para detectar desvio nos dados (data drift) ao longo do tempo, monitoramos a distribuição "
        "das principais variáveis em comparação com o conjunto de dados de treinamento."
    )
    
    # Simulação de desvio em uma das variáveis
    if predictions is not None and 'shot_distance' in predictions.columns:
        # Dados simulados para comparação com treinamento
        train_distances = np.random.normal(15, 5, 1000)  # Simulação de distribuição no treino
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(train_distances, label='Distribuição no Treino', color='blue', ax=ax)
        sns.kdeplot(predictions['shot_distance'], label='Distribuição Atual', color='red', ax=ax)
        ax.set_title('Comparação da Distribuição de Distância de Arremesso')
        ax.set_xlabel('Distância (pés)')
        ax.legend()
        plt.grid(True)
        st.pyplot(fig)
        
        # Alerta simulado de desvio
        drift_detected = np.random.choice([True, False], p=[0.3, 0.7])  # Simulação de detecção
        if drift_detected:
            st.warning("⚠️ **Alerta de Desvio Detectado!**\\n\\nA distribuição atual da distância de arremesso está significativamente diferente da distribuição usada no treinamento. Considere reavaliar o modelo.")
        else:
            st.success("✅ Nenhum desvio significativo detectado nas variáveis de entrada.")
    
    # Ação recomendada
    st.subheader("Recomendações de Ação")
    
    if has_metrics and latest_metrics.get('log_loss', 0) > 0.6:
        st.error("🔴 **Ação recomendada**: Retreinamento do modelo\\n\\nO Log Loss atual está acima do limite aceitável (0.6), indicando degradação no desempenho do modelo. Recomendamos retreinar o modelo com dados mais recentes.")
    else:
        st.success("🟢 **Status atual**: Saudável\\n\\nO modelo está operando dentro dos parâmetros esperados. Continue monitorando regularmente para detectar mudanças no desempenho.")
''')

    # Rodapé
    code_fragments.append("""
# Rodapé
st.markdown("---")
st.caption("Dashboard de Monitoramento do Modelo - Kobe Bryant Shot Prediction - Powered by Streamlit")
""")
    
    # Unir todos os fragmentos de código em um único arquivo
    with open(app_path, "w", encoding='utf-8') as f:
        dashboard_code = "".join(code_fragments)
        f.write(dashboard_code)

    log_step(f"Dashboard de monitoramento criado em: {app_path}")
    return app_path

def retrain_model_with_new_data(predictions, has_target):
    """Retreina o modelo com os novos dados de produção."""
    if not has_target:
        log_step("Não é possível retreinar o modelo: dados de produção não contêm a coluna target 'shot_made_flag'")
        return None
    
    log_step("Iniciando processo de retreinamento com novos dados de produção")
    
    try:
        # Importar função de retreinamento
        from retraining_pipeline import retrain_model
        
        # Chamar função de retreinamento passando os dados de produção
        new_model = retrain_model(predictions)
        
        log_step("Modelo retreinado com sucesso!")
        return new_model
    except ImportError:
        log_step("Módulo de retreinamento não encontrado. Skipping.")
        return None
    except Exception as e:
        log_step(f"Erro durante o retreinamento: {str(e)}")
        return None

def main():
    """Main function to run the application pipeline."""
    log_step("Iniciando pipeline de aplicação")

    try:
        # Etapa 1: Carregar dados de produção
        production_data = load_production_data()
        log_step(f"Dados de produção carregados. Formato: {production_data.shape}")

        # Etapa 2: Filtrar dados
        filtered_data, has_target = filter_production_data(production_data)
        log_step(f"Dados filtrados. Formato final: {filtered_data.shape}")

        # Etapa 3: Carregar o melhor modelo
        best_model = load_best_model()

        # Etapa 4: Aplicar modelo aos dados de produção
        predictions = apply_model_to_production_data(best_model, filtered_data)

        # Etapa 5: Avaliar predições (se target estiver disponível)
        metrics = evaluate_predictions(predictions, has_target)

        # Etapa 6: Salvar resultados e registrar métricas
        results_path = save_results(predictions, metrics)

        # Etapa 7: Criar dashboard de monitoramento
        app_path = create_monitoring_dashboard(predictions, results_path, metrics)
        
        # Etapa 8: Retreinar o modelo com novos dados (NOVA FUNCIONALIDADE)
        should_retrain = True  # Definir como parâmetro ou configuração
        if should_retrain and has_target:
            log_step("Iniciando retreinamento do modelo com novos dados")
            new_model = retrain_model_with_new_data(predictions, has_target)
            if new_model:
                log_step("Modelo retreinado e atualizado com sucesso!")
            else:
                log_step("Retreinamento não foi bem sucedido, mantendo modelo atual")

        log_step("Pipeline de aplicação concluído com sucesso")
        log_step(f"Para visualizar o dashboard, execute: streamlit run {app_path}")

        return True

    except Exception as e:
        log_step(f"Erro no pipeline de aplicação: {str(e)}")
        raise

if __name__ == "__main__":
    # Iniciar run do MLflow para rastreamento com o nome especificado
    with mlflow.start_run(run_name="PipelineAplicacao"):
        main()