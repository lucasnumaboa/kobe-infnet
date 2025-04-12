import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

# Set page title and layout
st.set_page_config(
    page_title="Preditor de Arremessos do Kobe Bryant",
    layout="wide"
)

# Helper function to get project root path
def get_project_root():
    """Get the absolute path to the project root directory."""
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent
    return project_root

# Load model info
@st.cache_resource
def load_model_info():
    try:
        project_root = get_project_root()
        model_info_path = os.path.join(project_root, "models", "deployment", "model_info.yaml")
        with open(model_info_path, "r") as f:
            model_info = yaml.safe_load(f)
        return model_info
    except Exception as e:
        st.error(f"Erro ao carregar informações do modelo: {e}")
        return None

@st.cache_resource
def load_model(model_info):
    try:
        project_root = get_project_root()
        # For local development, load directly from the joblib file
        if model_info["model_type"] == "Regression":
            model_path = os.path.join(project_root, "models", "regression", f"{model_info['model_name']}.joblib")
            model = joblib.load(model_path)
        else:
            model_path = os.path.join(project_root, "models", "classification", f"{model_info['model_name']}.pkl")
            model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

# Load sample data for reference
@st.cache_data
def load_sample_data():
    try:
        project_root = get_project_root()
        data_path = os.path.join(project_root, "data", "processed", "base_test.parquet")
        df = pd.read_parquet(data_path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados de exemplo: {e}")
        return None

# Main title
st.title("Preditor de Arremessos do Kobe Bryant")

# Sidebar
st.sidebar.title("Sobre")
st.sidebar.info("Este aplicativo prevê se o Kobe Bryant acertaria ou erraria um arremesso de basquete com base em diversas características.")

# Load model info and model
model_info = load_model_info()
if model_info:
    model = load_model(model_info)
    st.sidebar.success(f"Modelo {model_info['model_type']} carregado: {model_info['model_name']}")
    
    # Display model info
    st.sidebar.subheader("Informações do Modelo")
    st.sidebar.write(f"**Tipo de Modelo:** {model_info['model_type']}")
    st.sidebar.write(f"**Nome do Modelo:** {model_info['model_name']}")
    st.sidebar.write(f"**Implantado em:** {model_info['timestamp']}")
else:
    st.sidebar.error("Nenhuma informação de modelo encontrada. Por favor, execute o pipeline de implantação primeiro.")
    model = None

# Load sample data
sample_data = load_sample_data()

# Main content
tabs = st.tabs(["Preditor", "Desempenho do Modelo", "Dados de Exemplo"])

with tabs[0]:
    st.header("Prever Resultado do Arremesso")
    
    if model:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detalhes do Arremesso")
            action_type = st.selectbox(
                "Tipo de Ação",
                options=[
                    "Jump Shot",
                    "Layup",
                    "Dunk",
                    "Fadeaway Jump Shot",
                    "Hook Shot"
                ]
            )
            
            shot_type = st.selectbox(
                "Tipo de Arremesso",
                options=["Lance de 2 Pontos", "Lance de 3 Pontos"]
            )
            
            shot_zone_basic = st.selectbox(
                "Zona Básica de Arremesso",
                options=[
                    "Acima do Intervalo 3",
                    "Canto Esquerdo 3",
                    "Média Distância",
                    "Área Restrita",
                    "Canto Direito 3",
                    "Na Pintura (Não-RA)"
                ]
            )
        
        with col2:
            st.subheader("Contexto do Jogo")
            period = st.slider("Período (Quarto)", 1, 4, 2)
            minutes_remaining = st.slider("Minutos Restantes", 0, 12, 6)
            seconds_remaining = st.slider("Segundos Restantes", 0, 59, 30)
            shot_distance = st.slider("Distância do Arremesso (pés)", 0, 40, 15)
            playoffs = st.checkbox("Playoffs")
            
        # Make prediction button
        if st.button("Prever Resultado"):
            # Create features based on inputs
            # Note: This is a simplified example - in a real app, we'd need to
            # match the exact feature engineering done during training
            input_features = pd.DataFrame({
                'period': [period],
                'minutes_remaining': [minutes_remaining],
                'seconds_remaining': [seconds_remaining],
                'shot_distance': [shot_distance],
                'playoffs': [1 if playoffs else 0],
                'action_type': [action_type],
                'shot_type': [shot_type],
                'shot_zone_basic': [shot_zone_basic]
            })
            
            # For a real app, we'd need to one-hot encode categorical variables 
            # exactly as was done during training
            st.info("Em um aplicativo de produção, isso realizaria exatamente o mesmo pré-processamento de features que o pipeline de treinamento.")
            
            # Simulate prediction (replace with actual prediction code that handles feature preprocessing)
            try:
                if model_info["model_type"] == "Regression":
                    result_value = np.random.random()  # Placeholder
                    shot_made = result_value > 0.5
                    st.write(f"**Probabilidade de Acerto**: {result_value:.1%}")
                else:  # Classification
                    shot_made = np.random.choice([True, False])  # Placeholder
                
                # Display prediction
                if shot_made:
                    st.success("**Previsão: ARREMESSO CONVERTIDO!**")
                else:
                    st.error("**Previsão: ARREMESSO ERRADO**")
                
                # Show a Kobe image
                st.image("https://media.giphy.com/media/l0MYwdebx8o0XI56E/giphy.gif", width=400, caption="Black Mamba")
            
            except Exception as e:
                st.error(f"Erro ao fazer previsão: {e}")
    else:
        st.warning("Modelo não carregado. Por favor, execute o pipeline de implantação primeiro.")

with tabs[1]:
    st.header("Desempenho do Modelo")
    
    project_root = get_project_root()
    eval_path = os.path.join(project_root, "models", "evaluation")
    
    if os.path.exists(eval_path):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Comparação de Modelos")
            f1_comparison_path = os.path.join(eval_path, "f1_score_comparison.png")
            if os.path.exists(f1_comparison_path):
                st.image(f1_comparison_path, caption="Comparação de F1 Score")
            
            auc_comparison_path = os.path.join(eval_path, "auc_comparison.png")
            if os.path.exists(auc_comparison_path):
                st.image(auc_comparison_path, caption="Comparação de AUC")
        
        with col2:
            st.subheader("Desempenho do Melhor Modelo")
            if model_info:
                model_name = model_info["model_name"]
                model_type = model_info["model_type"]
                
                # Show confusion matrix of the best model
                if model_type == "Regression":
                    cm_path = os.path.join(eval_path, f"{model_name}_reg_confusion_matrix.png")
                    roc_path = os.path.join(eval_path, f"{model_name}_reg_roc_curve.png")
                else:
                    cm_path = os.path.join(eval_path, f"{model_name}_class_confusion_matrix.png")
                    roc_path = os.path.join(eval_path, f"{model_name}_class_roc_curve.png")
                
                if os.path.exists(cm_path):
                    st.image(cm_path, caption=f"Matriz de Confusão - {model_name}")
                
                if os.path.exists(roc_path):
                    st.image(roc_path, caption=f"Curva ROC - {model_name}")
    else:
        st.warning("Resultados de avaliação do modelo não encontrados. Por favor, execute o pipeline de avaliação primeiro.")

with tabs[2]:
    st.header("Dados de Exemplo")
    
    if sample_data is not None:
        st.write(f"Formato dos dados de exemplo: {sample_data.shape}")
        st.dataframe(sample_data.head(100))
        
        # Plot shot distribution
        st.subheader("Distribuição de Arremessos")
        
        if 'shot_made_flag' in sample_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='shot_made_flag', data=sample_data, ax=ax)
            ax.set_xlabel("Arremesso Convertido (1) vs Errado (0)")
            ax.set_ylabel("Contagem")
            ax.set_title("Distribuição de Arremessos Convertidos vs Errados")
            st.pyplot(fig)
    else:
        st.warning("Dados de exemplo não carregados.")

# Add footer
st.markdown("---")
st.markdown("Projeto de Previsão de Arremessos do Kobe Bryant usando MLflow | Criado com Streamlit")
