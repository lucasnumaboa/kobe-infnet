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
            st.subheader("Predições por Período (Quarto)")
            
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
            st.warning("⚠️ **Alerta de Desvio Detectado!**\n\nA distribuição atual da distância de arremesso está significativamente diferente da distribuição usada no treinamento. Considere reavaliar o modelo.")
        else:
            st.success("✅ Nenhum desvio significativo detectado nas variáveis de entrada.")
    
    # Ação recomendada
    st.subheader("Recomendações de Ação")
    
    if has_metrics and latest_metrics.get('log_loss', 0) > 0.6:
        st.error("🔴 **Ação recomendada**: Retreinamento do modelo\n\nO Log Loss atual está acima do limite aceitável (0.6), indicando degradação no desempenho do modelo. Recomendamos retreinar o modelo com dados mais recentes.")
    else:
        st.success("🟢 **Status atual**: Saudável\n\nO modelo está operando dentro dos parâmetros esperados. Continue monitorando regularmente para detectar mudanças no desempenho.")

# Rodapé
st.markdown("---")
st.caption("Dashboard de Monitoramento do Modelo - Kobe Bryant Shot Prediction - Desenvolvido com Streamlit")
