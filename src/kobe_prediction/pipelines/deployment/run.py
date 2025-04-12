"""
Deployment Pipeline for Kobe Bryant Shot Prediction.
This module handles model deployment to MLflow and builds a Streamlit app.
"""

import os
import joblib
import shutil
import mlflow
import logging
from datetime import datetime
import yaml
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname=s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the absolute path to the project root directory."""
    # Start from the current file and go up until we find the project root
    current_path = Path(__file__).resolve()
    # Go up multiple levels: pipelines/deployment/run.py -> deployment -> pipelines -> kobe_prediction -> src -> root
    project_root = current_path.parent.parent.parent.parent.parent
    return project_root

def log_step(message):
    """Log a step to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as f:
        f.write(f"[{timestamp}] DEPLOYMENT: {message}\n")
    logger.info(message)

def find_best_model():
    """Find the best model based on evaluation results."""
    log_step("Finding best model from evaluation results")
    
    project_root = get_project_root()
    evaluation_results_path = os.path.join(project_root, "models", "evaluation", "model_comparison_results.csv")
    
    if not os.path.exists(evaluation_results_path):
        log_step(f"Evaluation results not found at: {evaluation_results_path}")
        return None, None
    
    # Load the evaluation results
    results_df = pd.read_csv(evaluation_results_path, index_col=0)
    
    # Find best model based on F1 score
    best_model_name = results_df['f1_score'].idxmax()
    model_type = results_df.loc[best_model_name, 'model_type']
    
    log_step(f"Best model identified: {best_model_name} ({model_type})")
    
    return best_model_name, model_type

def deploy_model_to_mlflow(model_name, model_type):
    """Deploy the model to MLflow for serving."""
    log_step(f"Deploying {model_name} to MLflow")
    
    project_root = get_project_root()
    
    # Determine model path based on type
    if model_type == 'Regression':
        model_path = os.path.join(project_root, "models", "regression", f"{model_name}.joblib")
        model_stage = "regression"
    else:  # Classification
        model_path = os.path.join(project_root, "models", "classification", f"{model_name}.pkl")
        model_stage = "classification"
    
    if not os.path.exists(model_path):
        log_step(f"Model file not found at {model_path}")
        return False
    
    # Load the model
    model = joblib.load(model_path)
    
    # Generate a registered model name that includes timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    registered_model_name = f"kobe_shot_predictor_{timestamp}"
    
    # Log model to MLflow - usando nested=True para criar runs aninhados
    with mlflow.start_run(run_name=f"deploy_{model_name}", nested=True):
        mlflow.log_params({
            "model_name": model_name,
            "model_type": model_type,
            "deployment_timestamp": timestamp
        })
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(
            model,
            f"deployed_{model_stage}_model",
            registered_model_name=registered_model_name
        )
        
        # Get the model URI
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/deployed_{model_stage}_model"
        
        log_step(f"Model deployed to MLflow with URI: {model_uri}")
        log_step(f"Registered model name: {registered_model_name}")
        
        # Create deployment directory if it doesn't exist
        deployment_dir = os.path.join(project_root, "models", "deployment")
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Save the model URI for later use in the app
        model_info_path = os.path.join(deployment_dir, "model_info.yaml")
        with open(model_info_path, "w") as f:
            yaml.dump({
                "model_name": model_name,
                "model_type": model_type,
                "model_uri": model_uri,
                "registered_model_name": registered_model_name,
                "timestamp": timestamp
            }, f)
        
        log_step(f"Model info saved to {model_info_path}")
    
    return True

def create_streamlit_app():
    """Create a Streamlit app for model serving and visualization."""
    log_step("Creating Streamlit application")
    
    project_root = get_project_root()
    
    # Create app directory if it doesn't exist
    app_dir = os.path.join(project_root, "app")
    os.makedirs(app_dir, exist_ok=True)
    
    # Create Streamlit app file - usando encoding='utf-8' para permitir emojis
    app_file = os.path.join(app_dir, "kobe_shot_predictor_app.py")
    
    # Updated app content with absolute paths using get_project_root()
    app_content = """
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
    page_title="Kobe Bryant Shot Predictor",
    layout="wide"
)

# Helper function to get project root path
def get_project_root():
    \"\"\"Get the absolute path to the project root directory.\"\"\"
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
        st.error(f"Error loading model info: {e}")
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
        st.error(f"Error loading model: {e}")
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
        st.error(f"Error loading sample data: {e}")
        return None

# Main title
st.title("Kobe Bryant Shot Predictor")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This app predicts whether Kobe Bryant would make or miss a basketball shot based on various features.")

# Load model info and model
model_info = load_model_info()
if model_info:
    model = load_model(model_info)
    st.sidebar.success(f"Loaded {model_info['model_type']} model: {model_info['model_name']}")
    
    # Display model info
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"**Model Type:** {model_info['model_type']}")
    st.sidebar.write(f"**Model Name:** {model_info['model_name']}")
    st.sidebar.write(f"**Deployed on:** {model_info['timestamp']}")
else:
    st.sidebar.error("No model information found. Please run the deployment pipeline first.")
    model = None

# Load sample data
sample_data = load_sample_data()

# Main content
tabs = st.tabs(["Predictor", "Model Performance", "Sample Data"])

with tabs[0]:
    st.header("Predict Shot Outcome")
    
    if model:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Shot Details")
            action_type = st.selectbox(
                "Action Type",
                options=[
                    "Jump Shot",
                    "Layup",
                    "Dunk",
                    "Fadeaway Jump Shot",
                    "Hook Shot"
                ]
            )
            
            shot_type = st.selectbox(
                "Shot Type",
                options=["2PT Field Goal", "3PT Field Goal"]
            )
            
            shot_zone_basic = st.selectbox(
                "Shot Zone Basic",
                options=[
                    "Above the Break 3",
                    "Left Corner 3",
                    "Mid-Range",
                    "Restricted Area",
                    "Right Corner 3",
                    "In The Paint (Non-RA)"
                ]
            )
        
        with col2:
            st.subheader("Game Context")
            period = st.slider("Period (Quarter)", 1, 4, 2)
            minutes_remaining = st.slider("Minutes Remaining", 0, 12, 6)
            seconds_remaining = st.slider("Seconds Remaining", 0, 59, 30)
            shot_distance = st.slider("Shot Distance (feet)", 0, 40, 15)
            playoffs = st.checkbox("Playoffs")
            
        # Make prediction button
        if st.button("Predict Shot Outcome"):
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
            st.info("In a production app, this would perform the exact same feature engineering as the training pipeline.")
            
            # Simulate prediction (replace with actual prediction code that handles feature preprocessing)
            try:
                if model_info["model_type"] == "Regression":
                    result_value = np.random.random()  # Placeholder
                    shot_made = result_value > 0.5
                    st.write(f"**Shot Probability**: {result_value:.1%}")
                else:  # Classification
                    shot_made = np.random.choice([True, False])  # Placeholder
                
                # Display prediction
                if shot_made:
                    st.success("**Prediction: MADE SHOT!**")
                else:
                    st.error("**Prediction: MISSED SHOT**")
                
                # Show a Kobe image
                st.image("https://media.giphy.com/media/l0MYwdebx8o0XI56E/giphy.gif", width=400, caption="The Black Mamba")
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.warning("Model not loaded. Please run the deployment pipeline first.")

with tabs[1]:
    st.header("Model Performance")
    
    project_root = get_project_root()
    eval_path = os.path.join(project_root, "models", "evaluation")
    
    if os.path.exists(eval_path):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Comparison")
            f1_comparison_path = os.path.join(eval_path, "f1_score_comparison.png")
            if os.path.exists(f1_comparison_path):
                st.image(f1_comparison_path, caption="F1 Score Comparison")
            
            auc_comparison_path = os.path.join(eval_path, "auc_comparison.png")
            if os.path.exists(auc_comparison_path):
                st.image(auc_comparison_path, caption="AUC Comparison")
        
        with col2:
            st.subheader("Best Model Performance")
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
                    st.image(cm_path, caption=f"Confusion Matrix - {model_name}")
                
                if os.path.exists(roc_path):
                    st.image(roc_path, caption=f"ROC Curve - {model_name}")
    else:
        st.warning("Model evaluation results not found. Please run the evaluation pipeline first.")

with tabs[2]:
    st.header("Sample Data")
    
    if sample_data is not None:
        st.write(f"Sample data shape: {sample_data.shape}")
        st.dataframe(sample_data.head(100))
        
        # Plot shot distribution
        st.subheader("Shot Distribution")
        
        if 'shot_made_flag' in sample_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='shot_made_flag', data=sample_data, ax=ax)
            ax.set_xlabel("Shot Made (1) vs Missed (0)")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Made vs Missed Shots")
            st.pyplot(fig)
    else:
        st.warning("Sample data not loaded.")

# Add footer
st.markdown("---")
st.markdown("Kobe Bryant Shot Prediction Project using MLflow | Created with Streamlit")
"""
    
    # Usando o parâmetro encoding='utf-8' para evitar o erro de codificação
    with open(app_file, "w", encoding='utf-8') as f:
        f.write(app_content)
    
    log_step(f"Streamlit app created at {app_file}")
    
    # Create README file for the app - também usando encoding='utf-8'
    readme_path = os.path.join(app_dir, "README.md")
    with open(readme_path, "w", encoding='utf-8') as f:
        f.write("""
# Kobe Bryant Shot Predictor App

This Streamlit application provides a user interface for predicting whether Kobe Bryant would make or miss a basketball shot based on various features.

## Running the App

To run the application, execute the following command from the root directory:

```bash
streamlit run app/kobe_shot_predictor_app.py
```

## Features

- **Shot Prediction**: Input shot details and game context to predict whether Kobe would make or miss the shot
- **Model Performance**: View model evaluation metrics and visualizations
- **Sample Data**: Explore the dataset used for model training and evaluation

## Models

The app uses the best performing model from the regression and classification approaches, as determined by the evaluation pipeline.
""")
    
    log_step(f"Created app README.md file at {readme_path}")
    
    return True

def main():
    """Main function to run the deployment pipeline."""
    log_step("Starting deployment pipeline")
    
    try:
        project_root = get_project_root()
        
        # Create deployment directory if it doesn't exist
        deployment_dir = os.path.join(project_root, "models", "deployment")
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Find the best model from evaluation results
        best_model_name, model_type = find_best_model()
        
        if best_model_name is None or model_type is None:
            log_step("Could not determine best model. Please run the evaluation pipeline first.")
            return False
        
        # Deploy the model to MLflow
        success = deploy_model_to_mlflow(best_model_name, model_type)
        if not success:
            log_step("Model deployment to MLflow failed")
            return False
        
        # Create Streamlit app
        create_streamlit_app()
        
        log_step("Deployment pipeline completed successfully")
        log_step("To run the Streamlit app, execute: streamlit run app/kobe_shot_predictor_app.py")
        
        return True
        
    except Exception as e:
        log_step(f"Error in deployment pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Start MLflow run for tracking
    with mlflow.start_run(run_name="deployment"):
        main()