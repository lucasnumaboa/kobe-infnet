
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
