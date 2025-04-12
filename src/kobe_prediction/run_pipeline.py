"""
Main Pipeline Runner for Kobe Bryant Shot Prediction Project.
This module coordinates the execution of all pipeline components.
"""

import argparse
import importlib
import logging
import mlflow
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def log_step(message):
    """Log a step to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as f:
        f.write(f"[{timestamp}] PIPELINE: {message}\n")
    logger.info(message)

def run_pipeline_component(component_name, parent_run_id=None):
    """Run a specific pipeline component by importing its module and executing the main function."""
    log_step(f"Running pipeline component: {component_name}")
    
    module_mapping = {
        'data_processing': 'src.kobe_prediction.pipelines.data_processing.run',
        'train_regression': 'src.kobe_prediction.pipelines.model_training.run_regression',
        'train_classification': 'src.kobe_prediction.pipelines.model_training.run_classification',
        'evaluate': 'src.kobe_prediction.pipelines.model_evaluation.run',
        'deploy': 'src.kobe_prediction.pipelines.deployment.run'
    }
    
    if component_name not in module_mapping:
        log_step(f"Unknown pipeline component: {component_name}")
        return False
    
    try:
        # Import the module
        module_name = module_mapping[component_name]
        module = importlib.import_module(module_name)
        
        # Execute the main function
        if hasattr(module, 'main'):
            # Use nested runs if we're in a parent run
            if parent_run_id:
                with mlflow.start_run(run_name=f"pipeline_{component_name}", nested=True):
                    result = module.main()
            else:
                # Make sure there are no active runs before starting a new one
                if mlflow.active_run():
                    mlflow.end_run()
                    
                with mlflow.start_run(run_name=f"pipeline_{component_name}"):
                    result = module.main()
                    
            log_step(f"Completed pipeline component: {component_name}, result: {result}")
            return result
        else:
            log_step(f"No main function found in module: {module_name}")
            return False
    except Exception as e:
        log_step(f"Error running pipeline component {component_name}: {str(e)}")
        # Make sure we end any active run on error
        if mlflow.active_run():
            mlflow.end_run()
        return False

def initialize_mlflow():
    """Initialize MLflow experiment."""
    log_step("Initializing MLflow experiment")
    
    # Set up MLflow tracking
    if not os.path.exists("mlruns"):
        os.makedirs("mlruns")
    
    # Set the tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create or get the experiment
    experiment_name = "kobe_shot_prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        log_step(f"Creating new experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        log_step(f"Using existing experiment: {experiment_name}")
        experiment_id = experiment.experiment_id
    
    # Set as active experiment
    mlflow.set_experiment(experiment_name)
    
    # End any active run that might exist
    if mlflow.active_run():
        mlflow.end_run()
    
    return experiment_id

def run_full_pipeline():
    """Run the complete pipeline from data processing to deployment."""
    log_step("Starting full pipeline execution")
    
    # Initialize MLflow
    experiment_id = initialize_mlflow()
    log_step(f"Using experiment ID: {experiment_id}")
    
    # Define the pipeline components in order
    components = [
        'data_processing',
        'train_regression',
        'train_classification',
        'evaluate',
        'deploy'
    ]
    
    # Run each component sequentially
    # Make sure there are no active runs
    if mlflow.active_run():
        mlflow.end_run()
        
    with mlflow.start_run(run_name="full_pipeline") as parent_run:
        parent_run_id = parent_run.info.run_id
        success = True
        
        for component in components:
            log_step(f"Starting component: {component}")
            result = run_pipeline_component(component, parent_run_id=parent_run_id)
            if not result:
                log_step(f"Pipeline component {component} failed. Stopping pipeline.")
                success = False
                break
            log_step(f"Component {component} completed successfully")
    
    if success:
        log_step("Full pipeline completed successfully")
        log_step("To run the Streamlit app, execute: streamlit run app/kobe_shot_predictor_app.py")
    else:
        log_step("Pipeline execution failed")
    
    return success

def main():
    """Main function to parse arguments and run the appropriate pipeline component."""
    parser = argparse.ArgumentParser(description='Kobe Bryant Shot Prediction Pipeline Runner')
    parser.add_argument(
        'component', 
        nargs='?', 
        default='full',
        choices=['full', 'data_processing', 'train_regression', 'train_classification', 'evaluate', 'deploy'],
        help='Pipeline component to run (default: full pipeline)'
    )
    
    args = parser.parse_args()
    component = args.component
    
    # Initialize MLflow
    initialize_mlflow()
    
    if component == 'full':
        return run_full_pipeline()
    else:
        return run_pipeline_component(component)

if __name__ == "__main__":
    main()