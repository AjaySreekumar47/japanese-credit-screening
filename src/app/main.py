#!/usr/bin/env python
"""
Japanese Credit Screening - Main Script

This script runs the complete pipeline for the Japanese Credit Screening project.
It downloads the data, performs preprocessing, trains models, evaluates performance,
and generates visualization and reports.
"""

import os
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
from src.data.download import download_dataset, download_metadata
from src.data.preprocessing import load_data, preprocess_data, get_processed_data
from src.features.build_features import create_features, get_feature_names
from src.models.train import train_model, train_all_models
from src.models.evaluate import evaluate_model, compare_models, find_optimal_threshold
from src.models.predict import load_model, predict_single_sample
from src.visualization.visualize import (
    plot_class_distribution, 
    plot_correlation_matrix, 
    plot_feature_distributions, 
    plot_categorical_features,
    plot_feature_importance,
    create_dashboard
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("credit_screening.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Japanese Credit Screening Pipeline')
    
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory for data files')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for output files')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory for model files')
    parser.add_argument('--visualizations-dir', type=str, default='visualizations',
                        help='Directory for visualization files')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--tune-hyperparams', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training (use existing models)')
    parser.add_argument('--best-model', type=str, default=None,
                        help='Best model to use for deployment')
    
    return parser.parse_args()

def setup_directories(args):
    """Create necessary directories."""
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.visualizations_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(args.data_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'evaluation'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'feature_importance'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'deployment'), exist_ok=True)
    
    logger.info("Directories created")

def download_data(args):
    """Download the dataset."""
    logger.info("Downloading dataset...")
    
    data_path = download_dataset(output_dir=os.path.join(args.data_dir, 'raw'))
    meta_path = download_metadata(output_dir=os.path.join(args.data_dir, 'raw'))
    
    if data_path and meta_path:
        logger.info("Dataset downloaded successfully")
        return data_path
    else:
        logger.error("Failed to download dataset")
        return None

def prepare_data(data_path, args):
    """Load and preprocess the data."""
    logger.info("Preparing data...")
    
    # Load data
    data = load_data(data_path)
    if data is None:
        logger.error("Failed to load data")
        return None, None, None, None, None
    
    # Save raw data
    data.to_csv(os.path.join(args.data_dir, 'processed', 'credit_data_raw.csv'), index=False)
    
    # Create additional features
    enhanced_data = create_features(data)
    if enhanced_data is not None:
        logger.info(f"Created features. Shape: {enhanced_data.shape}")
        enhanced_data.to_csv(os.path.join(args.data_dir, 'processed', 'credit_data_enhanced.csv'), index=False)
    else:
        logger.warning("Failed to create features, using original data")
        enhanced_data = data
    
    # Preprocess data
    preprocessor_path = os.path.join(args.models_dir, 'preprocessor.pkl')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        enhanced_data, 
        test_size=args.test_size, 
        random_state=args.random_state,
        save_preprocessor=preprocessor_path
    )
    
    if X_train is None:
        logger.error("Failed to preprocess data")
        return None, None, None, None, None
    
    logger.info("Data prepared successfully")
    return data, X_train, X_test, y_train, y_test, preprocessor

def explore_data(data, args):
    """Perform exploratory data analysis."""
    logger.info("Performing exploratory data analysis...")
    
    # Create visualizations
    plot_class_distribution(data, output_dir=args.visualizations_dir)
    plot_correlation_matrix(data, output_dir=args.visualizations_dir)
    plot_feature_distributions(data, output_dir=args.visualizations_dir)
    plot_categorical_features(data, output_dir=args.visualizations_dir)
    
    logger.info("EDA completed successfully")

def train_models(X_train, y_train, preprocessor, args):
    """Train predictive models."""
    if args.skip_training:
        logger.info("Skipping model training as requested")
        return None
    
    logger.info("Training models...")
    
    # Train all models
    models = train_all_models(
        X_train, y_train, 
        preprocessor=preprocessor,
        tune_hyperparams=args.tune_hyperparams,
        output_dir=args.models_dir
    )
    
    if not models:
        logger.error("Failed to train models")
        return None
    
    logger.info(f"Trained {len(models)} models successfully")
    return models

def load_trained_models(args):
    """Load pre-trained models."""
    logger.info("Loading trained models...")
    
    # Model types
    model_types = ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'SVM']
    
    # Dictionary to store loaded models
    models = {}
    
    # Load each model
    for model_type in model_types:
        model_path = os.path.join(args.models_dir, f'{model_type}_model.pkl')
        
        # Check if tuned model exists
        tuned_model_path = os.path.join(args.models_dir, f'{model_type}_tuned_model.pkl')
        if os.path.exists(tuned_model_path):
            model_path = tuned_model_path
            model_type = f"{model_type}_tuned"
        
        if os.path.exists(model_path):
            model = load_model(model_path)
            if model is not None:
                models[model_type] = model
                logger.info(f"Loaded {model_type} model")
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    if not models:
        logger.error("No models loaded")
        return None
    
    logger.info(f"Loaded {len(models)} models successfully")
    return models

def evaluate_models(models, X_test, y_test, args):
    """Evaluate models and compare performance."""
    if models is None:
        logger.error("No models to evaluate")
        return None
    
    logger.info("Evaluating model performance...")
    
    # Create evaluation directory
    eval_dir = os.path.join(args.output_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Compare models
    comparison = compare_models(models, X_test, y_test, output_dir=eval_dir)
    
    if comparison is None:
        logger.error("Failed to compare models")
        return None
    
    # Save comparison to CSV
    comparison.to_csv(os.path.join(eval_dir, 'model_comparison.csv'), index=False)
    
    # Determine best model if not specified
    best_model_name = args.best_model
    if best_model_name is None:
        best_model_name = comparison.loc[comparison['Accuracy'].idxmax(), 'Model']
    
    logger.info(f"Best model: {best_model_name}")
    
    # Find optimal threshold for best model
    if best_model_name in models:
        best_model = models[best_model_name]
        optimal_threshold = find_optimal_threshold(
            best_model, X_test, y_test, 
            criterion='f1', 
            output_dir=os.path.join(eval_dir, 'threshold_analysis')
        )
        logger.info(f"Optimal threshold for {best_model_name}: {optimal_threshold:.2f}")
    
    # Create feature importance visualizations
    for name, model in models.items():
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            
            # For tree-based models
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                feature_names = get_feature_names(model.named_steps['preprocessor'])
                
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Save to CSV
                feature_importance_df.to_csv(
                    os.path.join(args.output_dir, 'feature_importance', f'{name}_feature_importance.csv'),
                    index=False
                )
                
                # Plot
                plot_feature_importance(
                    feature_importance_df, 
                    model_name=name,
                    output_dir=os.path.join(args.output_dir, 'feature_importance')
                )
            
            # For linear models
            elif hasattr(classifier, 'coef_'):
                coefficients = classifier.coef_[0]
                feature_names = get_feature_names(model.named_steps['preprocessor'])
                
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                })
                feature_importance_df['Abs_Coefficient'] = feature_importance_df['Coefficient'].abs()
                feature_importance_df = feature_importance_df.sort_values('Abs_Coefficient', ascending=False)
                
                # Save to CSV
                feature_importance_df.to_csv(
                    os.path.join(args.output_dir, 'feature_importance', f'{name}_coefficients.csv'),
                    index=False
                )
                
                # Plot
                plot_feature_importance(
                    feature_importance_df, 
                    model_name=name,
                    output_dir=os.path.join(args.output_dir, 'feature_importance')
                )
    
    logger.info("Model evaluation completed successfully")
    return comparison, best_model_name

def create_deployment_package(models, best_model_name, preprocessor, args):
    """Create a deployment package for the best model."""
    if models is None or best_model_name not in models:
        logger.error("Best model not available")
        return
    
    logger.info(f"Creating deployment package for {best_model_name}...")
    
    # Directory for deployment
    deploy_dir = os.path.join(args.output_dir, 'deployment')
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Get the best model
    best_model = models[best_model_name]
    
    # Save the model
    import joblib
    joblib.dump(best_model, os.path.join(deploy_dir, 'best_model.pkl'))
    
    # Create a prediction script
    prediction_script = f"""
import joblib
import pandas as pd
from datetime import datetime

# Load the model
model = joblib.load('best_model.pkl')

def predict_credit_approval(data):
    \"\"\"
    Predict credit approval for the input data.
    
    Args:
        data (dict): Input data as a dictionary with feature names as keys
        
    Returns:
        dict: Prediction results
    \"\"\"
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    # Get probability
    probabilities = model.predict_proba(df)[0]
    probability = probabilities[1]  # Probability of positive class
    
    # Return results
    result = {{
        'approved': bool(prediction),
        'approval_probability': float(probability),
        'timestamp': datetime.now().isoformat()
    }}
    
    return result

# Example usage
if __name__ == '__main__':
    # Example input data (adjust feature names based on your dataset)
    sample_data = {{
        'A1': 'b', 
        'A2': 30.0,
        # Add other features here as needed
    }}
    
    # Get prediction
    result = predict_credit_approval(sample_data)
    print(f"Approved: {{result['approved']}}")
    print(f"Approval Probability: {{result['approval_probability']:.4f}}")
    print(f"Timestamp: {{result['timestamp']}}")
"""
    
    # Save the prediction script
    with open(os.path.join(deploy_dir, 'predict.py'), 'w') as f:
        f.write(prediction_script)
    
    # Create a README file
    readme_content = f"""
# Japanese Credit Screening - Deployment Package

## Overview
This package contains a trained model for credit screening based on the Japanese Credit Screening dataset from UCI.

## Model Information
- Model: {best_model_name}
- Features: {len(get_feature_names(preprocessor))} features after preprocessing
- Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Files
- `best_model.pkl`: The trained model
- `predict.py`: Script for making predictions
- `requirements.txt`: Required dependencies

## Usage
```python
from predict import predict_credit_approval

# Example input data
data = {{
    'A1': 'b',
    'A2': 30.0,
    # Add other features as needed
}}

# Get prediction
result = predict_credit_approval(data)
print(f"Approved: {{result['approved']}}")
print(f"Approval Probability: {{result['approval_probability']:.4f}}")
```

## Requirements
- Python 3.7+
- scikit-learn
- pandas
- numpy
- joblib
"""
    
    # Save the README
    with open(os.path.join(deploy_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    # Create a requirements file
    requirements_content = """
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
joblib>=1.0.0
"""
    
    # Save the requirements
    with open(os.path.join(deploy_dir, 'requirements.txt'), 'w') as f:
        f.write(requirements_content)
    
    logger.info(f"Deployment package created in {deploy_dir}")

def create_dashboard_visualizations(data, models, comparison, args):
    """Create a dashboard with key visualizations."""
    logger.info("Creating dashboard visualizations...")
    
    # Prepare model results for dashboard
    model_results = {
        'comparison': comparison
    }
    
    # Get feature importance for the best model if available
    if comparison is not None:
        best_model_name = comparison.loc[comparison['Accuracy'].idxmax(), 'Model']
        
        if best_model_name in models:
            best_model = models[best_model_name]
            
            if hasattr(best_model, 'named_steps') and 'classifier' in best_model.named_steps:
                classifier = best_model.named_steps['classifier']
                
                if hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_
                    feature_names = get_feature_names(best_model.named_steps['preprocessor'])
                    
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    model_results['feature_importance'] = feature_importance_df
    
    # Create dashboard
    create_dashboard(
        data, 
        model_results=model_results, 
        output_dir=args.visualizations_dir
    )
    
    logger.info("Dashboard created successfully")

def generate_final_report(args, comparison=None, best_model_name=None):
    """Generate a final report summarizing the project."""
    logger.info("Generating final report...")
    
    # Create report directory
    report_dir = os.path.join(args.output_dir, 'report')
    os.makedirs(report_dir, exist_ok=True)
    
    # Report content
    report_content = """
# Japanese Credit Screening Project - Final Report

## 1. Introduction
This report presents the results of the Japanese Credit Screening project, which aims to develop models for predicting credit approval based on applicant attributes.

## 2. Dataset
The Japanese Credit Screening dataset from the UCI Machine Learning Repository was used for this project. The dataset contains information about credit applications, including both categorical and numerical features that represent various attributes of the applicants.

## 3. Methodology
The following steps were performed in this project:
1. Data preprocessing and feature engineering
2. Statistical analysis
3. Model training and evaluation
4. Feature importance analysis
5. Model deployment

## 4. Results
"""
    
    # Add model comparison if available
    if comparison is not None:
        report_content += """
### 4.1 Model Performance
The following models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
"""
        
        for _, row in comparison.iterrows():
            report_content += f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1 Score']:.4f} |\n"
        
        if best_model_name is not None:
            report_content += f"\nThe best performing model was **{best_model_name}**.\n"
    
    report_content += """
### 4.2 Key Findings
- [List key findings from the analysis]
- [Discuss important features for credit approval]
- [Include other insights discovered during the project]

## 5. Deployment
A deployment package was created for the best performing model. The package includes:
- The trained model
- A prediction script
- Documentation on how to use the model

## 6. Conclusion
This project demonstrated the application of machine learning techniques for credit risk assessment. The models developed can be used to predict credit approval with good accuracy.

## 7. Next Steps
- [Suggest improvements to the current approach]
- [Recommend additional features to collect]
- [Propose strategies for model monitoring and updating]
"""
    
    # Save the report
    with open(os.path.join(report_dir, 'final_report.md'), 'w') as f:
        f.write(report_content)
    
    logger.info(f"Final report generated in {report_dir}")

def main():
    """Main function to run the pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup directories
    setup_directories(args)
    
    # Download data
    data_path = download_data(args)
    if data_path is None:
        return
    
    # Prepare data
    data, X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_path, args)
    if data is None:
        return
    
    # Explore data
    explore_data(data, args)
    
    # Train or load models
    models = None
    if args.skip_training:
        models = load_trained_models(args)
    else:
        models = train_models(X_train, y_train, preprocessor, args)
    
    if models is None:
        return
    
    # Evaluate models
    eval_results = evaluate_models(models, X_test, y_test, args)
    if eval_results is None:
        return
    
    comparison, best_model_name = eval_results
    
    # Create deployment package
    create_deployment_package(models, best_model_name, preprocessor, args)
    
    # Create dashboard visualizations
    create_dashboard_visualizations(data, models, comparison, args)
    
    # Generate final report
    generate_final_report(args, comparison, best_model_name)
    
    logger.info("Pipeline completed successfully")
    print(f"All outputs saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
_model
    if best_model_name is None:
        best_model_name = comparison.loc[comparison['Accuracy'].idxmax(), 'Model']
    
    logger.info(f"Best model: {best_model_name}")
    
    # Find optimal threshold for best model
    if best_model_name in models:
        best_model = models[best_model_name]
        optimal_threshold = find_optimal_threshold(
            best_model, X_test, y_test, 
            criterion='f1', 
            output_dir=os.path.join(eval_dir, 'threshold_analysis')
        )
        logger.info(f"Optimal threshold for {best_model_name}: {optimal_threshold:.2f}")
    
    # Create feature importance visualizations
    for name, model in models.items():
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            
            # For tree-based models
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                feature_names = get_feature_names(model.named_steps['preprocessor'])
                
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Save to CSV
                feature_importance_df.to_csv(
                    os.path.join(args.output_dir, 'feature_importance', f'{name}_feature_importance.csv'),
                    index=False
                )
                
                # Plot
                plot_feature_importance(
                    feature_importance_df, 
                    model_name=name,
                    output_dir=os.path.join(args.output_dir, 'feature_importance')
                )
            
            # For linear models
            elif hasattr(classifier, 'coef_'):
                coefficients = classifier.coef_[0]
                feature_names = get_feature_names(model.named_steps['preprocessor'])
                
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                })
                feature_importance_df['Abs_Coefficient'] = feature_importance_df['Coefficient'].abs()
                feature_importance_df = feature_importance_df.sort_values('Abs_Coefficient', ascending=False)
                
                # Save to CSV
                feature_importance_df.to_csv(
                    os.path.join(args.output_dir, 'feature_importance', f'{name}_coefficients.csv'),
                    index=False
                )
                
                # Plot
                plot_feature_importance(
                    feature_importance_df, 
                    model_name=name,
                    output_dir=os.path.join(args.output_dir, 'feature_importance')
                )
    
    logger.info("Model evaluation completed successfully")
    return comparison, best_model_name