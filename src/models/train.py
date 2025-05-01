"""Model training utilities for the Japanese Credit Screening project."""

import os
import logging
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Configure logging
logger = logging.getLogger(__name__)

def get_model_configs():
    """
    Get configuration for different models.
    
    Returns:
        dict: Dictionary of model configurations
    """
    model_configs = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear']
            }
        }
    }
    
    return model_configs

def train_model(X_train, y_train, model_type='LogisticRegression', preprocessor=None, 
                tune_hyperparams=False, cv=5, output_dir=None):
    """
    Train a model on the given data.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        model_type (str): Type of model to train
        preprocessor (ColumnTransformer, optional): Preprocessor to use
        tune_hyperparams (bool): Whether to perform hyperparameter tuning
        cv (int): Number of cross-validation folds for hyperparameter tuning
        output_dir (str, optional): Directory to save the model
        
    Returns:
        object: Trained model
    """
    logger.info(f"Training {model_type} model...")
    
    try:
        # Get model configurations
        model_configs = get_model_configs()
        
        if model_type not in model_configs:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        # Get the model and parameters
        model_config = model_configs[model_type]
        model = model_config['model']
        
        # Create a pipeline with the preprocessor if provided
        if preprocessor:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
        else:
            pipeline = Pipeline([
                ('classifier', model)
            ])
        
        # Perform hyperparameter tuning if requested
        if tune_hyperparams:
            logger.info(f"Performing hyperparameter tuning for {model_type}...")
            
            # Prepare parameter grid for the pipeline
            param_grid = {f"classifier__{key}": value for key, value in model_config['params'].items()}
            
            # Create and fit the grid search
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get the best parameters and model
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            trained_model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best cross-validation score: {best_score:.4f}")
            
        else:
            # Train the model without hyperparameter tuning
            pipeline.fit(X_train, y_train)
            trained_model = pipeline
        
        # Save the model if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            model_filename = os.path.join(output_dir, f"{model_type}_model.pkl")
            joblib.dump(trained_model, model_filename)
            logger.info(f"Model saved to {model_filename}")
        
        return trained_model
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

def train_all_models(X_train, y_train, preprocessor=None, tune_hyperparams=False, 
                    cv=5, output_dir=None):
    """
    Train all available models on the given data.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        preprocessor (ColumnTransformer, optional): Preprocessor to use
        tune_hyperparams (bool): Whether to perform hyperparameter tuning
        cv (int): Number of cross-validation folds for hyperparameter tuning
        output_dir (str, optional): Directory to save the models
        
    Returns:
        dict: Dictionary of trained models
    """
    logger.info("Training all models...")
    
    try:
        # Get model configurations
        model_configs = get_model_configs()
        
        # Dictionary to store trained models
        trained_models = {}
        
        # Train each model
        for model_type in model_configs.keys():
            trained_models[model_type] = train_model(
                X_train, y_train, 
                model_type=model_type, 
                preprocessor=preprocessor,
                tune_hyperparams=tune_hyperparams,
                cv=cv,
                output_dir=output_dir
            )
        
        return trained_models
    
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return {}

if __name__ == "__main__":
    # Example usage when run as a script
    from ..data.download import download_dataset
    from ..data.preprocessing import get_processed_data
    
    # Download and process data
    data_path = download_dataset()
    if data_path:
        X_train, X_test, y_train, y_test, preprocessor = get_processed_data(
            data_path, 
            save_preprocessor='models/preprocessor.pkl'
        )
        
        # Train models
        models = train_all_models(
            X_train, y_train, 
            preprocessor=preprocessor,
            output_dir='models'
        )
        
        print(f"Trained {len(models)} models successfully")