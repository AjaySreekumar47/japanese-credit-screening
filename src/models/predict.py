"""Prediction utilities for the Japanese Credit Screening project."""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        object: Loaded model or None if loading failed
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def predict(model, data, return_proba=False):
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model with predict method
        data (pandas.DataFrame): Data to make predictions on
        return_proba (bool): Whether to return probability estimates
        
    Returns:
        numpy.ndarray: Predictions or None if prediction failed
    """
    try:
        # Make predictions
        if return_proba and hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(data)
            logger.info(f"Made probability predictions with shape {predictions.shape}")
            return predictions
        else:
            predictions = model.predict(data)
            logger.info(f"Made predictions with shape {predictions.shape}")
            return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None

def batch_predict(model, data, batch_size=1000, return_proba=False):
    """
    Make predictions in batches to handle large datasets.
    
    Args:
        model: Trained model with predict method
        data (pandas.DataFrame): Data to make predictions on
        batch_size (int): Size of each batch
        return_proba (bool): Whether to return probability estimates
        
    Returns:
        numpy.ndarray: Predictions or None if prediction failed
    """
    try:
        # Calculate number of batches
        n_samples = len(data)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        logger.info(f"Making predictions in {n_batches} batches...")
        
        # Initialize list to store predictions
        predictions_list = []
        
        # Process each batch
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_data = data.iloc[start_idx:end_idx]
            
            # Make predictions for the batch
            if return_proba and hasattr(model, 'predict_proba'):
                batch_predictions = model.predict_proba(batch_data)
            else:
                batch_predictions = model.predict(batch_data)
            
            predictions_list.append(batch_predictions)
            
            logger.debug(f"Processed batch {i+1}/{n_batches}")
        
        # Combine predictions from all batches
        if return_proba and hasattr(model, 'predict_proba'):
            predictions = np.vstack(predictions_list)
        else:
            predictions = np.concatenate(predictions_list)
        
        logger.info(f"Completed batch predictions with shape {predictions.shape}")
        return predictions
    
    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        return None

def predict_with_explanation(model, data, feature_names=None):
    """
    Make predictions and provide explanations if possible.
    
    Args:
        model: Trained model
        data (pandas.DataFrame): Data to make predictions on
        feature_names (list, optional): Names of features
        
    Returns:
        dict: Dictionary with predictions and explanations
    """
    try:
        # Make predictions
        predictions = predict(model, data)
        
        result = {'predictions': predictions}
        
        # Try to get feature importance for explanation
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            
            # For tree-based models
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Create explanation with feature names if provided
                if feature_names is not None:
                    explanation = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                else:
                    explanation = pd.DataFrame({
                        'Feature_Index': range(len(importances)),
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                
                result['explanation'] = explanation
            
            # For linear models
            elif hasattr(classifier, 'coef_'):
                coefficients = classifier.coef_[0]
                
                # Create explanation with feature names if provided
                if feature_names is not None:
                    explanation = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coefficients
                    })
                    explanation['Abs_Coefficient'] = explanation['Coefficient'].abs()
                    explanation = explanation.sort_values('Abs_Coefficient', ascending=False)
                else:
                    explanation = pd.DataFrame({
                        'Feature_Index': range(len(coefficients)),
                        'Coefficient': coefficients
                    })
                    explanation['Abs_Coefficient'] = explanation['Coefficient'].abs()
                    explanation = explanation.sort_values('Abs_Coefficient', ascending=False)
                
                result['explanation'] = explanation
        
        return result
    
    except Exception as e:
        logger.error(f"Error making predictions with explanations: {e}")
        return {'predictions': None, 'explanation': None}

def predict_single_sample(model, sample_data, return_proba=True):
    """
    Make a prediction for a single sample.
    
    Args:
        model: Trained model
        sample_data (dict): Sample data as a dictionary
        return_proba (bool): Whether to return probability estimates
        
    Returns:
        dict: Prediction results
    """
    try:
        # Convert dictionary to DataFrame
        df = pd.DataFrame([sample_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get probability if requested
        probability = None
        if return_proba and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            probability = probabilities[1]  # Probability of positive class
        
        # Return results
        result = {
            'prediction': int(prediction),
            'approved': bool(prediction),
            'probability': float(probability) if probability is not None else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error making prediction for single sample: {e}")
        return None

if __name__ == "__main__":
    # Example usage when run as a script
    from ..data.download import download_dataset
    from ..data.preprocessing import get_processed_data
    
    # Download and process data
    data_path = download_dataset()
    if data_path:
        X_train, X_test, y_train, y_test, preprocessor = get_processed_data(data_path)
        
        # Load a trained model (assuming it exists)
        model_path = 'models/LogisticRegression_model.pkl'
        if os.path.exists(model_path):
            model = load_model(model_path)
            
            # Make predictions
            predictions = predict(model, X_test)
            print(f"Made predictions with shape: {predictions.shape}")
            
            # Example of a single prediction
            sample = X_test.iloc[0].to_dict()
            result = predict_single_sample(model, sample)
            print(f"Single sample prediction: {result}")