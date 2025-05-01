"""Feature engineering for the Japanese Credit Screening project."""

import logging
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def create_features(data):
    """
    Create additional features from the original dataset.
    
    Args:
        data (pandas.DataFrame): The original dataset
        
    Returns:
        pandas.DataFrame: Dataset with additional engineered features
    """
    logger.info("Creating additional features...")
    
    if data is None:
        logger.error("No data provided for feature engineering")
        return None
    
    try:
        # Make a copy of the data to avoid modifying the original
        enhanced_data = data.copy()
        
        # Example feature engineering (these would be based on domain knowledge)
        # Note: In a real project, you'd add meaningful features based on domain expertise
        # and the specific attributes in the Japanese Credit Screening dataset
        
        # Example 1: Create a feature for missing values count
        enhanced_data['missing_count'] = enhanced_data.isnull().sum(axis=1)
        
        # Example 2: If there are numerical features, you could create ratio features
        numeric_cols = enhanced_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove the target and newly created features from numeric columns
        if 'class' in numeric_cols:
            numeric_cols.remove('class')
        if 'missing_count' in numeric_cols:
            numeric_cols.remove('missing_count')
            
        # Create example ratio feature if at least two numeric columns exist
        if len(numeric_cols) >= 2:
            # This is just an example; in a real project, you'd choose financially meaningful ratios
            enhanced_data['ratio_feature'] = enhanced_data[numeric_cols[0]] / enhanced_data[numeric_cols[1]].replace(0, 0.001)
            
        logger.info(f"Created {len(enhanced_data.columns) - len(data.columns)} new features")
        return enhanced_data
    
    except Exception as e:
        logger.error(f"Error creating features: {e}")
        return data  # Return original data if feature creation fails

def get_feature_names(preprocessor):
    """
    Get feature names after preprocessing.
    
    Args:
        preprocessor (ColumnTransformer): The fitted preprocessor
        
    Returns:
        list: List of feature names after transformation
    """
    try:
        # Get feature names from the preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            return preprocessor.get_feature_names_out()
        
        # Fallback for older scikit-learn versions
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name != 'remainder':
                if hasattr(transformer, 'get_feature_names_out'):
                    trans_feature_names = transformer.get_feature_names_out(columns)
                else:
                    trans_feature_names = [f"{name}_{feature}" for feature in columns]
                feature_names.extend(trans_feature_names)
        
        return feature_names
    
    except Exception as e:
        logger.error(f"Error getting feature names: {e}")
        return []

def get_engineered_data(data_path, preprocessor_path=None):
    """
    Load data and perform feature engineering in a single step.
    
    Args:
        data_path (str): Path to the data file
        preprocessor_path (str, optional): Path to a saved preprocessor
        
    Returns:
        tuple: Enhanced data and preprocessor
    """
    from ..data.preprocessing import load_data
    import joblib
    
    # Load the data
    data = load_data(data_path)
    
    if data is None:
        return None, None
    
    # Create features
    enhanced_data = create_features(data)
    
    # Load the preprocessor if provided
    preprocessor = None
    if preprocessor_path:
        try:
            preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
    
    return enhanced_data, preprocessor

if __name__ == "__main__":
    # Example usage when run as a script
    from ..data.download import download_dataset
    from ..data.preprocessing import load_data
    
    # Download and process data
    data_path = download_dataset()
    if data_path:
        data = load_data(data_path)
        enhanced_data = create_features(data)
        print(f"Features created successfully. New dataset shape: {enhanced_data.shape}")