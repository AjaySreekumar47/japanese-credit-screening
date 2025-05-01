"""Data preprocessing utilities for the Japanese Credit Screening project."""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Configure logging
logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    Load the Japanese Credit Screening dataset.
    
    Args:
        data_path (str): Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded data or None if loading failed
    """
    logger.info(f"Loading data from {data_path}...")
    
    try:
        # Define column names based on the UCI documentation
        column_names = [f'A{i}' for i in range(1, 16)] + ['class']
        
        # Load the data
        data = pd.read_csv(data_path, header=None, names=column_names, na_values='?')
        
        # Convert class labels to binary (1 for '+' approved, 0 for '-' denied)
        data['class'] = data['class'].map({'+': 1, '-': 0})
        
        logger.info(f"Data loaded successfully with shape: {data.shape}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def identify_column_types(data):
    """
    Identify categorical and numerical columns in the dataset.
    
    Args:
        data (pandas.DataFrame): The dataset
        
    Returns:
        tuple: Lists of categorical and numerical column names
    """
    # Identify categorical and numerical columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target variable from feature lists if present
    if 'class' in num_cols:
        num_cols.remove('class')
    
    return cat_cols, num_cols

def preprocess_data(data, test_size=0.2, random_state=42, save_preprocessor=None):
    """
    Preprocess the data and split into training and testing sets.
    
    Args:
        data (pandas.DataFrame): The dataset
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        save_preprocessor (str, optional): Path to save the preprocessor
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    logger.info("Preprocessing data...")
    
    if data is None:
        logger.error("No data provided for preprocessing")
        return None, None, None, None, None
    
    try:
        # Identify column types
        cat_cols, num_cols = identify_column_types(data)
        logger.info(f"Categorical columns: {cat_cols}")
        logger.info(f"Numerical columns: {num_cols}")
        
        # Get features and target
        X = data.drop('class', axis=1)
        y = data['class']
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ],
            remainder='passthrough'
        )
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Fit the preprocessor on the training data
        preprocessor.fit(X_train)
        
        # Save the preprocessor if a path is provided
        if save_preprocessor:
            os.makedirs(os.path.dirname(save_preprocessor), exist_ok=True)
            joblib.dump(preprocessor, save_preprocessor)
            logger.info(f"Preprocessor saved to {save_preprocessor}")
        
        logger.info("Data preprocessing completed successfully")
        return X_train, X_test, y_train, y_test, preprocessor
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None, None, None, None, None

def get_processed_data(data_path, test_size=0.2, random_state=42, save_preprocessor=None):
    """
    Load and preprocess data in a single function.
    
    Args:
        data_path (str): Path to the data file
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        save_preprocessor (str, optional): Path to save the preprocessor
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    # Load the data
    data = load_data(data_path)
    
    # Preprocess the data
    return preprocess_data(data, test_size, random_state, save_preprocessor)

if __name__ == "__main__":
    # Example usage when run as a script
    from download import download_dataset
    
    # Download and preprocess the data
    data_path = download_dataset()
    if data_path:
        X_train, X_test, y_train, y_test, preprocessor = get_processed_data(
            data_path, 
            save_preprocessor='models/preprocessor.pkl'
        )
        print(f"Data processed successfully. Training set size: {X_train.shape[0]}")