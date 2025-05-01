"""Data downloading utilities for the Japanese Credit Screening project."""

import os
import logging
from pathlib import Path
import requests

# Configure logging
logger = logging.getLogger(__name__)

def download_dataset(output_dir='data/raw', file_name='crx.data'):
    """
    Download the Japanese Credit Screening dataset from UCI Repository.
    
    Args:
        output_dir (str): Directory to save the downloaded data
        file_name (str): Name of the file to save
        
    Returns:
        str: Path to the downloaded file or None if download failed
    """
    logger.info("Downloading Japanese Credit Screening dataset...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URL to the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    
    # Full path for saving the file
    file_path = os.path.join(output_dir, file_name)
    
    try:
        # Check if file already exists
        if os.path.exists(file_path):
            logger.info(f"File already exists at {file_path}")
            return file_path
        
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Dataset successfully downloaded to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

def download_metadata(output_dir='data/raw', file_name='crx.names'):
    """
    Download the metadata file for the Japanese Credit Screening dataset.
    
    Args:
        output_dir (str): Directory to save the downloaded data
        file_name (str): Name of the file to save
        
    Returns:
        str: Path to the downloaded file or None if download failed
    """
    logger.info("Downloading dataset metadata...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URL to the metadata file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names"
    
    # Full path for saving the file
    file_path = os.path.join(output_dir, file_name)
    
    try:
        # Check if file already exists
        if os.path.exists(file_path):
            logger.info(f"File already exists at {file_path}")
            return file_path
        
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Metadata successfully downloaded to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error downloading metadata: {e}")
        return None

if __name__ == "__main__":
    # When run as a script, download both files
    download_dataset()
    download_metadata()