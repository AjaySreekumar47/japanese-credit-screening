"""Visualization utilities for the Japanese Credit Screening project."""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

# Configure logging
logger = logging.getLogger(__name__)

def plot_class_distribution(data, output_dir=None):
    """
    Plot the distribution of the target variable.
    
    Args:
        data (pandas.DataFrame): The dataset with a 'class' column
        output_dir (str, optional): Directory to save the plot
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='class', data=data)
        plt.title('Credit Approval Distribution')
        plt.xlabel('Approved (1) / Denied (0)')
        plt.ylabel('Count')
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
            plt.close()
        else:
            plt.show()
            
        # Get the count and percentage for each class
        class_counts = data['class'].value_counts()
        class_percentages = data['class'].value_counts(normalize=True) * 100
        
        # Create a DataFrame with the statistics
        stats_df = pd.DataFrame({
            'Count': class_counts,
            'Percentage': class_percentages
        })
        
        return stats_df
    
    except Exception as e:
        logger.error(f"Error plotting class distribution: {e}")
        return None

def plot_correlation_matrix(data, output_dir=None):
    """
    Plot a correlation matrix for numerical features.
    
    Args:
        data (pandas.DataFrame): The dataset
        output_dir (str, optional): Directory to save the plot
    """
    try:
        # Select only numerical columns
        numerical_data = data.select_dtypes(include=['int64', 'float64'])
        
        # Remove target variable if present
        if 'class' in numerical_data.columns:
            numerical_data = numerical_data.drop('class', axis=1)
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
            plt.close()
            
            # Save correlation matrix to CSV
            corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
        else:
            plt.show()
        
        return corr_matrix
    
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {e}")
        return None

def plot_feature_distributions(data, output_dir=None):
    """
    Plot distributions of numerical features.
    
    Args:
        data (pandas.DataFrame): The dataset
        output_dir (str, optional): Directory to save the plots
    """
    try:
        # Select only numerical columns
        numerical_data = data.select_dtypes(include=['int64', 'float64'])
        
        # Remove target variable if present
        if 'class' in numerical_data.columns:
            numerical_data = numerical_data.drop('class', axis=1)
        
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot histograms
        numerical_data.hist(figsize=(12, 10), bins=20)
        plt.suptitle('Histograms of Numerical Features')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'numerical_histograms.png'))
            plt.close()
        else:
            plt.show()
        
        # Plot box plots
        plt.figure(figsize=(12, 10))
        for i, col in enumerate(numerical_data.columns):
            plt.subplot(int(np.ceil(len(numerical_data.columns)/3)), 3, i+1)
            sns.boxplot(x=data['class'], y=data[col])
            plt.title(f'Distribution of {col} by Class')
            plt.xlabel('Approved (1) / Denied (0)')
            plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'numerical_boxplots.png'))
            plt.close()
        else:
            plt.show()
        
        # Create individual plots for each feature
        for col in numerical_data.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            sns.histplot(data=data, x=col, hue='class', ax=ax1, kde=True)
            ax1.set_title(f'Histogram of {col}')
            
            # Box plot
            sns.boxplot(x='class', y=col, data=data, ax=ax2)
            ax2.set_title(f'Box Plot of {col} by Class')
            ax2.set_xlabel('Approved (1) / Denied (0)')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'feature_{col}.png'))
                plt.close()
            else:
                plt.show()
        
        return True
    
    except Exception as e:
        logger.error(f"Error plotting feature distributions: {e}")
        return False

def plot_categorical_features(data, output_dir=None):
    """
    Plot distributions of categorical features.
    
    Args:
        data (pandas.DataFrame): The dataset
        output_dir (str, optional): Directory to save the plots
    """
    try:
        # Select only categorical columns (object type)
        categorical_data = data.select_dtypes(include=['object'])
        
        # If no categorical features, return
        if categorical_data.empty:
            logger.info("No categorical features found")
            return False
        
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot count plots for each categorical feature
        for col in categorical_data.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, hue='class', data=data)
            plt.title(f'Count of {col} by Class')
            plt.xticks(rotation=45)
            plt.legend(['Denied (0)', 'Approved (1)'])
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'categorical_{col}.png'))
                plt.close()
            else:
                plt.show()
        
        return True
    
    except Exception as e:
        logger.error(f"Error plotting categorical features: {e}")
        return False

def plot_feature_importance(feature_importance_df, model_name='Model', top_n=15, output_dir=None):
    """
    Plot feature importance.
    
    Args:
        feature_importance_df (pandas.DataFrame): DataFrame with feature importance data
        model_name (str): Name of the model
        top_n (int): Number of top features to display
        output_dir (str, optional): Directory to save the plot
    """
    try:
        # Get the top N features
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Check if we have coefficients (linear model)
        if 'Coefficient' in top_features.columns:
            # Sort by absolute value for better visualization
            top_features = top_features.reindex(
                top_features['Coefficient'].abs().sort_values(ascending=False).index
            )
            
            # Create a color map based on coefficient values (blue for positive, red for negative)
            colors = ['blue' if c > 0 else 'red' for c in top_features['Coefficient']]
            
            # Plot
            ax = sns.barplot(x='Coefficient', y='Feature', data=top_features, palette=colors)
            plt.title(f'Top {top_n} Feature Coefficients - {model_name}')
            
            # Add a zero line
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
        else:  # For tree-based models with feature importances
            # Sort by importance
            top_features = top_features.sort_values('Importance', ascending=False).head(top_n)
            
            # Plot
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title(f'Top {top_n} Feature Importance - {model_name}')
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'feature_importance_{model_name}.png'))
            plt.close()
        else:
            plt.show()
        
        return True
    
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")
        return False

def plot_pca_scatter(X, y, output_dir=None):
    """
    Plot PCA scatter plot for feature visualization.
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target variable
        output_dir (str, optional): Directory to save the plot
    """
    try:
        # Standardize features
        X_std = StandardScaler().fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)
        
        # Create a DataFrame for plotting
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df['class'] = y.values
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='class', data=pca_df, palette=['red', 'blue'], alpha=0.7)
        plt.title('PCA Scatter Plot')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend(['Denied (0)', 'Approved (1)'])
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'pca_scatter.png'))
            plt.close()
        else:
            plt.show()
        
        return pca_df
    
    except Exception as e:
        logger.error(f"Error plotting PCA scatter: {e}")
        return None

def create_dashboard(data, model_results=None, output_dir=None):
    """
    Create a comprehensive dashboard of visualizations.
    
    Args:
        data (pandas.DataFrame): The dataset
        model_results (dict, optional): Dictionary of model evaluation results
        output_dir (str, optional): Directory to save the plots
    """
    try:
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create a multi-panel figure
        fig = plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(3, 3)
        
        # 1. Class Distribution
        ax1 = plt.subplot(gs[0, 0])
        sns.countplot(x='class', data=data, ax=ax1)
        ax1.set_title('Credit Approval Distribution')
        ax1.set_xlabel('Approved (1) / Denied (0)')
        
        # 2. Correlation Matrix
        ax2 = plt.subplot(gs[0, 1:])
        numerical_data = data.select_dtypes(include=['int64', 'float64'])
        if 'class' in numerical_data.columns:
            numerical_data = numerical_data.drop('class', axis=1)
        corr_matrix = numerical_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax2)
        ax2.set_title('Correlation Matrix of Numerical Features')
        
        # 3. PCA Scatter
        ax3 = plt.subplot(gs[1, 0:2])
        X = data.drop('class', axis=1).select_dtypes(include=['int64', 'float64'])
        y = data['class']
        X_std = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df['class'] = y.values
        sns.scatterplot(x='PC1', y='PC2', hue='class', data=pca_df, palette=['red', 'blue'], alpha=0.7, ax=ax3)
        ax3.set_title('PCA Scatter Plot')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # 4. Model Performance (if provided)
        ax4 = plt.subplot(gs[1, 2])
        if model_results and 'comparison' in model_results:
            comparison_df = model_results['comparison']
            sns.barplot(x='Accuracy', y='Model', data=comparison_df, ax=ax4)
            ax4.set_title('Model Accuracy Comparison')
        else:
            ax4.set_title('Model Performance (No Data)')
            ax4.axis('off')
        
        # 5-6. Feature Distributions (select a couple of interesting features)
        if len(numerical_data.columns) >= 1:
            ax5 = plt.subplot(gs[2, 0])
            feature1 = numerical_data.columns[0]
            sns.histplot(data=data, x=feature1, hue='class', kde=True, ax=ax5)
            ax5.set_title(f'Distribution of {feature1}')
        
        if len(numerical_data.columns) >= 2:
            ax6 = plt.subplot(gs[2, 1])
            feature2 = numerical_data.columns[1]
            sns.histplot(data=data, x=feature2, hue='class', kde=True, ax=ax6)
            ax6.set_title(f'Distribution of {feature2}')
        
        # 7. Feature Importance (if provided)
        ax7 = plt.subplot(gs[2, 2])
        if model_results and 'feature_importance' in model_results:
            feature_importance = model_results['feature_importance']
            top_features = feature_importance.head(10)
            sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax7)
            ax7.set_title('Top 10 Feature Importance')
        else:
            ax7.set_title('Feature Importance (No Data)')
            ax7.axis('off')
        
        plt.tight_layout()
        plt.suptitle('Japanese Credit Screening - Data Insights Dashboard', fontsize=16, y=1.02)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'dashboard.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        return False

if __name__ == "__main__":
    # Example usage when run as a script
    from ..data.download import download_dataset
    from ..data.preprocessing import load_data
    
    # Download and load data
    data_path = download_dataset()
    if data_path:
        data = load_data(data_path)
        
        # Create visualizations
        plot_class_distribution(data, output_dir='visualizations')
        plot_correlation_matrix(data, output_dir='visualizations')
        plot_feature_distributions(data, output_dir='visualizations')
        plot_categorical_features(data, output_dir='visualizations')
        
        # Create a simple dashboard
        create_dashboard(data, output_dir='visualizations')