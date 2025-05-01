"""Model evaluation utilities for the Japanese Credit Screening project."""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# Configure logging
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, threshold=0.5, output_dir=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model with predict method
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        threshold (float): Probability threshold for binary classification
        output_dir (str, optional): Directory to save evaluation results
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    try:
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Apply threshold to probabilities if provided
            if threshold != 0.5:
                y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'cm': confusion_matrix(y_test, y_pred)
        }
        
        # Calculate ROC and AUC if probabilities are available
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_test, y_proba)
            metrics['fpr'], metrics['tpr'], metrics['thresholds'] = roc_curve(y_test, y_proba)
            metrics['average_precision'] = average_precision_score(y_test, y_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            metrics['precision_curve'] = precision
            metrics['recall_curve'] = recall
        
        # Log metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        if 'auc' in metrics:
            logger.info(f"AUC: {metrics['auc']:.4f}")
        
        # Save results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC' if 'auc' in metrics else None],
                'Value': [
                    metrics['accuracy'], 
                    metrics['precision'], 
                    metrics['recall'], 
                    metrics['f1'],
                    metrics.get('auc', None)
                ]
            }).dropna()
            
            metrics_df.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
            
            # Save classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
            
            # Plot and save confusion matrix
            plot_confusion_matrix(metrics['cm'], output_dir)
            
            # Plot and save ROC curve if available
            if 'auc' in metrics:
                plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'], output_dir)
                plot_precision_recall_curve(
                    metrics['precision_curve'], 
                    metrics['recall_curve'], 
                    metrics['average_precision'],
                    output_dir
                )
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None

def plot_confusion_matrix(cm, output_dir=None):
    """
    Plot and optionally save a confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        output_dir (str, optional): Directory to save the plot
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            plt.close()
        else:
            plt.show()
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")

def plot_roc_curve(fpr, tpr, auc_score, output_dir=None):
    """
    Plot and optionally save a ROC curve.
    
    Args:
        fpr (numpy.ndarray): False positive rates
        tpr (numpy.ndarray): True positive rates
        auc_score (float): Area under the ROC curve
        output_dir (str, optional): Directory to save the plot
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
            plt.close()
        else:
            plt.show()
    
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")

def plot_precision_recall_curve(precision, recall, avg_precision, output_dir=None):
    """
    Plot and optionally save a precision-recall curve.
    
    Args:
        precision (numpy.ndarray): Precision values
        recall (numpy.ndarray): Recall values
        avg_precision (float): Average precision score
        output_dir (str, optional): Directory to save the plot
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
            plt.close()
        else:
            plt.show()
    
    except Exception as e:
        logger.error(f"Error plotting precision-recall curve: {e}")

def compare_models(models_dict, X_test, y_test, output_dir=None):
    """
    Compare multiple models on test data.
    
    Args:
        models_dict (dict): Dictionary of trained models
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        output_dir (str, optional): Directory to save comparison results
        
    Returns:
        pandas.DataFrame: DataFrame of comparison metrics
    """
    try:
        logger.info("Comparing model performance...")
        
        # Initialize list to store results
        results = []
        
        # Evaluate each model
        for name, model in models_dict.items():
            if model is None:
                logger.warning(f"Model {name} is None, skipping evaluation")
                continue
                
            logger.info(f"Evaluating {name}...")
            
            # Create a subdirectory for this model's results if output_dir is provided
            model_dir = None
            if output_dir:
                model_dir = os.path.join(output_dir, name)
                os.makedirs(model_dir, exist_ok=True)
            
            # Evaluate the model
            metrics = evaluate_model(model, X_test, y_test, output_dir=model_dir)
            
            if metrics is None:
                logger.warning(f"Failed to evaluate {name}, skipping")
                continue
            
            # Add to results
            results.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'AUC': metrics.get('auc', np.nan)
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Save comparison to CSV if output directory is provided
        if output_dir:
            comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
            
            # Create comparison plots
            plot_model_comparison(comparison_df, metric='Accuracy', output_dir=output_dir)
            plot_model_comparison(comparison_df, metric='F1 Score', output_dir=output_dir)
            if not comparison_df['AUC'].isna().all():
                plot_model_comparison(comparison_df, metric='AUC', output_dir=output_dir)
        
        return comparison_df
    
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        return None

def plot_model_comparison(comparison_df, metric='Accuracy', output_dir=None):
    """
    Plot and optionally save a model comparison for a specific metric.
    
    Args:
        comparison_df (pandas.DataFrame): DataFrame with model comparison metrics
        metric (str): Metric to compare
        output_dir (str, optional): Directory to save the plot
    """
    try:
        if metric not in comparison_df.columns:
            logger.warning(f"Metric {metric} not found in comparison DataFrame")
            return
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y=metric, data=comparison_df)
        plt.title(f'Model Comparison - {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'model_comparison_{metric.lower().replace(" ", "_")}.png'))
            plt.close()
        else:
            plt.show()
    
    except Exception as e:
        logger.error(f"Error plotting model comparison: {e}")

def find_optimal_threshold(model, X_test, y_test, criterion='f1', output_dir=None):
    """
    Find the optimal probability threshold for binary classification.
    
    Args:
        model: Trained model with predict_proba method
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        criterion (str): Criterion to optimize ('f1', 'accuracy', 'precision', 'recall')
        output_dir (str, optional): Directory to save results
        
    Returns:
        float: Optimal threshold
    """
    try:
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not support probability predictions")
            return 0.5
        
        logger.info(f"Finding optimal threshold based on {criterion}...")
        
        # Get probability predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Define thresholds to test
        thresholds = np.arange(0.1, 1.0, 0.05)
        
        # Initialize lists to store results
        results = []
        
        # Evaluate each threshold
        for threshold in thresholds:
            # Apply threshold
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Add to results
            results.append({
                'Threshold': threshold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold based on criterion
        if criterion == 'f1':
            best_idx = results_df['F1 Score'].idxmax()
        elif criterion == 'accuracy':
            best_idx = results_df['Accuracy'].idxmax()
        elif criterion == 'precision':
            best_idx = results_df['Precision'].idxmax()
        elif criterion == 'recall':
            best_idx = results_df['Recall'].idxmax()
        else:
            logger.warning(f"Unknown criterion: {criterion}, using F1 Score")
            best_idx = results_df['F1 Score'].idxmax()
        
        optimal_threshold = results_df.loc[best_idx, 'Threshold']
        
        logger.info(f"Optimal threshold: {optimal_threshold:.2f} (criterion: {criterion})")
        
        # Save results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save threshold analysis to CSV
            results_df.to_csv(os.path.join(output_dir, 'threshold_analysis.csv'), index=False)
            
            # Plot and save threshold analysis
            plot_threshold_analysis(results_df, output_dir)
        
        return optimal_threshold
    
    except Exception as e:
        logger.error(f"Error finding optimal threshold: {e}")
        return 0.5

def plot_threshold_analysis(results_df, output_dir=None):
    """
    Plot and optionally save threshold analysis results.
    
    Args:
        results_df (pandas.DataFrame): DataFrame with threshold analysis results
        output_dir (str, optional): Directory to save the plot
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot metrics by threshold
        plt.plot(results_df['Threshold'], results_df['Accuracy'], label='Accuracy')
        plt.plot(results_df['Threshold'], results_df['Precision'], label='Precision')
        plt.plot(results_df['Threshold'], results_df['Recall'], label='Recall')
        plt.plot(results_df['Threshold'], results_df['F1 Score'], label='F1 Score')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics by Threshold')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'))
            plt.close()
        else:
            plt.show()
    
    except Exception as e:
        logger.error(f"Error plotting threshold analysis: {e}")

if __name__ == "__main__":
    # Example usage when run as a script
    from ..data.download import download_dataset
    from ..data.preprocessing import get_processed_data
    from .train import train_all_models
    
    # Download and process data
    data_path = download_dataset()
    if data_path:
        X_train, X_test, y_train, y_test, preprocessor = get_processed_data(data_path)
        
        # Train models
        models = train_all_models(X_train, y_train, preprocessor=preprocessor, output_dir='models')
        
        # Compare models
        if models:
            comparison = compare_models(models, X_test, y_test, output_dir='evaluation')
            print(comparison)