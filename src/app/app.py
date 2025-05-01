"""
Web application for the Japanese Credit Screening project.

This script creates a Flask web application that allows users to input
credit application data and get predictions from the trained model.
"""

import os
import sys
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
from datetime import datetime

# Add the parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# Load the model
MODEL_PATH = os.environ.get('MODEL_PATH', '../models/best_model.pkl')
model = None

def load_model():
    """Load the trained model."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load the model when the application starts
load_model()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on the input data."""
    if not model:
        return jsonify({
            'error': 'Model not loaded. Please check the logs.'
        }), 500
    
    try:
        # Get input data from the form
        data = {}
        for key, value in request.form.items():
            if key.startswith('A'):
                # Try to convert numerical values
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get probability
        probability = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            probability = probabilities[1]  # Probability of positive class
        
        # Create result
        result = {
            'approved': bool(prediction),
            'approval_probability': float(probability) if probability is not None else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions."""
    if not model:
        return jsonify({
            'error': 'Model not loaded. Please check the logs.'
        }), 500
    
    try:
        # Get input data from JSON
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get probability
        probability = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            probability = probabilities[1]  # Probability of positive class
        
        # Create result
        result = {
            'approved': bool(prediction),
            'approval_probability': float(probability) if probability is not None else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)