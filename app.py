"""
Flask Web Interface for Federated Learning NIDS
Real-time attack detection and visualization dashboard
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import json
from pathlib import Path
import logging
from datetime import datetime

# Import project modules
import sys
sys.path.insert(0, './src')

from preprocessing import DataPreprocessor
from federated_server import FederatedServer
from predict import IntrustionDetectionSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global variables for loaded models
NIDS = None
PREPROCESSOR = None
CLASS_NAMES = None
FEATURE_COLUMNS = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    """Load trained global model and initialize NIDS"""
    global NIDS, PREPROCESSOR, CLASS_NAMES, FEATURE_COLUMNS
    
    logger.info("Loading trained model...")
    
    try:
        # Load preprocessor
        PREPROCESSOR = DataPreprocessor('./data/combinenew.csv')
        
        # Load trained model
        model_path = './models/global_model.pt'
        if not Path(model_path).exists():
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Please run training first: python src/train.py")
            return False
        
        # Create server and load model
        server = FederatedServer(input_size=50, num_classes=6, device=DEVICE)
        server.load_model(model_path)
        
        # Initialize NIDS
        CLASS_NAMES = ['BENIGN', 'DoS', 'DDoS', 'Brute Force', 'Web Attack', 'Botnet']
        NIDS = IntrustionDetectionSystem(
            global_model=server.global_model,
            preprocessor=PREPROCESSOR,
            class_names=CLASS_NAMES,
            device=DEVICE
        )
        
        # Load feature columns
        with open('./results/metadata.json', 'r') as f:
            metadata = json.load(f)
            # Feature columns would be stored here in production
        
        logger.info("✓ Model loaded successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status and configuration"""
    try:
        with open('./results/summary_report.json', 'r') as f:
            summary = json.load(f)
        
        with open('./results/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return jsonify({
            'status': 'ok',
            'model_loaded': NIDS is not None,
            'device': DEVICE,
            'model_accuracy': summary.get('model_accuracy', 0),
            'num_clients': metadata.get('num_clients', 0),
            'num_rounds': metadata.get('num_rounds', 0),
            'training_time': metadata.get('training_time', 0)
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on network traffic"""
    if NIDS is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        # Support both single and batch predictions
        if 'features' in data:
            # Single prediction with feature array
            features = np.array(data['features'], dtype=np.float32)
            
            if features.ndim == 1:
                prediction = NIDS.predict_single(features)
            else:
                predictions = NIDS.predict_batch(features)
                return jsonify({
                    'success': True,
                    'predictions': predictions,
                    'timestamp': datetime.now().isoformat()
                })
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            })
        
        elif 'traffic_samples' in data:
            # Batch prediction
            samples = np.array(data['traffic_samples'], dtype=np.float32)
            predictions = NIDS.predict_batch(samples)
            summary = NIDS._get_summary(predictions)
            
            return jsonify({
                'success': True,
                'total_samples': summary['total_samples'],
                'normal_traffic': summary['normal_traffic'],
                'attack_traffic': summary['attack_traffic'],
                'attack_percentage': summary['attack_percentage'],
                'attack_types': summary['attack_types'],
                'avg_confidence_normal': summary['avg_confidence_normal'],
                'avg_confidence_attack': summary['avg_confidence_attack'],
                'predictions': predictions[:10],  # Return first 10 for brevity
                'timestamp': datetime.now().isoformat()
            })
        
        else:
            return jsonify({'error': 'Invalid request format'}), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction from CSV data"""
    if NIDS is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save and predict
        import pandas as pd
        df = pd.read_csv(file)
        
        # Remove label column if present
        if 'Label' in df.columns:
            df = df.drop('Label', axis=1)
        
        # Preprocess
        X = PREPROCESSOR.preprocess_inference_data(df)
        
        # Predict
        predictions = NIDS.predict_batch(X)
        summary = NIDS._get_summary(predictions)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'sample_predictions': predictions[:20],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/features', methods=['GET'])
def get_features():
    """Get available features for prediction"""
    if PREPROCESSOR is None or PREPROCESSOR.feature_columns is None:
        return jsonify({'error': 'Preprocessor not initialized'}), 503
    
    return jsonify({
        'feature_count': len(PREPROCESSOR.feature_columns),
        'features': PREPROCESSOR.feature_columns,
        'classes': CLASS_NAMES
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get training history"""
    try:
        with open('./results/training_history.json', 'r') as f:
            history = json.load(f)
        
        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get evaluation metrics"""
    try:
        with open('./results/summary_report.json', 'r') as f:
            metrics = json.load(f)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Federated Learning NIDS Web Interface...")
    
    # Load model on startup
    if load_model():
        logger.info("✓ Ready to accept predictions")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model. Please train first.")
        logger.info("Run: python src/train.py")
