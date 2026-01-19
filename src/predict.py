"""
Prediction and Inference Module
Real-time attack detection and classification
"""

import numpy as np
import logging
from typing import Dict, Tuple, List
import torch
from pathlib import Path
import json

from preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class IntrustionDetectionSystem:
    """Real-time network intrusion detection system"""
    
    def __init__(self, global_model, preprocessor: DataPreprocessor,
                 class_names: List[str], device: str = 'cpu'):
        """
        Initialize NIDS
        
        Args:
            global_model: Trained PyTorch model
            preprocessor: Fitted data preprocessor
            class_names: List of class names
            device: 'cpu' or 'cuda'
        """
        self.model = global_model
        self.preprocessor = preprocessor
        self.class_names = class_names
        self.device = device
        
        self.model.eval()
        logger.info(f"NIDS initialized with {len(class_names)} attack classes")
    
    def predict_single(self, traffic_features: np.ndarray) -> Dict:
        """
        Predict for single traffic sample
        
        Args:
            traffic_features: 1D array of feature values (must match training features)
            
        Returns:
            Dictionary with prediction details
        """
        # Preprocess the single sample
        X = traffic_features.reshape(1, -1)
        X = self.preprocessor.normalize_features(X, fit=False)
        
        # Make prediction
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
        
        # Determine if attack or normal
        is_attack = predicted_class != 0  # Assuming class 0 is 'Normal'
        
        prediction = {
            'predicted_class': str(self.class_names[predicted_class]),
            'predicted_class_id': int(predicted_class),
            'confidence': float(confidence),
            'is_attack': bool(is_attack),
            'all_probabilities': {
                str(self.class_names[i]): float(probabilities[i])
                for i in range(len(self.class_names))
            }
        }
        
        return prediction
    
    def predict_batch(self, traffic_data: np.ndarray) -> List[Dict]:
        """
        Predict for batch of traffic samples
        
        Args:
            traffic_data: 2D array of shape (n_samples, n_features)
            
        Returns:
            List of prediction dictionaries
        """
        # Preprocess batch
        X = self.preprocessor.normalize_features(traffic_data, fit=False)
        
        # Make predictions
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_classes = np.argmax(probabilities, axis=1)
        
        predictions = []
        for i in range(len(traffic_data)):
            pred_class = predicted_classes[i]
            confidence = float(probabilities[i, pred_class])
            is_attack = pred_class != 0
            
            prediction = {
                'sample_id': i,
                'predicted_class': str(self.class_names[pred_class]),
                'predicted_class_id': int(pred_class),
                'confidence': confidence,
                'is_attack': bool(is_attack),
                'all_probabilities': {
                    str(self.class_names[j]): float(probabilities[i, j])
                    for j in range(len(self.class_names))
                }
            }
            predictions.append(prediction)
        
        return predictions
    
    def predict_csv(self, csv_file: str, skip_label: bool = True) -> Dict:
        """
        Predict from CSV file
        
        Args:
            csv_file: Path to CSV file with traffic features
            skip_label: If True, skip 'Label' column if present
            
        Returns:
            Dictionary with batch predictions
        """
        logger.info(f"Reading traffic data from {csv_file}")
        
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        # Remove label column if present
        if skip_label and 'Label' in df.columns:
            df = df.drop('Label', axis=1)
        
        # Preprocess
        X = self.preprocessor.preprocess_inference_data(df)
        
        # Predict
        predictions = self.predict_batch(X)
        
        return {
            'num_samples': len(predictions),
            'predictions': predictions,
            'summary': self._get_summary(predictions)
        }
    
    def predict_manual_input(self, features_dict: Dict) -> Dict:
        """
        Predict from manually entered feature values
        
        Args:
            features_dict: Dictionary with feature names and values
            
        Returns:
            Prediction details
        """
        # Create array with features in correct order
        X = np.array([features_dict.get(feat, 0) for feat in self.preprocessor.feature_columns])
        
        return self.predict_single(X)
    
    def _get_summary(self, predictions: List[Dict]) -> Dict:
        """Get summary statistics from batch predictions"""
        attacks = sum(1 for p in predictions if p['is_attack'])
        normal = len(predictions) - attacks
        
        attack_types = {}
        for p in predictions:
            if p['is_attack']:
                attack_class = p['predicted_class']
                attack_types[attack_class] = attack_types.get(attack_class, 0) + 1
        
        avg_confidence_attack = np.mean([p['confidence'] for p in predictions if p['is_attack']]) if attacks > 0 else 0.0
        avg_confidence_normal = np.mean([p['confidence'] for p in predictions if not p['is_attack']]) if normal > 0 else 0.0
        
        return {
            'total_samples': len(predictions),
            'normal_traffic': normal,
            'attack_traffic': attacks,
            'attack_percentage': (attacks / len(predictions) * 100) if len(predictions) > 0 else 0.0,
            'attack_types': attack_types,
            'avg_confidence_normal': float(avg_confidence_normal),
            'avg_confidence_attack': float(avg_confidence_attack)
        }
    
    def detect_anomalies(self, predictions: List[Dict], 
                        confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Filter predictions with low confidence (potential anomalies)
        
        Args:
            predictions: List of predictions
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of low-confidence predictions
        """
        anomalies = [p for p in predictions if p['confidence'] < confidence_threshold]
        logger.info(f"Detected {len(anomalies)} anomalies (confidence < {confidence_threshold})")
        return anomalies


def save_predictions(predictions: Dict, output_file: str) -> None:
    """Save predictions to JSON file"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    pass
