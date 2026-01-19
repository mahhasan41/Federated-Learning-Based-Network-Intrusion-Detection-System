"""
Evaluation Module
Comprehensive evaluation metrics and visualizations
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import pickle

import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, global_model, device: str = 'cpu'):
        """
        Initialize evaluator
        
        Args:
            global_model: PyTorch model to evaluate
            device: 'cpu' or 'cuda'
        """
        self.model = global_model
        self.device = device
        self.results = {}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data
        
        Args:
            X: Input features
            
        Returns:
            (predictions, probabilities)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Comprehensive evaluation
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")
        
        predictions, probabilities = self.predict(X)
        
        # Basic metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        
        # ROC-AUC for binary classification
        roc_auc = None
        if len(np.unique(y)) == 2:
            try:
                roc_auc = roc_auc_score(y, probabilities[:, 1])
            except:
                roc_auc = None
        
        # Classification report
        class_report = classification_report(y, predictions, output_dict=True, zero_division=0)
        
        self.results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        if roc_auc is not None:
            logger.info(f"ROC-AUC:   {roc_auc:.4f}")
        
        return self.results
    
    def evaluate_by_class(self, X: np.ndarray, y: np.ndarray, 
                         class_names: list) -> Dict:
        """
        Per-class evaluation metrics
        
        Args:
            X: Test features
            y: Test labels
            class_names: Names of classes
            
        Returns:
            Per-class metrics
        """
        predictions, _ = self.predict(X)
        
        per_class_metrics = {}
        
        for class_idx, class_name in enumerate(class_names):
            mask = y == class_idx
            if mask.sum() == 0:
                continue
            
            class_acc = accuracy_score(y[mask], predictions[mask])
            class_support = mask.sum()
            
            per_class_metrics[class_name] = {
                'accuracy': float(class_acc),
                'support': int(class_support)
            }
        
        return per_class_metrics
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filepath}")


def evaluate_federated_clients(clients: list, X_test: np.ndarray, 
                               y_test: np.ndarray) -> Dict:
    """
    Evaluate each federated client on test set
    
    Args:
        clients: List of FederatedClient objects
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of per-client metrics
    """
    client_eval_results = {}
    
    for client in clients:
        evaluator = ModelEvaluator(client.model, device=client.device)
        metrics = evaluator.evaluate(X_test, y_test)
        client_eval_results[f'client_{client.client_id}'] = metrics
        
        logger.info(f"Client {client.client_id} - Accuracy: {metrics['accuracy']:.4f}")
    
    return client_eval_results


def generate_evaluation_report(training_results: Dict, eval_results: Dict,
                              class_names: list, output_file: str) -> None:
    """
    Generate comprehensive evaluation report
    
    Args:
        training_results: Results from federated training
        eval_results: Evaluation metrics
        class_names: List of class names
        output_file: Output file path
    """
    report = {
        'training_summary': {
            'num_rounds': len(training_results['history']['rounds']),
            'num_clients': len(training_results['clients']),
            'training_time': training_results['training_time'],
            'total_samples': sum(len(c.X_train) for c in training_results['clients'])
        },
        'final_metrics': eval_results,
        'convergence': {
            'rounds': training_results['history']['rounds'],
            'global_metrics': training_results['history']['global_metrics']
        }
    }
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    pass
